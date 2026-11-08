"""
Validação experimental do artefato por injeção sintética de viés (DSR — fase de avaliação).

Protocolo:
  1. Reproduz o split treino/teste de 03_trainer.py (80/20, stratify=y, random_state=42).
  2. Para um grupo-alvo (Parda ou Indígena), "rebaixa" (1 -> 0) uma fração p dos rótulos
     positivos DO TREINO desse grupo, simulando discriminação histórica nos rótulos.
     p ∈ {0%, 10%, 25%, 50%}.
  3. Retreina o pipeline com os MESMOS hiperparâmetros do modelo de produção
     (sem novo GridSearch, para isolar o efeito da injeção).
  4. Audita no conjunto de TESTE limpo (rótulos originais) com o mesmo motor de
     fairness de 04_audit_engine.py: DI, TPR/FPR e SHAP do grupo-alvo.
  5. Controle negativo: o mesmo número de rótulos é rebaixado aleatoriamente entre
     TODOS os candidatos positivos (sem foco em grupo). O DI do grupo-alvo não deve
     cair de forma sistemática.

Hipótese de sensibilidade: no cenário "injection", o DI do grupo-alvo decresce de forma
monotônica com p, e o SHAP da variável de raça do grupo fica mais negativo; no cenário
"control", ambos permanecem próximos da linha de base.

Saídas:
  data/outputs/bias_injection_runs.parquet     (uma linha por execução/repetição)
  data/outputs/bias_injection_summary.parquet  (média, desvio-padrão e variação vs. p=0)
"""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import clone
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "data" / "outputs"

# Reutiliza o motor de fairness oficial para garantir que o indicador validado
# seja exatamente o mesmo exibido pela ferramenta.
_spec = spec_from_file_location("audit_engine", Path(__file__).resolve().parent / "04_audit_engine.py")
audit_engine = module_from_spec(_spec)
_spec.loader.exec_module(audit_engine)

FEATURE_COLS = audit_engine.FEATURE_COLS
TARGET_COL = audit_engine.TARGET_COL
TEST_SIZE = audit_engine.TEST_SIZE
RANDOM_STATE = audit_engine.RANDOM_STATE

YEARS = [2019, 2020, 2021]
RACE_COL = "TP_COR_RACA"
TARGET_GROUPS = {"3": "Parda", "5": "Indígena"}
INTENSITIES = [0.00, 0.10, 0.25, 0.50]
SCENARIOS = {"injection": 0, "control": 1}
N_REPEATS = 5


def run_seed(year: int, group: str, intensity: float, scenario: str, rep: int) -> np.random.Generator:
    """Gerador determinístico derivado da semente-mestra 42 (reprodutível por execução)."""
    return np.random.default_rng(
        [RANDOM_STATE, year, int(group), int(round(intensity * 100)), SCENARIOS[scenario], rep]
    )


def inject_label_bias(X_train, y_train, group, intensity, scenario, rng):
    """
    Devolve uma cópia de y_train com rótulos positivos rebaixados para 0.
    - injection: rebaixa round(p * N_pos_grupo) positivos do grupo-alvo;
    - control:   rebaixa a MESMA quantidade entre todos os positivos (qualquer grupo).
    """
    race = X_train[RACE_COL].map(audit_engine.normalize_group_label)
    target_pos_idx = y_train.index[(race == group) & (y_train == 1)]
    n_flip = int(round(intensity * len(target_pos_idx)))

    if scenario == "injection":
        pool = target_pos_idx
    else:
        pool = y_train.index[y_train == 1]

    y_biased = y_train.copy()
    if n_flip > 0:
        flip_idx = rng.choice(pool.to_numpy(), size=n_flip, replace=False)
        y_biased.loc[flip_idx] = 0
    return y_biased, n_flip, len(target_pos_idx)


def race_dummy_column(feature_names, group: str) -> str:
    prefix = f"cat__{RACE_COL}_"
    for name in feature_names:
        if name.startswith(prefix):
            suffix = name[len(prefix):]
            try:
                if audit_engine.normalize_group_label(float(suffix)) == group:
                    return name
            except ValueError:
                continue
    raise KeyError(f"Coluna one-hot do grupo {group} não encontrada.")


def evaluate_run(pipeline, X_test, y_test, group, year) -> dict:
    """Audita o modelo retreinado no teste limpo, com o motor oficial de fairness."""
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    X_trans = preprocessor.transform(X_test)
    feature_names = list(preprocessor.get_feature_names_out())

    y_pred = classifier.predict(X_trans)
    y_proba = classifier.predict_proba(X_trans)[:, 1]

    df_eval = X_test.copy()
    df_eval["y_true"] = y_test.to_numpy()
    df_eval["y_pred"] = y_pred
    fairness = audit_engine.calculate_fairness_metrics(df_eval, year)
    race_rows = fairness[fairness["protected_attribute"] == RACE_COL].set_index("group")
    target = race_rows.loc[group]

    contribs = classifier.get_booster().predict(
        xgb.DMatrix(X_trans, feature_names=feature_names), pred_contribs=True
    )
    dummy_idx = feature_names.index(race_dummy_column(feature_names, group))
    phi = contribs[:, dummy_idx]
    is_member = X_trans[:, dummy_idx] == 1

    return {
        "test_group_n": int(target["total_count"]),
        "selection_rate": float(target["selection_rate"]),
        "reference_group": str(target["reference_group"]),
        "disparate_impact": float(target["disparate_impact"]),
        "alert_di_low": bool(target["alert_di_low"]),
        "tpr": float(target["tpr"]),
        "fpr": float(target["fpr"]),
        "tpr_ratio": float(target["tpr_ratio"]),
        "shap_mean_abs_group_dummy": float(np.abs(phi).mean()),
        "shap_mean_group_members": float(phi[is_member].mean()),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }


def run_injection_for_year(year: int) -> pd.DataFrame:
    data_file = PROCESSED_DATA_DIR / f"sampled_{year}.parquet"
    model_file = MODELS_DIR / f"xgboost_{year}.joblib"
    if not data_file.exists() or not model_file.exists():
        print(f"[AVISO] Arquivos do ano {year} ausentes. Pule.")
        return pd.DataFrame()

    print(f"\n--- Injeção sintética de viés - ENEM {year} ---")
    df = pd.read_parquet(data_file)
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # Mesmo pipeline e hiperparâmetros do modelo de produção, sem ajuste prévio
    template = clone(joblib.load(model_file))

    records = []
    for group, label in TARGET_GROUPS.items():
        for intensity in INTENSITIES:
            for scenario in SCENARIOS:
                for rep in range(N_REPEATS):
                    rng = run_seed(year, group, intensity, scenario, rep)
                    y_biased, n_flip, n_pos = inject_label_bias(
                        X_train, y_train, group, intensity, scenario, rng
                    )
                    model = clone(template).fit(X_train, y_biased)
                    metrics = evaluate_run(model, X_test, y_test, group, year)
                    records.append({
                        "year": year,
                        "eval_split": "test",
                        "target_group": group,
                        "target_label": label,
                        "scenario": scenario,
                        "intensity": intensity,
                        "repeat": rep,
                        "n_labels_flipped": n_flip,
                        "n_group_train_positives": n_pos,
                        **metrics,
                    })
        inj = [r for r in records if r["target_group"] == group and r["scenario"] == "injection"]
        means = pd.DataFrame(inj).groupby("intensity")["disparate_impact"].mean().round(3).to_dict()
        print(f"  {label:9s} DI médio (injection) por intensidade: {means}")
    return pd.DataFrame(records)


def summarize(df_runs: pd.DataFrame) -> pd.DataFrame:
    keys = ["year", "target_group", "target_label", "scenario", "intensity"]
    metrics = ["disparate_impact", "tpr", "fpr", "shap_mean_group_members",
               "shap_mean_abs_group_dummy", "accuracy", "roc_auc"]
    summary = df_runs.groupby(keys)[metrics].agg(["mean", "std"])
    summary.columns = [f"{m}_{stat}" for m, stat in summary.columns]
    summary = summary.reset_index()
    summary["n_labels_flipped"] = df_runs.groupby(keys)["n_labels_flipped"].first().to_numpy()
    summary["alert_rate"] = df_runs.groupby(keys)["alert_di_low"].mean().to_numpy()

    baseline = (summary[summary["intensity"] == 0.0]
                .set_index(["year", "target_group", "scenario"])["disparate_impact_mean"])
    summary["di_delta_vs_baseline"] = summary.apply(
        lambda r: r["disparate_impact_mean"] - baseline.loc[(r["year"], r["target_group"], r["scenario"])],
        axis=1,
    )

    # Efeito líquido = Δ(injection) − Δ(control): isola o efeito do viés DIRECIONADO ao grupo
    # do efeito genérico de reduzir a prevalência global de positivos (deslocamento de limiar).
    control_delta = (summary[summary["scenario"] == "control"]
                     .set_index(["year", "target_group", "intensity"])["di_delta_vs_baseline"])
    summary["di_net_effect"] = summary.apply(
        lambda r: (r["di_delta_vs_baseline"] - control_delta.loc[(r["year"], r["target_group"], r["intensity"])])
        if r["scenario"] == "injection" else np.nan,
        axis=1,
    )
    return summary


def monotonicity_report(summary: pd.DataFrame) -> pd.DataFrame:
    """Verifica se o DI médio decresce estritamente com a intensidade em cada cenário."""
    rows = []
    for (year, group, label, scenario), g in summary.groupby(
        ["year", "target_group", "target_label", "scenario"]
    ):
        di = g.sort_values("intensity")["disparate_impact_mean"].to_numpy()
        rows.append({
            "year": year, "target_label": label, "scenario": scenario,
            "di_p0": di[0], "di_p50": di[-1], "delta_p50": di[-1] - di[0],
            "strictly_decreasing": bool(np.all(np.diff(di) < 0)),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    all_runs = pd.concat([run_injection_for_year(yr) for yr in YEARS], ignore_index=True)
    if all_runs.empty:
        raise SystemExit("[ERRO] Nenhum ano processado.")

    df_summary = summarize(all_runs)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    all_runs.to_parquet(OUTPUTS_DIR / "bias_injection_runs.parquet", engine="pyarrow", index=False)
    df_summary.to_parquet(OUTPUTS_DIR / "bias_injection_summary.parquet", engine="pyarrow", index=False)

    print("\n=== Curva de sensibilidade (DI médio do grupo-alvo) ===")
    with pd.option_context("display.width", 200):
        print(monotonicity_report(df_summary).round(3).to_string(index=False))
    print(f"\n[SUCESSO] {len(all_runs)} execuções salvas em {OUTPUTS_DIR / 'bias_injection_runs.parquet'}")
    print(f"[SUCESSO] Resumo salvo em {OUTPUTS_DIR / 'bias_injection_summary.parquet'}")
