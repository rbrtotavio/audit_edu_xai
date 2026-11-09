from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# Diretórios base
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "data" / "outputs"

# Atributos do Data Contract
FEATURE_COLS = ["TP_COR_RACA", "Q006", "Q001", "Q002", "Q024", "Q025"]
PROTECTED_ATTRIBUTES = ["TP_COR_RACA", "Q006"]
TARGET_COL = "target"

# Parâmetros do split — DEVEM ser idênticos aos de src/03_trainer.py
TEST_SIZE = 0.20
RANDOM_STATE = 42
DI_THRESHOLD = 0.80  # Regra dos 80% (EEOC)
BOOTSTRAP_B = 2000   # Réplicas do bootstrap do teste (ICs 95% percentílicos)


def reproduce_test_split(df: pd.DataFrame, seed: int = RANDOM_STATE):
    """
    Reproduz exatamente o split treino/teste de 03_trainer.py e devolve
    apenas a partição de TESTE (20%). Os índices originais da amostra
    (sampled_{ano}.parquet) são preservados para rastreabilidade (row_id).
    """
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=seed
    )
    return X_test, y_test


def normalize_group_label(value) -> str:
    """Converte códigos numéricos (ex.: 1.0) em rótulos inteiros ('1')."""
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _safe_rate(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else float("nan")


def draw_bootstrap_indices(n: int, b: int, seed: int = RANDOM_STATE, chunk: int = 250) -> np.ndarray:
    """
    Sorteia B reamostragens com reposição de n linhas (matriz B x n de índices, int32).
    As mesmas réplicas são usadas para todos os atributos protegidos do ano.
    """
    rng = np.random.default_rng(seed)
    out = np.empty((b, n), dtype=np.int32)
    for start in range(0, b, chunk):
        stop = min(start + chunk, b)
        out[start:stop] = rng.integers(0, n, size=(stop - start, n), dtype=np.int32)
    return out


def bootstrap_group_counts(codes: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
                           n_groups: int, boot_idx: np.ndarray, chunk: int = 250) -> np.ndarray:
    """
    Para cada réplica, conta as observações por (grupo, y_true, y_pred) com um único
    np.bincount vetorizado. Retorna um array (B, n_groups, 4) em que o último eixo é
    a célula 2*y_true + y_pred: 0 = (y=0, ŷ=0), 1 = (y=0, ŷ=1), 2 = (y=1, ŷ=0), 3 = (y=1, ŷ=1).
    """
    key = (codes.astype(np.int64) * 4 + 2 * y_true.astype(np.int64) + y_pred.astype(np.int64))
    width = n_groups * 4
    b = boot_idx.shape[0]
    counts = np.empty((b, width), dtype=np.int64)
    for start in range(0, b, chunk):
        stop = min(start + chunk, b)
        offsets = (np.arange(stop - start, dtype=np.int64) * width)[:, None]
        flat = (key[boot_idx[start:stop]] + offsets).ravel()
        counts[start:stop] = np.bincount(flat, minlength=(stop - start) * width).reshape(stop - start, width)
    return counts.reshape(b, n_groups, 4)


def _nan_percentiles(values: np.ndarray) -> tuple:
    """Percentis 2,5 e 97,5 ignorando NaN (réplicas em que a taxa é indefinida)."""
    finite = values[~np.isnan(values)]
    if finite.size == 0:
        return float("nan"), float("nan")
    low, high = np.percentile(finite, [2.5, 97.5])
    return float(low), float(high)


def bootstrap_fairness_ci(codes: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
                          n_groups: int, ref_code: int, boot_idx: np.ndarray) -> list:
    """
    Intervalos de confiança percentílicos (95%) por bootstrap não paramétrico do conjunto
    de TESTE, com o modelo FIXO (mede apenas a variabilidade amostral do teste, não a do
    treinamento). A identidade do grupo de referência é mantida fixa (a observada na
    amostra original); em cada réplica DI_g = TS_g / TS_ref.
    Réplicas em que uma taxa é indefinida (denominador 0) são ignoradas (NaN).
    """
    counts = bootstrap_group_counts(codes, y_true, y_pred, n_groups, boot_idx).astype(np.float64)
    n_g = counts.sum(axis=2)
    sel = counts[:, :, 1] + counts[:, :, 3]
    pos = counts[:, :, 2] + counts[:, :, 3]
    neg = counts[:, :, 0] + counts[:, :, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        sr = np.where(n_g > 0, sel / n_g, np.nan)
        tpr = np.where(pos > 0, counts[:, :, 3] / pos, np.nan)
        fpr = np.where(neg > 0, counts[:, :, 1] / neg, np.nan)
        ref_sr = sr[:, [ref_code]]
        di = np.where(ref_sr > 0, sr / ref_sr, np.nan)
        # DI dos rótulos com a MESMA referência fixa (grupo de maior taxa de seleção do
        # modelo na amostra original) e as MESMAS réplicas: P(y=1|g) / P(y=1|ref)
        br = np.where(n_g > 0, pos / n_g, np.nan)
        ref_br = br[:, [ref_code]]
        label_di = np.where(ref_br > 0, br / ref_br, np.nan)
        amp = di - label_di  # NaN se qualquer um dos dois for indefinido

    results = []
    for k in range(n_groups):
        di_k = di[:, k]
        valid = ~np.isnan(di_k)
        ldi_k = label_di[:, k]
        lvalid = ~np.isnan(ldi_k)
        results.append({
            "di_ci": _nan_percentiles(di_k),
            "tpr_ci": _nan_percentiles(tpr[:, k]),
            "fpr_ci": _nan_percentiles(fpr[:, k]),
            # Fração das réplicas válidas (DI definido) com DI abaixo do limiar de 0,8
            "p_di_below": float((di_k[valid] < DI_THRESHOLD).mean()) if valid.any() else float("nan"),
            "label_di_ci": _nan_percentiles(ldi_k),
            "p_label_di_below": float((ldi_k[lvalid] < DI_THRESHOLD).mean()) if lvalid.any() else float("nan"),
            "amp_ci": _nan_percentiles(amp[:, k]),
        })
    return results


def calculate_fairness_metrics(df_eval: pd.DataFrame, year: int, bootstrap_b: int = 0,
                               seed: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Calcula, por subgrupo de cada atributo protegido (somente no conjunto de teste):
      - Taxa de seleção P(ŷ=1 | G=g) e Impacto Disparate (DI) — Paridade Demográfica;
      - Taxa base real P(y=1 | G=g) e DI dos rótulos (para medir amplificação do viés);
      - TPR = P(ŷ=1 | y=1, G=g) e FPR = P(ŷ=1 | y=0, G=g) — Equal Opportunity / Equalized Odds.
    O grupo de referência é o subgrupo de maior taxa de seleção.

    bootstrap_b > 0 acrescenta ICs 95% por bootstrap (B réplicas, np.random.default_rng(seed))
    do DI, TPR, FPR, DI dos rótulos e di_amplification (todos das mesmas réplicas) e o
    nº de rótulos negativos por grupo. Com bootstrap_b = 0 (padrão, usado por
    05_bias_injection.py) o esquema e os valores de saída são exatamente os originais.
    """
    records = []
    boot_idx = None
    if bootstrap_b > 0:
        boot_idx = draw_bootstrap_indices(len(df_eval), bootstrap_b, seed)
        y_true_all = df_eval["y_true"].to_numpy().astype(np.int64)
        y_pred_all = df_eval["y_pred"].to_numpy().astype(np.int64)

    for attr in PROTECTED_ATTRIBUTES:
        rows = []
        for group_val, g in df_eval.groupby(attr, observed=True):
            n = len(g)
            pos_true = g["y_true"] == 1
            neg_true = ~pos_true
            rows.append({
                "group_raw": group_val,
                "total_count": n,
                "positive_count": int(g["y_pred"].sum()),
                "selection_rate": _safe_rate(g["y_pred"].sum(), n),
                "true_positive_label_count": int(pos_true.sum()),
                "base_rate": _safe_rate(pos_true.sum(), n),
                "tpr": _safe_rate(g.loc[pos_true, "y_pred"].sum(), pos_true.sum()),
                "fpr": _safe_rate(g.loc[neg_true, "y_pred"].sum(), neg_true.sum()),
            })
        grouped = pd.DataFrame(rows)

        ref = grouped.loc[grouped["selection_rate"].idxmax()]
        ref_rate = ref["selection_rate"]
        ref_base_rate = ref["base_rate"]
        ref_tpr = ref["tpr"]
        ref_fpr = ref["fpr"]

        boot = None
        if boot_idx is not None:
            # Códigos 0..G-1 na mesma ordem de `grouped` (ordem do groupby)
            code_of = {v: k for k, v in enumerate(grouped["group_raw"])}
            codes = df_eval[attr].map(code_of).to_numpy()
            if pd.isna(codes).any():
                raise ValueError(f"Valores de {attr} sem grupo correspondente no bootstrap.")
            boot = bootstrap_fairness_ci(codes.astype(np.int64), y_true_all, y_pred_all,
                                         len(grouped), code_of[ref["group_raw"]], boot_idx)

        for k, (_, row) in enumerate(grouped.iterrows()):
            di = _safe_rate(row["selection_rate"], ref_rate) if ref_rate > 0 else 1.0
            label_di = _safe_rate(row["base_rate"], ref_base_rate)
            rec = {
                "year": year,
                "eval_split": "test",
                "protected_attribute": attr,
                "group": normalize_group_label(row["group_raw"]),
                "total_count": int(row["total_count"]),
                "positive_count": int(row["positive_count"]),
                "selection_rate": float(row["selection_rate"]),
                "reference_group": normalize_group_label(ref["group_raw"]),
                "reference_selection_rate": float(ref_rate),
                "disparate_impact": float(di),
                "alert_di_low": bool(di < DI_THRESHOLD),
                # Viés presente nos rótulos reais (para comparar com o DI do modelo)
                "true_positive_label_count": int(row["true_positive_label_count"]),
                "base_rate": float(row["base_rate"]),
                "label_disparate_impact": float(label_di),
                "di_amplification": float(di - label_di),  # < 0 => modelo amplia a disparidade
                # Equal Opportunity / Equalized Odds
                "tpr": float(row["tpr"]),
                "fpr": float(row["fpr"]),
                "reference_tpr": float(ref_tpr),
                "reference_fpr": float(ref_fpr),
                "tpr_ratio": _safe_rate(row["tpr"], ref_tpr),
                "equal_opportunity_diff": float(row["tpr"] - ref_tpr),
                "fpr_diff": float(row["fpr"] - ref_fpr),
            }
            if boot is not None:
                b_k = boot[k]
                rec.update({
                    "negative_label_count": int(row["total_count"] - row["true_positive_label_count"]),
                    "di_ci_low": b_k["di_ci"][0],
                    "di_ci_high": b_k["di_ci"][1],
                    "tpr_ci_low": b_k["tpr_ci"][0],
                    "tpr_ci_high": b_k["tpr_ci"][1],
                    "fpr_ci_low": b_k["fpr_ci"][0],
                    "fpr_ci_high": b_k["fpr_ci"][1],
                    "p_di_below_threshold": b_k["p_di_below"],
                    "bootstrap_b": int(bootstrap_b),
                    # Acrescentadas depois (I1, parecer nº 04): IC95 do DI dos rótulos e da
                    # amplificação, das mesmas réplicas e com a mesma referência fixa
                    "label_di_ci_low": b_k["label_di_ci"][0],
                    "label_di_ci_high": b_k["label_di_ci"][1],
                    "p_label_di_below_threshold": b_k["p_label_di_below"],
                    "di_amplification_ci_low": b_k["amp_ci"][0],
                    "di_amplification_ci_high": b_k["amp_ci"][1],
                })
            records.append(rec)

    return pd.DataFrame(records)


def calculate_shap_feature_summary(shap_matrix: np.ndarray, X_trans: np.ndarray,
                                   feature_names: list, year: int) -> pd.DataFrame:
    """
    Resumo SHAP por coluna One-Hot (teste), comparável entre edições:
      - mean_abs_phi: média de |φ| (importância global usual; depende da prevalência);
      - prevalence: média da dummy no teste;
      - mean_phi_when_1 / mean_phi_when_0: média de φ entre quem tem / não tem a categoria;
      - conditional_gap = mean_phi_when_0 − mean_phi_when_1 (positivo ⇒ a categoria reduz o escore);
      - rank_mean_abs: posição pela média de |φ| (1 = maior).
    """
    X = np.asarray(X_trans, dtype=np.float64)
    phi = np.asarray(shap_matrix, dtype=np.float64)
    is1 = X == 1
    n1 = is1.sum(axis=0)
    n0 = len(X) - n1
    sum1 = np.where(is1, phi, 0.0).sum(axis=0)
    sum0 = phi.sum(axis=0) - sum1
    with np.errstate(divide="ignore", invalid="ignore"):
        mean1 = np.where(n1 > 0, sum1 / n1, np.nan)
        mean0 = np.where(n0 > 0, sum0 / n0, np.nan)
    df = pd.DataFrame({
        "year": year,
        "feature": feature_names,
        "mean_abs_phi": np.abs(phi).mean(axis=0),
        "prevalence": X.mean(axis=0),
        "n_when_1": n1.astype(np.int64),
        "mean_phi_when_1": mean1,
        "mean_phi_when_0": mean0,
        "conditional_gap": mean0 - mean1,
    })
    df["rank_mean_abs"] = df["mean_abs_phi"].rank(ascending=False, method="min").astype(int)
    return df.sort_values("rank_mean_abs").reset_index(drop=True)


def calculate_model_metrics(y_true, y_pred, y_proba, pipeline, year: int, base_value: float) -> pd.DataFrame:
    """Métricas de desempenho do modelo-alvo no conjunto de teste (persistidas para o texto)."""
    clf_params = pipeline.named_steps["classifier"].get_params()
    return pd.DataFrame([{
        "year": year,
        "eval_split": "test",
        "n_test": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "n_estimators": int(clf_params["n_estimators"]),
        "max_depth": int(clf_params["max_depth"]),
        "learning_rate": float(clf_params["learning_rate"]),
        "shap_base_value": float(base_value),
    }])


def run_audit_for_year(year: int) -> None:
    """
    Executa a auditoria em lote (Fairness + SHAP) para um ano, EXCLUSIVAMENTE
    sobre o conjunto de teste (20%), exportando Parquets estáticos.
    """
    data_file = PROCESSED_DATA_DIR / f"sampled_{year}.parquet"
    model_file = MODELS_DIR / f"xgboost_{year}.joblib"

    if not data_file.exists() or not model_file.exists():
        print(f"[AVISO] Arquivos do ano {year} ausentes ({data_file.name} ou {model_file.name}). Pule.")
        return

    print(f"\n--- Iniciando Motor de Auditoria (SHAP + Fairness) - ENEM {year} ---")

    df = pd.read_parquet(data_file)
    pipeline = joblib.load(model_file)

    X_test, y_test = reproduce_test_split(df)
    print(f"Conjunto de teste reproduzido: {len(X_test):,} de {len(df):,} registros "
          f"(test_size={TEST_SIZE}, random_state={RANDOM_STATE}).")

    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    # Transformação One-Hot Encoding
    X_trans = preprocessor.transform(X_test)
    feature_names = list(preprocessor.get_feature_names_out())

    y_true = y_test.to_numpy()
    y_pred = classifier.predict(X_trans)
    y_proba = classifier.predict_proba(X_trans)[:, 1]

    df_eval = X_test.copy()
    df_eval["y_true"] = y_true
    df_eval["y_pred"] = y_pred
    df_eval["y_proba"] = y_proba

    # 1. Métricas de Fairness (Paridade Demográfica, DI, TPR/FPR)
    print("Calculando Paridade Demográfica, Impacto Disparate e Equal Opportunity (TPR/FPR)...")
    print(f"  + ICs 95% por bootstrap do teste (B={BOOTSTRAP_B}, seed={RANDOM_STATE}, modelo fixo)...")
    df_fairness = calculate_fairness_metrics(df_eval, year, bootstrap_b=BOOTSTRAP_B)

    # 2. Cálculo Nativo SHAP via C++ do XGBoost (pred_contribs=True)
    print("Gerando matriz SHAP com o motor C++ nativo do XGBoost...")
    booster = classifier.get_booster()
    dmatrix = xgb.DMatrix(X_trans, feature_names=feature_names)

    # Retorna matriz no formato (N_amostras, N_features + 1); última coluna = valor base
    contribs = booster.predict(dmatrix, pred_contribs=True)
    shap_matrix = contribs[:, :-1]
    base_val = float(contribs[0, -1])

    # Montagem da Tabela SHAP (row_id = índice original em sampled_{ano}.parquet)
    row_ids = X_test.index.to_numpy()
    df_shap = pd.DataFrame(shap_matrix.astype("float32"), columns=feature_names)
    df_shap.insert(0, "row_id", row_ids)
    df_shap.insert(1, "base_value", base_val)

    df_features_trans = pd.DataFrame(X_trans.astype("uint8"), columns=feature_names)
    df_features_trans.insert(0, "row_id", row_ids)

    # Resumo SHAP por atributo One-Hot (prevalência e diferença condicional de φ)
    df_shap_summary = calculate_shap_feature_summary(shap_matrix, X_trans, feature_names, year)

    # 3. Métricas de desempenho do modelo no teste
    df_model = calculate_model_metrics(y_true, y_pred, y_proba, pipeline, year, base_val)

    # 4. Persistência dos Arquivos Parquet Estáticos
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "fairness_metrics": df_fairness,
        "shap_values": df_shap,
        "shap_features": df_features_trans,
        "model_metrics": df_model,
        "shap_feature_summary": df_shap_summary,
    }
    for name, frame in outputs.items():
        frame.to_parquet(OUTPUTS_DIR / f"{name}_{year}.parquet", engine="pyarrow", index=False)

    m = df_model.iloc[0]
    print(f"Desempenho (teste): acc={m.accuracy:.4f} f1={m.f1:.4f} auc={m.roc_auc:.4f}")
    print(f"[SUCESSO] Auditoria concluída para o ENEM {year}!")
    for name in outputs:
        print(f"  - {OUTPUTS_DIR / f'{name}_{year}.parquet'}")


if __name__ == "__main__":
    for yr in [2019, 2020, 2021]:
        run_audit_for_year(yr)
