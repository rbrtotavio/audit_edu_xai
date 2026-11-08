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


def calculate_fairness_metrics(df_eval: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Calcula, por subgrupo de cada atributo protegido (somente no conjunto de teste):
      - Taxa de seleção P(ŷ=1 | G=g) e Impacto Disparate (DI) — Paridade Demográfica;
      - Taxa base real P(y=1 | G=g) e DI dos rótulos (para medir amplificação do viés);
      - TPR = P(ŷ=1 | y=1, G=g) e FPR = P(ŷ=1 | y=0, G=g) — Equal Opportunity / Equalized Odds.
    O grupo de referência é o subgrupo de maior taxa de seleção.
    """
    records = []

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

        for _, row in grouped.iterrows():
            di = _safe_rate(row["selection_rate"], ref_rate) if ref_rate > 0 else 1.0
            label_di = _safe_rate(row["base_rate"], ref_base_rate)
            records.append({
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
            })

    return pd.DataFrame(records)


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
    df_fairness = calculate_fairness_metrics(df_eval, year)

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

    # 3. Métricas de desempenho do modelo no teste
    df_model = calculate_model_metrics(y_true, y_pred, y_proba, pipeline, year, base_val)

    # 4. Persistência dos Arquivos Parquet Estáticos
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "fairness_metrics": df_fairness,
        "shap_values": df_shap,
        "shap_features": df_features_trans,
        "model_metrics": df_model,
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
