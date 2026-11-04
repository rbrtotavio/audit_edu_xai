from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

# Diretórios base
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "data" / "outputs"

# Atributos do Data Contract
FEATURE_COLS = ["TP_COR_RACA", "Q006", "Q001", "Q002", "Q024", "Q025"]
PROTECTED_ATTRIBUTES = ["TP_COR_RACA", "Q006"]
TARGET_COL = "target"

def calculate_fairness_metrics(df_eval: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Calcula as taxas de seleção (Demographic Parity) e o Impacto Disparate (DI)
    para cada subgrupo dos atributos protegidos.
    """
    fairness_records = []

    for attr in PROTECTED_ATTRIBUTES:
        grouped = df_eval.groupby(attr, observed=True).agg(
            total_count=("y_pred", "count"),
            positive_count=("y_pred", "sum"),
            selection_rate=("y_pred", "mean")
        ).reset_index()

        max_rate_row = grouped.loc[grouped["selection_rate"].idxmax()]
        ref_group = max_rate_row[attr]
        ref_rate = max_rate_row["selection_rate"]

        for _, row in grouped.iterrows():
            group_val = row[attr]
            group_rate = row["selection_rate"]
            
            # Cálculo do Impacto Disparate (DI = Taxa Grupo / Taxa Referência)
            di = (group_rate / ref_rate) if ref_rate > 0 else 1.0

            fairness_records.append({
                "year": year,
                "protected_attribute": attr,
                "group": str(group_val),
                "total_count": int(row["total_count"]),
                "positive_count": int(row["positive_count"]),
                "selection_rate": float(group_rate),
                "reference_group": str(ref_group),
                "reference_selection_rate": float(ref_rate),
                "disparate_impact": float(di),
                "alert_di_low": bool(di < 0.8)  # Regra dos 80% (EEOC)
            })

    return pd.DataFrame(fairness_records)

def run_audit_for_year(year: int) -> None:
    """
    Executa a auditoria em lote (SHAP + Fairness) para um determinado ano,
    exportando as matrizes pré-calculadas em formato Parquet estático.
    """
    data_file = PROCESSED_DATA_DIR / f"sampled_{year}.parquet"
    model_file = MODELS_DIR / f"xgboost_{year}.joblib"

    if not data_file.exists() or not model_file.exists():
        print(f"[AVISO] Arquivos do ano {year} ausentes ({data_file.name} ou {model_file.name}). Pule.")
        return

    print(f"\n--- Iniciando Motor de Auditoria (SHAP + Fairness) - ENEM {year} ---")
    
    df = pd.read_parquet(data_file)
    pipeline = joblib.load(model_file)

    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    X = df[FEATURE_COLS].copy()
    y_true = df[TARGET_COL].values

    # Transformação One-Hot Encoding
    X_trans = preprocessor.transform(X)
    feature_names = preprocessor.get_feature_names_out()
    
    y_pred = classifier.predict(X_trans)
    y_proba = classifier.predict_proba(X_trans)[:, 1]

    df_eval = X.copy()
    df_eval["y_true"] = y_true
    df_eval["y_pred"] = y_pred
    df_eval["y_proba"] = y_proba

    # 1. Métricas de Fairness
    print("Calculando métricas de Paridade Demográfica e Impacto Disparate...")
    df_fairness = calculate_fairness_metrics(df_eval, year)

    # 2. Cálculo Nativo SHAP via C++ do XGBoost (pred_contribs=True)
    print("Gerando matriz SHAP com o motor C++ nativo do XGBoost...")
    booster = classifier.get_booster()
    dmatrix = xgb.DMatrix(X_trans, feature_names=list(feature_names))
    
    # Retorna matriz no formato (N_amostras, N_features + 1)
    contribs = booster.predict(dmatrix, pred_contribs=True)
    
    # As primeiras N_features colunas contêm os valores SHAP
    shap_matrix = contribs[:, :-1]
    
    # A última coluna contém o valor base (esperado/bias) do modelo
    base_val = float(contribs[0, -1])

    # Montagem da Tabela SHAP
    df_shap = pd.DataFrame(shap_matrix, columns=feature_names)
    df_shap.insert(0, "row_id", df.index)
    df_shap.insert(1, "base_value", base_val)

    # Features pós-encoding para visualização local/global
    df_features_trans = pd.DataFrame(X_trans, columns=feature_names)
    df_features_trans.insert(0, "row_id", df.index)

    # 3. Persistência dos Arquivos Parquet Estáticos
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    df_fairness.to_parquet(OUTPUTS_DIR / f"fairness_metrics_{year}.parquet", engine="pyarrow", index=False)
    df_shap.to_parquet(OUTPUTS_DIR / f"shap_values_{year}.parquet", engine="pyarrow", index=False)
    df_features_trans.to_parquet(OUTPUTS_DIR / f"shap_features_{year}.parquet", engine="pyarrow", index=False)

    print(f"[SUCESSO] Auditoria concluída para o ENEM {year}!")
    print(f"  - Métricas de Fairness: {OUTPUTS_DIR / f'fairness_metrics_{year}.parquet'}")
    print(f"  - Matriz SHAP:          {OUTPUTS_DIR / f'shap_values_{year}.parquet'}")
    print(f"  - Matriz de Features:   {OUTPUTS_DIR / f'shap_features_{year}.parquet'}")

if __name__ == "__main__":
    for yr in [2019, 2020, 2021]:
        run_audit_for_year(yr)