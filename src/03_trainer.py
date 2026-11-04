from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

# Diretorios base do projeto
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

# Atributos preditivos e Target definidos no Data Contract
FEATURE_COLS = ["TP_COR_RACA", "Q006", "Q001", "Q002", "Q024", "Q025"]
TARGET_COL = "target"
RANDOM_STATE = 42

def train_model_for_year(year: int, seed: int = RANDOM_STATE) -> None:
    """
    Treina o classificador XGBoost via GridSearchCV (k=5) com validação cruzada,
    avalia o desempenho no conjunto de teste e salva o pipeline ajustado.
    """
    input_file = PROCESSED_DATA_DIR / f"sampled_{year}.parquet"
    model_output_file = MODELS_DIR / f"xgboost_{year}.joblib"

    if not input_file.exists():
        print(f"[AVISO] Arquivo processado não encontrado: {input_file}. Execute 02_preprocessing.py.")
        return

    print(f"\n--- Iniciando Treinamento do Modelo - ENEM {year} ---")
    df = pd.read_parquet(input_file)
    print(f"Dataset carregado: {len(df):,} linhas.")

    # Separação de Features (X) e Target (y)
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()

    # Separação Treino/Teste (80/20) estratificada pelo Target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=seed
    )

    # Preprocessador para codificação das variáveis categóricas
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), FEATURE_COLS)
        ],
        remainder="drop"
    )

    # Pipeline integrando pré-processamento e XGBoost
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            random_state=seed,
            eval_metric="logloss",
            n_jobs=-1
        ))
    ])

    # Grade de Hiperparâmetros para GridSearch (k-fold = 5)
    param_grid = {
        "classifier__n_estimators": [50, 100],
        "classifier__max_depth": [3, 5],
        "classifier__learning_rate": [0.05, 0.1]
    }

    print("Executando GridSearchCV (k=5)...")
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"Melhores hiperparâmetros encontrados: {grid_search.best_params_}")

    # Avaliação do melhor modelo no conjunto de teste
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)

    print("\nMétricas de Avaliação no Conjunto de Teste:")
    print(f"  - Acurácia:  {acc:.4f}")
    print(f"  - Precisão:  {prec:.4f}")
    print(f"  - Recall:    {rec:.4f}")
    print(f"  - F1-Score:  {f1:.4f}")
    print(f"  - ROC AUC:   {roc_auc:.4f}")

    # Persistência do Artefato
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_output_file)
    print(f"[SUCESSO] Modelo salvo com sucesso em: {model_output_file}")

if __name__ == "__main__":
    for yr in [2019, 2020, 2021]:
        train_model_for_year(yr)