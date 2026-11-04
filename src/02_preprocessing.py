from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Configurações de diretórios base
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

# Constantes e Atributos definidos na Especificação Mestre
CATEGORICAL_COLS = ["TP_COR_RACA", "Q006", "Q001", "Q002", "Q024", "Q025"]
GRADE_COLS = ["NU_NOTA_CN", "NU_NOTA_CH", "NU_NOTA_LC", "NU_NOTA_MT", "NU_NOTA_REDACAO"]
STRATA_COL = "TP_COR_RACA"
SAMPLE_SIZE = 50_000
RANDOM_STATE = 42

def preprocess_enem_year(year: int, sample_size: int = SAMPLE_SIZE, seed: int = RANDOM_STATE) -> None:
    """
    Realiza a imputação por moda, cálculo na NMG, binarização do target
    e amostragem aleatória estratificada do dataseet do ENEM.
    """

    input_file = PROCESSED_DATA_DIR / f"raw_filtered_{year}.parquet"
    output_file = PROCESSED_DATA_DIR / f"sampled_{year}.parquet"

    if not input_file.exists():
        print(f"[AVISO] Arquivo bruto filtrado não encontrado: {input_file}. Execute primeiro o 01_etl.py.")
        return

    print(f"\n--- Iniciando pré-processamento do ENEM {year} ---")
    df = pd.read_parquet(input_file)
    print(f"Registros brutos filtrados carregados: {len(df):,}")

    # 1. Imputação por Moda para Variáveis Categóricas Ausentes
    print("Aplicando imputação pela moda nas variáveis categóricas...")
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            mode_val = df[col].mode(dropna=True)[0]
            df[col] = df[col].fillna(mode_val).astype("category")

    # 2. Cálculo da Nota Média Geral (NMG)
    print("Calculando a Nota Média Geral (NMG)...")
    df["NMG"] = df[GRADE_COLS].mean(axis=1).astype("float32")

    # 3. Binarização da Variável Alvo (Y = 1 se NMG >= mediana, senão 0)
    median_nmg = float(df["NMG"].median())
    df["target"] = (df["NMG"] >= median_nmg).astype("uint8")
    print(f"mediana da NMG calculada: {median_nmg:.2f}")

    # 4. Amostragem Estratificada por Atributo Protegido (TP_COR_RACA)
    target_n = min(sample_size, len(df))
    if target_n < len(df):
        print(f"Realizando amostragem estratificada ({target_n:,} registros com semente {seed})...")
        try:
            df_sampled, _ = train_test_split(
                df,
                train_size=target_n,
                stratify=df[STRATA_COL],
                random_state=seed,
                shuffle=True
            )
        except ValueError:
            # Fallback seguro para amostras pequenas (ex: dados sintéticos de teste)
            print("[INFO] Estratificação estrita não aplicável pela quantidade de classes; executando amostragem simples.")
            df_sampled = df.sampled(n=target_n, random_state=seed)
        df_sampled = df_sampled.reset_index(drop=True)
    else:
        print(f"O tamanho do dataset ({len(df):,}) é menor ou igual ao limite de amostragem. Mantendo todos os registros.")
        df_sampled = df.copy()
        
    # 5. Persistência em Parquet
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_sampled.to_parquet(output_file, engine="pyarrow", index=False)
    print(f"[SUCESSO] Processamento concluído. Salvo em: {output_file}")

if __name__ == "__main__":
    for yr in [2019, 2020, 2021]:
        preprocess_enem_year(yr)