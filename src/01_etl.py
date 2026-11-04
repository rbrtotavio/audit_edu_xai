from pathlib import Path
import pandas as pd

# Definir caminhos relativos à raiz do projeto
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

# Colunas especificadas no Data Contract
COLUMNS_TO_KEEP = [
    "TP_PRESENCA_CN", "TP_PRESENCA_CH", "TP_PRESENCA_LC", "TP_PRESENCA_MT",
    "NU_NOTA_CN", "NU_NOTA_CH", "NU_NOTA_LC", "NU_NOTA_MT", "NU_NOTA_REDACAO",
    "TP_COR_RACA", "Q006", "Q001", "Q002", "Q024", "Q025"
]

def process_enem_year(year: int, chunksize: int = 100_000) -> None:
    """
    Processa o CSV bruto de um ano específico do ENEM em chunks,
    filtrando candidatos ausentes e salvando em Parquet.
    """
    raw_file = RAW_DATA_DIR / f"MICRODADOS_ENEM_{year}.csv"
    output_file = PROCESSED_DATA_DIR / f"raw_filtered_{year}.parquet"

    if not raw_file.exists():
        print(f"[AVISO] Arquivo não encontrado: {raw_file}. Pule ou insira o CSV.")
        return

    print(f"Iniciando processamento em lote do ENEM {year}...")
    filtered_chunks = []

    # Leitura em blocos para não estourar os 8 GB de RAM
    for chunk in pd.read_csv(
        raw_file,
        sep=";",
        encoding="latin1",
        usecols=COLUMNS_TO_KEEP,
        chunksize=chunksize,
        low_memory=False
    ):
        # Filtro estrito: candidato presente em todas as 4 provas (valor == 1)
        presence_mask = (
            (chunk["TP_PRESENCA_CN"] == 1) &
            (chunk["TP_PRESENCA_CH"] == 1) &
            (chunk["TP_PRESENCA_LC"] == 1) &
            (chunk["TP_PRESENCA_MT"] == 1)
        )
        filtered_chunk = chunk[presence_mask].copy()

        # Downcasting de tipos para economia de memória
        floats_cols = ["NU_NOTA_CN", "NU_NOTA_CH", "NU_NOTA_LC", "NU_NOTA_MT", "NU_NOTA_REDACAO"]
        filtered_chunk[floats_cols] = filtered_chunk[floats_cols].astype("float32")

        filtered_chunks.append(filtered_chunk)

    # Consolidação e exportação para Parquet
    df_final = pd.concat(filtered_chunks, ignore_index=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_final.to_parquet(output_file, engine="pyarrow", index=False)

    print(f"[SUCESSO] processado ENEM {year}: {len(df_final):,} registros salvos em {output_file}")

if __name__ == "__main__":
    # Processa cada ano do escopo
    for yr in [2019, 2020, 2021]:
        process_enem_year(yr)