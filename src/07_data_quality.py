"""
Qualidade dos dados e impacto da imputação por moda (evidência para a banca).

Para cada edição, a partir de data/processed/raw_filtered_{ano}.parquet (população filtrada
pelo absenteísmo, ANTES da imputação e da amostragem), calcula:
  - n_filtered: nº de registros da base usada na imputação e na mediana da NMG;
  - nº de nulos por coluna categórica do modelo;
  - n_all_questionnaire_null: linhas com Q001, Q002, Q006, Q024 e Q025 TODOS nulos;
  - moda (valor imputado, mesma regra de 02_preprocessing.py), sua participação e a da
    segunda categoria mais frequente (entre os valores não nulos), e a margem em
    contagem entre as duas — se a margem excede o tamanho do teste (10.000), nenhuma
    partição de teste poderia alterar a moda, mesmo que fosse excluída do cálculo;
  - nmg_median_population: mediana da NMG (média das 5 notas), como em 02_preprocessing.py;
  - test_fraction_of_imputation_base = 10.000 / n_filtered.

Formato de saída (LONGO): data/outputs/data_quality.parquet, uma linha por (year, column),
com as colunas de nível de ano (n_filtered, n_all_questionnaire_null,
nmg_median_population, n_test, test_fraction_of_imputation_base) repetidas em todas as
linhas do ano.

Memória: lê apenas as 11 colunas necessárias; as categóricas são convertidas para
`category` na leitura (strings_to_categorical), o que mantém o pico bem abaixo de 8 GB.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
OUTPUTS_DIR = BASE_DIR / "data" / "outputs"

YEARS = [2019, 2020, 2021]
CATEGORICAL_COLS = ["TP_COR_RACA", "Q006", "Q001", "Q002", "Q024", "Q025"]
QUESTIONNAIRE_COLS = ["Q001", "Q002", "Q006", "Q024", "Q025"]
GRADE_COLS = ["NU_NOTA_CN", "NU_NOTA_CH", "NU_NOTA_LC", "NU_NOTA_MT", "NU_NOTA_REDACAO"]
N_TEST = 10_000  # 20% de 50.000 (split de 03_trainer.py)


def _label(value) -> str:
    if isinstance(value, (float, np.floating, int, np.integer)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def quality_for_year(year: int) -> list:
    path = PROCESSED_DATA_DIR / f"raw_filtered_{year}.parquet"
    if not path.exists():
        print(f"[AVISO] {path} ausente. Pule.")
        return []

    table = pq.read_table(path, columns=CATEGORICAL_COLS + GRADE_COLS)
    df = table.to_pandas(strings_to_categorical=True)
    del table
    n = len(df)

    # Mediana populacional da NMG — mesma expressão de 02_preprocessing.py
    nmg = df[GRADE_COLS].mean(axis=1).astype("float32")
    median_nmg = float(nmg.median())
    del nmg

    all_null = int(df[QUESTIONNAIRE_COLS].isna().all(axis=1).sum())

    rows = []
    for col in CATEGORICAL_COLS:
        s = df[col]
        n_null = int(s.isna().sum())
        counts = s.value_counts(dropna=True)  # ordem decrescente de frequência
        n_valid = int(counts.sum())
        mode_val = s.mode(dropna=True)[0]  # regra de imputação de 02_preprocessing.py
        mode_count = int(counts.loc[mode_val])
        second = counts.drop(index=mode_val)
        second_val = second.index[0] if len(second) else None
        second_count = int(second.iloc[0]) if len(second) else 0
        margin = mode_count - second_count
        rows.append({
            "year": year,
            "column": col,
            "n_filtered": n,
            "n_null": n_null,
            "null_share": n_null / n,
            "n_categories": int((counts > 0).sum()),
            "mode_value": _label(mode_val),
            "mode_count": mode_count,
            "mode_share": mode_count / n_valid,
            "second_value": _label(second_val) if second_val is not None else None,
            "second_count": second_count,
            "second_share": second_count / n_valid,
            "mode_margin_count": margin,
            # Retirar até N_TEST linhas da moda não a deixaria abaixo da 2ª categoria
            "mode_robust_to_test_removal": bool(margin > N_TEST),
            "n_all_questionnaire_null": all_null,
            "nmg_median_population": median_nmg,
            "n_test": N_TEST,
            "test_fraction_of_imputation_base": N_TEST / n,
        })
    del df
    return rows


def main() -> None:
    records = []
    for year in YEARS:
        print(f"--- Qualidade dos dados — ENEM {year} ---")
        rows = quality_for_year(year)
        records += rows
        for r in rows:
            print(f"  {r['column']:12s} nulos={r['n_null']:>7,} moda={r['mode_value']} "
                  f"({r['mode_share']:.3f}) 2a={r['second_value']} ({r['second_share']:.3f}) "
                  f"margem={r['mode_margin_count']:,}")
        if rows:
            r = rows[0]
            print(f"  n_filtered={r['n_filtered']:,} todos_nulos={r['n_all_questionnaire_null']:,} "
                  f"mediana_NMG={r['nmg_median_population']:.4f} "
                  f"fração_teste={r['test_fraction_of_imputation_base']:.5f}")

    df = pd.DataFrame(records)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUTS_DIR / "data_quality.parquet"
    df.to_parquet(out, engine="pyarrow", index=False)
    print(f"[SUCESSO] {out}")


if __name__ == "__main__":
    main()
