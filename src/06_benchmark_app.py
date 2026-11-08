"""
Benchmark da camada online (app.py) — evidência para o requisito de tempo de resposta (< 2 s).

Executa o script do Streamlit em modo headless (streamlit.testing.AppTest) para cada
combinação edição × atributo protegido:
  - cold: cache do Streamlit limpo antes da execução (inclui leitura dos Parquets);
  - warm: reexecuções com os dados já em cache (interação típica do usuário).
O tempo medido é o de execução completa do script no servidor, incluindo a
renderização do summary plot SHAP e dos gráficos Plotly.

Saída: data/outputs/app_benchmark.parquet
"""
import os
import platform
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

BASE_DIR = Path(__file__).resolve().parent.parent
APP = BASE_DIR / "app.py"
OUTPUTS_DIR = BASE_DIR / "data" / "outputs"
YEARS = [2019, 2020, 2021]
ATTRS = ["TP_COR_RACA", "Q006"]
WARM_REPEATS = 5
SLA_SECONDS = 2.0


def outputs_size_mb(year: int) -> float:
    files = [OUTPUTS_DIR / f"{n}_{year}.parquet" for n in ("fairness_metrics", "shap_values", "shap_features")]
    return sum(f.stat().st_size for f in files) / 1e6


def timed_run(at: AppTest) -> float:
    t0 = time.perf_counter()
    at.run()
    return time.perf_counter() - t0


def main() -> None:
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()  # aquecimento do interpretador/imports (não medido)

    records = []
    for year in YEARS:
        for attr in ATTRS:
            st.cache_data.clear()
            at.sidebar.selectbox[0].set_value(year)
            at.sidebar.selectbox[1].set_value(attr)
            runs = [("cold", 0, timed_run(at))]
            runs += [("warm", r, timed_run(at)) for r in range(WARM_REPEATS)]
            for mode, rep, sec in runs:
                records.append({
                    "year": year, "protected_attribute": attr, "mode": mode, "repeat": rep,
                    "seconds": sec, "within_sla": sec < SLA_SECONDS, "exceptions": len(at.exception),
                    "outputs_size_mb": outputs_size_mb(year), "cpu_count": os.cpu_count(),
                    "platform": platform.platform(), "python": platform.python_version(),
                    "streamlit": st.__version__,
                })
            cold = runs[0][2]
            warm = sum(r[2] for r in runs[1:]) / WARM_REPEATS
            print(f"  {year} {attr:12s} cold={cold:.2f}s warm(média)={warm:.2f}s exceções={len(at.exception)}")

    df = pd.DataFrame(records)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUTS_DIR / "app_benchmark.parquet", engine="pyarrow", index=False)
    print(f"\nPior caso: {df.seconds.max():.2f}s | dentro do SLA: {df.within_sla.mean():.0%}")
    print(f"[SUCESSO] {OUTPUTS_DIR / 'app_benchmark.parquet'}")


if __name__ == "__main__":
    main()
