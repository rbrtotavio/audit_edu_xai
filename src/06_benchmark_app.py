"""
Benchmark da camada online (app.py) — evidência para o requisito de tempo de resposta (< 2 s).

Executa o script do Streamlit em modo headless (streamlit.testing.AppTest) para cada
combinação edição × atributo protegido:
  - cold: cache do Streamlit limpo antes da execução (inclui leitura dos Parquets);
  - warm: reexecuções com os dados já em cache (interação típica do usuário).
O tempo medido é o de execução completa do script no servidor, incluindo a
renderização do summary plot SHAP e dos gráficos Plotly.

Modos de CPU (coluna `cpu_limit`):
  python src/06_benchmark_app.py            -> cpu_limit = "all" (todos os núcleos)
  python src/06_benchmark_app.py --cpus 2   -> cpu_limit = "2": simula o alvo de 2 vCPUs
    * a afinidade do PROCESSO é restrita aos núcleos lógicos 0 e 1 via Win32
      SetProcessAffinityMask (ctypes; psutil não está no venv). O AppTest executa o
      script do app no mesmo processo, então a restrição vale para ele. Em Linux usa
      os.sched_setaffinity;
    * OMP_NUM_THREADS (e OPENBLAS/MKL_NUM_THREADS) = 2 são definidos ANTES de importar
      pandas/numpy/streamlit.
    Observação: é uma aproximação local do hardware de destino (mesma CPU física, sem
    a contenção e a virtualização do Hugging Face Spaces).

O script ACUMULA as configurações: as linhas existentes de outros `cpu_limit` em
app_benchmark.parquet são preservadas e apenas as da configuração atual são substituídas.

Saída: data/outputs/app_benchmark.parquet
"""
import os
import sys

# --- Restrição de CPU: precisa ocorrer antes de importar bibliotecas numéricas ---
CPU_LIMIT = None
if "--cpus" in sys.argv:
    CPU_LIMIT = int(sys.argv[sys.argv.index("--cpus") + 1])
    for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_var] = str(CPU_LIMIT)


def restrict_affinity(n_cpus: int) -> int:
    """Restringe o processo aos núcleos lógicos 0..n_cpus-1. Retorna o nº efetivo de núcleos."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
        kernel32.SetProcessAffinityMask.restype = wintypes.BOOL
        handle = kernel32.GetCurrentProcess()
        if not kernel32.SetProcessAffinityMask(handle, (1 << n_cpus) - 1):
            raise OSError(ctypes.get_last_error(), "SetProcessAffinityMask falhou")
        return current_affinity_count()
    os.sched_setaffinity(0, set(range(n_cpus)))
    return len(os.sched_getaffinity(0))


def current_affinity_count() -> int:
    """Nº de núcleos lógicos na máscara de afinidade atual do processo."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.GetProcessAffinityMask.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
        proc_mask, sys_mask = ctypes.c_size_t(), ctypes.c_size_t()
        kernel32.GetProcessAffinityMask(kernel32.GetCurrentProcess(),
                                        ctypes.byref(proc_mask), ctypes.byref(sys_mask))
        return bin(proc_mask.value).count("1")
    return len(os.sched_getaffinity(0))


if CPU_LIMIT is not None:
    AFFINITY_CPUS = restrict_affinity(CPU_LIMIT)
else:
    AFFINITY_CPUS = current_affinity_count()

import platform  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
APP = BASE_DIR / "app.py"
OUTPUTS_DIR = BASE_DIR / "data" / "outputs"
OUTPUT_FILE = OUTPUTS_DIR / "app_benchmark.parquet"
YEARS = [2019, 2020, 2021]
ATTRS = ["TP_COR_RACA", "Q006"]
WARM_REPEATS = 5
SLA_SECONDS = 2.0
CPU_LIMIT_LABEL = "all" if CPU_LIMIT is None else str(CPU_LIMIT)


def outputs_size_mb(year: int) -> float:
    files = [OUTPUTS_DIR / f"{n}_{year}.parquet" for n in ("fairness_metrics", "shap_values", "shap_features")]
    return sum(f.stat().st_size for f in files) / 1e6


def timed_run(at: AppTest) -> float:
    t0 = time.perf_counter()
    at.run()
    return time.perf_counter() - t0


def main() -> None:
    print(f"Configuração: cpu_limit={CPU_LIMIT_LABEL} | núcleos na afinidade={AFFINITY_CPUS} "
          f"| OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', '(não definido)')}")
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
                    "cpu_limit": CPU_LIMIT_LABEL,
                    "affinity_cpus": AFFINITY_CPUS,
                    "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
                })
            cold = runs[0][2]
            warm = sum(r[2] for r in runs[1:]) / WARM_REPEATS
            print(f"  {year} {attr:12s} cold={cold:.2f}s warm(média)={warm:.2f}s exceções={len(at.exception)}")

    df = pd.DataFrame(records)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    if OUTPUT_FILE.exists():
        previous = pd.read_parquet(OUTPUT_FILE)
        # Arquivos antigos (sem cpu_limit) são substituídos; os demais modos são preservados
        if "cpu_limit" in previous.columns:
            previous = previous[previous["cpu_limit"] != CPU_LIMIT_LABEL]
            df = pd.concat([previous, df], ignore_index=True)
    order = {"all": 0}
    df = df.sort_values(["cpu_limit", "year", "protected_attribute", "mode", "repeat"],
                        key=lambda s: s.map(lambda v: order.get(v, 1)) if s.name == "cpu_limit" else s,
                        kind="stable").reset_index(drop=True)
    df.to_parquet(OUTPUT_FILE, engine="pyarrow", index=False)
    cur = df[df["cpu_limit"] == CPU_LIMIT_LABEL]
    print(f"\nPior caso (cpu_limit={CPU_LIMIT_LABEL}): {cur.seconds.max():.2f}s | "
          f"dentro do SLA: {cur.within_sla.mean():.0%}")
    print(f"Configurações no arquivo: {sorted(df['cpu_limit'].unique())}")
    print(f"[SUCESSO] {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
