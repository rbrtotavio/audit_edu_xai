"""
Manifesto de integridade (SHA-256) dos dados, modelos e saídas do pipeline.

Deve ser executado POR ÚLTIMO, depois de todos os scripts que escrevem em data/ e models/.

Cobertura (caminhos relativos à raiz do projeto, separador "/"):
  data/raw/*.csv              (microdados INEP; nome e tamanho em bytes em comentário)
  data/processed/*.parquet
  models/*.joblib
  data/outputs/*.parquet

Formato: compatível com `sha256sum -c` — uma linha "<hash>  <caminho>" por arquivo; as
linhas iniciadas por "#" (cabeçalho com data e versões das bibliotecas, e o tamanho de cada
arquivo) são comentários. Verificação, a partir da raiz do projeto:
  grep -v '^#' data/MANIFEST.sha256 | sha256sum -c

Saída: data/MANIFEST.sha256
"""
import hashlib
import platform
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MANIFEST = BASE_DIR / "data" / "MANIFEST.sha256"
PATTERNS = [
    ("data/raw", "*.csv"),
    ("data/processed", "*.parquet"),
    ("models", "*.joblib"),
    ("data/outputs", "*.parquet"),
]
PACKAGES = ["pandas", "numpy", "pyarrow", "xgboost", "scikit-learn", "shap", "streamlit"]
CHUNK = 8 * 1024 * 1024  # leitura em blocos de 8 MB (os CSVs brutos têm GBs)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(CHUNK):
            h.update(block)
    return h.hexdigest()


def pkg_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "não instalado"


def main() -> None:
    now = datetime.now(timezone.utc).astimezone()
    lines = [
        "# AuditEdu-XAI — manifesto de integridade SHA-256 (formato sha256sum)",
        f"# Gerado em: {now.isoformat(timespec='seconds')}",
        f"# Plataforma: {platform.platform()}",
        f"# Python: {platform.python_version()}",
    ]
    lines += [f"# {p}: {pkg_version(p)}" for p in PACKAGES]
    lines.append("# Verificar (na raiz do projeto): grep -v '^#' data/MANIFEST.sha256 | sha256sum -c")

    n_files = 0
    for rel_dir, pattern in PATTERNS:
        files = sorted((BASE_DIR / rel_dir).glob(pattern), key=lambda p: p.name.lower())
        lines.append("#")
        lines.append(f"# --- {rel_dir}/{pattern} ({len(files)} arquivos) ---")
        for f in files:
            rel = f.relative_to(BASE_DIR).as_posix()
            size = f.stat().st_size
            print(f"  hash: {rel} ({size:,} bytes)")
            lines.append(f"# {rel}  {size} bytes")
            lines.append(f"{sha256_of(f)}  {rel}")
            n_files += 1

    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"[SUCESSO] {n_files} arquivos -> {MANIFEST}")


if __name__ == "__main__":
    main()
