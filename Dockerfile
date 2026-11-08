# syntax=docker/dockerfile:1
# =============================================================================
# AuditEdu-XAI — imagem da CAMADA ONLINE (dashboard Streamlit)
# Alvo: Hugging Face Spaces (SDK Docker, CPU Basic: 2 vCPUs / 16 GB RAM, porta 7860)
#
# A imagem contém APENAS o app.py e os Parquets pré-calculados de data/outputs/.
# Nenhum modelo é treinado e nenhum SHAP é calculado em tempo de requisição:
# o pipeline offline (src/01..05) roda fora do contêiner.
# =============================================================================

ARG PYTHON_VERSION=3.12

# -----------------------------------------------------------------------------
# Estágio 1 — builder: resolve e instala as dependências num venv isolado
# -----------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Copiado isoladamente para aproveitar o cache de camadas do Docker
COPY requirements-app.txt /tmp/requirements-app.txt
RUN pip install --upgrade pip \
 && pip install -r /tmp/requirements-app.txt

# -----------------------------------------------------------------------------
# Estágio 2 — runner: imagem final enxuta, sem pip cache nem artefatos de build
# -----------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runner

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/home/user \
    # Diretórios graváveis por usuário não-root (matplotlib e numba/SHAP)
    MPLCONFIGDIR=/tmp/matplotlib \
    NUMBA_CACHE_DIR=/tmp/numba \
    # Limita threads ao orçamento de 2 vCPUs do CPU Basic
    OMP_NUM_THREADS=2 \
    # Configuração do Streamlit para o Spaces
    STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# O Hugging Face Spaces executa o contêiner com UID 1000
RUN useradd --create-home --uid 1000 user

COPY --from=builder /opt/venv /opt/venv

WORKDIR /home/user/app
COPY --chown=user:user app.py ./
COPY --chown=user:user data/outputs/ ./data/outputs/

USER user
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://localhost:7860/_stcore/health', timeout=4).status == 200 else 1)"

CMD ["streamlit", "run", "app.py"]
