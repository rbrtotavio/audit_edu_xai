# syntax=docker/dockerfile:1
# =============================================================================
# AuditEdu-XAI — imagem da CAMADA ONLINE (dashboard Streamlit)
# Artefato de PORTABILIDADE, validado em CI (.github/workflows/docker.yml): roda em
# qualquer infraestrutura de contêineres. O deploy público NÃO usa esta imagem: é feito
# no Streamlit Community Cloud (https://auditeduxai.streamlit.app/), direto do repositório.
# A porta 7860 é herança do planejamento inicial (Hugging Face Spaces), mantida por
# compatibilidade; é configurável via STREAMLIT_SERVER_PORT.
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
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip \
 && pip install -r /tmp/requirements.txt

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
    # Limita as threads numéricas a 2, compatível com ambientes de poucos núcleos
    OMP_NUM_THREADS=2 \
    # Servidor Streamlit headless; porta 7860 mantida do planejamento inicial
    # (Hugging Face Spaces) por compatibilidade
    STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Usuário não privilegiado (UID 1000): boa prática de segurança exigida por diversas
# plataformas de contêiner
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
