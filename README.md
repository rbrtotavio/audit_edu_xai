# ⚖️ AuditEdu-XAI — Artefato de Auditoria Algorítmica e Explicabilidade Educacional

O **AuditEdu-XAI** é uma solução de auditoria algorítmica e explicabilidade (XAI) projetada para detectar e quantificar vieses socioeconômicos e demográficos em modelos de machine learning treinados com microdados de exames públicos de larga escala (**ENEM**).

A ferramenta combina processamento em lote desacoplado, métricas rigorosas de equidade (*Fairness*) e explicabilidade baseada em valores de Shapley (SHAP) apresentadas em um dashboard interativo em Streamlit.

---

## 📐 1. Arquitetura e Stack Tecnológica

O projeto adota a metodologia **Design Science Research (DSR)** com estrita separação entre o **Pipeline Offline em Lote** e a **Camada Visual de Consumo (Streamlit)**.

```
[Microdados ENEM (CSV)]
       │
       ▼ (01_etl.py - Chunk Processing + Absenteísmo)
[data/processed/raw_filtered_{ano}.parquet]
       │
       ▼ (02_preprocessing.py - Amostragem 50k + Imputação Moda + NMG Target)
[data/processed/sampled_{ano}.parquet]
       │
       ▼ (03_trainer.py - XGBoost + GridSearchCV k=5)
[models/xgboost_{ano}.joblib]
       │
       ▼ (04_audit_engine.py - C++ TreeSHAP + Disparate Impact)
[data/outputs/*.parquet]
       │
       ▼ (app.py - Interface Streamlit < 2s SLA)
[Dashboard Interativo UI]
```

### Stack Tecnológica:
- **Linguagem:** Python 3.10+
- **ETL e Processamento:** Pandas, PyArrow, NumPy
- **Modelagem e MLOps:** XGBoost, Scikit-Learn, Joblib
- **Auditoria e XAI:** XGBoost Native TreeSHAP (`pred_contribs`), Fairlearn / Rotinas Customizadas de Fairness
- **Visualização & UI:** Streamlit, Plotly Express, Matplotlib
- **Containerização:** Docker (Hugging Face Spaces CPU Basic: 2 vCPUs, 16 GB RAM)

---

## 🗂️ 2. Estrutura do Repositório (WBS)

```text
audit_edu_xai/
├── data/
│   ├── raw/                 # CSVs brutos do ENEM (ignorado no git)
│   ├── processed/           # Arquivos limpos e amostrados (.parquet)
│   └── outputs/             # Matrizes SHAP e métricas de Fairness estáticas (.parquet)
├── models/                  # Pipelines XGBoost serializados (.joblib)
├── src/
│   ├── __init__.py
│   ├── 01_etl.py            # Extração em chunks e filtro de absenteísmo
│   ├── 02_preprocessing.py  # Amostragem estratificada, imputação por moda e NMG
│   ├── 03_trainer.py        # Treinamento XGBoost com GridSearchCV 5-fold
│   ├── 04_audit_engine.py   # Auditoria no TESTE: DI, TPR/FPR, IC bootstrap, TreeSHAP
│   ├── 05_bias_injection.py # Validação por injeção sintética de viés
│   ├── 06_benchmark_app.py  # Tempo de resposta do dashboard (--cpus 2)
│   ├── 07_data_quality.py   # Ausentes e robustez da imputação
│   └── 08_build_manifest.py # Manifesto SHA-256 dos artefatos
├── app.py                   # Dashboard Streamlit (só lê data/outputs/*.parquet)
├── requirements.txt         # Dependências do dashboard (versões fixadas) — Streamlit Cloud e Docker
├── requirements-pipeline.txt # Dependências do pipeline offline (inclui requirements.txt)
├── Dockerfile / .dockerignore # Imagem multi-stage da camada online
├── .github/workflows/docker.yml # CI: build + teste de fumaça da imagem
└── README.md                # Documentação técnica mestre
```

---

## 📋 3. Contrato de Dados (Data Contract)

O pipeline mapeia e processa estritamente as seguintes colunas do ENEM:

| Coluna Bruta | Tipo de Dado | Função no Pipeline | Regra de Tratamento |
| :--- | :--- | :--- | :--- |
| `TP_PRESENCA_CN/CH/LC/MT` | `uint8` | Filtro de Absenteísmo | Manter apenas candidatos presentes em todas as 4 provas ($=1$) |
| `NU_NOTA_CN/CH/LC/MT/RED` | `float32` | Construção do Target | Cálculo da Nota Média Geral ($\text{NMG}$) |
| `TP_COR_RACA` | `category` | Atributo Protegido 1 | 0=Não Declarado, 1=Branca, 2=Preta, 3=Parda, 4=Amarela, 5=Indígena |
| `Q006` | `category` | Atributo Protegido 2 | Renda mensal familiar (A a Q); Imputação pela moda |
| `Q001` / `Q002` | `category` | Feature Socioeconômica | Escolaridade do Pai/Mãe; Imputação pela moda |
| `Q024` / `Q025` | `category` | Feature Tecnológica | Acesso a Computador/Internet; Imputação pela moda |

---

## 📊 4. Métricas de Auditoria e XAI

### 4.1 Equidade Algorítmica (*Fairness*)
- **Taxa de Seleção (Paridade Demográfica):** Proporção de candidatos rotulados com $Y=1$ (Nota Média Geral acima da mediana) em cada subgrupo demográfico ou socioeconômico.
  $$\text{Taxa de Seleção}_g = \frac{\sum \hat{Y}_g}{N_g}$$

- **Impacto Disparate ($\text{DI}$):** Razão entre a taxa de seleção do subgrupo analisado e a taxa de seleção do grupo de maior desempenho (referência).
  $$\text{DI}_g = \frac{\text{Taxa de Seleção}_g}{\text{Taxa de Seleção}_{\text{ref}}}$$
- **Regra dos 80% (EEOC):** Um valor de $\text{DI} < 0.80$ dispara um **alerta visual automático de viés algorítmico**, indicando disparidade adversa contra o grupo.

### 4.2 Explicabilidade Algorítmica (SHAP)
- **TreeSHAP Nativo (C++):** Utiliza a decomposição exata em árvore para calcular os valores de Shapley de cada atributo no log-odds de aprovação.
- **Explicabilidade Global (Summary Plot):** Apresenta o impacto relativo e a direção de cada feature na predição do modelo.
- **Explicabilidade Local (Waterfall Plot):** Decompõe a decisão individual de um candidato específico a partir do valor base (*expected value*).

---

## 🚀 5. Guia de Execução Passo a Passo

### 5.1 Configuração do Ambiente Virtual

```bash
# Clonar o repositório e acessar a pasta
cd audit_edu_xai

# Criar e ativar o ambiente virtual
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows (PowerShell)
.\venv\Scripts\activate

# Instalar dependências (pipeline completo + dashboard)
pip install --upgrade pip
pip install -r requirements-pipeline.txt

# Apenas o dashboard
pip install -r requirements.txt
```

### 5.2 Execução do Pipeline Offline em Lote

O pipeline deve ser executado em ordem sequencial:

```bash
# Passo 1: Extração e filtragem em chunks dos CSVs do ENEM (data/raw/)
python src/01_etl.py

# Passo 2: Imputação por moda, NMG e amostragem estratificada de 50.000 linhas
python src/02_preprocessing.py

# Passo 3: Treinamento XGBoost Classifier via GridSearch (k-fold = 5)
python src/03_trainer.py

# Passo 4: Pré-cálculo estático SHAP e métricas de Fairness (Parquet)
python src/04_audit_engine.py
```

### 5.3 Inicialização da Aplicação Web (Streamlit UI)

```bash
streamlit run app.py
```
Acesse `http://localhost:8501` no navegador. O dashboard realiza apenas leituras assíncronas dos arquivos `.parquet` salvos em `data/outputs/`, garantindo tempo de resposta inferior a 2 segundos.

---

## 🚀 6. Implantação

### 6.1 Streamlit Community Cloud (hospedagem pública gratuita)

O dashboard é hospedado diretamente a partir deste repositório no GitHub, sem Docker:

1. Acesse https://share.streamlit.io e entre com a conta do GitHub.
2. **Create app** → *Deploy a public app from GitHub* → repositório `rbrtotavio/audit_edu_xai`, branch de trabalho, arquivo principal `app.py`.
3. Em **Advanced settings**, selecione **Python 3.12** (o `numpy==1.26.4` fixado não tem *wheels* para versões mais novas).
4. **Deploy.** O Cloud instala `requirements.txt` (apenas as dependências do dashboard) e lê os Parquets versionados em `data/outputs/`.

> O app hiberna após 12 h sem acesso; o primeiro acesso seguinte leva alguns segundos para acordá-lo. Abra a URL antes de uma apresentação.
> Pico de memória medido localmente: ~380 MB (todas as edições, atributos e beeswarms).

### 6.2 Imagem Docker (portabilidade)

O `Dockerfile` é **multi-stage** (`builder` instala `requirements.txt` num venv; `runner` copia só o venv, `app.py` e `data/outputs/`), roda como usuário não-root UID 1000 e expõe a porta `7860`. O `.dockerignore` é uma lista de permissão: `data/raw/`, `venv/`, `models/` e `article/` não entram no contexto de build.

```bash
docker build -t audit-edu-xai .
docker run --cpus=2 -p 7860:7860 audit-edu-xai   # http://localhost:7860
```

A cada mudança em `Dockerfile`, `requirements.txt`, `app.py` ou `data/outputs/`, o workflow `.github/workflows/docker.yml` constrói a imagem, sobe o contêiner com 2 CPUs, verifica o *health check* e executa o app (AppTest) dentro dele.

> A mesma imagem pode ser publicada em plataformas baseadas em contêiner (ex.: Hugging Face Spaces com SDK Docker, que exige plano pago para novas contas).

> O pipeline offline (`src/01`–`08`) **não** roda no contêiner nem no Cloud: execute-o localmente e versione os Parquets resultantes.

---

## 📄 Licença e Uso Acadêmico

Este projeto foi construído sob os princípios de transparência algorítmica e ciência aberta. Sinta-se à vontade para utilizar, modificar e expandir esta ferramenta para pesquisas educacionais e auditorias de interesse público.