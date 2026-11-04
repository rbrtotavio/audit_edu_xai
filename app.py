from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st

# Configuração da Página
st.set_page_config(
    page_title="AuditEdu-XAI | Auditoria Algorítmica",
    page_icon="⚖️",
    layout="wide"
)

# Caminho dos artefatos pré-calculados
BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "data" / "outputs"

# Dicionários de mapeamento legível
RACA_MAP = {
    "0": "Não Declarado",
    "1": "Branca",
    "2": "Preta",
    "3": "Parda",
    "4": "Amarela",
    "5": "Indígena"
}

ATTR_LABELS = {
    "TP_COR_RACA": "Raça / Cor",
    "Q006": "Renda Familiar Mensal"
}

# -----------------------------------------------------------------------------
# Camada de Leitura Assíncrona / Cacheada (Apenas Parquet Estático)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_audit_data(year: int):
    fairness_path = OUTPUTS_DIR / f"fairness_metrics_{year}.parquet"
    shap_path = OUTPUTS_DIR / f"shap_values_{year}.parquet"
    features_path = OUTPUTS_DIR / f"shap_features_{year}.parquet"

    if not (fairness_path.exists() and shap_path.exists() and features_path.exists()):
        return None, None, None

    df_fairness = pd.read_parquet(fairness_path)
    df_shap = pd.read_parquet(shap_path)
    df_features = pd.read_parquet(features_path)
    return df_fairness, df_shap, df_features


# -----------------------------------------------------------------------------
# Interface do Usuário (Sidebar & Filtros)
# -----------------------------------------------------------------------------
st.title("⚖️ AuditEdu-XAI — Auditoria Algorítmica e Explicabilidade")
st.caption("Plataforma de detecção de viés socioeconômico e demográfico em modelos preditivos do ENEM.")

st.sidebar.header("⚙️ Parâmetros da Auditoria")

# Seleção de Ano
selected_year = st.sidebar.selectbox("Ano da Edição do ENEM", [2019, 2020, 2021], index=1)

# Carregamento dos dados pré-calculados
df_fairness, df_shap, df_features = load_audit_data(selected_year)

if df_fairness is None:
    st.error(f"⚠️ Os arquivos do ano {selected_year} não foram encontrados em `data/outputs/`. Execute o script `src/04_audit_engine.py` primeiro.")
    st.stop()

# Seleção do Atributo Protegido
selected_attr = st.sidebar.selectbox(
    "Atributo Protegido em Análise",
    options=list(ATTR_LABELS.keys()),
    format_func=lambda x: ATTR_LABELS[x]
)

st.sidebar.divider()
st.sidebar.markdown("**Status da Infraestrutura:**")
st.sidebar.success("⚡ Consumo Ativo: Matrizes Parquet Estáticas (< 2s SLA)")

# -----------------------------------------------------------------------------
# Painel 1: Auditoria de Equidade Algorítmica (Fairness)
# -----------------------------------------------------------------------------
st.header("1. Módulo de Equidade Algorítmica (Fairness)")

df_fairness_filtered = df_fairness[df_fairness["protected_attribute"] == selected_attr].copy()

if selected_attr == "TP_COR_RACA":
    df_fairness_filtered["group_label"] = df_fairness_filtered["group"].map(RACA_MAP).fillna(df_fairness_filtered["group"])
else:
    df_fairness_filtered["group_label"] = "Classe " + df_fairness_filtered["group"]

# Indicadores de Visão Geral (KPIs)
min_di = df_fairness_filtered["disparate_impact"].min()
ref_group = df_fairness_filtered["reference_group"].iloc[0]
if selected_attr == "TP_COR_RACA":
    ref_group = RACA_MAP.get(ref_group, ref_group)

col1, col2, col3 = st.columns(3)
col1.metric("Menor Impacto Disparate (DI)", f"{min_di:.3f}", delta="Viés Detectado" if min_di < 0.8 else "Dentro do Limite", delta_color="inverse" if min_di < 0.8 else "normal")
col2.metric("Grupo de Referência", str(ref_group))
col3.metric("Limiar da Regra dos 80%", "0.800")

st.subheader("Tabela de Impacto Disparate (DI) por Subgrupo")

# Função de estilização para destacar DI < 0.8
def highlight_low_di(val):
    color = "#ff4b4b" if val < 0.8 else "#28a745"
    return f"background-color: {color}; color: white; font-weight: bold;"

display_df = df_fairness_filtered[[
    "group_label", "total_count", "positive_count", "selection_rate", "disparate_impact", "alert_di_low"
]].rename(columns={
    "group_label": "Subgrupo",
    "total_count": "Total de Candidatos",
    "positive_count": "Aprovados (Y=1)",
    "selection_rate": "Taxa de Seleção",
    "disparate_impact": "Impacto Disparate (DI)",
    "alert_di_low": "Alerta de Viés (< 0.8)"
})

st.dataframe(
    display_df.style.map(highlight_low_di, subset=["Impacto Disparate (DI)"])
    .format({"Taxa de Seleção": "{:.2%}", "Impacto Disparate (DI)": "{:.3f}"}),
    use_container_width=True
)

# Gráfico Comparativo de Taxas de Seleção
fig_di = px.bar(
    df_fairness_filtered,
    x="group_label",
    y="selection_rate",
    color="alert_di_low",
    color_discrete_map={True: "#ff4b4b", False: "#0068c9"},
    labels={"group_label": "Subgrupo", "selection_rate": "Taxa de Seleção", "alert_di_low": "Alerta de Viés"},
    title=f"Taxa de Seleção por {ATTR_LABELS[selected_attr]} (Vermelho = DI < 0.8)"
)
fig_di.add_hline(y=df_fairness_filtered["reference_selection_rate"].iloc[0] * 0.8, line_dash="dash", line_color="red", annotation_text="Limiar 80%")
st.plotly_chart(fig_di, use_container_width=True)

st.divider()

# -----------------------------------------------------------------------------
# Painel 2: Explicabilidade Algorítmica (XAI com SHAP)
# -----------------------------------------------------------------------------
st.header("2. Módulo de Explicabilidade Algorítmica (SHAP)")

feature_cols = [c for c in df_shap.columns if c not in ["row_id", "base_value"]]

tab_global, tab_local = st.tabs(["📊 Explicabilidade Global (Summary Plot)", "🔍 Explicabilidade Local (Waterfall Plot)"])

with tab_global:
    st.subheader("Importância Global das Features (SHAP Summary Plot)")
    
    # Preparação das matrizes SHAP para plot
    X_shap_mat = df_shap[feature_cols].values
    X_feat_mat = df_features[feature_cols].values

    fig_shap, ax = plt.subplots(figsize=(10, 5))
    shap.summary_plot(X_shap_mat, X_feat_mat, feature_names=feature_cols, show=False)
    st.pyplot(fig_shap, bbox_inches="tight")
    plt.close(fig_shap)

with tab_local:
    st.subheader("Análise Individual de Impacto Algorítmico")
    
    candidate_idx = st.number_input(
        "Selecione o ID do Candidato (Índice da Amostra)",
        min_value=0,
        max_value=len(df_shap) - 1,
        value=0,
        step=1
    )

    row_shap = df_shap.iloc[candidate_idx]
    row_feat = df_features.iloc[candidate_idx]
    base_val = row_shap["base_value"]

    shap_vals_cand = row_shap[feature_cols].values
    feat_vals_cand = row_feat[feature_cols].values

    # Gráfico Waterfall via Plotly
    df_waterfall = pd.DataFrame({
        "Feature": feature_cols,
        "SHAP_Value": shap_vals_cand,
        "Feature_Value": feat_vals_cand
    }).sort_values(by="SHAP_Value", key=abs, ascending=True)

    fig_waterfall = go.Figure(go.Waterfall(
        name="SHAP",
        orientation="h",
        measure=["relative"] * len(df_waterfall),
        y=df_waterfall["Feature"] + " = " + df_waterfall["Feature_Value"].astype(str),
        x=df_waterfall["SHAP_Value"],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": "#ff4b4b"}},
        decreasing={"marker": {"color": "#0068c9"}}
    ))

    fig_waterfall.update_layout(
        title=f"Contribuição das Features para o Candidato #{candidate_idx} (Base Value = {base_val:.3f})",
        xaxis_title="Impacto no Valor de Output do Modelo (Log-Odds SHAP)",
        height=500
    )

    st.plotly_chart(fig_waterfall, use_container_width=True)