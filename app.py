"""
AuditEdu-XAI — camada online (dashboard Streamlit).

Regras (CLAUDE.md / .claude/agents/04_ui_ux_designer.md):
- só LÊ os Parquets estáticos de data/outputs/ (nenhum treino, predição, SHAP ou bootstrap aqui);
- toda leitura passa por @st.cache_data; gráficos matplotlib são cacheados como PNG;
- os dois primeiros seletores da barra lateral (edição, atributo) são usados por src/06_benchmark_app.py.
"""
import io
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st

st.set_page_config(page_title="AuditEdu-XAI | Auditoria Algorítmica", page_icon="⚖️", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "data" / "outputs"
YEARS = [2019, 2020, 2021]
DI_THRESHOLD = 0.8
SMALL_GROUP_POSITIVES = 30  # abaixo disso, TPR/IC têm baixa precisão

# Paleta semântica (sempre acompanhada de ícone + texto: nunca só cor)
RED, GREEN, BLUE, GRAY, INK = "#ff4b4b", "#28a745", "#0068c9", "#8a8984", "#31333f"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6", "#e87ba4"]

# -----------------------------------------------------------------------------
# Dicionário de rótulos (INEP) — idêntico ao usado na monografia (gerar_resultados.py)
# -----------------------------------------------------------------------------
RACA = {"0": "Não declarada", "1": "Branca", "2": "Preta", "3": "Parda", "4": "Amarela", "5": "Indígena"}
RACA_ORDER = ["1", "2", "3", "4", "5", "0"]
RENDA = {
    "A": "Nenhuma renda", "B": "Até 1 salário mínimo", "C": "1 a 1,5 SM", "D": "1,5 a 2 SM",
    "E": "2 a 2,5 SM", "F": "2,5 a 3 SM", "G": "3 a 4 SM", "H": "4 a 5 SM", "I": "5 a 6 SM",
    "J": "6 a 7 SM", "K": "7 a 8 SM", "L": "8 a 9 SM", "M": "9 a 10 SM", "N": "10 a 12 SM",
    "O": "12 a 15 SM", "P": "15 a 20 SM", "Q": "Acima de 20 SM",
}
ESCOLARIDADE = {
    "A": "nunca estudou", "B": "fundamental I incompleto", "C": "fundamental I completo",
    "D": "fundamental II completo", "E": "ensino médio completo", "F": "superior completo",
    "G": "pós-graduação", "H": "não sabe",
}
COMPUTADOR = {"A": "Sem computador em casa", "B": "1 computador em casa", "C": "2 computadores em casa",
              "D": "3 computadores em casa", "E": "4 ou mais computadores em casa"}
INTERNET = {"A": "Sem internet em casa", "B": "Com internet em casa"}
ATTR_LABELS = {"TP_COR_RACA": "Raça / Cor", "Q006": "Renda familiar mensal"}
VAR_LABELS = {"TP_COR_RACA": "Raça/cor", "Q006": "Renda familiar", "Q001": "Escolaridade do pai",
              "Q002": "Escolaridade da mãe", "Q024": "Computador", "Q025": "Internet"}

HELP = {
    "di": "Impacto Disparate: taxa de seleção do grupo ÷ taxa do grupo de referência (o de maior taxa). "
          "Abaixo de 0,8 dispara o alerta da regra dos quatro quintos — um indicador de TRIAGEM de "
          "disparidade estatística, não um teste jurídico de discriminação.",
    "ts": "Proporção de candidatos que o modelo classifica como Ŷ = 1, isto é, com Nota Média Geral prevista "
          "igual ou superior à MEDIANA da edição. É desempenho relativo ao ano — não aprovação.",
    "tpr": "Taxa de verdadeiros positivos (Igualdade de Oportunidades): entre os candidatos que de fato ficaram "
           "acima da mediana, a fração que o modelo reconhece.",
    "fpr": "Taxa de falsos positivos: entre os que ficaram abaixo da mediana, a fração que o modelo classifica acima.",
    "ic": "Intervalo de confiança de 95% por bootstrap (2.000 reamostragens do conjunto de teste, modelo fixo).",
    "dil": "DI calculado sobre as notas REAIS. Se o DI do modelo for menor, o modelo AMPLIA a desigualdade dos dados.",
    "shap": "Valores SHAP medem quanto cada atributo empurra a previsão para cima ou para baixo, em log-odds "
            "(escala logarítmica da chance de ficar acima da mediana). Descrevem o modelo, não causas.",
    "auc": "Área sob a curva ROC no conjunto de teste: desempenho global do modelo (0,5 = acaso; 1 = perfeito).",
}


def br(x, nd=3):
    """Número com vírgula decimal."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.{nd}f}".replace(".", ",")


def pct(x, nd=1):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.{nd}f}%".replace(".", ",")


def ci(lo, hi):
    if pd.isna(lo) or pd.isna(hi):
        return "—"
    return f"[{br(lo)}; {br(hi)}]"


def group_label(attr: str, g: str) -> str:
    return RACA.get(g, g) if attr == "TP_COR_RACA" else f"{g} · {RENDA.get(g, g)}"


def split_feature(col: str):
    var, val = col.replace("cat__", "").rsplit("_", 1)
    if var == "TP_COR_RACA":
        val = str(int(float(val)))
    return var, val


def value_label(var: str, val: str) -> str:
    if var == "TP_COR_RACA":
        return RACA.get(val, val)
    if var == "Q006":
        return RENDA.get(val, val)
    if var in ("Q001", "Q002"):
        return ESCOLARIDADE.get(val, val).capitalize()
    if var == "Q024":
        return COMPUTADOR.get(val, val)
    if var == "Q025":
        return INTERNET.get(val, val)
    return val


def feature_label(col: str) -> str:
    var, val = split_feature(col)
    if var in ("Q024", "Q025"):
        return value_label(var, val)
    if var == "Q001":
        return f"Pai: {ESCOLARIDADE.get(val, val)}"
    if var == "Q002":
        return f"Mãe: {ESCOLARIDADE.get(val, val)}"
    return f"{VAR_LABELS[var]}: {value_label(var, val)}"


def status_of(row) -> str:
    if row["group"] == row["reference_group"]:
        return "🔷 Referência"
    return "⚠️ Viés (DI < 0,8)" if row["alert_di_low"] else "✅ Conforme"


def add_legend_line(fig, name, color=INK, dash="dash", width=1.5):
    """Entrada de legenda para linhas de referência (evita texto solto sobre o gráfico)."""
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=name, hoverinfo="skip",
                             line=dict(color=color, dash=dash, width=width)))


def style_fig(fig, height=None):
    # A legenda é ancorada na borda inferior do CONTÊINER (yref="container"), não em fração da área do
    # gráfico: assim ela nunca invade o título do eixo x, qualquer que seja a altura do gráfico.
    fig.update_layout(separators=",.", margin=dict(l=10, r=10, t=50, b=85), plot_bgcolor="white",
                      font=dict(color=INK),
                      legend=dict(orientation="h", yref="container", y=0.0, yanchor="bottom", x=0, xanchor="left"))
    fig.update_xaxes(gridcolor="#ececec", zeroline=False)
    fig.update_yaxes(gridcolor="#ececec", zeroline=False)
    if height:
        fig.update_layout(height=height)
    return fig


# -----------------------------------------------------------------------------
# Leitura cacheada (apenas Parquet estático)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_audit_data(year: int):
    paths = {k: OUTPUTS_DIR / f"{k}_{year}.parquet"
             for k in ("fairness_metrics", "shap_values", "shap_features", "model_metrics", "shap_feature_summary")}
    if not all(p.exists() for p in paths.values()):
        return None
    return {k: pd.read_parquet(p) for k, p in paths.items()}


@st.cache_data(ttl=3600)
def load_all_years(name: str) -> pd.DataFrame:
    frames = [pd.read_parquet(OUTPUTS_DIR / f"{name}_{y}.parquet") for y in YEARS
              if (OUTPUTS_DIR / f"{name}_{y}.parquet").exists()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


@st.cache_data(ttl=3600)
def load_injection() -> pd.DataFrame:
    p = OUTPUTS_DIR / "bias_injection_summary.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


@st.cache_data(ttl=3600)
def candidate_positions_by_race(year: int) -> dict:
    """Posições (linhas do teste) de cada grupo de raça/cor — calculadas uma vez por edição."""
    feat_df = load_audit_data(year)["shap_features"]
    race_cols = [c for c in feat_df.columns if c.startswith("cat__TP_COR_RACA_")]
    race_of_row = feat_df[race_cols].idxmax(axis=1).map(lambda c: split_feature(c)[1]).to_numpy()
    out = {"Todos": list(range(len(feat_df)))}
    for g in RACA_ORDER:
        out[g] = [i for i, r in enumerate(race_of_row) if r == g]
    return out


@st.cache_data(ttl=3600, show_spinner="Renderizando o gráfico detalhado (apenas na primeira vez)…")
def render_shap_beeswarm_png(year: int) -> bytes:
    """Beeswarm SHAP com rótulos legíveis — renderizado uma vez por edição e reutilizado como PNG."""
    data = load_audit_data(year)
    cols = [c for c in data["shap_values"].columns if c not in ("row_id", "base_value")]
    plt.figure(figsize=(10, 6))
    shap.summary_plot(data["shap_values"][cols].values, data["shap_features"][cols].values,
                      feature_names=[feature_label(c) for c in cols], max_display=15, show=False)
    buffer = io.BytesIO()
    plt.gcf().savefig(buffer, format="png", bbox_inches="tight", dpi=110)
    plt.close("all")
    return buffer.getvalue()


# -----------------------------------------------------------------------------
# Barra lateral (os dois primeiros seletores são usados pelo benchmark)
# -----------------------------------------------------------------------------
st.sidebar.title("⚖️ AuditEdu-XAI")
selected_year = st.sidebar.selectbox("Edição do ENEM", YEARS, index=1,
                                     help="Cada edição tem seu próprio modelo, treinado e auditado separadamente.")
selected_attr = st.sidebar.selectbox("Atributo protegido", list(ATTR_LABELS), format_func=ATTR_LABELS.get,
                                     help="Grupo demográfico ou socioeconômico em relação ao qual o viés é medido.")
st.sidebar.divider()
st.sidebar.markdown(
    "**Legenda**\n\n"
    "⚠️ **Viés** — DI abaixo de 0,8  \n"
    "✅ **Conforme** — DI de 0,8 ou mais  \n"
    "🔷 **Referência** — grupo de maior taxa de seleção  \n"
    "🔸 **Baixa precisão** — menos de 30 casos positivos"
)
st.sidebar.divider()
with st.sidebar.expander("ℹ️ Sobre esta ferramenta"):
    st.markdown(
        "Audita classificadores **XGBoost** que preveem se um candidato do ENEM ficará **acima da mediana** "
        "da Nota Média Geral da edição, usando apenas variáveis socioeconômicas.\n\n"
        "- Métricas calculadas **só no conjunto de teste** (10.000 candidatos por edição, nunca vistos no treino).\n"
        "- Todos os resultados são **pré-calculados** em lote; o painel apenas os exibe.\n"
        "- O modelo é **objeto de auditoria**, não um instrumento de decisão sobre candidatos reais."
    )

data = load_audit_data(selected_year)
if data is None:
    st.error(f"⚠️ Os arquivos da edição {selected_year} não foram encontrados em `data/outputs/`. "
             "Execute `src/04_audit_engine.py`.")
    st.stop()

fair_all = data["fairness_metrics"]
fair = fair_all[fair_all["protected_attribute"] == selected_attr].copy()
order = RACA_ORDER if selected_attr == "TP_COR_RACA" else sorted(fair["group"])
fair["order"] = fair["group"].map({g: i for i, g in enumerate(order)})
fair = fair.sort_values("order")
fair["label"] = [group_label(selected_attr, g) for g in fair["group"]]
fair["status"] = fair.apply(status_of, axis=1)
fair["small"] = fair["true_positive_label_count"] < SMALL_GROUP_POSITIVES
ref_row = fair[fair["group"] == fair["reference_group"]].iloc[0]
non_ref = fair[fair["group"] != fair["reference_group"]]
metrics = data["model_metrics"].iloc[0]

# -----------------------------------------------------------------------------
# Cabeçalho e KPIs
# -----------------------------------------------------------------------------
st.title("Auditoria de viés em modelos preditivos do ENEM")
st.caption(f"Edição **{selected_year}** · atributo protegido: **{ATTR_LABELS[selected_attr]}** · "
           "métricas do conjunto de teste (10.000 candidatos)")

worst = non_ref.loc[non_ref["disparate_impact"].idxmin()]
n_alert = int(non_ref["alert_di_low"].sum())
overall_rate = fair["positive_count"].sum() / fair["total_count"].sum()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Menor Impacto Disparate", br(worst["disparate_impact"]),
          delta=f"{worst['disparate_impact'] - DI_THRESHOLD:+.3f} vs. limiar 0,8".replace(".", ","),
          delta_color="normal", help=HELP["di"], border=True)
k1.caption(f"{'⚠️' if worst['alert_di_low'] else '✅'} {worst['label']}")
k2.metric("Grupos em alerta", f"{n_alert} de {len(non_ref)}",
          help="Subgrupos (exceto a referência) com DI abaixo de 0,8.", border=True)
k2.caption(f"🔷 Referência: {ref_row['label']}")
k3.metric("Taxa de seleção (Ŷ = 1)", pct(overall_rate), help=HELP["ts"], border=True)
k3.caption("acima da mediana da edição")
k4.metric("AUC-ROC do modelo", br(metrics["roc_auc"]), help=HELP["auc"], border=True)
k4.caption(f"acurácia {pct(metrics['accuracy'])}")

tab_eq, tab_xai, tab_inj = st.tabs(["📊 Visão Geral de Equidade", "🔍 Explicabilidade (SHAP)",
                                     "🧪 Injeção Sintética de Viés"])

# =============================================================================
# ABA 1 — Equidade
# =============================================================================
with tab_eq:
    col_chart, col_text = st.columns([3, 2])
    with col_chart:
        colors = [BLUE if s.startswith("🔷") else RED if s.startswith("⚠️") else GREEN for s in fair["status"]]
        is_ref = fair["group"] == fair["reference_group"]
        err_plus = (fair["di_ci_high"] - fair["disparate_impact"]).where(~is_ref).fillna(0)
        err_minus = (fair["disparate_impact"] - fair["di_ci_low"]).where(~is_ref).fillna(0)
        ci_txt = ["— (referência)" if r else ci(a, b) for r, a, b in zip(is_ref, fair["di_ci_low"], fair["di_ci_high"])]
        fig = go.Figure(go.Bar(
            x=fair["disparate_impact"], y=fair["label"], orientation="h", marker_color=colors, showlegend=False,
            error_x=dict(type="data", array=err_plus, arrayminus=err_minus, color=GRAY, thickness=1.5, width=5),
            customdata=list(zip(fair["status"], ci_txt, [br(v) for v in fair["disparate_impact"]],
                                [f"{int(n):,}".replace(",", ".") for n in fair["total_count"]])),
            hovertemplate="<b>%{y}</b><br>%{customdata[0]}<br>DI: %{customdata[2]}<br>"
                          "IC 95%: %{customdata[1]}<br>Candidatos no teste: %{customdata[3]}<extra></extra>",
        ))
        # Rótulos posicionados DEPOIS da ponta do intervalo de confiança (não colidem com a linha do IC) e com
        # fundo branco, para que a linha tracejada do limiar passe POR TRÁS do texto em vez de cortá-lo.
        label_x = pd.concat([fair["disparate_impact"] + err_plus, fair["disparate_impact"]], axis=1).max(axis=1) + 0.02
        for lx, lab, s, v in zip(label_x, fair["label"], fair["status"], fair["disparate_impact"]):
            fig.add_annotation(x=lx, y=lab, text=f"{s.split()[0]} {br(v)}", showarrow=False, xanchor="left",
                               bgcolor="rgba(255,255,255,0.92)", borderpad=1, font=dict(size=12, color=INK))
        fig.add_vline(x=DI_THRESHOLD, line_dash="dash", line_color=INK, line_width=1.5, layer="below")
        add_legend_line(fig, "Limiar 0,8 (regra dos 4/5)")
        add_legend_line(fig, "Intervalo de confiança 95%", color=GRAY, dash="solid", width=2)
        fig.update_layout(title="Impacto Disparate por grupo", xaxis_title="Impacto Disparate",
                          xaxis_range=[0, 1.32], yaxis_autorange="reversed", showlegend=True)
        st.plotly_chart(style_fig(fig, height=max(340, 34 * len(fair) + 150)), width="stretch")
        st.caption("A linha cinza na ponta de cada barra é o **intervalo de confiança de 95%**: a faixa em que o DI "
                   "provavelmente estaria se repetíssemos a amostragem. Linha longa = estimativa incerta "
                   "(grupos pequenos). Se a linha inteira está à esquerda do limiar, o alerta é robusto.")

    with col_text:
        st.subheader("O que este gráfico mostra")
        st.markdown(
            f"Para cada grupo, o **DI** compara a chance de o modelo prever *acima da mediana* com a do grupo "
            f"de referência (**{ref_row['label']}**, taxa de {pct(ref_row['selection_rate'])}).\n\n"
            f"- **{n_alert} de {len(non_ref)}** grupos ficam abaixo de 0,8.\n"
            f"- O pior caso é **{worst['label']}**: DI de **{br(worst['disparate_impact'])}**, ou seja, "
            f"a chance de seleção é cerca de **{pct(worst['disparate_impact'], 0)}** da do grupo de referência."
        )
        robust = non_ref[non_ref["alert_di_low"] & (non_ref["p_di_below_threshold"] >= 0.95)]
        if n_alert:
            st.success(f"🔒 {len(robust)} de {n_alert} alertas se mantêm em pelo menos 95% das reamostragens bootstrap.",
                       icon=None)
        if fair["small"].any():
            st.warning("🔸 Grupos com poucos casos positivos (" +
                       ", ".join(fair.loc[fair["small"], "label"]) +
                       ") têm estimativas imprecisas: veja a largura dos intervalos.")
        with st.expander("📏 Como ler a regra dos 80%"):
            st.markdown(
                "A regra dos quatro quintos vem das diretrizes da *Equal Employment Opportunity Commission* (EUA): "
                "uma taxa de seleção inferior a 80% da do grupo mais favorecido indica **impacto adverso**.\n\n"
                "Aqui ela é usada como **indicador de triagem de disparidade estatística**: um alerta sinaliza que "
                "a disparidade merece investigação, mas **não** estabelece ilicitude ou intenção discriminatória."
            )

    st.divider()
    st.subheader("O modelo apenas reproduz a desigualdade dos dados, ou a amplia?")
    amp = non_ref.copy()
    fig_amp = go.Figure()
    for _, r in amp.iterrows():
        fig_amp.add_trace(go.Scatter(x=[r["label_disparate_impact"], r["disparate_impact"]], y=[r["label"], r["label"]],
                                     mode="lines", line=dict(color=GRAY, width=2), showlegend=False, hoverinfo="skip"))
    fig_amp.add_trace(go.Scatter(x=amp["label_disparate_impact"], y=amp["label"], mode="markers", name="Notas reais (dados)",
                                 marker=dict(size=11, color="white", line=dict(color=BLUE, width=2.5)),
                                 hovertemplate="<b>%{y}</b><br>DI nas notas reais: %{x:.3f}<extra></extra>"))
    fig_amp.add_trace(go.Scatter(x=amp["disparate_impact"], y=amp["label"], mode="markers", name="Previsões do modelo",
                                 marker=dict(size=11, color=BLUE, symbol="diamond"),
                                 hovertemplate="<b>%{y}</b><br>DI do modelo: %{x:.3f}<extra></extra>"))
    fig_amp.add_vline(x=DI_THRESHOLD, line_dash="dash", line_color=INK, line_width=1.5)
    add_legend_line(fig_amp, "Limiar 0,8")
    fig_amp.update_layout(xaxis_title="Impacto Disparate", xaxis_range=[0, 1.05], yaxis_autorange="reversed")
    c_amp, c_amp_txt = st.columns([3, 2])
    c_amp.plotly_chart(style_fig(fig_amp, height=max(300, 30 * len(amp) + 130)), width="stretch")
    n_amp = int((amp["disparate_impact"] < amp["label_disparate_impact"]).sum())
    crossed = amp[(amp["label_disparate_impact"] >= DI_THRESHOLD) & (amp["disparate_impact"] < DI_THRESHOLD)]
    c_amp_txt.markdown(
        f"Em **{n_amp} de {len(amp)}** grupos, o losango (modelo) fica à **esquerda** do círculo (dados): "
        "o modelo concentra ainda mais as previsões positivas no grupo de referência."
    )
    if len(crossed):
        c_amp_txt.error("⚠️ Alerta criado pelo modelo: " + ", ".join(crossed["label"]) +
                        " — conforme nos dados, abaixo de 0,8 nas previsões.")
    c_amp_txt.caption(HELP["dil"])

    st.divider()
    st.subheader("Tabela detalhada")
    table = pd.DataFrame({
        "Grupo": [("🔸 " if s else "") + l for l, s in zip(fair["label"], fair["small"])],
        "Situação": fair["status"],
        "Candidatos": fair["total_count"],
        "Acima da mediana (real)": fair["true_positive_label_count"],
        "Taxa de seleção": [pct(v) for v in fair["selection_rate"]],
        "DI": [br(v) for v in fair["disparate_impact"]],
        "IC 95% do DI": [("—" if g == r else ci(a, b)) for g, r, a, b in
                         zip(fair["group"], fair["reference_group"], fair["di_ci_low"], fair["di_ci_high"])],
        "DI nas notas reais": [br(v) for v in fair["label_disparate_impact"]],
        "TPR": [br(v) for v in fair["tpr"]],
        "IC 95% da TPR": [ci(a, b) for a, b in zip(fair["tpr_ci_low"], fair["tpr_ci_high"])],
        "FPR": [br(v) for v in fair["fpr"]],
    })
    st.dataframe(table, hide_index=True, width="stretch", column_config={
        "Taxa de seleção": st.column_config.TextColumn(help=HELP["ts"]),
        "DI": st.column_config.TextColumn(help=HELP["di"]),
        "IC 95% do DI": st.column_config.TextColumn(help=HELP["ic"]),
        "DI nas notas reais": st.column_config.TextColumn(help=HELP["dil"]),
        "TPR": st.column_config.TextColumn(help=HELP["tpr"]),
        "IC 95% da TPR": st.column_config.TextColumn(help=HELP["ic"]),
        "FPR": st.column_config.TextColumn(help=HELP["fpr"]),
    })
    export = fair[["label", "status", "total_count", "true_positive_label_count", "selection_rate", "disparate_impact",
                   "di_ci_low", "di_ci_high", "label_disparate_impact", "tpr", "tpr_ci_low", "tpr_ci_high", "fpr"]]
    st.download_button("⬇️ Baixar tabela (CSV)", export.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
                       file_name=f"auditedu_equidade_{selected_attr}_{selected_year}.csv", mime="text/csv")

    with st.expander("📈 Evolução entre as edições (2019–2021)"):
        hist = load_all_years("fairness_metrics")
        hist = hist[(hist["protected_attribute"] == selected_attr) & (hist["group"] != hist["reference_group"])]
        if selected_attr == "Q006":
            hist = hist[hist["group"].isin(list("ABCDEF"))]
            st.caption("Exibidas as faixas A–F (até 3 salários mínimos), onde se concentram os alertas.")
        fig_ev = go.Figure()
        for i, (g, h) in enumerate(hist.sort_values("year").groupby("group", sort=False)):
            fig_ev.add_trace(go.Scatter(
                x=h["year"], y=h["disparate_impact"], mode="lines+markers+text", name=group_label(selected_attr, g),
                line=dict(color=SERIES[i % len(SERIES)], width=2), marker=dict(size=8),
                error_y=dict(type="data", array=h["di_ci_high"] - h["disparate_impact"],
                             arrayminus=h["disparate_impact"] - h["di_ci_low"], thickness=1),
                text=["", "", group_label(selected_attr, g)], textposition="middle right",
                customdata=[ci(a, b) for a, b in zip(h["di_ci_low"], h["di_ci_high"])],
                hovertemplate="<b>" + group_label(selected_attr, g) + "</b><br>%{x}: DI %{y:.3f}"
                              "<br>IC 95%: %{customdata}<extra></extra>"))
        fig_ev.add_hline(y=DI_THRESHOLD, line_dash="dash", line_color=INK, line_width=1.5)
        add_legend_line(fig_ev, "Limiar 0,8")
        fig_ev.update_layout(xaxis=dict(tickvals=YEARS, range=[2018.8, 2021.9]), yaxis_title="Impacto Disparate",
                             yaxis_range=[0, 1.05])
        st.plotly_chart(style_fig(fig_ev, height=420), width="stretch")

# =============================================================================
# ABA 2 — Explicabilidade
# =============================================================================
with tab_xai:
    sub_global, sub_local = st.tabs(["🌐 O que pesa nas previsões", "👤 Explicar um candidato"])

    with sub_global:
        summ = data["shap_feature_summary"].sort_values("mean_abs_phi", ascending=False).head(12)
        fig_imp = go.Figure(go.Bar(
            x=summ["mean_abs_phi"], y=[feature_label(f) for f in summ["feature"]], orientation="h",
            marker_color=[RED if f.endswith("Q024_A") else BLUE for f in summ["feature"]],
            text=[br(v) for v in summ["mean_abs_phi"]], textposition="outside", cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>importância média: %{x:.3f}<extra></extra>"))
        fig_imp.update_layout(title=f"Os 12 atributos mais influentes — {selected_year}",
                              xaxis_title="Importância média |SHAP| (log-odds)", yaxis_autorange="reversed")
        c_imp, c_imp_txt = st.columns([3, 2])
        c_imp.plotly_chart(style_fig(fig_imp, height=480), width="stretch")
        with c_imp_txt:
            st.markdown("**Como ler:** quanto maior a barra, mais o atributo altera, em média, a previsão do modelo.",
                        help=HELP["shap"])
            allsum = load_all_years("shap_feature_summary")
            q = allsum[allsum["feature"].str.endswith("Q024_A")].sort_values("year")
            st.markdown("##### 💻 Destaque: exclusão digital")
            st.markdown("Não ter computador em casa é o atributo **mais influente nas três edições**:")
            cols_q = st.columns(len(q))
            for c, (_, r) in zip(cols_q, q.iterrows()):
                c.metric(str(int(r["year"])), br(r["conditional_gap"]),
                         help="Diferença média de SHAP entre quem tem e quem não tem computador (log-odds).")
                c.caption(f"{pct(r['prevalence'], 0)} sem computador")
            st.caption("O peso por candidato cresce enquanto a proporção de candidatos sem computador cai, "
                       "o que descarta um efeito de composição. A coincidência com a pandemia é **associativa**, "
                       "não causal.")
        if st.toggle("Mostrar gráfico detalhado (beeswarm SHAP)",
                     help="Cada ponto é um candidato; a cor indica se ele possui o atributo. "
                          "Renderizado uma vez por edição."):
            st.image(render_shap_beeswarm_png(selected_year), width="stretch")

    with sub_local:
        shap_df, feat_df = data["shap_values"], data["shap_features"]
        feat_cols = [c for c in shap_df.columns if c not in ("row_id", "base_value")]
        c_f1, c_f2 = st.columns([1, 2])
        race_filter = c_f1.selectbox("Filtrar por raça/cor", ["Todos"] + RACA_ORDER,
                                     format_func=lambda g: g if g == "Todos" else RACA[g])
        positions = candidate_positions_by_race(selected_year)[race_filter]
        pick = c_f2.slider(f"Candidato ({len(positions):,} disponíveis)".replace(",", "."),
                           0, len(positions) - 1, 0, help="Índice do candidato dentro do filtro escolhido.")
        pos = positions[pick]
        phi = shap_df.iloc[pos][feat_cols].astype(float)
        fx = feat_df.iloc[pos][feat_cols]
        base = float(shap_df.iloc[pos]["base_value"])
        logit = base + float(phi.sum())
        prob = 1 / (1 + math.exp(-logit))

        profile = {}
        for c in feat_cols:
            if fx[c] == 1:
                var, val = split_feature(c)
                profile[VAR_LABELS[var]] = value_label(var, val)
        c_prof, c_pred = st.columns([3, 1])
        c_prof.markdown("**Perfil do candidato** (identificado só pelo índice; dados anonimizados)  \n" +
                        " · ".join(f"**{k}:** {v}" for k, v in profile.items()))
        c_pred.metric("Chance estimada", pct(prob),
                      delta="previsto: acima da mediana" if prob >= 0.5 else "previsto: abaixo da mediana",
                      delta_color="off", border=True,
                      help="Probabilidade, segundo o modelo, de o candidato ficar acima da mediana da Nota Média "
                           "Geral da edição. Obtida somando ao ponto de partida as contribuições SHAP abaixo.")

        top = phi.reindex(phi.abs().sort_values(ascending=False).index)
        shown, rest = top.head(8), top.iloc[8:]
        names = [(feature_label(c) if fx[c] == 1 else f"não: {feature_label(c)}") for c in shown.index]
        y = ["Ponto de partida (média)"] + names + [f"Demais {len(rest)} atributos", "Previsão final"]
        x = [base] + list(shown.values) + [float(rest.sum()), 0]
        hover = ([f"<b>Ponto de partida</b><br>média do modelo: {br(base)} log-odds"] +
                 [f"<b>{n}</b><br>contribuição: {'+' if v >= 0 else ''}{br(v)} log-odds" for n, v in zip(names, shown.values)] +
                 [f"<b>Demais {len(rest)} atributos</b><br>soma: {'+' if rest.sum() >= 0 else ''}{br(float(rest.sum()))} log-odds",
                  f"<b>Previsão final</b><br>{br(logit)} log-odds = {pct(prob)} de chance"])
        fig_w = go.Figure(go.Waterfall(
            orientation="h", measure=["absolute"] + ["relative"] * (len(shown) + 1) + ["total"], y=y, x=x,
            hovertext=hover, hoverinfo="text",
            text=[br(v) for v in x[:-1]] + [br(logit)], textposition="outside",
            increasing=dict(marker=dict(color=GREEN)), decreasing=dict(marker=dict(color=RED)),
            totals=dict(marker=dict(color=BLUE)), connector=dict(line=dict(color=GRAY))))
        fig_w.update_layout(title="Como cada atributo empurrou a previsão (log-odds)",
                            xaxis_title="Contribuição SHAP (log-odds)", yaxis_autorange="reversed", showlegend=False)
        st.plotly_chart(style_fig(fig_w, height=520), width="stretch")
        st.caption("🟩 aumenta a chance prevista · 🟥 reduz · \"não: …\" indica a ausência daquela característica. "
                   + HELP["shap"] + " A decomposição subsidia, mas não substitui, a explicação devida ao titular "
                   "pelo Art. 20 da LGPD.")

# =============================================================================
# ABA 3 — Injeção sintética de viés
# =============================================================================
with tab_inj:
    inj = load_injection()
    if inj.empty:
        st.info("Resultados do experimento não encontrados. Execute `src/05_bias_injection.py`.")
    else:
        st.markdown(
            "**Teste de sensibilidade da ferramenta.** Inserimos de propósito um viés de tamanho conhecido: uma "
            "fração dos candidatos de um grupo que ficaram *acima* da mediana é rotulada como *abaixo* no treino. "
            "Depois o modelo é retreinado e auditado no teste original. O **controle** rebaixa o mesmo número de "
            "rótulos, mas sorteados entre **todos** os grupos."
        )
        target = st.radio("Grupo-alvo da injeção", ["Parda", "Indígena"], horizontal=True,
                          help="Resultados da edição selecionada na barra lateral; independem do atributo protegido.")
        sub = inj[(inj["year"] == selected_year) & (inj["target_label"] == target)].sort_values("intensity")
        injc, ctrl = sub[sub["scenario"] == "injection"], sub[sub["scenario"] == "control"]
        at50 = injc[injc["intensity"] == 0.5].iloc[0]
        at0 = injc[injc["intensity"] == 0.0].iloc[0]

        m1, m2, m3 = st.columns(3)
        m1.metric(f"DI do grupo {target} (0% → 50%)", f"{br(at0['disparate_impact_mean'])} → {br(at50['disparate_impact_mean'])}",
                  delta=br(at50["disparate_impact_mean"] - at0["disparate_impact_mean"]), border=True)
        m2.metric("AUC-ROC (0% → 50%)", f"{br(at0['roc_auc_mean'])} → {br(at50['roc_auc_mean'])}",
                  delta=br(at50["roc_auc_mean"] - at0["roc_auc_mean"]), delta_color="off", border=True,
                  help="Variação pequena: a métrica global quase não percebe o viés injetado.")
        m3.metric("Efeito líquido do direcionamento", br(at50["di_net_effect"]), border=True,
                  help="Queda do DI na injeção menos a queda no controle, com 50% de intensidade.")

        # Eixo numérico (espaçamento proporcional à intensidade) com marcas explícitas e sufixo "%"
        xs = [int(round(100 * p)) for p in injc["intensity"]]
        x_axis = dict(tickvals=xs, ticktext=[f"{x}%" for x in xs], range=[-3, max(xs) + 3])
        c_di, c_auc = st.columns(2)
        fig_i = go.Figure()
        for df_, name, dash, sym in [(injc, "Injeção direcionada", "solid", "circle"), (ctrl, "Controle negativo", "dash", "circle-open")]:
            fig_i.add_trace(go.Scatter(x=xs, y=df_["disparate_impact_mean"], name=name, mode="lines+markers",
                                       line=dict(color=RED if dash == "solid" else GRAY, width=2.5, dash=dash),
                                       marker=dict(size=10, symbol=sym),
                                       error_y=dict(type="data", array=df_["disparate_impact_std"], thickness=1),
                                       customdata=[br(v) for v in df_["disparate_impact_std"]],
                                       hovertemplate="<b>" + name + "</b><br>intensidade %{x}%<br>DI médio: %{y:.3f}"
                                                     "<br>desvio-padrão: %{customdata}<extra></extra>"))
        fig_i.add_hline(y=DI_THRESHOLD, line_dash="dot", line_color=INK, line_width=1.5)
        add_legend_line(fig_i, "Limiar 0,8", dash="dot")
        fig_i.update_layout(title=f"DI do grupo {target}: a ferramenta percebe o viés", yaxis_title="Impacto Disparate",
                            xaxis_title="Intensidade da injeção", yaxis_range=[0, 0.9], xaxis=x_axis)
        c_di.plotly_chart(style_fig(fig_i, height=400), width="stretch")
        fig_a = go.Figure(go.Scatter(x=xs, y=injc["roc_auc_mean"], mode="lines+markers", name="AUC-ROC",
                                     line=dict(color=BLUE, width=2.5), marker=dict(size=10),
                                     hovertemplate="intensidade %{x}%<br>AUC-ROC: %{y:.3f}<extra></extra>"))
        fig_a.update_layout(title="AUC-ROC: a métrica convencional mal se move", yaxis_title="AUC-ROC",
                            xaxis_title="Intensidade da injeção", yaxis_range=[0.5, 0.8], xaxis=x_axis)
        c_auc.plotly_chart(style_fig(fig_a, height=400), width="stretch")
        st.info(f"Com 50% de injeção, o DI do grupo {target} cai de {br(at0['disparate_impact_mean'])} para "
                f"{br(at50['disparate_impact_mean'])}, enquanto a AUC-ROC varia {br(at50['roc_auc_mean'] - at0['roc_auc_mean'])}. "
                "**Um modelo validado só por métricas agregadas poderia ser aprovado mesmo excluindo esse grupo.**")
        with st.expander("🔬 Protocolo do experimento"):
            st.markdown(
                "- Intensidades: 0%, 10%, 25% e 50% dos rótulos positivos de treino do grupo-alvo.\n"
                "- 5 repetições por cenário (barras de erro: ±1 desvio-padrão); 240 execuções no total.\n"
                "- Mesmos hiperparâmetros do modelo de produção; avaliação no conjunto de teste **sem alteração**.\n"
                "- Efeito líquido = queda do DI na injeção − queda do DI no controle."
            )
