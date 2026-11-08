"""
Gera as tabelas (article/tabelas/*.tex) e figuras (article/figuras/resultados/*.pdf)
do capítulo de Resultados lendo EXCLUSIVAMENTE data/outputs/*.parquet (CLAUDE.md, Regra 3).

Uso (na raiz do repositório):
    venv/Scripts/python.exe article/scripts/gerar_resultados.py

Nenhum número das tabelas é digitado à mão: rerodar o pipeline e este script
mantém o texto sincronizado com os artefatos.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "outputs"
TAB_DIR = ROOT / "article" / "tabelas"
FIG_DIR = ROOT / "article" / "figuras" / "resultados"
YEARS = [2019, 2020, 2021]

# ---------------------------------------------------------------------------
# Rótulos (dicionário de dados do INEP, questionário socioeconômico 2019-2021)
# ---------------------------------------------------------------------------
RACA = {"0": "Não declarada", "1": "Branca", "2": "Preta", "3": "Parda", "4": "Amarela", "5": "Indígena"}
RACA_ORDER = ["1", "2", "3", "4", "5", "0"]
RENDA = {
    "A": "Nenhuma renda", "B": "Até 1 SM", "C": "1 a 1,5 SM", "D": "1,5 a 2 SM", "E": "2 a 2,5 SM",
    "F": "2,5 a 3 SM", "G": "3 a 4 SM", "H": "4 a 5 SM", "I": "5 a 6 SM", "J": "6 a 7 SM",
    "K": "7 a 8 SM", "L": "8 a 9 SM", "M": "9 a 10 SM", "N": "10 a 12 SM", "O": "12 a 15 SM",
    "P": "15 a 20 SM", "Q": "Acima de 20 SM",
}
ESCOLARIDADE = {
    "A": "nunca estudou", "B": "fund. I incompleto", "C": "fund. I completo", "D": "fund. II completo",
    "E": "médio completo", "F": "superior completo", "G": "pós-graduação", "H": "não sabe",
}
COMPUTADOR = {"A": "nenhum computador", "B": "um computador", "C": "dois computadores",
              "D": "três computadores", "E": "quatro ou mais computadores"}
INTERNET = {"A": "sem internet", "B": "com internet"}

# Paleta categórica (validada; modo claro — figuras impressas). Uma cor por ano.
YEAR_COLOR = {2019: "#2a78d6", 2020: "#eb6834", 2021: "#1baf7a"}
YEAR_MARKER = {2019: "o", 2020: "s", 2021: "^"}
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"

plt.rcParams.update({
    "font.family": "Arial", "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "figure.dpi": 150,
    "axes.formatter.use_locale": False,
})


def br(v: float, nd: int = 3) -> str:
    """Formata número com vírgula decimal (padrão ABNT/pt-BR)."""
    if pd.isna(v):
        return "--"
    v = round(float(v), nd) + 0.0  # evita "-0,000"
    return f"{v:.{nd}f}".replace(".", ",").replace("-", "−")


def br_signed(v: float, nd: int = 3) -> str:
    v = round(float(v), nd) + 0.0
    return ("+" if v > 0 else "") + br(v, nd)


def br_int(v) -> str:
    return f"{int(v):,}".replace(",", ".")


def comma_axis(ax, axis="y", nd=1):
    fmt = matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:.{nd}f}".replace(".", ","))
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def feature_label(col: str) -> str:
    name = col.replace("cat__", "")
    var, val = name.rsplit("_", 1)
    if var == "TP_COR_RACA":
        val = str(int(float(val)))
        return f"Raça/cor: {RACA[val].lower()}"
    if var == "Q001":
        return f"Escol. do pai: {ESCOLARIDADE.get(val, val)}"
    if var == "Q002":
        return f"Escol. da mãe: {ESCOLARIDADE.get(val, val)}"
    if var == "Q006":
        return f"Renda: {RENDA.get(val, val).lower().replace('sm', 'SM')}"
    if var == "Q024":
        return f"Computador: {COMPUTADOR.get(val, val)}"
    if var == "Q025":
        return f"Internet: {INTERNET.get(val, val)}"
    return name


def code_label(col: str) -> str:
    name = col.replace("cat__", "")
    var, val = name.rsplit("_", 1)
    if var == "TP_COR_RACA":
        val = str(int(float(val)))
    return f"{var}\\_{val}"


def write_tex(name: str, body: str) -> None:
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    header = (f"% ARQUIVO GERADO AUTOMATICAMENTE por article/scripts/gerar_resultados.py\n"
              f"% Fonte: data/outputs/*.parquet — NÃO EDITAR À MÃO.\n")
    (TAB_DIR / f"{name}.tex").write_text(header + body, encoding="utf-8")


def save_fig(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight", dpi=200)  # pré-visualização
    plt.close(fig)


def threshold_line(ax, orient="h"):
    if orient == "h":
        ax.axhline(0.8, color=INK_2, lw=1, ls=(0, (4, 3)), zorder=1)
    else:
        ax.axvline(0.8, color=INK_2, lw=1, ls=(0, (4, 3)), zorder=1)


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------
model = pd.concat([pd.read_parquet(OUT / f"model_metrics_{y}.parquet") for y in YEARS], ignore_index=True)
fair = pd.concat([pd.read_parquet(OUT / f"fairness_metrics_{y}.parquet") for y in YEARS], ignore_index=True)
raca = fair[fair.protected_attribute == "TP_COR_RACA"].copy()
renda = fair[fair.protected_attribute == "Q006"].copy()
shap_imp = {}
for y in YEARS:
    s = pd.read_parquet(OUT / f"shap_values_{y}.parquet").drop(columns=["row_id", "base_value"])
    shap_imp[y] = s.abs().mean().sort_values(ascending=False)
inj = pd.read_parquet(OUT / "bias_injection_summary.parquet")

# ---------------------------------------------------------------------------
# Tabela 1 — Desempenho
# ---------------------------------------------------------------------------
rows = []
for _, m in model.sort_values("year").iterrows():
    rows.append(
        f"{m.year} & {br(m.accuracy, 4)} & {br(m.precision, 4)} & {br(m.recall, 4)} & {br(m.f1, 4)} & "
        f"{br(m.roc_auc, 4)} & {m.n_estimators} / {m.max_depth} / {br(m.learning_rate, 2)} & "
        f"{br_int(m.n_test)}" + r" \\"
    )
write_tex("tab_desempenho", r"""\begin{table}[htbp]
\centering
\caption{Desempenho do modelo-alvo (XGBoost) no conjunto de teste}
\label{tab:desempenho}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lccccccc}
\toprule
\textbf{Edição} & \textbf{Acurácia} & \textbf{Precisão} & \textbf{Recall} & \textbf{F1} & \textbf{AUC-ROC} & \textbf{n\_est. / prof. / taxa} & \textbf{$n_{\text{teste}}$} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\fonte{Elaborado pelo autor a partir de \texttt{model\_metrics\_\{ano\}.parquet}.}
\end{table}
""")

# ---------------------------------------------------------------------------
# Tabela 2 — Raça/cor: DI, TPR, FPR
# ---------------------------------------------------------------------------
body = []
for y in YEARS:
    body.append(rf"\multicolumn{{7}}{{l}}{{\textbf{{ENEM {y}}}}} \\")
    sub = raca[raca.year == y].set_index("group")
    for g in RACA_ORDER:
        r = sub.loc[g]
        ref = r" (ref.)" if g == r.reference_group else ""
        alert = r"\textbf{Sim}" if r.alert_di_low else "Não"
        body.append(
            f"\\quad {RACA[g]}{ref} & {br_int(r.total_count)} & {br(r.selection_rate)} & {br(r.disparate_impact)} & "
            f"{br(r.tpr)} & {br(r.fpr)} & {alert} \\\\"
        )
    if y != YEARS[-1]:
        body.append(r"\midrule")
write_tex("tab_di_raca", r"""\begin{table}[htbp]
\centering
\caption{Impacto Disparate e taxas de erro por raça/cor no conjunto de teste}
\label{tab:di_raca}
\begin{tabular}{lcccccc}
\toprule
\textbf{Grupo} & \textbf{$n$} & \textbf{Taxa de seleção} & \textbf{DI} & \textbf{TPR} & \textbf{FPR} & \textbf{DI $<$ 0,8} \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\fonte{Elaborado pelo autor a partir de \texttt{fairness\_metrics\_\{ano\}.parquet}. Grupo de referência: maior taxa de seleção.}
\end{table}
""")

# ---------------------------------------------------------------------------
# Tabela 3 — Amplificação (DI dos rótulos vs DI previsto)
# ---------------------------------------------------------------------------
body = []
for g in [g for g in RACA_ORDER if g != "1"]:
    cells = []
    for y in YEARS:
        r = raca[(raca.year == y) & (raca.group == g)].iloc[0]
        cells += [br(r.label_disparate_impact), br(r.disparate_impact), br_signed(r.di_amplification)]
    body.append(f"{RACA[g]} & " + " & ".join(cells) + r" \\")
write_tex("tab_amplificacao", r"""\begin{table}[htbp]
\centering
\caption{Impacto Disparate dos rótulos reais ($y$) versus das predições ($\hat{y}$) por raça/cor}
\label{tab:amplificacao}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l*{9}{c}}
\toprule
 & \multicolumn{3}{c}{\textbf{2019}} & \multicolumn{3}{c}{\textbf{2020}} & \multicolumn{3}{c}{\textbf{2021}} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}
\textbf{Grupo} & $\mathrm{DI}_y$ & $\mathrm{DI}_{\hat{y}}$ & $\Delta$ & $\mathrm{DI}_y$ & $\mathrm{DI}_{\hat{y}}$ & $\Delta$ & $\mathrm{DI}_y$ & $\mathrm{DI}_{\hat{y}}$ & $\Delta$ \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}}
\fonte{Elaborado pelo autor a partir de \texttt{fairness\_metrics\_\{ano\}.parquet}. Referência: Branca. $\Delta = \mathrm{DI}_{\hat{y}} - \mathrm{DI}_y$ (valores negativos indicam amplificação da disparidade pelo modelo).}
\end{table}
""")

# ---------------------------------------------------------------------------
# Tabela 4 — Renda (Q006)
# ---------------------------------------------------------------------------
body = []
for f in sorted(RENDA):
    cells = []
    for y in YEARS:
        r = renda[(renda.year == y) & (renda.group == f)].iloc[0]
        di = br(r.disparate_impact)
        cells.append(rf"\textbf{{{di}}}" if r.alert_di_low else di)
    for y in YEARS:
        r = renda[(renda.year == y) & (renda.group == f)].iloc[0]
        cells.append(br(r.tpr))
    body.append(f"{f} & {RENDA[f]} & " + " & ".join(cells) + r" \\")
refs = ", ".join(f"{y}: {renda[renda.year == y].reference_group.iloc[0]}" for y in YEARS)
write_tex("tab_di_renda", r"""\begin{table}[htbp]
\centering
\caption{Impacto Disparate e TPR por faixa de renda familiar (Q006) no conjunto de teste}
\label{tab:di_renda}
\begin{tabular}{llcccccc}
\toprule
 & & \multicolumn{3}{c}{\textbf{DI}} & \multicolumn{3}{c}{\textbf{TPR}} \\
\cmidrule(lr){3-5}\cmidrule(lr){6-8}
\textbf{Faixa} & \textbf{Renda mensal} & 2019 & 2020 & 2021 & 2019 & 2020 & 2021 \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\fonte{Elaborado pelo autor a partir de \texttt{fairness\_metrics\_\{ano\}.parquet}. SM = salário mínimo vigente no ano. Em negrito: DI $<$ 0,8. Faixa de referência (maior taxa de seleção) --- """ + refs + r""".}
\end{table}
""")

# ---------------------------------------------------------------------------
# Tabela 5 — Ranking SHAP global
# ---------------------------------------------------------------------------
TOP = 8
body = []
for i in range(TOP):
    cells = []
    for y in YEARS:
        col = shap_imp[y].index[i]
        cells += [f"{feature_label(col)}", br(shap_imp[y].iloc[i])]
    body.append(f"{i + 1} & " + " & ".join(cells) + r" \\")
write_tex("tab_shap", r"""\begin{table}[htbp]
\centering
\caption{Ranking de importância global SHAP (média de $|\phi_i|$, escala log-odds) no conjunto de teste}
\label{tab:shap}
\resizebox{\textwidth}{!}{%
\begin{tabular}{clclclc}
\toprule
 & \multicolumn{2}{c}{\textbf{2019}} & \multicolumn{2}{c}{\textbf{2020}} & \multicolumn{2}{c}{\textbf{2021}} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
\textbf{\#} & \textbf{Atributo} & $\overline{|\phi|}$ & \textbf{Atributo} & $\overline{|\phi|}$ & \textbf{Atributo} & $\overline{|\phi|}$ \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}}
\fonte{Elaborado pelo autor a partir de \texttt{shap\_values\_\{ano\}.parquet}.}
\end{table}
""")

# ---------------------------------------------------------------------------
# Tabela 6 — Injeção: curva DI / AUC
# ---------------------------------------------------------------------------
INT = [0.0, 0.10, 0.25, 0.50]
body = []
for label in ["Parda", "Indígena"]:
    body.append(rf"\multicolumn{{10}}{{l}}{{\textbf{{Grupo-alvo: {label}}}}} \\")
    for p in INT:
        cells = []
        for y in YEARS:
            q = inj[(inj.year == y) & (inj.target_label == label) & (inj.intensity == p)].set_index("scenario")
            i_, c_ = q.loc["injection"], q.loc["control"]
            cells += [f"{br(i_.disparate_impact_mean)} $\\pm$ {br(i_.disparate_impact_std)}",
                      br(c_.disparate_impact_mean), br(i_.roc_auc_mean)]
        body.append(f"\\quad {int(p * 100)}\\% & " + " & ".join(cells) + r" \\")
    if label == "Parda":
        body.append(r"\midrule")
write_tex("tab_injecao", r"""\begin{table}[htbp]
\centering
\caption{Curva de sensibilidade: DI médio do grupo-alvo e AUC-ROC sob injeção sintética de viés (teste)}
\label{tab:injecao}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l*{9}{c}}
\toprule
 & \multicolumn{3}{c}{\textbf{2019}} & \multicolumn{3}{c}{\textbf{2020}} & \multicolumn{3}{c}{\textbf{2021}} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}
\textbf{Intensidade $p$} & DI (injeção) & DI (controle) & AUC & DI (injeção) & DI (controle) & AUC & DI (injeção) & DI (controle) & AUC \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}}
\fonte{Elaborado pelo autor a partir de \texttt{bias\_injection\_summary.parquet}. Média $\pm$ desvio-padrão de 5 repetições. AUC do cenário de injeção.}
\end{table}
""")

# Tabela 7 — Efeito a 50%
body = []
for label in ["Parda", "Indígena"]:
    for y in YEARS:
        q = inj[(inj.year == y) & (inj.target_label == label)]
        b = q[(q.scenario == "injection") & (q.intensity == 0.0)].iloc[0]
        i_ = q[(q.scenario == "injection") & (q.intensity == 0.5)].iloc[0]
        c_ = q[(q.scenario == "control") & (q.intensity == 0.5)].iloc[0]
        body.append(
            f"{label} & {y} & {br_int(i_.n_labels_flipped)} & {br_signed(i_.di_delta_vs_baseline)} & "
            f"{br_signed(c_.di_delta_vs_baseline)} & {br_signed(i_.di_net_effect)} & "
            f"{br_signed(i_.roc_auc_mean - b.roc_auc_mean)} & {br(b.tpr_mean)} $\\rightarrow$ {br(i_.tpr_mean)} & "
            f"{br(b.shap_mean_group_members_mean)} $\\rightarrow$ {br(i_.shap_mean_group_members_mean)} \\\\"
        )
write_tex("tab_injecao_efeito", r"""\begin{table}[htbp]
\centering
\caption{Efeito da injeção de 50\% de viés sobre os indicadores do artefato}
\label{tab:injecao_efeito}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccccccc}
\toprule
\textbf{Grupo} & \textbf{Edição} & \textbf{Rótulos alterados} & $\Delta\mathrm{DI}_{\text{inj}}$ & $\Delta\mathrm{DI}_{\text{ctrl}}$ & \textbf{Efeito líquido} & $\Delta\mathrm{AUC}$ & \textbf{TPR (0\%$\rightarrow$50\%)} & $\bar{\phi}_{\text{raça}}$ \textbf{(0\%$\rightarrow$50\%)} \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}}
\fonte{Elaborado pelo autor a partir de \texttt{bias\_injection\_summary.parquet}. $\bar{\phi}_{\text{raça}}$: valor SHAP médio da variável indicadora do grupo-alvo entre os membros do grupo.}
\end{table}
""")

# ---------------------------------------------------------------------------
# Tabela 8 — Benchmark da camada online (app_benchmark.parquet)
# ---------------------------------------------------------------------------
bench = pd.read_parquet(OUT / "app_benchmark.parquet")
ATTR_LBL = {"TP_COR_RACA": "Raça/cor", "Q006": "Renda (Q006)"}
body = []
for (y, a), g in bench.groupby(["year", "protected_attribute"], sort=True):
    cold = g[g["mode"] == "cold"].seconds.iloc[0]
    warm = g[g["mode"] == "warm"].seconds
    body.append(f"{y} & {ATTR_LBL[a]} & {br(cold, 2)} & {br(warm.mean(), 3)} & {br(warm.max(), 3)} & {br(g.outputs_size_mb.iloc[0], 2)} \\\\")
n_ok, n_all = int(bench.within_sla.sum()), len(bench)
write_tex("tab_app_benchmark", r"""\begin{table}[htbp]
\centering
\caption{Tempo de execução da interface por interação (segundos)}
\label{tab:app_benchmark}
\begin{tabular}{llcccc}
\toprule
\textbf{Edição} & \textbf{Atributo} & \textbf{Primeira carga} & \textbf{Interação (média)} & \textbf{Interação (máx.)} & \textbf{Parquets (MB)} \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\fonte{Elaborado pelo autor a partir de \texttt{app\_benchmark.parquet}. """ + f"{n_ok} de {n_all}" + r""" execuções abaixo de 2~s. Ambiente: """ + f"{bench.platform.iloc[0].split('-')[0]} {bench.platform.iloc[0].split('-')[1]}, {int(bench.cpu_count.iloc[0])} CPUs lógicas, Python {bench.python.iloc[0]}, Streamlit {bench.streamlit.iloc[0]}" + r""".}
\end{table}
""")

# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------
def year_legend(ax, **kw):
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.13), handlelength=1.4, **kw)


# Fig. 1 — DI por raça/cor
groups = [g for g in RACA_ORDER if g != "1"]
fig, ax = plt.subplots(figsize=(6.3, 3.0))
w = 0.26
x = np.arange(len(groups))
for k, y in enumerate(YEARS):
    vals = [raca[(raca.year == y) & (raca.group == g)].disparate_impact.iloc[0] for g in groups]
    ax.bar(x + (k - 1) * w, vals, width=w - 0.03, color=YEAR_COLOR[y], label=str(y), zorder=2)
threshold_line(ax)
ax.text(len(groups) - 0.5, 0.81, "limiar 0,8 (regra dos 4/5)", ha="right", va="bottom", color=INK_2, fontsize=8)
ax.set_xticks(x, [RACA[g] for g in groups])
ax.set_ylabel("Impacto Disparate (ref.: Branca)")
ax.set_ylim(0, 1.0)
comma_axis(ax)
year_legend(ax)
save_fig(fig, "fig_di_raca")

# Fig. 2 — TPR e FPR por raça/cor (mesma escala, dois painéis)
fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.9), sharey=True)
for ax, metric, title in zip(axes, ["tpr", "fpr"], ["TPR — taxa de verdadeiros positivos", "FPR — taxa de falsos positivos"]):
    xx = np.arange(len(RACA_ORDER))
    for k, y in enumerate(YEARS):
        vals = [raca[(raca.year == y) & (raca.group == g)][metric].iloc[0] for g in RACA_ORDER]
        ax.bar(xx + (k - 1) * w, vals, width=w - 0.03, color=YEAR_COLOR[y], label=str(y), zorder=2)
    ax.set_xticks(xx, [RACA[g].replace("Não declarada", "Não decl.") for g in RACA_ORDER], rotation=30, ha="right")
    ax.set_title(title, color=INK, loc="left")
    ax.set_ylim(0, 1)
    comma_axis(ax)
axes[0].set_ylabel("Taxa")
axes[0].legend(ncol=3, loc="upper right", handlelength=1.2, fontsize=8)
fig.tight_layout()
save_fig(fig, "fig_tpr_fpr_raca")

# Fig. 3 — Amplificação (dumbbell, pequenos múltiplos por ano)
fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.6), sharey=True, sharex=True)
for ax, y in zip(axes, YEARS):
    sub = raca[(raca.year == y)].set_index("group")
    for i, g in enumerate(groups):
        a, b = sub.loc[g, "label_disparate_impact"], sub.loc[g, "disparate_impact"]
        ax.plot([a, b], [i, i], color=MUTED, lw=1.5, zorder=1)
        ax.scatter(a, i, s=36, facecolor="white", edgecolor=YEAR_COLOR[y], lw=1.5, zorder=3,
                   label="Rótulos reais ($y$)" if i == 0 else None)
        ax.scatter(b, i, s=36, color=YEAR_COLOR[y], zorder=3, label="Predições ($\\hat{y}$)" if i == 0 else None)
    threshold_line(ax, "v")
    ax.set_title(str(y), color=INK)
    ax.set_xlim(0, 1)
    comma_axis(ax, "x")
    ax.grid(axis="y", visible=False)
axes[0].set_yticks(range(len(groups)), [RACA[g] for g in groups])
axes[0].invert_yaxis()
axes[1].set_xlabel("Impacto Disparate (ref.: Branca)")
from matplotlib.lines import Line2D
proxies = [
    Line2D([], [], ls="", marker="o", ms=6, mfc="white", mec=INK_2, mew=1.5, label="Rótulos reais ($y$)"),
    Line2D([], [], ls="", marker="o", ms=6, mfc=INK_2, mec=INK_2, label="Predições ($\\hat{y}$)"),
]
fig.legend(handles=proxies, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.08))
fig.tight_layout()
save_fig(fig, "fig_amplificacao")

# Fig. 4 — DI por faixa de renda
fig, ax = plt.subplots(figsize=(6.3, 2.9))
faixas = sorted(RENDA)
for y in YEARS:
    sub = renda[renda.year == y].set_index("group").reindex(faixas)
    ax.plot(faixas, sub.disparate_impact, color=YEAR_COLOR[y], lw=2, marker=YEAR_MARKER[y], ms=5, label=str(y))
threshold_line(ax)
ax.text(16, 0.77, "limiar 0,8", ha="right", va="top", color=INK_2, fontsize=8)
ax.set_xlabel("Faixa de renda familiar (Q006): A = sem renda … Q = acima de 20 SM")
ax.set_ylabel("Impacto Disparate")
ax.set_ylim(0, 1.05)
comma_axis(ax)
year_legend(ax)
save_fig(fig, "fig_di_renda")

# Fig. 5 — SHAP: atributos mais importantes (união do top-6), por ano
top_cols = []
for y in YEARS:
    for c in shap_imp[y].index[:6]:
        if c not in top_cols:
            top_cols.append(c)
top_cols = sorted(top_cols, key=lambda c: -shap_imp[2021].get(c, 0))
fig, ax = plt.subplots(figsize=(6.3, 3.6))
yy = np.arange(len(top_cols))
h = 0.26
for k, y in enumerate(YEARS):
    ax.barh(yy + (k - 1) * h, [shap_imp[y].get(c, 0) for c in top_cols], height=h - 0.03,
            color=YEAR_COLOR[y], label=str(y), zorder=2)
ax.set_yticks(yy, [feature_label(c) for c in top_cols])
ax.invert_yaxis()
ax.set_xlabel("Importância global SHAP — média de |φ| (log-odds)")
ax.grid(axis="y", visible=False)
comma_axis(ax, "x", 2)
ax.legend(loc="lower right", handlelength=1.2)
save_fig(fig, "fig_shap_global")

# Fig. 6 — Injeção: DI (linha de cima) e AUC (linha de baixo), por grupo-alvo
fig, axes = plt.subplots(2, 2, figsize=(6.3, 4.6), sharex=True,
                         gridspec_kw={"height_ratios": [1.6, 1]})
xs = [p * 100 for p in INT]
for col, label in enumerate(["Parda", "Indígena"]):
    for y in YEARS:
        q = inj[(inj.year == y) & (inj.target_label == label)]
        for scen, ls, mfc in [("injection", "-", None), ("control", (0, (3, 2)), "white")]:
            s = q[q.scenario == scen].sort_values("intensity")
            axes[0, col].errorbar(xs, s.disparate_impact_mean, yerr=s.disparate_impact_std, color=YEAR_COLOR[y],
                                  ls=ls, lw=1.8, marker=YEAR_MARKER[y], ms=5, mfc=mfc or YEAR_COLOR[y],
                                  capsize=2, elinewidth=1)
        s = q[q.scenario == "injection"].sort_values("intensity")
        axes[1, col].plot(xs, s.roc_auc_mean, color=YEAR_COLOR[y], lw=1.8, marker=YEAR_MARKER[y], ms=5)
    threshold_line(axes[0, col])
    axes[0, col].set_title(f"Grupo-alvo: {label}", color=INK, loc="left")
    axes[0, col].set_ylim(0, 0.85)
    axes[1, col].set_ylim(0.5, 0.8)
    axes[1, col].set_xticks(xs, [f"{int(v)}%" for v in xs])
    axes[1, col].set_xlabel("Intensidade da injeção ($p$)")
    comma_axis(axes[0, col], nd=1)
    comma_axis(axes[1, col], nd=2)
axes[0, 0].set_ylabel("DI do grupo-alvo")
axes[1, 0].set_ylabel("AUC-ROC")
from matplotlib.lines import Line2D
handles = [Line2D([], [], color=YEAR_COLOR[y], marker=YEAR_MARKER[y], lw=1.8, label=str(y)) for y in YEARS]
handles += [Line2D([], [], color=INK_2, lw=1.8, label="Injeção direcionada"),
            Line2D([], [], color=INK_2, lw=1.8, ls=(0, (3, 2)), marker="o", mfc="white", label="Controle negativo")]
fig.legend(handles=handles, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.04), handlelength=1.8, fontsize=8)
fig.tight_layout()
save_fig(fig, "fig_injecao")

print(f"[OK] Tabelas em {TAB_DIR}")
print(f"[OK] Figuras em {FIG_DIR}")
