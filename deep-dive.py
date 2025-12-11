import streamlit as st
import pandas as pd
import altair as alt

# ==========================
# CONFIG
# ==========================
st.set_page_config(
    page_title="Nova Visão – Deep Dive",
    layout="wide",
)

# CSV direto do GitHub (layout real do seu arquivo)
GITHUB_CSV_URL = (
    "https://raw.githubusercontent.com/"
    "cbeninatto/performance-moveleiro-v2/main/data/relatorio_faturamento.csv"
)

# Nome lógico da coluna de status calculado
STATUS_COL = "StatusCarteira"


# ==========================
# HELPERS
# ==========================
def format_brl(value: float) -> str:
    if pd.isna(value):
        return "R$ 0,00"
    return "R$ " + f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


@st.cache_data(show_spinner=True)
def load_data() -> pd.DataFrame:
    """
    Lê o CSV do GitHub exatamente no layout:

    Codigo,Descricao,Quantidade,Valor,Mes,Ano,ClienteCodigo,Cliente,
    Estado,Cidade,RepresentanteCodigo,Representante,Categoria,SourcePDF
    """
    df = pd.read_csv(GITHUB_CSV_URL)

    expected = [
        "Codigo", "Descricao", "Quantidade", "Valor", "Mes", "Ano",
        "ClienteCodigo", "Cliente", "Estado", "Cidade",
        "RepresentanteCodigo", "Representante", "Categoria", "SourcePDF",
    ]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(
            "CSV do GitHub não tem as colunas esperadas: "
            + ", ".join(missing)
        )

    # Tipagem básica
    df["Valor"] = pd.to_numeric(df["Valor"], errors="coerce").fillna(0.0)
    df["Quantidade"] = pd.to_numeric(df["Quantidade"], errors="coerce").fillna(0.0)
    df["Ano"] = pd.to_numeric(df["Ano"], errors="coerce").astype("Int64")
    df["MesNum"] = pd.to_numeric(df["Mes"], errors="coerce").astype("Int64")

    # Competência (primeiro dia do mês)
    df["Competencia"] = pd.to_datetime(
        dict(year=df["Ano"], month=df["MesNum"], day=1),
        errors="coerce",
    )

    return df


def compute_carteira_score(status_counts: pd.Series):
    """Retorna (score 0–100, label) a partir da contagem por status."""
    if status_counts is None or status_counts.empty:
        return 50.0, "Neutra"

    weights = {
        "Novo": 1,
        "Novos": 1,
        "Crescendo": 2,
        "CRESCENDO": 2,
        "Caindo": -1,
        "CAINDO": -1,
        "Estável": 1,
        "Estáveis": 1,
        "ESTAVEL": 1,
        "ESTAVEIS": 1,
        "Perdido": -2,
        "Perdidos": -2,
        "PERDIDO": -2,
        "PERDIDOS": -2,
    }

    score_total = 0
    n_clients = 0

    for status, qty in status_counts.items():
        w = weights.get(str(status), 0)
        score_total += w * qty
        n_clients += qty

    if n_clients == 0:
        return 50.0, "Neutra"

    avg = score_total / n_clients  # [-2, 2]
    score_0_100 = (avg + 2) / 4 * 100
    score_0_100 = max(0, min(100, score_0_100))

    if score_0_100 < 40:
        label = "Crítica"
    elif score_0_100 < 60:
        label = "Neutra"
    else:
        label = "Saudável"

    return score_0_100, label


# Mapeamento de meses
MONTH_MAP_NUM_TO_NAME = {
    1: "JAN", 2: "FEV", 3: "MAR", 4: "ABR",
    5: "MAI", 6: "JUN", 7: "JUL", 8: "AGO",
    9: "SET", 10: "OUT", 11: "NOV", 12: "DEZ",
}
MONTH_MAP_NAME_TO_NUM = {v: k for k, v in MONTH_MAP_NUM_TO_NAME.items()}


def build_carteira_status(df_all: pd.DataFrame,
                          rep: str,
                          start_comp: pd.Timestamp,
                          end_comp: pd.Timestamp) -> pd.DataFrame:
    """
    Calcula StatusCarteira (Novos / Perdidos / Crescendo / Caindo / Estáveis)
    comparando o período selecionado com a JANELA ANTERIOR de mesmo tamanho.

    Retorna um DF com colunas:
    Cliente, Estado, Cidade, ValorAtual, ValorAnterior, StatusCarteira
    """
    df_rep_all = df_all[df_all["Representante"] == rep].copy()
    if df_rep_all.empty:
        return pd.DataFrame(columns=[
            "Cliente", "Estado", "Cidade",
            "ValorAtual", "ValorAnterior", STATUS_COL
        ])

    # Quantidade de meses no período atual
    months_span = (end_comp.year - start_comp.year) * 12 + (end_comp.month - start_comp.month) + 1

    # Período anterior de mesmo tamanho
    prev_end = start_comp - pd.DateOffset(months=1)
    prev_start = prev_end - pd.DateOffset(months=months_span - 1)

    mask_curr = (df_rep_all["Competencia"] >= start_comp) & (df_rep_all["Competencia"] <= end_comp)
    mask_prev = (df_rep_all["Competencia"] >= prev_start) & (df_rep_all["Competencia"] <= prev_end)

    df_curr = df_rep_all.loc[mask_curr].copy()
    df_prev = df_rep_all.loc[mask_prev].copy()

    # Agrega por cliente (atual)
    curr_agg = (
        df_curr
        .groupby("Cliente", as_index=False)
        .agg({
            "Valor": "sum",
            "Estado": "first",
            "Cidade": "first",
        })
        .rename(columns={"Valor": "ValorAtual"})
    )

    # Agrega por cliente (anterior)
    prev_agg = (
        df_prev
        .groupby("Cliente", as_index=False)["Valor"]
        .sum()
        .rename(columns={"Valor": "ValorAnterior"})
    )

    # Junta tudo
    clientes = pd.merge(curr_agg, prev_agg, on="Cliente", how="outer")

    # Preenche NaNs
    clientes["ValorAtual"] = clientes["ValorAtual"].fillna(0.0)
    clientes["ValorAnterior"] = clientes["ValorAnterior"].fillna(0.0)
    clientes["Estado"] = clientes["Estado"].fillna("")
    clientes["Cidade"] = clientes["Cidade"].fillna("")

    def classify(row):
        va = row["ValorAtual"]
        vp = row["ValorAnterior"]

        if va > 0 and vp == 0:
            return "Novos"
        if va == 0 and vp > 0:
            return "Perdidos"
        if va > 0 and vp > 0:
            # Relação crescimento / queda
            ratio = va / vp if vp != 0 else 0
            if ratio >= 1.2:
                return "Crescendo"
            elif ratio <= 0.8:
                return "Caindo"
            else:
                return "Estáveis"
        # Sem movimento relevante nos dois períodos
        return "Estáveis"

    clientes[STATUS_COL] = clientes.apply(classify, axis=1)

    # Remove clientes totalmente "mortos" (sem movimento em nenhum dos dois períodos)
    clientes = clientes[(clientes["ValorAtual"] > 0) | (clientes["ValorAnterior"] > 0)]

    return clientes


# ==========================
# LOAD DATA
# ==========================
try:
    df = load_data()
except Exception as e:
    st.error(f"Erro ao carregar dados do GitHub: {e}")
    st.stop()

if df.empty:
    st.warning("O arquivo de dados está vazio.")
    st.stop()

# ==========================
# SIDEBAR – FILTROS
# ==========================
st.sidebar.title("Filtros – Deep Dive")

# Representantes
reps = sorted(df["Representante"].dropna().unique())
if not reps:
    st.error("Não foram encontrados representantes na base de dados.")
    st.stop()
rep_selected = st.sidebar.selectbox("Representante", reps)

# ----- Filtro de período: dropdowns de mês e ano -----
st.sidebar.markdown("### Período")

anos_disponiveis = sorted(df["Ano"].dropna().unique())
if not anos_disponiveis:
    st.error("Não foi possível identificar anos na base de dados.")
    st.stop()

last_year = int(anos_disponiveis[-1])
default_start_year = last_year
default_end_year = last_year

meses_ano_default = df.loc[df["Ano"] == last_year, "MesNum"].dropna().unique()
if len(meses_ano_default) > 0:
    default_start_month_num = int(meses_ano_default.min())
    default_end_month_num = int(meses_ano_default.max())
else:
    default_start_month_num = 1
    default_end_month_num = 12

month_names = [MONTH_MAP_NUM_TO_NAME[m] for m in range(1, 12 + 1)]

# Mês / Ano inicial
st.sidebar.caption("Período inicial")
col_mi, col_ai = st.sidebar.columns(2)
with col_mi:
    start_month_name = st.selectbox(
        "Mês",
        options=month_names,
        index=month_names.index(MONTH_MAP_NUM_TO_NAME[default_start_month_num]),
        key="start_month",
    )
with col_ai:
    start_year = st.selectbox(
        "Ano",
        options=[int(a) for a in anos_disponiveis],
        index=list(anos_disponiveis).index(default_start_year),
        key="start_year",
    )

# Mês / Ano final
st.sidebar.caption("Período final")
col_mf, col_af = st.sidebar.columns(2)
with col_mf:
    end_month_name = st.selectbox(
        "Mês ",
        options=month_names,
        index=month_names.index(MONTH_MAP_NUM_TO_NAME[default_end_month_num]),
        key="end_month",
    )
with col_af:
    end_year = st.selectbox(
        "Ano ",
        options=[int(a) for a in anos_disponiveis],
        index=list(anos_disponiveis).index(default_end_year),
        key="end_year",
    )

start_month = MONTH_MAP_NAME_TO_NUM[start_month_name]
end_month = MONTH_MAP_NAME_TO_NUM[end_month_name]

start_comp = pd.Timestamp(year=int(start_year), month=int(start_month), day=1)
end_comp = pd.Timestamp(year=int(end_year), month=int(end_month), day=1)

if start_comp > end_comp:
    st.sidebar.error("Período inicial não pode ser maior que o período final.")
    st.stop()

# Aplica período
mask_period = (df["Competencia"] >= start_comp) & (df["Competencia"] <= end_comp)
df_period = df.loc[mask_period].copy()

if df_period.empty:
    st.warning("Nenhuma venda no período selecionado.")
    st.stop()

df_rep = df_period[df_period["Representante"] == rep_selected].copy()

# ==========================
# CALCULA STATUS DA CARTEIRA
# ==========================
clientes_carteira = build_carteira_status(df, rep_selected, start_comp, end_comp)

# ==========================
# HEADER
# ==========================
st.title("Deep Dive – Representante")

st.subheader(f"Representante: **{rep_selected}**")
st.caption(
    f"Período selecionado: "
    f"{start_comp.strftime('%b %Y')} até {end_comp.strftime('%b %Y')}"
)

st.markdown("---")

# ==========================
# MÉTRICAS PRINCIPAIS
# ==========================
col1, col2, col3, col4, col5 = st.columns(5)

total_rep = df_rep["Valor"].sum()

if not df_rep.empty:
    meses_rep = (
        df_rep.groupby([df_rep["Ano"], df_rep["MesNum"]])["Valor"]
        .sum()
        .reset_index(name="ValorMes")
    )
    meses_com_venda = (meses_rep["ValorMes"] > 0).sum()
else:
    meses_com_venda = 0

meses_periodo = (
    df_period.groupby([df_period["Ano"], df_period["MesNum"]])["Valor"]
    .sum()
    .reset_index(name="ValorMes")
)
total_meses_periodo = len(meses_periodo)

if meses_com_venda > 0:
    media_mensal = total_rep / meses_com_venda
else:
    media_mensal = 0.0

total_periodo_geral = df_period["Valor"].sum()
if total_periodo_geral > 0:
    participacao = total_rep / total_periodo_geral
else:
    participacao = 0.0

# Saúde da carteira usando o DF calculado
if not clientes_carteira.empty:
    status_counts_series = (
        clientes_carteira.groupby(STATUS_COL)["Cliente"].nunique()
    )
    carteira_score, carteira_label = compute_carteira_score(status_counts_series)
else:
    carteira_score, carteira_label = 50.0, "Neutra"

col1.metric("Total período", format_brl(total_rep))
col2.metric("Média mensal", format_brl(media_mensal))
col3.metric("Meses com venda", f"{meses_com_venda} / {total_meses_periodo}")
col4.metric("Participação", f"{participacao:.1%}")
col5.metric("Saúde da carteira", f"{carteira_score:.0f} / 100", carteira_label)

st.markdown("---")

# ==========================
# COMBO CHART – BARRAS (FATURAMENTO) + LINHA (VOLUME)
# ==========================
st.subheader("Evolução – Faturamento (barras) e Volume (linha)")

if df_rep.empty:
    st.info("Este representante não possui vendas no período selecionado.")
else:
    ts_rep = (
        df_rep
        .groupby("Competencia", as_index=False)[["Valor", "Quantidade"]]
        .sum()
        .sort_values("Competencia")
    )

    base = alt.Chart(ts_rep).encode(
        x=alt.X(
            "Competencia:T",
            axis=alt.Axis(title="Competência", format="%b %Y"),
        )
    )

    bars = base.mark_bar().encode(
        y=alt.Y(
            "Valor:Q",
            axis=alt.Axis(title="Faturamento (R$)"),
        ),
        tooltip=[
            alt.Tooltip("Competencia:T", title="Competência", format="%b %Y"),
            alt.Tooltip("Valor:Q", title="Faturamento (R$)", format=",.2f"),
            alt.Tooltip("Quantidade:Q", title="Volume"),
        ],
    )

    line = base.mark_line(point=True).encode(
        y=alt.Y(
            "Quantidade:Q",
            axis=alt.Axis(title="Volume", orient="right"),
        ),
        tooltip=[
            alt.Tooltip("Competencia:T", title="Competência", format="%b %Y"),
            alt.Tooltip("Valor:Q", title="Faturamento (R$)", format=",.2f"),
            alt.Tooltip("Quantidade:Q", title="Volume"),
        ],
    )

    combo_chart = alt.layer(bars, line).resolve_scale(
        y="independent"
    ).properties(
        height=320,
    )

    st.altair_chart(combo_chart, use_container_width=True)

st.markdown("---")

# ==========================
# SAÚDE DA CARTEIRA – DETALHES
# ==========================
st.subheader("Saúde da carteira – Detalhes")

if clientes_carteira.empty:
    st.info("Não há clientes com movimento nos períodos atual / anterior para calcular a carteira.")
else:
    status_counts = (
        clientes_carteira.groupby(STATUS_COL)["Cliente"]
        .nunique()
        .reset_index()
        .rename(columns={"Cliente": "QtdClientes", STATUS_COL: "Status"})
    )
    total_clientes = status_counts["QtdClientes"].sum()
    status_counts["%Clientes"] = (
        status_counts["QtdClientes"] / total_clientes if total_clientes > 0 else 0
    )

    # Resumo em texto: "Novos X • Perdidos Y • Crescendo Z..."
    resumo_text = " • ".join(
        f"{row.Status} {int(row.QtdClientes)}"
        for _, row in status_counts.iterrows()
    )
    resumo_text += f" ({int(total_clientes)} clientes)"

    st.caption(resumo_text)

    col_pie, col_table = st.columns([1, 1.2])

    with col_pie:
        st.caption("Distribuição de clientes por status")
        if total_clientes == 0:
            st.info("Nenhum cliente com status definido.")
        else:
            chart_pie = (
                alt.Chart(status_counts)
                .mark_arc(outerRadius=120)
                .encode(
                    theta=alt.Theta("QtdClientes:Q"),
                    color=alt.Color("Status:N", legend=alt.Legend(title="Status")),
                    tooltip=[
                        alt.Tooltip("Status:N", title="Status"),
                        alt.Tooltip("QtdClientes:Q", title="Clientes"),
                        alt.Tooltip("%Clientes:Q", title="% Clientes", format=".1%"),
                    ],
                )
                .properties(height=320)
            )
            st.altair_chart(chart_pie, use_container_width=True)

    with col_table:
        st.caption("Resumo por status")
        status_counts_display = status_counts.copy()
        status_counts_display["%Clientes"] = status_counts_display["%Clientes"].map(
            lambda x: f"{x:.1%}"
        )
        st.dataframe(
            status_counts_display,
            hide_index=True,
            use_container_width=True,
        )

    st.markdown("### Lista de clientes da carteira")

    status_options = sorted(clientes_carteira[STATUS_COL].dropna().unique())
    status_selected = st.multiselect(
        "Filtrar por status",
        options=status_options,
        default=status_options,
    )

    df_clientes_view = clientes_carteira.copy()
    if status_selected:
        df_clientes_view = df_clientes_view[df_clientes_view[STATUS_COL].isin(status_selected)]

    df_clientes_view = df_clientes_view.rename(
        columns={
            "ValorAtual": "FaturamentoAtual",
            "ValorAnterior": "FaturamentoAnterior",
            STATUS_COL: "StatusCarteira",
        }
    )
    df_clientes_view["FaturamentoAtualFmt"] = df_clientes_view["FaturamentoAtual"].map(format_brl)
    df_clientes_view["FaturamentoAnteriorFmt"] = df_clientes_view["FaturamentoAnterior"].map(format_brl)

    df_clientes_view = df_clientes_view.sort_values(
        "FaturamentoAtual", ascending=False
    )[
        ["Cliente", "Estado", "Cidade", "StatusCarteira",
         "FaturamentoAtualFmt", "FaturamentoAnteriorFmt"]
    ]

    st.dataframe(
        df_clientes_view,
        hide_index=True,
        use_container_width=True,
    )
