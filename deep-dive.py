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

# CSV direto do GitHub (ajustado para o seu arquivo real)
GITHUB_CSV_URL = (
    "https://raw.githubusercontent.com/"
    "cbeninatto/performance-moveleiro-v2/main/data/relatorio_faturamento.csv"
)

STATUS_COL = "StatusCarteira"  # por enquanto não existe, então será só placeholder


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
    df["Ano"] = pd.to_numeric(df["Ano"], errors="coerce").astype("Int64")
    df["MesNum"] = pd.to_numeric(df["Mes"], errors="coerce").astype("Int64")

    # Competência (primeiro dia do mês)
    df["Competencia"] = pd.to_datetime(
        dict(year=df["Ano"], month=df["MesNum"], day=1),
        errors="coerce",
    )

    return df


def compute_carteira_score(status_counts: pd.Series):
    """Placeholder: se não houver coluna STATUS_COL, usa score neutro."""
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

# Representante
reps = sorted(df["Representante"].dropna().unique())
if not reps:
    st.error("Não foram encontrados representantes na base de dados.")
    st.stop()

rep_selected = st.sidebar.selectbox("Representante", reps)

# Período
valid_comp = df["Competencia"].dropna().sort_values().unique()
if len(valid_comp) == 0:
    st.error("Não foi possível identificar as competências (Ano/Mês).")
    st.stop()

default_start = valid_comp[max(0, len(valid_comp) - 12)]
default_end = valid_comp[-1]

start_comp, end_comp = st.sidebar.select_slider(
    "Período (competência)",
    options=list(valid_comp),
    value=(default_start, default_end),
    format_func=lambda x: x.strftime("%b %Y"),
)

mask_period = (df["Competencia"] >= start_comp) & (df["Competencia"] <= end_comp)
df_period = df.loc[mask_period].copy()

if df_period.empty:
    st.warning("Nenhuma venda no período selecionado.")
    st.stop()

df_rep = df_period[df_period["Representante"] == rep_selected].copy()

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

# Saúde da carteira (placeholder neutro, pois ainda não temos StatusCarteira)
carteira_score, carteira_label = 50.0, "Neutra"
if STATUS_COL in df_rep.columns:
    clientes_rep = (
        df_rep
        .dropna(subset=[STATUS_COL, "Cliente"])
        .groupby(["Cliente", STATUS_COL], as_index=False)
        .agg({"Valor": "sum"})
    )
    status_counts_series = clientes_rep.groupby(STATUS_COL)["Cliente"].nunique()
    carteira_score, carteira_label = compute_carteira_score(status_counts_series)

col1.metric("Total período", format_brl(total_rep))
col2.metric("Média mensal", format_brl(media_mensal))
col3.metric("Meses com venda", f"{meses_com_venda} / {total_meses_periodo}")
col4.metric("Participação", f"{participacao:.1%}")
col5.metric("Saúde da carteira", f"{carteira_score:.0f} / 100", carteira_label)

st.markdown("---")

# ==========================
# EVOLUÇÃO DE VENDAS
# ==========================
st.subheader("Evolução de vendas no período")

if df_rep.empty:
    st.info("Este representante não possui vendas no período selecionado.")
else:
    ts_rep = (
        df_rep
        .groupby("Competencia", as_index=False)["Valor"]
        .sum()
        .sort_values("Competencia")
    )

    chart_ts = (
        alt.Chart(ts_rep)
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "Competencia:T",
                axis=alt.Axis(title="Competência", format="%b %Y"),
            ),
            y=alt.Y(
                "Valor:Q",
                axis=alt.Axis(title="Faturamento"),
            ),
            tooltip=[
                alt.Tooltip("Competencia:T", title="Competência", format="%b %Y"),
                alt.Tooltip("Valor:Q", title="Faturamento", format=",.2f"),
            ],
        )
        .properties(height=260)
    )

    st.altair_chart(chart_ts, use_container_width=True)

st.markdown("---")

# ==========================
# SAÚDE DA CARTEIRA – DETALHES (placeholder)
# ==========================
st.subheader("Saúde da carteira – Detalhes")

if STATUS_COL not in df_rep.columns:
    st.info(
        "Ainda não existe coluna `StatusCarteira` no CSV. "
        "Quando você adicionar essa coluna (Novos, Perdidos, Crescendo, Caindo, Estáveis), "
        "vamos montar aqui o gráfico de pizza e a lista detalhada de clientes."
    )
else:
    st.write("Aqui entrariam os detalhes (pizza + tabela) usando StatusCarteira.")
