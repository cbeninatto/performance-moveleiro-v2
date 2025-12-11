import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

st.set_page_config(
    page_title="Nova Visão – Deep Dive",
    layout="wide",
)

# Nome da coluna de status de carteira no CSV
STATUS_COL = "StatusCarteira"

# 🔗 CSV direto do GitHub (ajuste o nome do arquivo se for diferente)
GITHUB_CSV_URL = (
    "https://raw.githubusercontent.com/"
    "cbeninatto/performance-moveleiro-v2/main/data/relatorio_faturamento.csv"
)
# Se o arquivo for outro, por exemplo:
# GITHUB_CSV_URL = (
#     "https://raw.githubusercontent.com/"
#     "cbeninatto/performance-moveleiro-v2/main/data/clientes_relatorio_faturamento.csv"
# )


def format_brl(value: float) -> str:
    if pd.isna(value):
        return "R$ 0,00"
    return "R$ " + f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def normalize_month_column(df: pd.DataFrame, month_col: str = "Mes") -> pd.DataFrame:
    month_map = {
        "JAN": 1, "FEV": 2, "MAR": 3, "ABR": 4,
        "MAI": 5, "JUN": 6, "JUL": 7, "AGO": 8,
        "SET": 9, "OUT": 10, "NOV": 11, "DEZ": 12,
    }

    df = df.copy()

    df["Ano"] = pd.to_numeric(df["Ano"], errors="coerce").astype("Int64")

    if df[month_col].dtype == "O":
        upper = df[month_col].astype(str).str.strip().str.upper()
        mes_num = upper.map(month_map)
        mask_na = mes_num.isna()
        if mask_na.any():
            mes_num.loc[mask_na] = pd.to_numeric(upper.loc[mask_na], errors="coerce")
        df["MesNum"] = mes_num.astype("Int64")
    else:
        df["MesNum"] = pd.to_numeric(df[month_col], errors="coerce").astype("Int64")

    df["Competencia"] = pd.to_datetime(
        dict(year=df["Ano"], month=df["MesNum"], day=1),
        errors="coerce"
    )

    return df


def compute_carteira_score(status_counts: pd.Series):
    """Retorna (score 0–100, label)."""
    if status_counts.empty:
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


def _normalize_col_name(col: str) -> str:
    """
    Remove BOM, espaços, faz lower e tira acentos básicos:
    ' Mês  ' -> 'mes'
    """
    s = str(col).strip().replace("\ufeff", "").lower()
    s = (
        s.replace("á", "a").replace("à", "a").replace("ã", "a").replace("â", "a")
         .replace("é", "e").replace("ê", "e")
         .replace("í", "i")
         .replace("ó", "o").replace("õ", "o").replace("ô", "o")
         .replace("ú", "u")
         .replace("ç", "c")
    )
    return s


def _load_core_csv(source, is_upload: bool) -> pd.DataFrame:
    """
    Core loader que lê o CSV (URL ou UploadedFile), trata separador, normaliza
    nomes de colunas e monta Ano/Mês/Valor/Representante/Cliente/Estado/Cidade.
    """
    last_error = None
    df = None

    # tenta ; depois ,
    for sep in [";", ","]:
        try:
            if is_upload:
                source.seek(0)
                df = pd.read_csv(source, sep=sep)
            else:
                df = pd.read_csv(source, sep=sep)
            break
        except Exception as e:
            last_error = e

    if df is None:
        raise RuntimeError(f"Erro ao ler CSV: {last_error}")

    # Normaliza nomes de colunas
    col_map = {_normalize_col_name(c): c for c in df.columns}

    needed = {
        "ano": ["ano", "year"],
        "mes": ["mes", "mesfat", "mes_fat", "competencia"],
        "valor": ["valor", "faturamento", "vl_total", "total"],
        "representante": ["representante", "vendedor", "rep"],
        "cliente": ["cliente", "razaosocial", "nomecliente", "cliente_nome"],
        "estado": ["estado", "uf"],
        "cidade": ["cidade", "municipio"],
    }

    rename_dict = {}
    missing_logical = []

    for logical_name, candidates in needed.items():
        found_original = None
        for cand in candidates:
            if cand in col_map:
                found_original = col_map[cand]
                break
        if found_original is None:
            missing_logical.append(logical_name)
        else:
            # padroniza em Português com inicial maiúscula
            rename_dict[found_original] = logical_name.capitalize()

    if missing_logical:
        raise KeyError(
            "Colunas obrigatórias não encontradas no CSV (depois de normalizar nomes): "
            + ", ".join(missing_logical)
        )

    df = df.rename(columns=rename_dict)

    # Converte Valor p/ número (suporta 1.234,56)
    df["Valor"] = (
        df["Valor"]
        .astype(str)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    df["Valor"] = pd.to_numeric(df["Valor"], errors="coerce").fillna(0.0)

    # Normaliza meses / competência
    df = normalize_month_column(df, "Mes")

    return df


@st.cache_data(show_spinner=True)
def load_data_from_github() -> pd.DataFrame:
    df = _load_core_csv(GITHUB_CSV_URL, is_upload=False)
    return df


@st.cache_data(show_spinner=False)
def load_data_from_upload(uploaded_file) -> pd.DataFrame:
    df = _load_core_csv(uploaded_file, is_upload=True)
    return df


# ==========================
# SIDEBAR – Filtros
# ==========================
st.sidebar.title("Filtros – Deep Dive")

mode = st.sidebar.radio(
    "Fonte de dados",
    ["GitHub (automático)", "Upload manual"],
    index=0,
)

if mode.startswith("GitHub"):
    st.sidebar.caption("Lendo dados diretamente do GitHub:")
    st.sidebar.code(GITHUB_CSV_URL, language="text")
    try:
        df = load_data_from_github()
    except Exception as e:
        st.error(f"Erro ao carregar dados do GitHub: {e}")
        st.stop()
else:
    uploaded_file = st.sidebar.file_uploader(
        "Envie um CSV equivalente ao relatorio_faturamento",
        type=["csv"],
    )
    if uploaded_file is None:
        st.info("Envie um arquivo CSV para continuar.")
        st.stop()
    try:
        df = load_data_from_upload(uploaded_file)
    except Exception as e:
        st.error(f"Erro ao carregar o CSV enviado: {e}")
        st.stop()

if df.empty:
    st.warning("O arquivo de dados está vazio.")
    st.stop()

# ==========================
# Filtros: representante + período
# ==========================
reps = sorted(df["Representante"].dropna().unique())
if not reps:
    st.error("Não foram encontrados representantes na base de dados.")
    st.stop()

rep_selected = st.sidebar.selectbox("Representante", reps)

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

if STATUS_COL in df_rep.columns:
    clientes_rep = (
        df_rep
        .dropna(subset=[STATUS_COL, "Cliente"])
        .groupby(["Cliente", STATUS_COL], as_index=False)
        .agg({"Valor": "sum"})
    )
    status_counts_series = clientes_rep.groupby(STATUS_COL)["Cliente"].nunique()
    carteira_score, carteira_label = compute_carteira_score(status_counts_series)
else:
    carteira_score, carteira_label = 50.0, "Neutra"

col1.metric("Total período", format_brl(total_rep))
col2.metric("Média mensal", format_brl(media_mensal))
col3.metric("Meses com venda", f"{meses_com_venda} / {total_meses_periodo}")
col4.metric("Participação", f"{participacao:.1%}")

with col5:
    st.metric(
        "Saúde da carteira",
        f"{carteira_score:.0f} / 100",
        carteira_label
    )

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
        .properties(
            height=260,
        )
    )

    st.altair_chart(chart_ts, use_container_width=True)

st.markdown("---")

# ==========================
# SAÚDE DA CARTEIRA – DETALHES
# ==========================
st.subheader("Saúde da carteira – Detalhes")

if STATUS_COL not in df_rep.columns:
    st.info(
        f"Coluna de status da carteira (`{STATUS_COL}`) não encontrada no dataframe. "
        "Adicione esta coluna no CSV para ver a distribuição de Novos/Perdidos/Crescendo/Caindo/Estáveis."
    )
else:
    clientes_rep = (
        df_rep
        .dropna(subset=[STATUS_COL, "Cliente"])
        .groupby(["Cliente", STATUS_COL, "Estado", "Cidade"], as_index=False)
        .agg({"Valor": "sum"})
    )

    status_counts = clientes_rep.groupby(STATUS_COL)["Cliente"].nunique().reset_index()
    status_counts = status_counts.rename(
        columns={"Cliente": "QtdClientes", STATUS_COL: "Status"}
    )
    total_clientes = status_counts["QtdClientes"].sum()
    status_counts["%Clientes"] = (
        status_counts["QtdClientes"] / total_clientes if total_clientes > 0 else 0
    )

    col_pie, col_table = st.columns([1, 1.2])

    with col_pie:
        st.caption("Distribuição de clientes por status")
        if total_clientes == 0:
            st.info("Nenhum cliente com status definido para este representante no período.")
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

    status_options = sorted(clientes_rep[STATUS_COL].dropna().unique())
    status_selected = st.multiselect(
        "Filtrar por status",
        options=status_options,
        default=status_options,
    )

    df_clientes_view = clientes_rep.copy()
    if status_selected:
        df_clientes_view = df_clientes_view[df_clientes_view[STATUS_COL].isin(status_selected)]

    df_clientes_view = df_clientes_view.rename(
        columns={
            "Valor": "Faturamento",
            STATUS_COL: "StatusCarteira"
        }
    )

    df_clientes_view["FaturamentoFmt"] = df_clientes_view["Faturamento"].map(format_brl)

    df_clientes_view = df_clientes_view.sort_values(
        "Faturamento", ascending=False
    )[
        ["Cliente", "Estado", "Cidade", "StatusCarteira", "FaturamentoFmt"]
    ]

    st.dataframe(
        df_clientes_view,
        hide_index=True,
        use_container_width=True,
    )
