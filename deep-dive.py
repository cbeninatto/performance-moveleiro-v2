import streamlit as st
import pandas as pd
import altair as alt
import folium
from streamlit_folium import st_folium
import plotly.express as px

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

CITY_GEO_CSV_URL = (
    "https://raw.githubusercontent.com/"
    "cbeninatto/performance-moveleiro-v2/main/data/cidades_br_geo.csv"
)

STATUS_COL = "StatusCarteira"


# ==========================
# HELPERS
# ==========================
def format_brl(value: float) -> str:
    if pd.isna(value):
        return "R$ 0,00"
    return "R$ " + f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def format_brl_compact(value: float) -> str:
    """Formato mais curto para cards (evita cortar número)."""
    if pd.isna(value):
        return "R$ 0"
    v = float(value)
    av = abs(v)
    if av >= 1_000_000_000:
        return "R$ " + f"{v/1_000_000_000:.1f} bi".replace(".", ",")
    if av >= 1_000_000:
        return "R$ " + f"{v/1_000_000:.1f} mi".replace(".", ",")
    if av >= 1_000:
        return "R$ " + f"{v/1_000:.1f} mil".replace(".", ",")
    return format_brl(v)


def load_data() -> pd.DataFrame:
    """Carrega SEM cache para sempre pegar a versão mais recente do CSV do GitHub."""
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

    df["Valor"] = pd.to_numeric(df["Valor"], errors="coerce").fillna(0.0)
    df["Quantidade"] = pd.to_numeric(df["Quantidade"], errors="coerce").fillna(0.0)
    df["Ano"] = pd.to_numeric(df["Ano"], errors="coerce").astype("Int64")

    # MesNum SEMPRE criado aqui
    df["MesNum"] = pd.to_numeric(df["Mes"], errors="coerce").astype("Int64")

    df["Competencia"] = pd.to_datetime(
        dict(year=df["Ano"], month=df["MesNum"], day=1),
        errors="coerce",
    )

    return df


def compute_carteira_score(clientes_carteira: pd.DataFrame):
    """
    Calcula o Índice de Saúde da Carteira (ISC) de 0 a 100 usando receita por status.

    Lógica:
      - Para cada cliente, define um "PesoReceita" = max(ValorAtual, ValorAnterior)
      - Soma PesoReceita por StatusCarteira
      - Converte em % da carteira (p_status)
      - Aplica pesos:
          Crescendo: +2
          Novos:     +1
          Estáveis:  +1
          Caindo:    -1
          Perdidos:  -2
      - Normaliza para 0–100:
          ISC = (score_bruto + 2) / 4 * 100
      - Regra extra: se churn de receita > 20%, não deixa o índice ficar em "Saudável"
    """
    if clientes_carteira is None or clientes_carteira.empty:
        return 50.0, "Neutra"

    df = clientes_carteira.copy()

    # Garantir colunas numéricas
    for col in ["ValorAtual", "ValorAnterior"]:
        df[col] = pd.to_numeric(df.get(col, 0.0), errors="coerce").fillna(0.0)

    # Peso da receita de cada cliente = tamanho típico dele na carteira
    df["PesoReceita"] = df[["ValorAtual", "ValorAnterior"]].max(axis=1)
    df["PesoReceita"] = df["PesoReceita"].clip(lower=0)

    # Soma por status
    if STATUS_COL not in df.columns:
        return 50.0, "Neutra"

    receita_status = df.groupby(STATUS_COL)["PesoReceita"].sum()
    R_total = float(receita_status.sum())

    if R_total <= 0:
        return 50.0, "Neutra"

    # Pesos por status (mais focado em receita do que em quantidade de clientes)
    pesos = {
        "Novos": 1,
        "Novo": 1,
        "Crescendo": 2,
        "CRESCENDO": 2,
        "Estáveis": 1,
        "Estável": 1,
        "ESTAVEIS": 1,
        "Caindo": -1,
        "CAINDO": -1,
        "Perdidos": -2,
        "Perdido": -2,
        "PERDIDOS": -2,
    }

    # Score bruto ponderado pela receita de cada status
    score_bruto = 0.0
    for status, receita in receita_status.items():
        w = pesos.get(str(status), 0)
        share = receita / R_total
        score_bruto += w * share

    # Normaliza para 0–100
    isc = (score_bruto + 2) / 4 * 100
    isc = max(0.0, min(100.0, isc))

    # --------- Regra extra: churn de receita ----------
    # Considera como "base anterior" quem tinha ValorAnterior > 0
    base_anterior = df[df["ValorAnterior"] > 0].copy()
    base_total = float(base_anterior["PesoReceita"].sum())

    # Receita perdida (status = Perdidos)
    perdidos_mask = df[STATUS_COL].astype(str).str.upper().isin(["PERDIDOS", "PERDIDO"])
    receita_perdida = float(df.loc[perdidos_mask, "PesoReceita"].sum())

    churn_receita = 0.0
    if base_total > 0:
        churn_receita = receita_perdida / base_total

    # Se churn > 20%, não permitir que a carteira seja classificada como "Saudável"
    if churn_receita > 0.20 and isc >= 70:
        isc = 69.0  # força ficar no máximo em "Neutra/Atenção"

    # --------- Traduz ISC em rótulo ----------
    if isc < 30:
        label = "Crítica"
    elif isc < 50:
        label = "Alerta"
    elif isc < 70:
        label = "Neutra"
    else:
        label = "Saudável"

    return float(isc), label


MONTH_MAP_NUM_TO_NAME = {
    1: "JAN", 2: "FEV", 3: "MAR", 4: "ABR",
    5: "MAI", 6: "JUN", 7: "JUL", 8: "AGO",
    9: "SET", 10: "OUT", 11: "NOV", 12: "DEZ",
}
MONTH_MAP_NAME_TO_NUM = {v: k for k, v in MONTH_MAP_NUM_TO_NAME.items()}


def format_period_label(start: pd.Timestamp, end: pd.Timestamp) -> str:
    def fmt(d: pd.Timestamp) -> str:
        return f"{MONTH_MAP_NUM_TO_NAME[d.month]} {str(d.year)[2:]}"
    if start.year == end.year and start.month == end.month:
        return fmt(start)
    return f"{fmt(start)} - {fmt(end)}"


def build_carteira_status(df_all: pd.DataFrame,
                          rep: str,
                          start_comp: pd.Timestamp,
                          end_comp: pd.Timestamp) -> pd.DataFrame:
    """
    Calcula StatusCarteira (Novos / Perdidos / Crescendo / Caindo / Estáveis)
    comparando o período selecionado com a JANELA ANTERIOR de mesmo tamanho.

    Estado/Cidade vêm tanto do período atual quanto do anterior,
    garantindo localização também para 'Perdidos'.
    """
    df_rep_all = df_all[df_all["Representante"] == rep].copy()
    if df_rep_all.empty:
        return pd.DataFrame(columns=[
            "Cliente", "Estado", "Cidade",
            "ValorAtual", "ValorAnterior", STATUS_COL
        ])

    months_span = (end_comp.year - start_comp.year) * 12 + (end_comp.month - start_comp.month) + 1

    prev_end = start_comp - pd.DateOffset(months=1)
    prev_start = prev_end - pd.DateOffset(months=months_span - 1)

    mask_curr = (df_rep_all["Competencia"] >= start_comp) & (df_rep_all["Competencia"] <= end_comp)
    mask_prev = (df_rep_all["Competencia"] >= prev_start) & (df_rep_all["Competencia"] <= prev_end)

    df_curr = df_rep_all.loc[mask_curr].copy()
    df_prev = df_rep_all.loc[mask_prev].copy()

    # Período atual: Valor + localização
    curr_agg = (
        df_curr
        .groupby("Cliente", as_index=False)
        .agg({
            "Valor": "sum",
            "Estado": "first",
            "Cidade": "first",
        })
        .rename(columns={
            "Valor": "ValorAtual",
            "Estado": "EstadoAtual",
            "Cidade": "CidadeAtual",
        })
    )

    # Período anterior: Valor + localização
    prev_agg = (
        df_prev
        .groupby("Cliente", as_index=False)
        .agg({
            "Valor": "sum",
            "Estado": "first",
            "Cidade": "first",
        })
        .rename(columns={
            "Valor": "ValorAnterior",
            "Estado": "EstadoAnterior",
            "Cidade": "CidadeAnterior",
        })
    )

    # Junta tudo
    clientes = pd.merge(curr_agg, prev_agg, on="Cliente", how="outer")

    # ValorAtual / ValorAnterior
    clientes["ValorAtual"] = clientes["ValorAtual"].fillna(0.0)
    clientes["ValorAnterior"] = clientes["ValorAnterior"].fillna(0.0)

    # Estado/Cidade: usa atual, se não tiver usa anterior
    clientes["Estado"] = clientes["EstadoAtual"].combine_first(clientes["EstadoAnterior"])
    clientes["Cidade"] = clientes["CidadeAtual"].combine_first(clientes["CidadeAnterior"])
    clientes["Estado"] = clientes["Estado"].fillna("")
    clientes["Cidade"] = clientes["Cidade"].fillna("")

    # Classificação de status
    def classify(row):
        va = row["ValorAtual"]
        vp = row["ValorAnterior"]
        if va > 0 and vp == 0:
            return "Novos"
        if va == 0 and vp > 0:
            return "Perdidos"
        if va > 0 and vp > 0:
            ratio = va / vp if vp != 0 else 0
            if ratio >= 1.2:
                return "Crescendo"
            elif ratio <= 0.8:
                return "Caindo"
            else:
                return "Estáveis"
        return "Estáveis"

    clientes[STATUS_COL] = clientes.apply(classify, axis=1)

    # Remove clientes sem movimento em nenhum dos dois períodos
    clientes = clientes[(clientes["ValorAtual"] > 0) | (clientes["ValorAnterior"] > 0)]

    # Mantém somente colunas finais
    clientes = clientes[[
        "Cliente",
        "Estado",
        "Cidade",
        "ValorAtual",
        "ValorAnterior",
        STATUS_COL,
    ]]

    return clientes


@st.cache_data(show_spinner=True)
def load_geo() -> pd.DataFrame:
    df_geo = pd.read_csv(CITY_GEO_CSV_URL, sep=None, engine="python")

    original_cols = list(df_geo.columns)
    df_geo.columns = [str(c).strip() for c in df_geo.columns]

    def find_col(candidates):
        for col in df_geo.columns:
            norm = str(col).strip().lower()
            if norm in candidates:
                return col
        return None

    estado_col = find_col({"estado", "uf"})
    cidade_col = find_col({"cidade", "municipio", "município"})
    lat_col = find_col({"lat", "latitude"})
    lon_col = find_col({"lon", "longitude", "long", "lng"})

    missing = []
    if estado_col is None:
        missing.append("Estado/UF")
    if cidade_col is None:
        missing.append("Cidade/Municipio")
    if lat_col is None:
        missing.append("Latitude (lat)")
    if lon_col is None:
        missing.append("Longitude (lon)")

    if missing:
        raise ValueError(
            "cidades_br_geo.csv está com colunas diferentes das esperadas.\n"
            f"Faltando: {', '.join(missing)}\n"
            f"Colunas encontradas: {', '.join(map(str, original_cols))}"
        )

    df_geo = df_geo[[estado_col, cidade_col, lat_col, lon_col]].rename(
        columns={
            estado_col: "Estado",
            cidade_col: "Cidade",
            lat_col: "lat",
            lon_col: "lon",
        }
    )

    df_geo["lat"] = pd.to_numeric(df_geo["lat"], errors="coerce")
    df_geo["lon"] = pd.to_numeric(df_geo["lon"], errors="coerce")
    df_geo = df_geo.dropna(subset=["lat", "lon"])

    df_geo["key"] = (
        df_geo["Estado"].astype(str).str.strip().str.upper()
        + "|"
        + df_geo["Cidade"].astype(str).str.strip().str.upper()
    )

    return df_geo[["key", "Estado", "Cidade", "lat", "lon"]]


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

month_names = [MONTH_MAP_NUM_TO_NAME[m] for m in range(1, 13)]

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

months_span_for_carteira = (end_comp.year - start_comp.year) * 12 + (end_comp.month - start_comp.month) + 1
prev_end = start_comp - pd.DateOffset(months=1)
prev_start = prev_end - pd.DateOffset(months=months_span_for_carteira - 1)

current_period_label = format_period_label(start_comp, end_comp)
previous_period_label = format_period_label(prev_start, prev_end)

mask_period = (df["Competencia"] >= start_comp) & (df["Competencia"] <= end_comp)
df_period = df.loc[mask_period].copy()

if df_period.empty:
    st.warning("Nenhuma venda no período selecionado.")
    st.stop()

# Representantes disponíveis APENAS no período selecionado
reps_period = sorted(df_period["Representante"].dropna().unique())
if not reps_period:
    st.error("Não há representantes com vendas no período selecionado.")
    st.stop()

rep_selected = st.sidebar.selectbox("Representante", reps_period)

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
# MÉTRICAS PRINCIPAIS – 5 COLUNAS
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

media_mensal = total_rep / meses_com_venda if meses_com_venda > 0 else 0.0

# Distribuição por clientes: N80, HHI, Top shares
if not df_rep.empty and total_rep > 0:
    df_clientes_tot = (
        df_rep.groupby("Cliente", as_index=False)["Valor"]
        .sum()
        .sort_values("Valor", ascending=False)
    )
    num_clientes_rep = df_clientes_tot["Cliente"].nunique()

    shares = df_clientes_tot["Valor"] / total_rep
    df_clientes_tot["Share"] = shares

    cum_share = shares.cumsum()
    n80_count = 0
    for i, val in enumerate(cum_share, start=1):
        n80_count = i
        if val >= 0.8:
            break
    n80_ratio = n80_count / num_clientes_rep if num_clientes_rep > 0 else 0.0

    hhi_value = float((shares ** 2).sum())
    if hhi_value < 0.10:
        hhi_label = "Baixa concentração"
    elif hhi_value < 0.20:
        hhi_label = "Concentração moderada"
    else:
        hhi_label = "Alta concentração"

    if "Baixa" in hhi_label:
        hhi_label_short = "Baixa"
    elif "moderada" in hhi_label:
        hhi_label_short = "Moderada"
    elif "Alta" in hhi_label:
        hhi_label_short = "Alta"
    else:
        hhi_label_short = hhi_label

    top1_share = shares.iloc[:1].sum()
    top3_share = shares.iloc[:3].sum()
    top10_share = shares.iloc[:10].sum()
else:
    num_clientes_rep = 0
    n80_count = 0
    n80_ratio = 0.0
    hhi_value = 0.0
    hhi_label = "Sem dados"
    hhi_label_short = "Sem dados"
    top1_share = 0.0
    top3_share = 0.0
    top10_share = 0.0

clientes_atendidos = num_clientes_rep
cidades_atendidas = (
    df_rep[["Estado", "Cidade"]]
    .dropna()
    .drop_duplicates()
    .shape[0]
)
estados_atendidos = df_rep["Estado"].dropna().nunique()

if not clientes_carteira.empty:
    carteira_score, carteira_label = compute_carteira_score(clientes_carteira)
else:
    carteira_score, carteira_label = 50.0, "Neutra"

col1.metric("Total período", format_brl_compact(total_rep))
col2.metric("Média mensal", format_brl_compact(media_mensal))
col3.metric("Distribuição por clientes", hhi_label_short, f"N80: {n80_count} clientes")
col4.metric("Saúde da carteira", f"{carteira_score:.0f} / 100", carteira_label)
col5.metric("Clientes atendidos", f"{clientes_atendidos}")

st.markdown("---")

# ==========================
# DESTAQUES DO PERÍODO
# ==========================
st.subheader("Destaques do período")

if df_rep.empty:
    st.info("Este representante não possui vendas no período selecionado.")
else:
    mensal_rep = (
        df_rep
        .groupby(["Ano", "MesNum"], as_index=False)[["Valor", "Quantidade"]]
        .sum()
    )
    mensal_rep["Competencia"] = pd.to_datetime(
        dict(year=mensal_rep["Ano"], month=mensal_rep["MesNum"], day=1)
    )
    mensal_rep["MesLabel"] = mensal_rep["Competencia"].dt.strftime("%b %Y")

    best_fat = mensal_rep.loc[mensal_rep["Valor"].idxmax()]
    worst_fat = mensal_rep.loc[mensal_rep["Valor"].idxmin()]
    best_vol = mensal_rep.loc[mensal_rep["Quantidade"].idxmax()]
    worst_vol = mensal_rep.loc[mensal_rep["Quantidade"].idxmin()]

    col_d1, col_d2 = st.columns(2)

    with col_d1:
        st.markdown("**Faturamento**")
        st.write(
            f"• Melhor mês: **{best_fat['MesLabel']}** — {format_brl(best_fat['Valor'])}"
        )
        st.write(
            f"• Pior mês: **{worst_fat['MesLabel']}** — {format_brl(worst_fat['Valor'])}"
        )

    with col_d2:
        st.markdown("**Volume**")
        st.write(
            f"• Melhor mês: **{best_vol['MesLabel']}** — "
            f"{int(best_vol['Quantidade']):,}".replace(",", ".")
        )
        st.write(
            f"• Pior mês: **{worst_vol['MesLabel']}** — "
            f"{int(worst_vol['Quantidade']):,}".replace(",", ".")
        )

st.markdown("---")

# ==========================
# MAPA DE CLIENTES
# ==========================
st.subheader("Mapa de Clientes")

# session_state para lembrar a última cidade clicada
if "selected_city_tooltip" not in st.session_state:
    st.session_state["selected_city_tooltip"] = None

if df_rep.empty:
    st.info("Este representante não possui vendas no período selecionado.")
else:
    try:
        df_geo = load_geo()

        df_cities = (
            df_rep.groupby(["Estado", "Cidade"], as_index=False)
            .agg(
                Valor=("Valor", "sum"),
                Quantidade=("Quantidade", "sum"),
                Clientes=("Cliente", "nunique"),
            )
        )

        df_cities["key"] = (
            df_cities["Estado"].astype(str).str.strip().str.upper()
            + "|"
            + df_cities["Cidade"].astype(str).str.strip().str.upper()
        )

        df_map = df_cities.merge(
            df_geo,
            on="key",
            how="inner",
            suffixes=("_fat", "_geo"),
        )

        if df_map.empty:
            st.info("Não há coordenadas de cidades para exibir no mapa.")
        else:
            # Texto de tooltip para cada cidade
            df_map["Tooltip"] = (
                df_map["Cidade_fat"].astype(str)
                + " - "
                + df_map["Estado_fat"].astype(str)
            )

            metric_choice = st.radio(
                "Métrica do mapa",
                ["Faturamento", "Volume"],
                horizontal=True,
            )

            if metric_choice == "Faturamento":
                metric_col = "Valor"
                metric_label = "Faturamento (R$)"
            else:
                metric_col = "Quantidade"
                metric_label = "Volume"

            if df_map[metric_col].max() <= 0:
                st.info("Sem dados para exibir no mapa nesse período.")
            else:
                values = df_map[metric_col]
                try:
                    df_map["bin"], bins = pd.qcut(
                        values, q=4, labels=False, retbins=True, duplicates="drop"
                    )
                except ValueError:
                    df_map["bin"] = 0
                    bins = [values.min(), values.max()]

                # Invertido: vermelho = menor valor, verde = maior valor
                colors = ["#ef4444", "#f97316", "#eab308", "#22c55e"]  # low → high

                # monta legenda das cores (quartis)
                legend_entries = []
                if len(bins) > 1:
                    for i in range(len(bins) - 1):
                        low = bins[i]
                        high = bins[i + 1]
                        if metric_col == "Valor":
                            label_range = f"{format_brl(low)} – {format_brl(high)}"
                        else:
                            low_i = int(round(low))
                            high_i = int(round(high))
                            label_range = (
                                f"{low_i:,} – {high_i:,}".replace(",", ".")
                            )
                        legend_entries.append((colors[i % len(colors)], label_range))

                col_map, col_stats = st.columns([0.8, 1.2])

                # ---------- MAPA ----------
                with col_map:
                    center = [df_map["lat"].mean(), df_map["lon"].mean()]
                    m = folium.Map(location=center, zoom_start=5, tiles="cartodbpositron")

                    for _, row in df_map.iterrows():
                        bin_idx = int(row["bin"]) if pd.notna(row["bin"]) else 0
                        color = colors[bin_idx % len(colors)]

                        if metric_col == "Valor":
                            metric_val_str = format_brl(row["Valor"])
                        else:
                            metric_val_str = f"{int(row['Quantidade']):,}".replace(",", ".")

                        popup_html = (
                            f"<b>{row['Cidade_fat']} - {row['Estado_fat']}</b><br>"
                            f"{metric_label}: {metric_val_str}<br>"
                            f"Clientes: {int(row['Clientes'])}"
                        )

                        folium.CircleMarker(
                            location=[row["lat"], row["lon"]],
                            radius=6,
                            color=color,
                            fill=True,
                            fill_color=color,
                            fill_opacity=0.8,
                            popup=folium.Popup(popup_html, max_width=300),
                            tooltip=row["Tooltip"],  # usado para identificar a cidade clicada
                        ).add_to(m)

                    map_data = st_folium(m, width=None, height=800)

                    # legenda embaixo do mapa
                    if legend_entries:
                        legend_html = "<div style='font-size:0.8rem; margin-top:0.5rem;'>"
                        legend_html += f"<b>Legenda – {metric_label}</b><br>"
                        for color, label_range in legend_entries:
                            legend_html += (
                                f"<span style='display:inline-block;width:12px;"
                                f"height:12px;background:{color};"
                                f"margin-right:4px;border-radius:2px;'></span>"
                                f"{label_range}<br>"
                            )
                        legend_html += "</div>"
                        st.markdown(legend_html, unsafe_allow_html=True)

                # ---------- COBERTURA + PRINCIPAIS CLIENTES + TABELA DA CIDADE ----------
                # descobre qual cidade foi clicada (tooltip do último objeto clicado)
                selected_label = None
                if isinstance(map_data, dict):
                    selected_label = map_data.get("last_object_clicked_tooltip")

                if selected_label:
                    st.session_state["selected_city_tooltip"] = selected_label
                else:
                    selected_label = st.session_state.get("selected_city_tooltip")

                with col_stats:
                    st.markdown("**Cobertura**")
                    cov1, cov2, cov3 = st.columns(3)
                    cov1.metric("Cidades atendidas", f"{cidades_atendidas}")
                    cov2.metric("Estados atendidos", f"{estados_atendidos}")
                    cov3.metric("Clientes atendidos", f"{clientes_atendidos}")

                    st.markdown("**Principais clientes**")

                    df_top_clients = (
                        df_rep.groupby(["Cliente", "Estado", "Cidade"], as_index=False)["Valor"]
                        .sum()
                        .sort_values("Valor", ascending=False)
                        .head(10)
                    )
                    df_top_clients["Faturamento"] = df_top_clients["Valor"].map(format_brl)
                    df_top_display = df_top_clients[
                        ["Cliente", "Cidade", "Estado", "Faturamento"]
                    ]

                    clientes_table_css = """
                    <style>
                    table.clientes-table {
                        width: 100%;
                        border-collapse: collapse;
                    }
                    table.clientes-table th,
                    table.clientes-table td {
                        padding: 0.25rem 0.5rem;
                        font-size: 0.85rem;
                        text-align: left;
                        border-bottom: 1px solid rgba(255,255,255,0.08);
                        white-space: nowrap;
                    }
                    table.clientes-table th:nth-child(1),
                    table.clientes-table td:nth-child(1) {
                        white-space: normal;
                    }
                    </style>
                    """
                    st.markdown(clientes_table_css, unsafe_allow_html=True)

                    cols_top = list(df_top_display.columns)
                    html_top = "<table class='clientes-table'><thead><tr>"
                    html_top += "".join(f"<th>{c}</th>" for c in cols_top)
                    html_top += "</tr></thead><tbody>"
                    for _, row in df_top_display.iterrows():
                        html_top += "<tr>" + "".join(f"<td>{row[c]}</td>" for c in cols_top) + "</tr>"
                    html_top += "</tbody></table>"

                    st.markdown(html_top, unsafe_allow_html=True)

                    # ---- TABELA DE CLIENTES DA CIDADE CLICADA ----
                    if selected_label:
                        row_city = df_map[df_map["Tooltip"] == selected_label].head(1)
                        if not row_city.empty:
                            cidade_sel = row_city["Cidade_fat"].iloc[0]
                            estado_sel = row_city["Estado_fat"].iloc[0]

                            df_city_clients = df_rep[
                                (df_rep["Cidade"] == cidade_sel)
                                & (df_rep["Estado"] == estado_sel)
                            ].copy()

                            if not df_city_clients.empty:
                                df_city_agg = (
                                    df_city_clients
                                    .groupby("Cliente", as_index=False)
                                    .agg(
                                        Valor=("Valor", "sum"),
                                        Quantidade=("Quantidade", "sum"),
                                    )
                                    .sort_values("Valor", ascending=False)
                                )
                                df_city_agg["Faturamento"] = df_city_agg["Valor"].map(format_brl)
                                display_city = df_city_agg[
                                    ["Cliente", "Quantidade", "Faturamento"]
                                ]

                                st.markdown(
                                    f"**Clientes em {cidade_sel} - {estado_sel}**"
                                )

                                # CSS específico para a tabela da cidade
                                city_clients_css = """
                                <style>
                                table.city-table {
                                    width: 100%;
                                    border-collapse: collapse;
                                }
                                table.city-table th,
                                table.city-table td {
                                    padding: 0.25rem 0.5rem;
                                    font-size: 0.85rem;
                                    border-bottom: 1px solid rgba(255,255,255,0.08);
                                }
                                /* Cabeçalhos de Quantidade e Faturamento centralizados */
                                table.city-table th:nth-child(2),
                                table.city-table th:nth-child(3) {
                                    text-align: center;
                                }
                                /* Valores de todas as colunas alinhados à esquerda */
                                table.city-table td {
                                    text-align: left;
                                }
                                </style>
                                """
                                st.markdown(city_clients_css, unsafe_allow_html=True)

                                with st.expander("Ver lista de clientes da cidade", expanded=True):
                                    cols_city = list(display_city.columns)
                                    html_city = "<table class='city-table'><thead><tr>"
                                    html_city += "".join(f"<th>{c}</th>" for c in cols_city)
                                    html_city += "</tr></thead><tbody>"
                                    for _, row in display_city.iterrows():
                                        html_city += "<tr>" + "".join(
                                            f"<td>{row[c]}</td>" for c in cols_city
                                        ) + "</tr>"
                                    html_city += "</tbody></table>"
                                    st.markdown(html_city, unsafe_allow_html=True)

    except Exception as e:
        st.info(f"Mapa de clientes ainda não disponível: {e}")

st.markdown("---")

# ==========================
# EVOLUÇÃO – FATURAMENTO x VOLUME
# ==========================
st.subheader("Evolução – Faturamento x Volume")

if df_rep.empty:
    st.info("Este representante não possui vendas no período selecionado.")
else:
    ts_rep = (
        df_rep
        .groupby("Competencia", as_index=False)[["Valor", "Quantidade"]]
        .sum()
        .sort_values("Competencia")
    )

    ts_rep["MesLabelBr"] = ts_rep["Competencia"].apply(
        lambda d: f"{MONTH_MAP_NUM_TO_NAME[d.month]} {str(d.year)[2:]}"
    )
    x_order = ts_rep["MesLabelBr"].tolist()

    base = alt.Chart(ts_rep).encode(
        x=alt.X(
            "MesLabelBr:N",
            sort=x_order,
            axis=alt.Axis(title=None),
        )
    )

    bars = base.mark_bar(color="#38bdf8").encode(
        y=alt.Y(
            "Valor:Q",
            axis=alt.Axis(title="Faturamento (R$)"),
        ),
        tooltip=[
            alt.Tooltip("MesLabelBr:N", title="Mês"),
            alt.Tooltip("Valor:Q", title="Faturamento (R$)", format=",.2f"),
            alt.Tooltip("Quantidade:Q", title="Volume"),
        ],
    )

    line = base.mark_line(
        color="#22c55e",
        strokeWidth=3,
        point=alt.OverlayMarkDef(color="#22c55e", filled=True, size=70),
    ).encode(
        y=alt.Y(
            "Quantidade:Q",
            axis=alt.Axis(title="Volume", orient="right"),
        ),
        tooltip=[
            alt.Tooltip("MesLabelBr:N", title="Mês"),
            alt.Tooltip("Valor:Q", title="Faturamento (R$)", format=",.2f"),
            alt.Tooltip("Quantidade:Q", title="Volume"),
        ],
    )

    combo_chart = alt.layer(bars, line).resolve_scale(
        y="independent"
    ).properties(
        height=320,
    )

    st.altair_chart(combo_chart, width="stretch")

st.markdown("---")

# ==========================
# DISTRIBUIÇÃO POR CLIENTES – SEÇÃO
# ==========================
st.subheader("Distribuição por clientes")

if df_rep.empty or clientes_atendidos == 0:
    st.info("Nenhum cliente com vendas no período selecionado.")
else:
    df_clientes = (
        df_rep.groupby("Cliente", as_index=False)[["Valor", "Quantidade"]]
        .sum()
        .sort_values("Valor", ascending=False)
    )
    total_rep_safe = total_rep if total_rep > 0 else 1.0
    df_clientes["Share"] = df_clientes["Valor"] / total_rep_safe

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("N80", f"{n80_count}", f"{n80_ratio:.0%} da carteira")
    k2.metric("Índice de concentração", hhi_label_short, f"HHI {hhi_value:.3f}")
    k3.metric("Top 1 cliente", f"{top1_share:.1%}")
    k4.metric("Top 3 clientes", f"{top3_share:.1%}")
    k5.metric("Top 10 clientes", f"{top10_share:.1%}")

    st.caption(
        f"{clientes_atendidos} clientes no período selecionado."
    )

    col_dc1, col_dc2 = st.columns([1.4, 1])

    with col_dc1:
        st.caption("Top 10 clientes por faturamento")
        chart_clients = (
            alt.Chart(df_clientes.head(10))
            .mark_bar()
            .encode(
                x=alt.X("Valor:Q", title="Faturamento (R$)"),
                y=alt.Y("Cliente:N", sort="-x", title="Cliente"),
                tooltip=[
                    alt.Tooltip("Cliente:N", title="Cliente"),
                    alt.Tooltip("Valor:Q", title="Faturamento", format=",.2f"),
                    alt.Tooltip("Quantidade:Q", title="Volume"),
                ],
            )
            .properties(height=420)
        )
        st.altair_chart(chart_clients, width="stretch")

    # Pizza com todos os clientes, Top 10 destacados (Plotly 6.5 compatível)
    with col_dc2:
        st.caption("Participação dos clientes (Top 10 destacados)")

        df_pie = df_clientes.copy()
        df_pie["Rank"] = df_pie["Valor"].rank(method="first", ascending=False)

        df_pie["Grupo"] = df_pie.apply(
            lambda r: r["Cliente"] if r["Rank"] <= 10 else "Outros",
            axis=1,
        )

        dist_df = (
            df_pie.groupby("Grupo", as_index=False)["Valor"]
            .sum()
        )
        dist_df["Share"] = dist_df["Valor"] / total_rep_safe
        dist_df = dist_df.sort_values("Share", ascending=False)

        dist_df["Legenda"] = dist_df.apply(
            lambda r: f"{r['Grupo']} {r['Share']*100:.1f}%",
            axis=1,
        )

        def make_text(row):
            if row["Share"] >= 0.07:
                return f"{row['Grupo']}<br>{row['Share']*100:.1f}%"
            else:
                return ""
        dist_df["Text"] = dist_df.apply(make_text, axis=1)

        order_legenda = dist_df["Legenda"].tolist()

        fig = px.pie(
            dist_df,
            values="Valor",
            names="Legenda",
            category_orders={"Legenda": order_legenda},
        )

        fig.update_traces(
            text=dist_df["Text"],
            textposition="inside",
            textinfo="text",
            insidetextorientation="radial",
        )

        fig.update_layout(
            legend=dict(
                title="Cliente (Top 10) / Outros",
                traceorder="normal",
            )
        )

        st.plotly_chart(fig, width="stretch")

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

    fat_status = (
        clientes_carteira.groupby(STATUS_COL)[["ValorAtual", "ValorAnterior"]]
        .sum()
        .reset_index()
        .rename(columns={STATUS_COL: "Status"})
    )
    fat_status["Faturamento"] = fat_status["ValorAtual"] - fat_status["ValorAnterior"]
    fat_status = fat_status[["Status", "Faturamento"]]

    status_counts = status_counts.merge(fat_status, on="Status", how="left")

    total_clientes = status_counts["QtdClientes"].sum()
    status_counts["%Clientes"] = (
        status_counts["QtdClientes"] / total_clientes if total_clientes > 0 else 0
    )

    status_order = ["Novos", "Crescendo", "Estáveis", "Caindo", "Perdidos"]
    status_counts["Status"] = pd.Categorical(
        status_counts["Status"], categories=status_order, ordered=True
    )
    status_counts = status_counts.sort_values("Status")

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
                    color=alt.Color(
                        "Status:N",
                        legend=alt.Legend(title="Status"),
                        scale=alt.Scale(
                            domain=["Perdidos", "Caindo", "Estáveis", "Crescendo", "Novos"],
                            range=["#ef4444", "#f97316", "#eab308", "#22c55e", "#3b82f6"],
                        ),
                    ),
                    tooltip=[
                        alt.Tooltip("Status:N", title="Status"),
                        alt.Tooltip("QtdClientes:Q", title="Clientes"),
                        alt.Tooltip("%Clientes:Q", title="% Clientes", format=".1%"),
                    ],
                )
                .properties(height=320)
            )
            st.altair_chart(chart_pie, width="stretch")

    with col_table:
        st.caption("Resumo por status")
        status_counts_display = status_counts.copy()
        status_counts_display["%Clientes"] = status_counts_display["%Clientes"].map(
            lambda x: f"{x:.1%}"
        )
        status_counts_display["Faturamento"] = status_counts_display["Faturamento"].map(
            format_brl
        )

        status_counts_display = status_counts_display[
            ["Status", "QtdClientes", "%Clientes", "Faturamento"]
        ]

        st.dataframe(
            status_counts_display,
            hide_index=True,
            width="stretch",
        )

    st.markdown("### Status dos clientes")

    table_css = """
    <style>
    table.status-table {
        width: 100%;
        border-collapse: collapse;
        margin-bottom: 0.75rem;
    }
    table.status-table col:nth-child(1) { width: 30%; }
    table.status-table col:nth-child(2) { width: 10%; }
    table.status-table col:nth-child(3) { width: 15%; }
    table.status-table col:nth-child(4) { width: 10%; }
    table.status-table col:nth-child(5) { width: 17.5%; }
    table.status-table col:nth-child(6) { width: 17.5%; }

    table.status-table th,
    table.status-table td {
        padding: 0.2rem 0.5rem;
        border-bottom: 1px solid rgba(255,255,255,0.08);
        font-size: 0.85rem;
        text-align: left;
    }
    table.status-table th {
        font-weight: 600;
    }
    </style>
    """
    st.markdown(table_css, unsafe_allow_html=True)

    ordered_statuses = ["Novos", "Crescendo", "Estáveis", "Caindo", "Perdidos"]

    for status_name in ordered_statuses:
        df_status = clientes_carteira[clientes_carteira[STATUS_COL] == status_name].copy()
        if df_status.empty:
            continue

        df_status["FaturamentoAtualFmt"] = df_status["ValorAtual"].map(format_brl)
        df_status["FaturamentoAnteriorFmt"] = df_status["ValorAnterior"].map(format_brl)

        df_status = df_status.sort_values("ValorAtual", ascending=False)

        display_df = df_status[
            ["Cliente", "Estado", "Cidade", STATUS_COL,
             "FaturamentoAtualFmt", "FaturamentoAnteriorFmt"]
        ].rename(
            columns={
                STATUS_COL: "Status",
                "FaturamentoAtualFmt": f"Faturamento {current_period_label}",
                "FaturamentoAnteriorFmt": f"Faturamento {previous_period_label}",
            }
        )

        cols_status = list(display_df.columns)

        html_status = "<h5>" + status_name + "</h5>"
        html_status += "<table class='status-table'><colgroup>"
        html_status += "<col><col><col><col><col><col></colgroup><thead><tr>"
        html_status += "".join(f"<th>{c}</th>" for c in cols_status)
        html_status += "</tr></thead><tbody>"
        for _, row in display_df.iterrows():
            html_status += "<tr>" + "".join(f"<td>{row[c]}</td>" for c in cols_status) + "</tr>"
        html_status += "</tbody></table>"

        st.markdown(html_status, unsafe_allow_html=True)
