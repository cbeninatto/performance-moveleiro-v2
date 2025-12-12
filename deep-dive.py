import streamlit as st
import pandas as pd
import altair as alt
import folium
from streamlit_folium import st_folium

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


@st.cache_data(show_spinner=True)
def load_data() -> pd.DataFrame:
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
    df["MesNum"] = pd.to_numeric(df["Mes"], errors="coerce").astype("Int64")

    df["Competencia"] = pd.to_datetime(
        dict(year=df["Ano"], month=df["MesNum"], day=1),
        errors="coerce",
    )

    return df


def compute_carteira_score(status_counts: pd.Series):
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

    prev_agg = (
        df_prev
        .groupby("Cliente", as_index=False)["Valor"]
        .sum()
        .rename(columns={"Valor": "ValorAnterior"})
    )

    clientes = pd.merge(curr_agg, prev_agg, on="Cliente", how="outer")

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
            ratio = va / vp if vp != 0 else 0
            if ratio >= 1.2:
                return "Crescendo"
            elif ratio <= 0.8:
                return "Caindo"
            else:
                return "Estáveis"
        return "Estáveis"

    clientes[STATUS_COL] = clientes.apply(classify, axis=1)
    clientes = clientes[(clientes["ValorAtual"] > 0) | (clientes["ValorAnterior"] > 0)]
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

reps = sorted(df["Representante"].dropna().unique())
if not reps:
    st.error("Não foram encontrados representantes na base de dados.")
    st.stop()
rep_selected = st.sidebar.selectbox("Representante", reps)

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
    status_counts_series = (
        clientes_carteira.groupby(STATUS_COL)["Cliente"].nunique()
    )
    carteira_score, carteira_label = compute_carteira_score(status_counts_series)
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
            # Cabeçalho: métrica do mapa (esq.) e cobertura (dir.)
            header_left, header_right = st.columns([1.2, 1])
            with header_left:
                metric_choice = st.radio(
                    "Métrica do mapa",
                    ["Faturamento", "Volume"],
                    horizontal=True,
                )
            with header_right:
                st.markdown("**Cobertura**")
                cov1, cov2, cov3 = st.columns(3)
                cov1.metric("Cidades atendidas", f"{cidades_atendidas}")
                cov2.metric("Estados atendidos", f"{estados_atendidos}")
                cov3.metric("Clientes atendidos", f"{clientes_atendidos}")

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

                colors = ["#22c55e", "#eab308", "#f97316", "#ef4444"]

                # Segunda linha: mapa mais estreito + stats com tabela
                col_map, col_stats = st.columns([1.1, 1])

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
                        ).add_to(m)

                    # mapa mais alto
                    st_folium(m, width=None, height=800)

                with col_stats:
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
                    st.table(df_top_display)

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

    st.altair_chart(combo_chart, use_container_width=True)

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

    # Mini KPIs (5 colunas) – usar label curto para não cortar
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
    st.altair_chart(chart_clients, use_container_width=True)
    
    # Pizza com clientes
    with col_dc2:
    st.caption("Participação dos clientes (Top 10 destacados)")

    # Usa TODOS os clientes no cálculo (100%),
    # mas só destaca os 10 maiores por nome (resto = "Outros")
    df_pie = df_clientes.copy()
    df_pie["Rank"] = df_pie["Valor"].rank(method="first", ascending=False)

    df_pie["Grupo"] = df_pie.apply(
        lambda r: r["Cliente"] if r["Rank"] <= 10 else "Outros",
        axis=1,
    )

    dist_df = (
        df_pie.groupby("Grupo", as_index=False)["Valor"]
        .sum()
        .sort_values("Valor", ascending=False)
    )
    dist_df["Share"] = dist_df["Valor"] / total_rep_safe

    # Só escreve o nome dentro das fatias grandes e que NÃO são "Outros"
    dist_df["LabelText"] = dist_df.apply(
        lambda r: r["Grupo"]
        if (r["Grupo"] != "Outros" and r["Share"] >= 0.07)
        else "",
        axis=1,
    )

    base_pie = alt.Chart(dist_df)

    pie = base_pie.mark_arc().encode(
        theta=alt.Theta("Share:Q"),
        color=alt.Color(
            "Grupo:N",
            legend=alt.Legend(title="Cliente (Top 10) / Outros"),
        ),
        tooltip=[
            alt.Tooltip("Grupo:N", title="Cliente / Grupo"),
            alt.Tooltip("Share:Q", title="% Faturamento", format=".1%"),
        ],
    )

    # Nomes dentro das fatias (para as maiores)
    text = base_pie.mark_text(radius=110, size=11).encode(
        theta=alt.Theta("Share:Q"),
        text="LabelText:N",
    )

    chart_pie = (pie + text).properties(height=320)

    st.altair_chart(chart_pie, use_container_width=True)

        
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

    # ==========================
    # STATUS DOS CLIENTES – TABELAS ALINHADAS
    # ==========================
    st.markdown("### Status dos clientes")

    # CSS para tabelas com colunas fixas
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

        cols = list(display_df.columns)

        # monta tabela HTML com colgroup fixo
        html = "<h5>" + status_name + "</h5>"
        html += "<table class='status-table'><colgroup>"
        html += "<col><col><col><col><col><col></colgroup><thead><tr>"
        html += "".join(f"<th>{c}</th>" for c in cols)
        html += "</tr></thead><tbody>"
        for _, row in display_df.iterrows():
            html += "<tr>" + "".join(f"<td>{row[c]}</td>" for c in cols) + "</tr>"
        html += "</tbody></table>"

        st.markdown(html, unsafe_allow_html=True)
