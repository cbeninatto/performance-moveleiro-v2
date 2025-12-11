@st.cache_data(show_spinner=True)
def load_geo() -> pd.DataFrame:
    """
    Carrega o CSV de coordenadas de cidades (cidades_br_geo.csv) do GitHub
    e normaliza colunas de Estado, Cidade, lat, lon.
    """
    df_geo = pd.read_csv(CITY_GEO_CSV_URL)

    # Detecta colunas estado/cidade ignorando caixa
    estado_col = next((c for c in df_geo.columns if c.lower() == "estado"), None)
    cidade_col = next((c for c in df_geo.columns if c.lower() == "cidade"), None)

    if estado_col is None or cidade_col is None:
        raise ValueError("cidades_br_geo.csv precisa ter colunas 'Estado' e 'Cidade'.")

    # Detecta colunas de latitude e longitude com nomes comuns
    lat_col = next(
        (c for c in df_geo.columns if c.lower() in ("lat", "latitude")),
        None,
    )
    lon_col = next(
        (c for c in df_geo.columns if c.lower() in ("lon", "longitude", "long")),
        None,
    )

    if lat_col is None or lon_col is None:
        raise ValueError("cidades_br_geo.csv precisa ter colunas de latitude/longitude.")

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
