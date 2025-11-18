import streamlit as st
import pdfplumber
import pandas as pd
import re
from io import BytesIO
import time

# -------------------------------------------------------
# STREAMLIT UI CONFIG
# -------------------------------------------------------
st.set_page_config(page_title="Billing Report Extractor", page_icon="📄")
st.title("📄 Billing Report PDF Extractor")

st.markdown(
    """
Upload the **full billing report PDF** exported from your BI.
The app will read all products and output clean **CSV** and **XLSX** files with:

- Product code and description
- Month and year
- Quantity and value (Brazilian formatting converted to float)
- **Client code** and **client name**
- **Sales representative code** and **sales representative name**
- **Client State and City** (joined from `clientes_relatorio_faturamento.csv`)
- **Categoria** (using the official Performance Moveleiro mapping)
    """
)

uploaded_file = st.file_uploader("📤 Choose the billing PDF file", type="pdf")

# -------------------------------------------------------
# SETTINGS
# -------------------------------------------------------
# Set to 1 to test only the first page; set to None to process the full PDF.
MAX_PAGES = None

# -------------------------------------------------------
# UTILS
# -------------------------------------------------------
def br_to_float(s: str) -> float:
    """Converts Brazilian number formatting to float.
    Example: '50,0000' -> 50.0 ; '1.234,56' -> 1234.56
    """
    return float(s.strip().replace(".", "").replace(",", "."))


# -------------------------------------------------------
# LOAD CATEGORY MAP CSV (no caching to always pick new rules)
# -------------------------------------------------------
def load_category_map():
    df = pd.read_csv("data/categorias_map.csv")
    df["pattern"] = df["pattern"].astype(str).str.upper().str.strip()
    df["categoria"] = df["categoria"].astype(str).str.strip()
    df["prioridade"] = df["prioridade"].astype(int)
    df = df.sort_values("prioridade")
    return df


CATEGORY_MAP = load_category_map()


# -------------------------------------------------------
# LOAD CLIENT GEO MAP (Estado / Cidade)
# -------------------------------------------------------
@st.cache_data
def load_client_geo_map():
    """
    Loads clientes_relatorio_faturamento.csv and returns a
    simplified mapping: ClienteCodigo -> EstadoNome, CidadeNome.
    Auto-detects delimiter.
    """
    df = pd.read_csv("data/clientes_relatorio_faturamento.csv", sep=None, engine="python")
    df["ClienteCodigo"] = df["ClienteCodigo"].astype(str).str.strip()
    df["EstadoNome"] = df["EstadoNome"].astype(str).str.strip()
    df["CidadeNome"] = df["CidadeNome"].astype(str).str.strip()
    df = df[["ClienteCodigo", "EstadoNome", "CidadeNome"]].drop_duplicates(
        subset=["ClienteCodigo"]
    )
    return df


CLIENT_GEO_MAP = load_client_geo_map()


# -------------------------------------------------------
# CATEGORY ENGINE
# -------------------------------------------------------
def map_categoria(desc: str) -> str:
    text = (str(desc) or "").upper().strip()
    for _, row in CATEGORY_MAP.iterrows():
        if row["pattern"] in text:
            return row["categoria"]
    return "Outros"


# -------------------------------------------------------
# REGEX DEFINITIONS FOR PDF PARSING
# -------------------------------------------------------
# PRODUTO: 9587-DOBRADICA 45° ...
prod_header_re = re.compile(
    r"^\s*PRODUTO:\s*(\d+)\s*-\s*(.+)$",
    re.IGNORECASE,
)

# MÊS: 10/2025-Outubro de 2025
mes_re = re.compile(
    r"^\s*MÊS\s*:\s*(\d{2})/(\d{4})",
    re.IGNORECASE,
)

# REPRESENTANTE: 4593-OPENFIELD BENTO
rep_re = re.compile(
    r"^\s*REPRESENTANTE:\s*(\d+)\s*-\s*(.+)$",
    re.IGNORECASE,
)

# CLIENTE : 8819 - AMBIENTAL MOVEIS ... 50,0000 100,00% 275,50 100,00%
cliente_line_re = re.compile(
    r"^\s*CLIENTE\s*:\s*(\d+)\s*-\s*(.+?)\s+([\d\.\,]+)\s+[\d\.\,]+%\s+([\d\.\,]+)\s+[\d\.\,]+%\s*$",
    re.IGNORECASE,
)


# -------------------------------------------------------
# PROCESS PDF
# -------------------------------------------------------
if uploaded_file is not None:
    records = []

    with pdfplumber.open(uploaded_file) as pdf:
        total_pages = len(pdf.pages)
        pages_to_process = (
            min(total_pages, MAX_PAGES) if MAX_PAGES is not None else total_pages
        )

        st.info(
            f"📄 PDF loaded with **{total_pages} pages**. "
            f"Processing **{pages_to_process} page(s)** for this run."
        )

        progress_bar = st.progress(0)
        status_text = st.empty()

        # Context variables
        current_code = None
        current_desc = None
        current_mes = None
        current_ano = None
        rep_code = None
        rep_name = None

        for i, page in enumerate(pdf.pages, start=1):
            if i > pages_to_process:
                break

            status_text.text(f"🔍 Reading page {i}/{pages_to_process}...")

            text = page.extract_text() or ""
            for raw in text.splitlines():
                line = raw.strip()
                if not line:
                    continue

                # 1) PRODUCT HEADER (reset month and rep)
                m = prod_header_re.match(line)
                if m:
                    current_code = m.group(1).strip()
                    current_desc = m.group(2).strip()
                    current_mes = None
                    current_ano = None
                    rep_code = None
                    rep_name = None
                    continue

                # 2) MONTH / YEAR (reset rep)
                m = mes_re.match(line)
                if m:
                    current_mes, current_ano = m.groups()
                    rep_code = None
                    rep_name = None
                    continue

                # 3) REPRESENTATIVE
                m = rep_re.match(line)
                if m:
                    rep_code, rep_name = m.groups()
                    rep_code = rep_code.strip()
                    rep_name = rep_name.strip()
                    continue

                # 4) CLIENT LINE WITH QTY + VALUE
                m = cliente_line_re.match(line)
                if m:
                    cli_code, cli_name, qty_str, val_str = m.groups()

                    if (
                        current_code
                        and current_desc is not None
                        and current_mes
                        and current_ano
                        and rep_code
                        and rep_name
                    ):
                        try:
                            record = {
                                "Codigo": current_code,
                                "Descricao": current_desc,
                                "Quantidade": br_to_float(qty_str),
                                "Valor": br_to_float(val_str),
                                "Mes": int(current_mes),
                                "Ano": int(current_ano),
                                # client code + name separated
                                "ClienteCodigo": cli_code.strip(),
                                "Cliente": cli_name.strip(),  # name only
                                # rep code + name separated
                                "RepresentanteCodigo": rep_code,
                                "Representante": rep_name,   # name only
                            }
                            records.append(record)
                        except Exception as e:
                            print(f"Error parsing line: {line} -> {e}")
                    continue

            progress_bar.progress(i / pages_to_process)
            time.sleep(0.01)

        status_text.text("📘 PDF scan finished — building DataFrame...")

    # -------------------------------------------------------
    # CREATE DATAFRAME
    # -------------------------------------------------------
    if not records:
        st.error(
            "No data rows were found. "
            "The PDF layout may have changed or the parsing patterns need to be adjusted."
        )
    else:
        df = pd.DataFrame(records)

        # ---------------------------
        # ENRICH WITH ESTADO / CIDADE
        # ---------------------------
        df["ClienteCodigo"] = df["ClienteCodigo"].astype(str).str.strip()

        df = df.merge(
            CLIENT_GEO_MAP,
            on="ClienteCodigo",
            how="left",
        )

        df.rename(
            columns={
                "EstadoNome": "Estado",
                "CidadeNome": "Cidade",
            },
            inplace=True,
        )

        # ---------------------------
        # APPLY CATEGORY LOGIC
        # ---------------------------
        df["Categoria"] = df["Descricao"].apply(map_categoria)

        # ---------------------------
        # FINAL COLUMN ORDER
        # ---------------------------
        df = df[
            [
                "Codigo",
                "Descricao",
                "Quantidade",
                "Valor",
                "Mes",
                "Ano",
                "ClienteCodigo",
                "Cliente",              # client name
                "Estado",
                "Cidade",
                "RepresentanteCodigo",  # rep code
                "Representante",        # rep name
                "Categoria",
            ]
        ]

        st.success(
            f"✅ Extraction finished — {len(df)} rows "
             f"({df['Codigo'].nunique()} unique products)."
        )
        st.dataframe(df.head(50))

        # -------------------------------------------------------
        # EXPORT CSV
        # -------------------------------------------------------
        csv_data = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇️ Download CSV",
            csv_data,
            "relatorio_faturamento.csv",
            "text/csv",
        )

        # -------------------------------------------------------
        # EXPORT XLSX
        # -------------------------------------------------------
        xlsx_io = BytesIO()
        with pd.ExcelWriter(xlsx_io, engine="openpyxl") as writer:
            df.to_excel(writer, index=False)

        st.download_button(
            "⬇️ Download XLSX",
            xlsx_io.getvalue(),
            "relatorio_faturamento.xlsx",
            "application/vnd.ms-excel",
        )

        st.info(
            "📊 Files are ready (including **ClienteCodigo**, **Cliente**, "
            "**Estado**, **Cidade**, **RepresentanteCodigo**, **Representante** "
            "and **Categoria**)."
        )
