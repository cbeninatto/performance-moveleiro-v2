import streamlit as st
import pdfplumber
import pandas as pd
import re
from io import BytesIO
import time

# -------------------------------------------------------
# STREAMLIT UI CONFIG
# -------------------------------------------------------
st.set_page_config(page_title="Relatório de Faturamento Extractor", page_icon="📄")
st.title("📄 Extrator de Relatório de Faturamento")

st.markdown(
    """
Envie o PDF completo do relatório de faturamento e o sistema extrairá automaticamente os dados para **CSV** e **XLSX**, incluindo a **categoria de cada produto** com a lógica oficial do Performance Moveleiro e as colunas **Cliente** e **Representante**.

Upload your complete billing report PDF below — the system extracts and classifies products into clean **CSV** and **XLSX** files, now including **Client** and **Sales Rep**.
"""
)

uploaded_file = st.file_uploader("📤 Escolha o arquivo PDF", type="pdf")


# -------------------------------------------------------
# UTILS
# -------------------------------------------------------
def br_to_float(s: str) -> float:
    """
    Converts Brazilian number formatting to float.
    Example: '1.234,56' -> 1234.56
    """
    return float(s.strip().replace(".", "").replace(",", "."))


# -------------------------------------------------------
# 📌 LOAD CATEGORY MAP CSV
# -------------------------------------------------------
@st.cache_data
def load_category_map():
    df = pd.read_csv("data/categorias_map.csv")
    df["pattern"] = df["pattern"].astype(str).str.upper()
    df["categoria"] = df["categoria"].astype(str)
    df["prioridade"] = df["prioridade"].astype(int)
    df = df.sort_values("prioridade")
    return df


CATEGORY_MAP = load_category_map()


# -------------------------------------------------------
# 🧠 OFFICIAL PERFORMANCE MOVELEIRO CATEGORY ENGINE
# -------------------------------------------------------
def map_categoria(desc: str) -> str:
    text = (str(desc) or "").upper()
    for _, row in CATEGORY_MAP.iterrows():
        if row["pattern"] in text:
            return row["categoria"]
    return "Outros"


# -------------------------------------------------------
# REGEX DEFINITIONS FOR PDF PARSING
# -------------------------------------------------------
# Example product header:
# "PRODUTO: 9587 - DOBRADICA 45° COM PISTAO ..."
prod_header_re = re.compile(r"^\s*PRODUTO:\s*(\d+)\s*-\s*(.+?)\s*$", re.IGNORECASE)

# Remove duplicated "Quantidade % Quantidade Valor % Valor" clutter from the description
cleanup_re = re.compile(r"\s*Quantidade\s*%\s*Quantidade\s*Valor\s*%\s*Valor\s*$", re.IGNORECASE)

# Example month line:
# "MÊS: 10/2025      50,0   100,0%   275,50   100,0%"
mes_line_re = re.compile(
    r"^\s*MÊS\s*:\s*(\d{2})/(\d{4}).*?\s([\d\.\,]+)\s+[\d\.\,]+%\s+([\d\.\,]+)\s+[\d\.\,]+%",
    re.IGNORECASE,
)

# New: CLIENTE and REPRESENTANTE headers
# We assume lines like:
# "CLIENTE: ALGUM NOME DE CLIENTE"
# "REPRESENTANTE: NOME DO REPRESENTANTE"
cliente_re = re.compile(r"^\s*CLIENTE\s*:\s*(.+?)\s*$", re.IGNORECASE)
representante_re = re.compile(r"^\s*REPRESENTANTE\s*:\s*(.+?)\s*$", re.IGNORECASE)


# -------------------------------------------------------
# 📘 PROCESS PDF
# -------------------------------------------------------
if uploaded_file:

    records = []

    current_code = None
    current_desc = None

    # These will be updated whenever we find CLIENTE / REPRESENTANTE lines
    current_cliente = None
    current_representante = None

    with pdfplumber.open(uploaded_file) as pdf:
        total_pages = len(pdf.pages)
        st.info(f"📄 PDF carregado com **{total_pages} páginas**.")

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, page in enumerate(pdf.pages, start=1):
            status_text.text(f"🔍 Lendo página {i}/{total_pages}...")

            text = page.extract_text() or ""
            for raw in text.splitlines():
                line = raw.strip()

                if not line:
                    continue

                # Ignore some known noise lines
                if "Subtotal PRODUTO" in line or "www.kunden.com.br" in line:
                    continue

                # -------------------------------
                # Detect CLIENTE line
                # -------------------------------
                m_cliente = cliente_re.match(line)
                if m_cliente:
                    current_cliente = m_cliente.group(1).strip()
                    # Move to next line
                    continue

                # -------------------------------
                # Detect REPRESENTANTE line
                # -------------------------------
                m_rep = representante_re.match(line)
                if m_rep:
                    current_representante = m_rep.group(1).strip()
                    # Move to next line
                    continue

                # -------------------------------
                # Detect product header
                # -------------------------------
                if line.upper().startswith("PRODUTO:"):
                    m_prod = prod_header_re.match(line)
                    if m_prod:
                        current_code = m_prod.group(1).strip()
                        desc_raw = m_prod.group(2)
                        desc_clean = cleanup_re.sub("", desc_raw).strip(" -")
                        current_desc = desc_clean
                    continue

                # -------------------------------
                # Detect month line (Mes/Ano + Qty + Value)
                # -------------------------------
                m_mes = mes_line_re.match(line)
                if m_mes and current_code:
                    mes, ano, qty, val = m_mes.groups()
                    try:
                        record = {
                            "Codigo": current_code,
                            "Descricao": current_desc,
                            "Quantidade": br_to_float(qty),
                            "Valor": br_to_float(val),
                            "Mes": int(mes),
                            "Ano": int(ano),
                            "Cliente": current_cliente,
                            "Representante": current_representante,
                        }
                        records.append(record)
                    except Exception:
                        # If a single line fails parsing, we just skip it.
                        # You can log this to a debug area if needed.
                        pass

            progress_bar.progress(i / total_pages)
            time.sleep(0.02)

        status_text.text("📘 Leitura concluída — processando dados...")


    # -------------------------------------------------------
    # CREATE DATAFRAME
    # -------------------------------------------------------
    if not records:
        st.error(
            "Nenhum dado encontrado. O PDF pode estar em formato inesperado "
            "ou as expressões regulares precisam ser ajustadas para o novo layout."
        )

    else:
        df = pd.DataFrame(records)

        # Ensure all expected columns exist even if some values are missing
        expected_cols = [
            "Codigo",
            "Descricao",
            "Quantidade",
            "Valor",
            "Mes",
            "Ano",
            "Cliente",
            "Representante",
        ]
        for col in expected_cols:
            if col not in df.columns:
                df[col] = None

        # 🔥 APPLY OFFICIAL CATEGORY LOGIC
        df["Categoria"] = df["Descricao"].apply(map_categoria)

        # Reorder columns to the official schema
        df = df[
            [
                "Codigo",
                "Descricao",
                "Quantidade",
                "Valor",
                "Mes",
                "Ano",
                "Cliente",
                "Representante",
                "Categoria",
            ]
        ]

        st.success(
            f"✅ Extração concluída — {len(df)} linhas "
            f"({df['Codigo'].nunique()} produtos)."
        )
        st.dataframe(df.head(50))

        # -------------------------------------------------------
        # EXPORT CSV
        # -------------------------------------------------------
        csv_data = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇️ Baixar CSV",
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
            "⬇️ Baixar XLSX",
            xlsx_io.getvalue(),
            "relatorio_faturamento.xlsx",
            "application/vnd.ms-excel",
        )

        st.info(
            "📊 Arquivos prontos para download (incluindo colunas "
            "**Cliente**, **Representante** e **Categoria**)."
        )
