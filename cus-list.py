import streamlit as st
import pdfplumber
import pandas as pd
import re
from io import BytesIO
import time

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Client Base Extractor",
    page_icon="📇",
    layout="wide"
)

st.title("📇 Client List Extractor (State / City / Client)")

st.markdown("""
Upload the **client list PDF** (billing report grouped by **Estado → Cidade → Cliente**)  
and this app will build a clean client base with:

- State code + name
- City code + name
- Client code + name
- Quantity and Billing (Valor) from the report

The output CSV is meant to be saved in:

`performance-moveleiro-v2/data/clientes_relatorio_faturamento.csv`
""")

# -----------------------------
# Helpers
# -----------------------------

# Regex for CLIENTE line
CLIENTE_PATTERN = re.compile(
    r'^CLIENTE\s*:\s*(-?\d+)\s*-\s*(.+?)\s+(-?[\d\.]+,\d{4})\s+'
    r'(-?\d{1,3},\d{2})%\s+(-?[\d\.]+,\d{2})\s+(-?\d{1,3},\d{2})%$'
)

def parse_br_number(s: str):
    """Convert Brazilian-formatted number to float (e.g. '75.940,0000' -> 75940.0)."""
    if s is None:
        return None
    s = s.strip().replace('.', '').replace('%', '')
    if not s:
        return None
    s = s.replace(',', '.')
    try:
        return float(s)
    except ValueError:
        return None

def extract_clientes_from_pdf(file) -> pd.DataFrame:
    """
    Parse the PDF in the format you provided and extract:
    Estado, Cidade, Cliente, Quantidade, %Quantidade, Valor, %Valor.
    """
    rows = []

    with pdfplumber.open(file) as pdf:
        total_pages = len(pdf.pages)

        estado_cod = None
        estado_nome = None
        cidade_cod = None
        cidade_nome = None

        progress = st.progress(0, text=f"Processing 0/{total_pages} pages...")

        for page_index, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            for raw_line in text.splitlines():
                line = raw_line.strip()
                if not line:
                    continue

                # Skip general headers
                if line == "RELATÓRIO DE FATURAMENTO":
                    continue
                if line.startswith("Quantidade % Quantidade Valor % Valor"):
                    continue
                if (
                    line.startswith("Subtotal CIDADE")
                    or line.startswith("Subtotal ESTADO")
                    or line.startswith("Total Geral")
                ):
                    continue

                # ESTADO line
                if line.startswith("ESTADO:"):
                    rest = line[len("ESTADO:"):].strip()
                    if "Subtotal" in rest:
                        rest = rest.split("Subtotal")[0].strip()
                    if "-" in rest:
                        estado_cod_str, estado_nome = rest.split("-", 1)
                        estado_cod = estado_cod_str.strip()
                    else:
                        estado_cod = None
                        estado_nome = rest
                    continue

                # CIDADE line
                if line.startswith("CIDADE:"):
                    rest = line[len("CIDADE:"):].strip()
                    rest = rest.split(" Quantidade")[0].strip()
                    if "-" in rest:
                        cidade_cod_str, cidade_nome = rest.split("-", 1)
                        cidade_cod = cidade_cod_str.strip()
                    else:
                        cidade_cod = None
                        cidade_nome = rest
                    continue

                # CLIENTE line
                if line.startswith("CLIENTE"):
                    m = CLIENTE_PATTERN.match(line)
                    if not m:
                        # If pattern doesn't match, skip this line silently
                        continue

                    (
                        cliente_cod,
                        cliente_nome,
                        qty_str,
                        pct_qty_str,
                        val_str,
                        pct_val_str,
                    ) = m.groups()

                    rows.append({
                        "EstadoCodigo": estado_cod,
                        "EstadoNome": estado_nome,
                        "CidadeCodigo": cidade_cod,
                        "CidadeNome": cidade_nome,
                        "ClienteCodigo": cliente_cod,
                        "ClienteNome": cliente_nome.strip(),
                        "Quantidade": parse_br_number(qty_str),
                        "PercQuantidade": parse_br_number(pct_qty_str),
                        "Valor": parse_br_number(val_str),
                        "PercValor": parse_br_number(pct_val_str),
                    })

            progress.progress(
                page_index / total_pages,
                text=f"Processing {page_index}/{total_pages} pages..."
            )

    df = pd.DataFrame(rows)

    # Sort exactly like the report: Estado > Cidade > Cliente
    sort_cols = ["EstadoNome", "CidadeNome", "ClienteNome"]
    existing_sort_cols = [c for c in sort_cols if c in df.columns]
    if existing_sort_cols:
        df = df.sort_values(existing_sort_cols).reset_index(drop=True)

    return df

def make_download_links(df: pd.DataFrame):
    """Create CSV and (optionally) XLSX buffers for download."""
    # CSV (UTF-8 with BOM for Excel PT-BR)
    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")

    xlsx_buffer = None
    try:
        # Try to create an XLSX file using xlsxwriter
        xlsx_buffer = BytesIO()
        with pd.ExcelWriter(xlsx_buffer, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name="Clientes")
        xlsx_buffer.seek(0)
    except Exception:
        # If xlsxwriter is not installed, we simply skip XLSX export
        xlsx_buffer = None

    return csv_bytes, xlsx_buffer


# -----------------------------
# Main UI
# -----------------------------
uploaded_file = st.file_uploader(
    "📎 Upload the client list PDF (grouped by Estado/Cidade/Cliente)",
    type=["pdf"]
)

if uploaded_file is not None:
    st.info("File uploaded. Click **Extract data** to process the PDF.")

    if st.button("📤 Extract data"):
        start = time.time()
        try:
            with st.spinner("Reading PDF and extracting client base..."):
                df_clientes = extract_clientes_from_pdf(uploaded_file)

            # Drop percentage columns from the exported dataset
            export_df = df_clientes.drop(
                columns=["PercQuantidade", "PercValor"], errors="ignore"
            )

            elapsed = time.time() - start

            st.success(
                f"Extraction completed! "
                f"Found **{len(export_df)} client records** "
                f"in **{export_df['EstadoNome'].nunique()} states** and "
                f"**{export_df['CidadeNome'].nunique()} cities** "
                f"(time: {elapsed:.1f}s)."
            )

            st.subheader("Preview of extracted data")
            st.dataframe(export_df.head(200))

            csv_bytes, xlsx_buffer = make_download_links(export_df)

            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "⬇️ Download CSV (clientes_relatorio_faturamento.csv)",
                    data=csv_bytes,
                    file_name="clientes_relatorio_faturamento.csv",
                    mime="text/csv",
                )
            with col2:
                if xlsx_buffer is not None:
                    st.download_button(
                        "⬇️ Download XLSX (clientes_relatorio_faturamento.xlsx)",
                        data=xlsx_buffer,
                        file_name="clientes_relatorio_faturamento.xlsx",
                        mime=(
                            "application/vnd.openxmlformats-officedocument."
                            "spreadsheetml.sheet"
                        ),
                    )
                else:
                    st.info(
                        "XLSX export not available. "
                        "To enable it, install the `xlsxwriter` package:\n\n"
                        "`pip install xlsxwriter`"
                    )

            st.markdown("""
> After downloading, place the CSV in  
> `performance-moveleiro-v2/data/clientes_relatorio_faturamento.csv`  
> so your other dashboards/apps can consume it.
            """)

        except Exception as e:
            st.error(f"Error processing PDF: {e}")

else:
    st.warning("No file uploaded yet. Please upload the PDF to begin.")
