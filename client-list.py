import streamlit as st
import pdfplumber
import pandas as pd
import re
from io import BytesIO
import time

# -----------------------------
# Configuração básica da página
# -----------------------------
st.set_page_config(
    page_title="Base de Clientes - Extrator",
    page_icon="📇",
    layout="wide"
)

st.title("📇 Extrator de Base de Clientes (Estado / Cidade / Cliente)")

st.markdown("""
Envie o **PDF da lista de clientes** (relatório de faturamento por cliente, 
agrupado por **Estado → Cidade → Cliente**) e o sistema vai extrair uma base 
estruturada com:

- Código e nome do **Estado**
- Código e nome da **Cidade**
- Código e nome do **Cliente**
- Quantidade e Faturamento deste relatório (opcional para uso em dashboards)

No final você poderá baixar os dados em **CSV** e **XLSX**.  
Sugestão de nome de arquivo para o repositório:  
`data/clientes_relatorio_faturamento.csv`
""")

# -----------------------------
# Funções auxiliares
# -----------------------------

# Regex para linhas de cliente (seguindo o padrão do PDF enviado)
CLIENTE_PATTERN = re.compile(
    r'^CLIENTE\s*:\s*(-?\d+)\s*-\s*(.+?)\s+(-?[\d\.]+,\d{4})\s+'
    r'(-?\d{1,3},\d{2})%\s+(-?[\d\.]+,\d{2})\s+(-?\d{1,3},\d{2})%$'
)

def parse_br_number(s: str):
    """Converte número no formato brasileiro para float (ex: '75.940,0000' -> 75940.0)."""
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
    Lê o PDF no formato do relatório enviado e extrai:
    Estado, Cidade, Cliente, Quantidade, %Quantidade, Valor, %Valor.
    """
    rows = []

    with pdfplumber.open(file) as pdf:
        total_pages = len(pdf.pages)

        estado_cod = None
        estado_nome = None
        cidade_cod = None
        cidade_nome = None

        progress = st.progress(0, text=f"Processando 0/{total_pages} páginas...")

        for page_index, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            for raw_line in text.splitlines():
                line = raw_line.strip()
                if not line:
                    continue

                # Ignorar cabeçalhos gerais
                if line == "RELATÓRIO DE FATURAMENTO":
                    continue
                if line.startswith("Quantidade % Quantidade Valor % Valor"):
                    continue
                if line.startswith("Subtotal CIDADE") or line.startswith("Subtotal ESTADO") or line.startswith("Total Geral"):
                    continue

                # Linha de ESTADO
                if line.startswith("ESTADO:"):
                    rest = line[len("ESTADO:"):].strip()
                    # Em geral é algo como "23-RIO DE JANEIRO"
                    if "Subtotal" in rest:
                        rest = rest.split("Subtotal")[0].strip()
                    if "-" in rest:
                        estado_cod_str, estado_nome = rest.split("-", 1)
                        estado_cod = estado_cod_str.strip()
                    else:
                        estado_cod = None
                        estado_nome = rest
                    continue

                # Linha de CIDADE
                if line.startswith("CIDADE:"):
                    rest = line[len("CIDADE:"):].strip()
                    # Às vezes vem com "Quantidade % Quantidade Valor % Valor" no final
                    rest = rest.split(" Quantidade")[0].strip()
                    if "-" in rest:
                        cidade_cod_str, cidade_nome = rest.split("-", 1)
                        cidade_cod = cidade_cod_str.strip()
                    else:
                        cidade_cod = None
                        cidade_nome = rest
                    continue

                # Linha de CLIENTE
                if line.startswith("CLIENTE"):
                    m = CLIENTE_PATTERN.match(line)
                    if not m:
                        # Se algum padrão fugir da regra, simplesmente pula.
                        # (Podemos logar aqui se você quiser debugar depois)
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
                text=f"Processando {page_index}/{total_pages} páginas..."
            )

    df = pd.DataFrame(rows)

    # Ordena exatamente na lógica do relatório: Estado > Cidade > Cliente
    sort_cols = ["EstadoNome", "CidadeNome", "ClienteNome"]
    existing_sort_cols = [c for c in sort_cols if c in df.columns]
    if existing_sort_cols:
        df = df.sort_values(existing_sort_cols).reset_index(drop=True)

    return df

def make_download_links(df: pd.DataFrame):
    """Cria buffers de CSV e XLSX para download."""
    # CSV (UTF-8 com BOM para Excel PT-BR)
    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")

    # XLSX
    xlsx_buffer = BytesIO()
    with pd.ExcelWriter(xlsx_buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Clientes")
    xlsx_buffer.seek(0)

    return csv_bytes, xlsx_buffer


# -----------------------------
# Interface principal
# -----------------------------
uploaded_file = st.file_uploader(
    "📎 Envie o PDF da lista de clientes (relatório por Estado/Cidade/Cliente)",
    type=["pdf"]
)

if uploaded_file is not None:
    st.info("Arquivo carregado. Clique em **Extrair dados** para processar o PDF.")

    if st.button("📤 Extrair dados"):
        start = time.time()
        try:
            with st.spinner("Lendo PDF e extraindo base de clientes..."):
                df_clientes = extract_clientes_from_pdf(uploaded_file)

            elapsed = time.time() - start

            st.success(
                f"Extração concluída com sucesso! "
                f"Foram encontrados **{len(df_clientes)} registros de clientes** "
                f"em **{df_clientes['EstadoNome'].nunique()} estados** e "
                f"**{df_clientes['CidadeNome'].nunique()} cidades** "
                f"(tempo: {elapsed:.1f}s)."
            )

            st.subheader("Prévia dos dados extraídos")
            st.dataframe(df_clientes.head(200))

            csv_bytes, xlsx_buffer = make_download_links(df_clientes)

            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "⬇️ Baixar CSV (clientes_relatorio_faturamento.csv)",
                    data=csv_bytes,
                    file_name="clientes_relatorio_faturamento.csv",
                    mime="text/csv",
                )
            with col2:
                st.download_button(
                    "⬇️ Baixar XLSX (clientes_relatorio_faturamento.xlsx)",
                    data=xlsx_buffer,
                    file_name="clientes_relatorio_faturamento.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

            st.markdown("""
            > 🔁 Depois de baixar, coloque o arquivo em  
            > `performance-moveleiro-v2/data/clientes_relatorio_faturamento.csv`  
            > para ser consumido pelos outros dashboards/apps.
            """)

        except Exception as e:
            st.error(f"Erro ao processar o PDF: {e}")

else:
    st.warning("Nenhum arquivo foi enviado ainda. Faça o upload do PDF para começar.")
