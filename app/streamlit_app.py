import os
import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path

DATA_DB = Path(os.getenv("DATA_DIR", "./data")).resolve() / "warehouse.duckdb"

st.set_page_config(page_title="Kaizen Poke — ETL & Análises", layout="wide")
st.title("Kaizen Poke — ETL & Análises")

@st.cache_data(show_spinner=False)
def load_table(name: str) -> pd.DataFrame:
    con = duckdb.connect(DATA_DB)
    try:
        df = con.execute(f"SELECT * FROM {name}").df()
    finally:
        con.close()
    return df

tabs = st.tabs(["Visão Geral", "Taxa de vitória por tipo", "Importância de atributos", "Explorar dados"])

with tabs[0]:
    st.markdown("### Tabelas disponíveis")
    con = duckdb.connect(DATA_DB)
    tables = con.execute("SHOW TABLES").df()
    con.close()
    st.dataframe(tables)

with tabs[1]:
    st.markdown("### Taxa de vitória por tipo")
    # Exemplo simples: ajuste a query conforme sua tabela
    con = duckdb.connect(DATA_DB)
    df = con.execute("""
        SELECT winner_type AS type, COUNT(*) AS wins
        FROM battles
        GROUP BY 1 ORDER BY wins DESC
    """).df()
    con.close()
    if len(df):
        fig = px.bar(df, x="type", y="wins")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df)
    else:
        st.info("Sem dados. Rode o ETL primeiro: `python -m src.etl`.")

with tabs[2]:
    st.markdown("### Importância de atributos / performance de modelos")
    st.write("Use o script `src/analyses.py` para gerar uma tabela `battles_flat` e métricas; em seguida, exponha aqui.")

with tabs[3]:
    st.markdown("### Explorar dados brutos")
    table = st.selectbox("Tabela", options=["pokemons", "battles", "teams"], index=0)
    try:
        df = load_table(table)
        st.dataframe(df.head(100))
    except Exception as e:
        st.warning(f"Falha ao carregar {table}: {e}")
