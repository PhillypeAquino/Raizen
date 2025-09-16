import duckdb
import pandas as pd
from pathlib import Path
from .config import settings

DB_PATH = settings.data_dir / "warehouse.duckdb"

def build_battle_features() -> pd.DataFrame:
    con = duckdb.connect(DB_PATH)
    # Ajuste estas queries conforme os nomes/colunas reais das tabelas
    # Exemplo: supõe tabela 'battles' com colunas: battle_id, winner_id, loser_id, winner_team_type, loser_team_type, etc.
    q = """
    SELECT *
    FROM battles
    """
    df = con.execute(q).df()
    con.close()
    # Engenharia simples: target binário 'win' e features numéricas/categóricas
    # Se a linha já estiver no formato por-plano (uma linha por combatente), adapte aqui.
    return df
