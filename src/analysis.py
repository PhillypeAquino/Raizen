import duckdb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from .config import settings

def winrate_by_type() -> pd.DataFrame:
    con = duckdb.connect(settings.data_dir / "warehouse.duckdb")
    # Exemplo: contar vitórias por 'type' (ajuste nomes)
    q = """
    WITH W AS (
      SELECT winner_type AS type FROM battles
      UNION ALL
      SELECT loser_type  AS type FROM battles WHERE 1=0 -- placeholder caso precise
    )
    SELECT type, COUNT(*) AS wins
    FROM (
      SELECT winner_type AS type FROM battles
    )
    GROUP BY 1
    ORDER BY wins DESC
    """
    df = con.execute(q).df()
    con.close()
    return df

def feature_importance() -> pd.DataFrame:
    con = duckdb.connect(settings.data_dir / "warehouse.duckdb")
    # Suponha tabela 'battles_flat' com target 'won' e atributos do pokémon/time
    # Você pode construir 'battles_flat' via uma VIEW no DuckDB ou no features.py
    q = "SELECT * FROM battles_flat"
    try:
        df = con.execute(q).df()
    finally:
        con.close()

    target = "won"
    y = df[target].astype(int)
    X = df.drop(columns=[target])
    cat_cols = [c for c in X.columns if X[c].dtype == 'object']
    num_cols = [c for c in X.columns if c not in cat_cols]

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols)
    ])

    # dois modelos para demonstrar pensamento crítico
    models = {
        "logreg": LogisticRegression(max_iter=200, n_jobs=None),
        "rf": RandomForestClassifier(n_estimators=200, random_state=42)
    }

    out_rows = []
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    for name, mdl in models.items():
        pipe = Pipeline([("pre", pre), ("clf", mdl)])
        pipe.fit(Xtr, ytr)
        proba = pipe.predict_proba(Xte)[:, 1]
        auc = roc_auc_score(yte, proba)
        out_rows.append({"model": name, "roc_auc": float(auc)})

    return pd.DataFrame(out_rows)
