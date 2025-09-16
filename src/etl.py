import json
from pathlib import Path
import pandas as pd
import duckdb
from tqdm import tqdm
from config import settings
from api_client import APIClient
print('test')
RAW = settings.data_dir / "raw"
PROC = settings.data_dir / "processed"
DB_PATH = settings.data_dir / "warehouse.duckdb"

# edite esta lista conforme os endpoints GET que você vir no /docs
DEFAULT_ENDPOINTS = [
    "/pokemons",
    "/battles",
    "/teams",
    "/matchups",
]

def dump_jsonl(path: Path, items):
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

def jsonl_to_df(path: Path) -> pd.DataFrame:
    rows = []
    with path.open() as f:
        for line in f:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    # limpa colunas aninhadas (explode dicts/listas)
    df = pd.json_normalize(df.to_dict(orient="records"))
    # tenta converter campos data/hora
    for c in df.columns:
        if any(k in c.lower() for k in ["created", "updated", "date", "time", "timestamp"]):
            with pd.option_context('mode.chained_assignment', None):
                try:
                    df[c] = pd.to_datetime(df[c], errors="ignore")
                except Exception:
                    pass
    return df

def run_extract(client: APIClient, endpoints=DEFAULT_ENDPOINTS):
    extracted = []
    for ep in endpoints:
        try:
            items = list(client.paginate_all(ep))
            if not items:
                # tenta simples GET sem paginação
                data = client.get(ep)
                items = data if isinstance(data, list) else [data]
            out = RAW / f"{ep.strip('/').replace('/', '_')}.jsonl"
            out.parent.mkdir(parents=True, exist_ok=True)
            dump_jsonl(out, items)
            extracted.append((ep, out, len(items)))
        except Exception as e:
            print(f"[WARN] Falha ao extrair {ep}: {e}")
    return extracted

def run_transform_load():
    con = duckdb.connect(DB_PATH)
    for jf in RAW.glob("*.jsonl"):
        name = jf.stem  # pokemons.jsonl -> pokemons
        df = jsonl_to_df(jf)
        df = normalize_df(df)
        pq = PROC / f"{name}.parquet"
        df.to_parquet(pq, index=False)
        con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM read_parquet('{pq.as_posix()}');")
        print(f"[LOAD] {name}: {len(df):,} linhas -> {pq.name}")
    con.close()

def main():
    client = APIClient(settings.api_base, settings.username, settings.password)
    print("[AUTH] autenticando...")
    token = client.authenticate()
    print(f"[AUTH] ok. token (prefixo): {token[:16]}...")
    # descobre endpoints GET (opcional) e mescla com DEFAULT_ENDPOINTS
    discovered = client.list_get_endpoints()
    # heurística: pega endpoints óbvios de interesse
    cand = [p for p in discovered if any(k in p.lower() for k in ["pokemon", "battle", "team", "match"])]
    endpoints = sorted(set(DEFAULT_ENDPOINTS + cand))
    print("[DISCOVER] GET endpoints:", endpoints)
    print("[EXTRACT] iniciando...")
    stats = run_extract(client, endpoints=endpoints)
    for ep, out, n in stats:
        print(f"  - {ep}: {n} registros -> {out.name}")
    print("[TRANSFORM/LOAD] ...")
    run_transform_load()
    print("[DONE] ETL concluído. DuckDB:", DB_PATH)

if __name__ == "__main__":
    main()
