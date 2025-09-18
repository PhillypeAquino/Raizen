import os
import json
import time
import base64
from pathlib import Path

import httpx
import backoff
import pandas as pd
from dotenv import load_dotenv

# ========= Config =========
load_dotenv()
BASE_URL = os.getenv("BASE_URL", "").rstrip("/")
USER = os.getenv("API_USERNAME")
PASS = os.getenv("API_PASSWORD")

RAW_DIR = Path("data/raw"); RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR = Path("data/processed"); PROC_DIR.mkdir(parents=True, exist_ok=True)
TOKEN_CACHE = Path(".token_cache.json")

# ========= Token cache (JWT) =========
def _b64url_decode(s: str) -> bytes:
    s += "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s.encode())

def _jwt_exp(token: str) -> int | None:
    try:
        h, p, s = token.split(".")
        payload = json.loads(_b64url_decode(p).decode())
        return int(payload.get("exp")) if "exp" in payload else None
    except Exception:
        return None

def load_cached_token() -> str | None:
    if not TOKEN_CACHE.exists():
        return None
    try:
        data = json.loads(TOKEN_CACHE.read_text(encoding="utf-8"))
        token = data.get("access_token")
        exp = data.get("exp") or _jwt_exp(token)
        if token and exp and time.time() < (exp - 30):
            return token
    except Exception:
        pass
    return None

def save_cached_token(token: str):
    TOKEN_CACHE.write_text(
        json.dumps({"access_token": token, "exp": _jwt_exp(token)}, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

# ========= API client =========
TIMEOUT = httpx.Timeout(connect=5.0, read=20.0, write=10.0, pool=5.0)

class API:
    def __init__(self, base_url: str):
        if not base_url:
            raise ValueError("BASE_URL não configurada.")
        self.client = httpx.Client(base_url=base_url, timeout=TIMEOUT)
        self.token = load_cached_token()

    def _auth_header(self) -> dict:
        exp = _jwt_exp(self.token) if self.token else None
        if not self.token or (exp and time.time() >= exp - 30):
            self.login()
        return {"Authorization": f"Bearer {self.token}"}

    @backoff.on_exception(backoff.expo, (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout), max_tries=3)
    def login(self):
        r = self.client.post("/login", json={"username": USER, "password": PASS})
        r.raise_for_status()
        data = r.json()
        token = data.get("access_token") or data.get("token") or data.get("jwt")
        if not token:
            raise RuntimeError(f"Login sem token: {data}")
        self.token = token
        save_cached_token(token)

    @backoff.on_exception(
        backoff.expo,
        (httpx.HTTPStatusError, httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout),
        max_tries=5,
        giveup=lambda e: isinstance(e, httpx.HTTPStatusError) and e.response is not None and e.response.status_code not in (429, 503, 504),
    )
    def get(self, path: str, params: dict | None = None) -> dict | list:
        r = self.client.get(path, params=params, headers=self._auth_header())
        if r.status_code == 401:
            self.login()
            r = self.client.get(path, params=params, headers=self._auth_header())
        if r.status_code == 429:
            ra = r.headers.get("Retry-After")
            if ra and ra.isdigit():
                time.sleep(float(ra))
        r.raise_for_status()
        return r.json()

# ========= Paginação (somente itens da lista) =========
def fetch_all_items(api: API, base_path: str, list_key: str) -> list[dict]:

    items: list[dict] = []

    # 1) Sonda sem params (usa defaults aceitos pela API)
    data0 = api.get(base_path)
    if not isinstance(data0, dict) or list_key not in data0:
        raise ValueError(f"Resposta inesperada de {base_path}. Chaves: {list(data0.keys()) if isinstance(data0, dict) else type(data0)}")

    page = data0.get("page", 1)
    per_page = 50
    total = data0.get("total", None)
    print('paginas totais:',total)
    items.extend(data0[list_key])

    # 2) Pagina mantendo per_page retornado
    while True:
        next_page = page + 1
        print("extrainddo dados de ", base_path, page)
        time.sleep(0.01)  # reduz 429
        data = api.get(base_path, params={"page": next_page, "per_page": per_page})
        chunk = data.get(list_key, [])
        if not chunk:
            break
        items.extend(chunk)
        page = data.get("page", next_page)

        if total is not None and len(items) >= total:
            break
        if len(chunk) < per_page:
            break

    return items
def fetch_pokemon_detail(api: API, pokemon_id: int | str) -> dict:
    """
    Busca o detalhe do Pokémon por ID.
    """
    pid = int(pokemon_id)

    pid = int(pokemon_id)
    data = api.get(f"/pokemon/{pid}")
    detail = data.get("pokemon", data) if isinstance(data, dict) else {}
    detail.setdefault("id", pid)

    return detail

def fetch_all_pokemon_details(api: API, pokemons: list[dict], pause: float = 0.05) -> list[dict]:
    """
    Para cada Pokémon da lista (que tem 'id'), busca o detalhe em /pokemon/{id} (fallback /pokemons/{id}).
    Retorna a lista de detalhes. Pausa curta entre chamadas para evitar 429.
    """
    details = []
    for i, p in enumerate(pokemons, start=1):
        pid = p.get("id") or p.get("pokemon_id")
        if pid is None:
            continue
        det = fetch_pokemon_detail(api, pid)
        # se o detalhe não tiver 'name', preenche com o da lista
        if "name" not in det and "name" in p:
            det["name"] = p["name"]
        details.append(det)

        if i % 100 == 0:
            print(f"...atributos coletados: {i}", flush=True)
        time.sleep(pause)  # educação com a API
    return details






# ========= ETL =========
def run_etl():
    api = API(BASE_URL)

    # 1) Lista de pokémons (apenas itens da chave 'pokemons')
    pokemons = fetch_all_items(api, base_path="/pokemon", list_key="pokemons")
    (RAW_DIR / "pokemons_raw.json").write_text(json.dumps(pokemons, ensure_ascii=False, indent=2), encoding="utf-8")

    # CSV só com id e name
    df_poke = pd.DataFrame(pokemons).rename(columns={"pokemon_id": "id", "pokemon_name": "name"})
    for col in ("id", "name"):
        if col not in df_poke.columns:
            df_poke[col] = pd.NA
    df_poke = df_poke[["id", "name"]].copy()
    df_poke["id"] = pd.to_numeric(df_poke["id"], errors="coerce").astype("Int64")
    df_poke["name"] = df_poke["name"].astype("string")
    df_poke = (df_poke.dropna(subset=["id"]).drop_duplicates(subset=["id"]).sort_values("id").reset_index(drop=True))
    df_poke.to_csv(PROC_DIR / "pokemons.csv", index=False)

    # 2) Detalhes de cada Pokémon: /pokemon/{id} (fallback /pokemons/{id})
    pokemon_details = fetch_all_pokemon_details(api, pokemons, pause=0.05)
    (RAW_DIR / "pokemon_details_raw.json").write_text(json.dumps(pokemon_details, ensure_ascii=False, indent=2), encoding="utf-8")

    df_details = pd.json_normalize(pokemon_details, max_level=2)
    # garante coluna id numérica e name (preenche pelo join se não veio no detalhe)
    if "id" not in df_details.columns:
        df_details["id"] = pd.NA
    df_details["id"] = pd.to_numeric(df_details["id"], errors="coerce").astype("Int64")

    if "name" not in df_details.columns:
        df_details = df_details.merge(df_poke[["id", "name"]], on="id", how="left")

    # ordena colunas: id, name, resto
    cols = df_details.columns.tolist()
    ordered = [c for c in ["id", "name"] if c in cols] + [c for c in cols if c not in ("id", "name")]
    df_details = df_details[ordered]
    df_details.to_csv(PROC_DIR / "pokemon_details.csv", index=False)

    # 3) Combates (se existir /combats; se der 404, ignore ou tente /battles)
    combats = []
    try:
        combats = fetch_all_items(api, base_path="/combats", list_key="combats")
    except httpx.HTTPStatusError as e:
        if e.response is not None and e.response.status_code == 404:
            combats = []  # se quiser, tente fetch_all_items(api, "/battles", "battles")

    (RAW_DIR / "combats_raw.json").write_text(json.dumps(combats, ensure_ascii=False, indent=2), encoding="utf-8")
    if combats:
        df_comb = pd.DataFrame(combats).rename(columns={
            "first_pokemon":  "first_pokemon_id",
            "second_pokemon": "second_pokemon_id",
            "winner":         "winner_id",
        })
        for c in ["first_pokemon_id","second_pokemon_id","winner_id"]:
            if c not in df_comb.columns: df_comb[c] = pd.NA
            df_comb[c] = pd.to_numeric(df_comb[c], errors="coerce").astype("Int64")
        mask_ok = (df_comb["winner_id"] == df_comb["first_pokemon_id"]) | (df_comb["winner_id"] == df_comb["second_pokemon_id"])
        df_comb = df_comb[mask_ok].reset_index(drop=True)
        df_comb.to_csv(PROC_DIR / "combats.csv", index=False)

    print("ETL concluído.")
    print(f"- data/raw/pokemons_raw.json ({len(df_poke)} itens)")
    print(f"- data/processed/pokemons.csv")
    print(f"- data/raw/pokemon_details_raw.json ({len(df_details)} itens)")
    print(f"- data/processed/pokemon_details.csv")
    print(f"- data/raw/combats_raw.json ({len(combats)} itens)")
    if combats:
        print(f"- data/processed/combats.csv")


if __name__ == "__main__":
    run_etl()
