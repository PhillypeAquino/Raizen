# app.py
from pathlib import Path
import ast
import numpy as np
import pandas as pd
import streamlit as st

# ================== Aparência ==================
st.set_page_config(page_title="Pokémon ETL • Análises", layout="wide")
st.markdown("""
<style>
/* leve polimento visual */
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
.small-muted { color: #6b7280; font-size: 0.9rem; }
.metric-card { padding: 0.75rem 1rem; border: 1px solid #e5e7eb; border-radius: 12px; }
.section-title { margin-top: 0.5rem; }
hr { margin: 1.2rem 0; }
</style>
""", unsafe_allow_html=True)

# ================== Caminhos & Leitura ==================
def find_processed_dir(start: Path, max_up: int = 3) -> Path:
    p = start
    for _ in range(max_up + 1):
        cand = p / "data" / "processed"
        if cand.exists():
            return cand
        p = p.parent
    return Path.cwd() / "data" / "processed"

HERE = Path(__file__).resolve().parent
PROC = find_processed_dir(HERE)
# === Helpers de limpeza ===
def _strip_df_strings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    obj_cols = df.select_dtypes(include=["object", "string"]).columns
    for c in obj_cols:
        df[c] = df[c].astype("string").str.strip()
    return df

def _clean_frames(df_poke: pd.DataFrame, df_comb: pd.DataFrame, df_detl: pd.DataFrame):
    # ---- pokemons ----
    if not df_poke.empty:
        df_poke = _strip_df_strings(df_poke)
        df_poke["id"] = pd.to_numeric(df_poke["id"], errors="coerce").astype("Int64")
        df_poke = (df_poke
                   .dropna(subset=["id"])
                   .drop_duplicates(subset=["id"])
                   .reset_index(drop=True))

    # ---- details ----
    if not df_detl.empty:
        df_detl = _strip_df_strings(df_detl)
        if "id" in df_detl.columns:
            df_detl["id"] = pd.to_numeric(df_detl["id"], errors="coerce").astype("Int64")
            df_detl = df_detl.dropna(subset=["id"])
        # normalizações opcionais, só se existirem:
        if "generation" in df_detl.columns:
            df_detl["generation"] = (df_detl["generation"]
                                     .replace({"Gen1":1,"Gen2":2,"Gen3":3,"Gen4":4,"Gen5":5,"Gen6":6})
                                     .pipe(lambda s: pd.to_numeric(s, errors="coerce")))
        if "legendary" in df_detl.columns:
            df_detl["legendary"] = df_detl["legendary"].replace({"No": False })

    # ---- combats ----
    if not df_comb.empty:
        for c in ["first_pokemon_id", "second_pokemon_id", "winner_id"]:
            if c in df_comb.columns:
                df_comb[c] = pd.to_numeric(df_comb[c], errors="coerce").astype("Int64")

        # remove nulos nas relações chave
        df_comb = df_comb.dropna(subset=["first_pokemon_id", "second_pokemon_id", "winner_id"])

        # winner precisa ser um dos dois lados
        df_comb = df_comb[
            (df_comb["winner_id"] == df_comb["first_pokemon_id"]) |
            (df_comb["winner_id"] == df_comb["second_pokemon_id"])
        ]

        # mantém só combates cujos IDs existem na tabela de pokemons
        valid_ids = set(df_poke["id"].dropna().astype("Int64")) if not df_poke.empty else set()
        if valid_ids:
            df_comb = df_comb[
                df_comb["first_pokemon_id"].isin(valid_ids) &
                df_comb["second_pokemon_id"].isin(valid_ids)
            ].reset_index(drop=True)

    return df_poke, df_comb, df_detl

@st.cache_data(show_spinner=False)
def load_data(proc_dir: Path):
    poke_path = proc_dir / "pokemons.csv"
    comb_path = proc_dir / "combats.csv"
    detl_path = proc_dir / "pokemon_details.csv"

    if not poke_path.exists():
        st.warning("`pokemons.csv` não encontrado em data/processed/. Rode o ETL primeiro.")
        df_poke = pd.DataFrame(columns=["id","name"])
    else:
        df_poke = pd.read_csv(poke_path, dtype={"id":"Int64","name":"string"})
    if comb_path.exists():
        df_comb = pd.read_csv(
            comb_path,
            dtype={"first_pokemon_id":"Int64","second_pokemon_id":"Int64","winner_id":"Int64"}
        )
    else:
        df_comb = pd.DataFrame(columns=["first_pokemon_id","second_pokemon_id","winner_id"])

    df_detl = pd.read_csv(detl_path) if detl_path.exists() else pd.DataFrame()
    df_detl["generation"] = df_detl["generation"].replace({"Gen2": 2})
    df_detl["legendary"] = df_detl["legendary"].replace({"No": "False"})
    
    df_poke, df_comb, df_detl = _clean_frames(df_poke, df_comb, df_detl)
    return df_poke, df_comb, df_detl

df_poke, df_comb, df_detl = load_data(PROC)
id2name = dict(zip(df_poke["id"].astype("Int64"), df_poke["name"].astype("string"))) if not df_poke.empty else {}

# ================== Header ==================
left, right = st.columns([0.75, 0.25])
with left:
    st.title("📊 Pokémon ETL — Dashboard")
    st.caption(f"Lendo de: `{PROC}`")
with right:
    if st.button("🔄 Recarregar dados"):
        st.cache_data.clear()
        st.rerun()

st.markdown("<hr/>", unsafe_allow_html=True)

# ================== Helpers ==================
def parse_types_column(df):
    """Extrai id -> type_name a partir de uma coluna de tipos, se existir."""
    if df.empty: return pd.DataFrame(columns=["id","type_name"])
    cand_cols = [c for c in df.columns if c.lower() in {"types","type","pokemon_types","type_name"}]
    if not cand_cols: return pd.DataFrame(columns=["id","type_name"])
    col = cand_cols[0]

    tmp = df[["id", col]].dropna().copy()
    def _to_list(x):
        if isinstance(x, list): return x
        if isinstance(x, str):
            s = x.strip()
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, list): return parsed
            except Exception:
                pass
            return [p.strip() for p in s.replace("[","").replace("]","").replace("'","").replace('"',"").split(",") if p.strip()]
        return [x]
    tmp[col] = tmp[col].apply(_to_list).explode(col).dropna()
    def _leaf(v):
        if isinstance(v, dict): return v.get("name") or v.get("type") or list(v.values())[0]
        return v
    tmp["type_name"] = tmp[col].apply(_leaf).astype("string")
    return tmp[["id","type_name"]].dropna()

def build_performance(df_comb, id2name_map):
    """Tabela id -> appearances, wins, win_rate, name."""
    if df_comb.empty:
        return pd.DataFrame(columns=["id","name","wins","appearances","win_rate"])
    a1 = df_comb["first_pokemon_id"].value_counts()
    a2 = df_comb["second_pokemon_id"].value_counts()
    apps = (a1.add(a2, fill_value=0)).astype(int).rename("appearances")
    wins = df_comb["winner_id"].value_counts().astype(int).rename("wins")
    perf = pd.concat([apps, wins], axis=1).fillna(0)
    perf["wins"] = perf["wins"].astype(int)
    perf["win_rate"] = np.where(perf["appearances"]>0, perf["wins"]/perf["appearances"], np.nan)
    perf.index.name = "id"
    perf = perf.reset_index()
    perf["name"] = perf["id"].map(id2name_map)
    return perf

types_map = parse_types_column(df_detl)
perf = build_performance(df_comb, id2name)

# ================== Sidebar: Filtros Globais ==================
st.sidebar.header("🎯 Filtros")
search = st.sidebar.text_input("Buscar por nome (contém)", value="").strip()
id_min, id_max = (int(df_poke["id"].min() or 1), int(df_poke["id"].max() or 1)) if not df_poke.empty else (1,1)
id_range = st.sidebar.slider("Faixa de ID", min_value=id_min, max_value=id_max, value=(id_min, id_max))

min_apps = st.sidebar.number_input("Mínimo de aparições", min_value=0, value=5, step=1)
wr_min, wr_max = st.sidebar.slider("Win rate (%)", 0, 100, (0, 100))
order_by = st.sidebar.selectbox("Ordenar por", ["win_rate", "wins", "appearances"])

type_filter = []
if not types_map.empty:
    type_options = sorted(types_map["type_name"].dropna().unique().tolist())
    type_filter = st.sidebar.multiselect("Filtrar por tipos", type_options, default=[])

st.sidebar.caption("Dica: combine busca por nome, faixa de ID e tipos para lapidar o ranking.")

# ================== 1) Taxas de vitória por Pokémon ==================
st.subheader("1) Taxas de vitória por Pokémon", anchor=False)

if df_comb.empty or df_poke.empty:
    st.info("Precisamos de **pokemons.csv** e **combats.csv** em `data/processed/` para esta seção.")
else:
    df = perf.copy()
    # aplica filtros
    df = df[(df["id"] >= id_range[0]) & (df["id"] <= id_range[1])]
    if search:
        df = df[df["name"].fillna("").str.contains(search, case=False, na=False)]
    if min_apps > 0:
        df = df[df["appearances"] >= min_apps]
    if wr_min > 0 or wr_max < 100:
        df = df[df["win_rate"].between(wr_min/100.0, wr_max/100.0, inclusive="both")]

    # filtro por tipo (join com types_map)
    if type_filter:
        df = df.merge(types_map, on="id", how="left")
        df = df[df["type_name"].isin(type_filter)]
        # agrupa por id caso tenha duplicado por múltiplos tipos
        df = df.groupby(["id","name","wins","appearances","win_rate"], as_index=False).first()


    # Cards de métricas
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Pokémons em combate", int((perf["appearances"]>0).sum()))
    with c2: st.metric("Total de combates", int(len(df_comb)))
    with c3: st.metric("Win rate médio (filtrado)", f"{(df['win_rate'].mean()*100 if not df.empty else 0):.1f}%")
    with c4: st.metric("Win rate máximo", f"{(df['win_rate'].max()*100 if not df.empty else 0):.1f}%")

    st.markdown("##### Ranking filtrado")
    st.dataframe(
        df.assign(win_rate=lambda d: (d["win_rate"]*100).round(1)).rename(columns={"win_rate":"win_rate_%"}),
        use_container_width=True, hide_index=True
    )

    # gráfico
    st.markdown("##### Top (gráfico)")
    if not df.empty:
        chart_df = df.sort_values("win_rate", ascending=False).set_index("name")["win_rate"]
        st.bar_chart(chart_df)

    # download
    st.download_button(
        "⬇️ Baixar ranking (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="ranking_filtrado.csv",
        mime="text/csv",
        use_container_width=True
    )

st.markdown("<hr/>", unsafe_allow_html=True)

# ================== 2) Taxa de vitória por tipo ==================
st.subheader("2) Taxa de vitória por tipo (se disponível)", anchor=False)
if df_comb.empty or df_poke.empty:
    st.info("Sem **combats.csv** e **pokemons.csv**, não dá para cruzar vitórias com tipos.")
else:
    if types_map.empty:
        st.info("Não encontrei coluna de **tipos** em `pokemon_details.csv` — seção desabilitada.")
    else:
        # monta base de performance
        wins = df_comb["winner_id"].value_counts().rename("wins").to_frame()
        wins.index.name = "id"; wins = wins.reset_index()
        apps = pd.concat([df_comb["first_pokemon_id"].value_counts(),
                          df_comb["second_pokemon_id"].value_counts()], axis=1)\
                         .fillna(0).sum(axis=1).astype(int).rename("apps").to_frame()
        apps.index.name = "id"; apps = apps.reset_index()
        perf2 = wins.merge(apps, on="id", how="outer").fillna(0)
        perf2["win_rate"] = np.where(perf2["apps"]>0, perf2["wins"]/perf2["apps"], np.nan)

        weight_mode = st.radio("Cálculo", ["Média simples", "Média ponderada por aparições"], horizontal=True)
        perft = perf2.merge(types_map, on="id", how="left")
        if weight_mode == "Média simples":
            rate_by_type = perft.groupby("type_name", dropna=True)["win_rate"].mean().sort_values(ascending=False)
        else:
            # pondera por aparições
            g = perft.dropna(subset=["type_name"]).copy()
            g["w"] = g["apps"].clip(lower=1)
            rate_by_type = (g["win_rate"] * g["w"]).groupby(g["type_name"]).sum() / g["w"].groupby(g["type_name"]).sum()
            rate_by_type = rate_by_type.sort_values(ascending=False)

        st.markdown("##### Taxas por tipo")
        st.bar_chart(rate_by_type)
        tbl_types = rate_by_type.reset_index().rename(columns={0:"win_rate"}).assign(win_rate=lambda d: (d["win_rate"]*100).round(1))
        st.dataframe(tbl_types.rename(columns={"win_rate":"win_rate_%"}), use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Baixar taxas por tipo (CSV)",
            data=tbl_types.to_csv(index=False).encode("utf-8"),
            file_name="winrate_por_tipo.csv",
            mime="text/csv",
            use_container_width=True
        )

st.markdown("<hr/>", unsafe_allow_html=True)

# ================== 3) Influência de atributos (se houver detalhes) ==================
st.subheader("3) Atributos que influenciam a vitória (se disponível)", anchor=False)

def attribute_influence(df_details, df_combats):
    if df_details.empty or df_combats.empty:
        return pd.DataFrame()

    cand = [c for c in df_details.columns if any(k in c.lower()
             for k in ["hp","attack","defense","speed","sp_attack","sp_defense","special-attack","special-defense","atk","def","spd"])]
    num_cols = df_details[cand].select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols: return pd.DataFrame()

    base = df_details[["id"] + num_cols].dropna(subset=["id"]).copy()
    base["id"] = pd.to_numeric(base["id"], errors="coerce").astype("Int64")

    a = df_combats.merge(base.rename(columns={"id":"first_pokemon_id"}), on="first_pokemon_id", how="left", suffixes=("","_a"))
    b = a.merge(base.rename(columns={"id":"second_pokemon_id"}), on="second_pokemon_id", how="left", suffixes=("_a","_b"))

    y = (b["winner_id"] == b["first_pokemon_id"]).astype(float).values

    rows = []
    for col in num_cols:
        ca, cb = f"{col}_a", f"{col}_b"
        if ca not in b.columns or cb not in b.columns: continue
        diff = (b[ca] - b[cb]).astype(float).values
        if np.all(np.isnan(diff)) or np.nanstd(diff) == 0: continue
        mask = ~np.isnan(diff) & ~np.isnan(y)
        if mask.sum() < 5: continue
        r = np.corrcoef(diff[mask], y[mask])[0,1]
        rows.append({"atributo": col, "correlacao": r, "importancia_absoluta": abs(r)})

    return pd.DataFrame(rows).sort_values("importancia_absoluta", ascending=False)

if df_detl.empty or df_comb.empty:
    st.info("Sem **pokemon_details.csv** e **combats.csv**, não dá para analisar atributos.")
else:
    df_inf = attribute_influence(df_detl, df_comb)
    if df_inf.empty:
        st.info("Nenhum atributo numérico encontrado — verifique o schema de `pokemon_details.csv`.")
    else:
        top_k = st.slider("Mostrar Top K atributos", 3, min(20, len(df_inf)), 10)
        st.dataframe(df_inf.head(top_k), use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Baixar correlações (CSV)",
            data=df_inf.to_csv(index=False).encode("utf-8"),
            file_name="atributos_influencia.csv",
            mime="text/csv",
            use_container_width=True
        )

st.markdown("<hr/>", unsafe_allow_html=True)

# ================== 4) Sugestão de time ==================
st.subheader("4) Sugestão de time (heurística simples)", anchor=False)

if df_comb.empty or df_poke.empty:
    st.info("Precisamos de **combats.csv** e **pokemons.csv** para sugerir um time.")
else:
    base = perf.copy()
    # respeita filtros globais chave
    base = base[(base["id"] >= id_range[0]) & (base["id"] <= id_range[1])]
    if min_apps > 0:
        base = base[base["appearances"] >= min_apps]
    if type_filter:
        base = base.merge(types_map, on="id", how="left")
        base = base[base["type_name"].isin(type_filter)]
        base = base.groupby(["id","name","wins","appearances","win_rate"], as_index=False).first()

    optimize_by = st.radio("Otimizar por", ["win_rate", "wins"], horizontal=True, index=0)
    team_size = st.selectbox("Tamanho do time", [3,4,5,6], index=3)
    team = base.sort_values([optimize_by, "appearances"], ascending=[False, False]).head(team_size)
    st.dataframe(
        team.assign(win_rate=lambda d: (d["win_rate"]*100).round(1)).rename(columns={"win_rate":"win_rate_%"}),
        use_container_width=True, hide_index=True
    )
    st.caption("Heurística: escolhe os melhores pelo critério selecionado, com mínimo de aparições aplicado nos filtros.")

st.markdown("<hr/>", unsafe_allow_html=True)
st.caption("App simples • lê CSVs do ETL • filtros globais para focar a análise • pronto para demonstração.")
