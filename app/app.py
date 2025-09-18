from __future__ import annotations
from pathlib import Path
import ast
import math
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ================== Config & Theme ==================
st.set_page_config(page_title="Pokémon ETL • Análises+", layout="wide")

st.markdown(
    """
<style>
:root {
  --card-bd: #e5e7eb; /* zinc-200 */
  --muted: #6b7280;  /* zinc-500 */
}
.block-container { padding-top: 1rem; padding-bottom: 2rem; }
.small-muted { color: var(--muted); font-size: 0.9rem; }
.metric-card { padding: 0.75rem 1rem; border: 1px solid var(--card-bd); border-radius: 12px; }
.section-title { margin-top: 0.5rem; }
hr { margin: 1.2rem 0; }
[data-testid="stMetric"] { background: rgba(0,0,0,0.02); padding: .5rem .75rem; border-radius: 12px; border: 1px solid var(--card-bd); }
.stButton>button { border-radius: 10px; }
.stDownloadButton>button { width: 100%; }
</style>
""",
    unsafe_allow_html=True,
)

POWERBI_URL = "https://app.powerbi.com/view?r=eyJrIjoiN2IyODQzNzktMGNhOS00OTY2LWFhNTUtMGI2MjM0Y2RkN2RlIiwidCI6IjFhOWM0NDgxLTVlYTUtNGI5OS1iOWE4LTI4NGRlZjI4YjYwNSJ9"

# ================== File Discovery ==================

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

# ================== Helpers ==================

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
        if "id" in df_poke.columns:
            df_poke["id"] = pd.to_numeric(df_poke["id"], errors="coerce").astype("Int64")
            df_poke = (
                df_poke.dropna(subset=["id"]).drop_duplicates(subset=["id"]).reset_index(drop=True)
            )

    # ---- details ----
    if not df_detl.empty:
        df_detl = _strip_df_strings(df_detl)
        if "id" in df_detl.columns:
            df_detl["id"] = pd.to_numeric(df_detl["id"], errors="coerce").astype("Int64")
            df_detl = df_detl.dropna(subset=["id"])
        # Normalizações: generation e legendary
        if "generation" in df_detl.columns:
            df_detl["generation"] = (
                df_detl["generation"]
                .replace({"Gen1": 1, "Gen2": 2, "Gen3": 3, "Gen4": 4, "Gen5": 5, "Gen6": 6})
                .pipe(lambda s: pd.to_numeric(s, errors="coerce"))
                .astype("Int64")
            )
        if "legendary" in df_detl.columns:
            s = df_detl["legendary"].astype("string").str.strip().str.lower()
            mapper = {"no": False, "false": False, "yes": True, "true": True, "0": False, "1": True}
            df_detl["legendary"] = s.map(mapper).astype("boolean")

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
                df_comb["first_pokemon_id"].isin(valid_ids) & df_comb["second_pokemon_id"].isin(valid_ids)
            ].reset_index(drop=True)

    return df_poke, df_comb, df_detl


@st.cache_data(show_spinner=True)
def load_data(proc_dir: Path):
    poke_path = proc_dir / "pokemons.csv"
    comb_path = proc_dir / "combats.csv"
    detl_path = proc_dir / "pokemon_details.csv"

    if not poke_path.exists():
        st.warning("`pokemons.csv` não encontrado em data/processed/. Rode o ETL primeiro.")
        df_poke = pd.DataFrame(columns=["id", "name"])
    else:
        df_poke = pd.read_csv(poke_path, dtype={"id": "Int64", "name": "string"})

    if comb_path.exists():
        df_comb = pd.read_csv(
            comb_path,
            dtype={"first_pokemon_id": "Int64", "second_pokemon_id": "Int64", "winner_id": "Int64"},
        )
    else:
        df_comb = pd.DataFrame(columns=["first_pokemon_id", "second_pokemon_id", "winner_id"])

    df_detl = pd.read_csv(detl_path) if detl_path.exists() else pd.DataFrame()

    # ajustes mínimos pedidos explicitamente
    if not df_detl.empty:
        if "generation" in df_detl.columns:
            df_detl["generation"] = df_detl["generation"].replace({"Gen2": 2})
        if "legendary" in df_detl.columns:
            s = df_detl["legendary"].astype("string").str.strip().str.lower()
            mapper = {"no": False, "false": False, "yes": True, "true": True, "0": False, "1": True}
            df_detl["legendary"] = s.map(mapper).astype("boolean")

    df_poke, df_comb, df_detl = _clean_frames(df_poke, df_comb, df_detl)
    return df_poke, df_comb, df_detl


df_poke, df_comb, df_detl = load_data(PROC)
id2name = dict(zip(df_poke["id"].astype("Int64"), df_poke["name"].astype("string"))) if not df_poke.empty else {}

# ================== Business Logic ==================

def parse_types_column(df):
    """Extrai id -> type_name a partir de uma coluna de tipos, se existir."""
    if df.empty:
        return pd.DataFrame(columns=["id", "type_name"])
    cand_cols = [c for c in df.columns if c.lower() in {"types", "type", "pokemon_types", "type_name"}]
    if not cand_cols:
        return pd.DataFrame(columns=["id", "type_name"])
    col = cand_cols[0]

    tmp = df[["id", col]].dropna().copy()
    def _to_list(x):
        if isinstance(x, list):
            return x
        if isinstance(x, str):
            s = x.strip()
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass
            return [p.strip() for p in s.replace("[", "").replace("]", "").replace("'", "").replace("\"", "").split(",") if p.strip()]
        return [x]

    tmp[col] = tmp[col].apply(_to_list).explode(col).dropna()

    def _leaf(v):
        if isinstance(v, dict):
            return v.get("name") or v.get("type") or list(v.values())[0]
        return v

    tmp["type_name"] = tmp[col].apply(_leaf).astype("string")
    return tmp[["id", "type_name"]].dropna()


def build_performance(df_comb, id2name_map):
    """Tabela id -> appearances, wins, win_rate, name."""
    if df_comb.empty:
        return pd.DataFrame(columns=["id", "name", "wins", "appearances", "win_rate"])
    a1 = df_comb["first_pokemon_id"].value_counts()
    a2 = df_comb["second_pokemon_id"].value_counts()
    apps = (a1.add(a2, fill_value=0)).astype(int).rename("appearances")
    wins = df_comb["winner_id"].value_counts().astype(int).rename("wins")
    perf = pd.concat([apps, wins], axis=1).fillna(0)
    perf["wins"] = perf["wins"].astype(int)
    perf["win_rate"] = np.where(perf["appearances"] > 0, perf["wins"] / perf["appearances"], np.nan)
    perf.index.name = "id"
    perf = perf.reset_index()
    perf["name"] = perf["id"].map(id2name_map)
    return perf


types_map = parse_types_column(df_detl)
perf = build_performance(df_comb, id2name)

# ================== Header ==================
left, right = st.columns([0.65, 0.35])
with left:
    st.title("📊 Pokémon ETL — Dashboard Plus")
    st.caption(f"Lendo de: `{PROC}`")
with right:
    st.button("")
    st.link_button("🔗 Abrir relatório Power BI", POWERBI_URL, use_container_width=True)
    if st.button("🔄 Recarregar dados", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


st.divider()

# ================== Sidebar Filters (Persistentes) ==================
with st.sidebar:
    st.header("🎯 Filtros globais")

    search = st.text_input("Buscar por nome (contém)", value="").strip()

    if not df_poke.empty:
        id_min, id_max = int(df_poke["id"].min() or 1), int(df_poke["id"].max() or 1)
    else:
        id_min, id_max = 1, 1
    id_range = st.slider("Faixa de ID", min_value=id_min, max_value=id_max, value=(id_min, id_max))

    min_apps = st.number_input("Mínimo de aparições", min_value=0, value=5, step=1)
    wr_min, wr_max = st.slider("Win rate (%)", 0, 100, (0, 100))

    order_by = st.selectbox("Ordenar por", ["win_rate", "wins", "appearances"], index=0)

    # Tipos com modo de correspondência (qualquer / todos)
    type_filter = []
    type_mode = "Qualquer"  # default
    if not types_map.empty:
        type_options = sorted(types_map["type_name"].dropna().unique().tolist())
        type_filter = st.multiselect("Filtrar por tipos", type_options, default=[])
        if type_filter:
            type_mode = st.radio("Modo de filtro de tipos", ["Qualquer", "Todos"], horizontal=True, index=0)

    # Filtros extras se houver detalhes
    gen_sel = None
    leg_mode = "Todos"
    if not df_detl.empty:
        gens = sorted([int(g) for g in df_detl.get("generation", pd.Series(dtype="Int64")).dropna().unique().tolist()])
        if gens:
            gen_sel = st.multiselect("Gerações", gens, default=[])
        if "legendary" in df_detl.columns:
            leg_mode = st.radio("Lendário?", ["Todos", "Somente lendários", "Somente não lendários"], horizontal=False, index=0)

    st.caption("Dica: combine busca, faixa de ID, tipos e geração para lapidar o ranking.")


# ================== Filter Application ==================

def apply_global_filters(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    # Por ID
    if {"id"}.issubset(df.columns):
        df = df[(df["id"] >= id_range[0]) & (df["id"] <= id_range[1])]

    # Por nome
    if search and "name" in df.columns:
        df = df[df["name"].fillna("").str.contains(search, case=False, na=False)]

    # Por aparições
    if "appearances" in df.columns and min_apps > 0:
        df = df[df["appearances"] >= min_apps]

    # Por win rate
    if "win_rate" in df.columns and (wr_min > 0 or wr_max < 100):
        df = df[df["win_rate"].between(wr_min / 100.0, wr_max / 100.0, inclusive="both")]

    # Tipos
    if type_filter:
        df = df.merge(types_map, on="id", how="left")
        if type_mode == "Qualquer":
            df = df[df["type_name"].isin(type_filter)]
            df = df.groupby([c for c in df.columns if c not in {"type_name"}], as_index=False).first()
        else:  # Todos os tipos selecionados
            # conta quantos tipos alvo cada id possui
            mask = df["type_name"].isin(type_filter)
            cnt = df[mask].groupby("id")["type_name"].nunique().rename("_hit")
            df = df.merge(cnt, on="id", how="left").fillna({"_hit": 0})
            df = df[df["_hit"] >= len(type_filter)].drop(columns=["_hit", "type_name"], errors="ignore").drop_duplicates()

    # Geração
    if gen_sel and not df_detl.empty:
        df = df.merge(df_detl[["id", "generation"]], on="id", how="left")
        df = df[df["generation"].isin(gen_sel)]
        df = df.drop(columns=["generation"]).drop_duplicates()

    # Lendário
    if not df_detl.empty and "legendary" in df_detl.columns:
        if leg_mode == "Somente lendários":
            df = df.merge(df_detl[["id", "legendary"]], on="id", how="left")
            df = df[df["legendary"] == True].drop(columns=["legendary"])  # noqa: E712
        elif leg_mode == "Somente não lendários":
            df = df.merge(df_detl[["id", "legendary"]], on="id", how="left")
            df = df[(df["legendary"] == False) | (df["legendary"].isna())].drop(columns=["legendary"])  # noqa: E712

    return df


# ================== Tabs ==================
t1, t2, t3, t4, t5, t6 = st.tabs([
    "Visão Geral",
    "Pokémons",
    "Tipos",
    "Atributos",
    "Time Sugerido",
    "Explorar Dados",
])

# ========== Tab 1: Overview ==========
with t1:
    st.subheader("Visão Geral")
    if df_comb.empty or df_poke.empty:
        st.info("Precisamos de **pokemons.csv** e **combats.csv** em `data/processed/`.")
    else:
        base = apply_global_filters(perf)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Pokémons em combate", int((perf["appearances"] > 0).sum()))
        with c2:
            st.metric("Total de combates", int(len(df_comb)))
        with c3:
            st.metric("Win rate médio (filtrado)", f"{(base['win_rate'].mean()*100 if not base.empty else 0):.1f}%")
        with c4:
            st.metric("Win rate máximo", f"{(base['win_rate'].max()*100 if not base.empty else 0):.1f}%")

        st.markdown("##### Distribuição de win rate (filtrado)")
        if not base.empty:
            hist = alt.Chart(base.assign(win_rate_pct=lambda d: d["win_rate"] * 100)).mark_bar().encode(
                x=alt.X("win_rate_pct:Q", bin=alt.Bin(maxbins=30), title="Win rate (%)"),
                y=alt.Y("count()", title="Qtd. Pokémons"),
                tooltip=[alt.Tooltip("count()", title="Qtd."), alt.Tooltip("win_rate_pct:Q", title="Faixa (%)")],
            ).properties(height=260)
            st.altair_chart(hist, use_container_width=True)

# ========== Tab 2: Pokémons ==========
with t2:
    st.subheader("1) Taxas de vitória por Pokémon")
    if df_comb.empty or df_poke.empty:
        st.info("Precisamos de **pokemons.csv** e **combats.csv** em `data/processed/` para esta seção.")
    else:
        df = apply_global_filters(perf)
        st.markdown("##### Ranking filtrado")
        st.dataframe(
            df.assign(win_rate=lambda d: (d["win_rate"] * 100).round(1)).rename(columns={"win_rate": "win_rate_%"}),
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "⬇️ Baixar ranking (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="ranking_filtrado.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.markdown("##### Top (gráfico)")
        if not df.empty:
            chart_df = df.sort_values("win_rate", ascending=False).head(30)
            chart = (
                alt.Chart(chart_df)
                .mark_bar()
                .encode(
                    x=alt.X("name:N", sort="-y", title="Pokémon"),
                    y=alt.Y("win_rate:Q", axis=alt.Axis(format="%"), title="Win rate"),
                    tooltip=["name", alt.Tooltip("wins:Q", title="Vitórias"), alt.Tooltip("appearances:Q", title="Aparições"), alt.Tooltip("win_rate:Q", title="Win rate", format=".1%")],
                )
                .properties(height=360)
            )
            st.altair_chart(chart, use_container_width=True)

# ========== Tab 3: Tipos ==========
with t3:
    st.subheader("2) Taxa de vitória por tipo (se disponível)")
    if df_comb.empty or df_poke.empty:
        st.info("Sem **combats.csv** e **pokemons.csv**, não dá para cruzar vitórias com tipos.")
    else:
        if types_map.empty:
            st.info("Não encontrei coluna de **tipos** em `pokemon_details.csv` — seção desabilitada.")
        else:
            wins = df_comb["winner_id"].value_counts().rename("wins").to_frame()
            wins.index.name = "id"; wins = wins.reset_index()
            apps = pd.concat([
                df_comb["first_pokemon_id"].value_counts(),
                df_comb["second_pokemon_id"].value_counts(),
            ], axis=1).fillna(0).sum(axis=1).astype(int).rename("apps").to_frame()
            apps.index.name = "id"; apps = apps.reset_index()
            perf2 = wins.merge(apps, on="id", how="outer").fillna(0)
            perf2["win_rate"] = np.where(perf2["apps"] > 0, perf2["wins"] / perf2["apps"], np.nan)

            weight_mode = st.radio("Cálculo", ["Média simples", "Média ponderada por aparições"], horizontal=True)
            perft = perf2.merge(types_map, on="id", how="left")
            if weight_mode == "Média simples":
                rate_by_type = perft.groupby("type_name", dropna=True)["win_rate"].mean().sort_values(ascending=False)
            else:
                g = perft.dropna(subset=["type_name"]).copy()
                g["w"] = g["apps"].clip(lower=1)
                rate_by_type = (g["win_rate"] * g["w"]).groupby(g["type_name"]).sum() / g["w"].groupby(g["type_name"]).sum()
                rate_by_type = rate_by_type.sort_values(ascending=False)

            st.markdown("##### Taxas por tipo")
            chart = (
                alt.Chart(rate_by_type.reset_index().rename(columns={0: "win_rate"}))
                .mark_bar()
                .encode(
                    x=alt.X("type_name:N", sort="-y", title="Tipo"),
                    y=alt.Y("win_rate:Q", axis=alt.Axis(format="%"), title="Win rate"),
                    tooltip=["type_name", alt.Tooltip("win_rate:Q", title="Win rate", format=".1%")],
                )
                .properties(height=360)
            )
            st.altair_chart(chart, use_container_width=True)

            tbl_types = rate_by_type.reset_index().rename(columns={0: "win_rate"}).assign(win_rate=lambda d: (d["win_rate"] * 100).round(1))
            st.dataframe(tbl_types.rename(columns={"win_rate": "win_rate_%"}), use_container_width=True, hide_index=True)
            st.download_button(
                "⬇️ Baixar taxas por tipo (CSV)",
                data=tbl_types.to_csv(index=False).encode("utf-8"),
                file_name="winrate_por_tipo.csv",
                mime="text/csv",
                use_container_width=True,
            )

# ========== Tab 4: Atributos ==========
with t4:
    st.subheader("3) Atributos que influenciam a vitória (se disponível)")

    def attribute_influence(df_details, df_combats):
        if df_details.empty or df_combats.empty:
            return pd.DataFrame()

        cand = [
            c for c in df_details.columns
            if any(k in c.lower() for k in [
                "hp", "attack", "defense", "speed", "sp_attack", "sp_defense",
                "special-attack", "special-defense", "atk", "def", "spd"
            ])
        ]
        num_cols = df_details[cand].select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            return pd.DataFrame()

        base = df_details[["id"] + num_cols].dropna(subset=["id"]).copy()
        base["id"] = pd.to_numeric(base["id"], errors="coerce").astype("Int64")

        a = df_combats.merge(base.rename(columns={"id": "first_pokemon_id"}), on="first_pokemon_id", how="left", suffixes=("", "_a"))
        b = a.merge(base.rename(columns={"id": "second_pokemon_id"}), on="second_pokemon_id", how="left", suffixes=("_a", "_b"))

        y = (b["winner_id"] == b["first_pokemon_id"]).astype(float).values

        rows = []
        for col in num_cols:
            ca, cb = f"{col}_a", f"{col}_b"
            if ca not in b.columns or cb not in b.columns:
                continue
            diff = (b[ca] - b[cb]).astype(float).values
            if np.all(np.isnan(diff)) or np.nanstd(diff) == 0:
                continue
            mask = ~np.isnan(diff) & ~np.isnan(y)
            if mask.sum() < 5:
                continue
            r = np.corrcoef(diff[mask], y[mask])[0, 1]
            rows.append({"atributo": col, "correlacao": r, "importancia_absoluta": abs(r)})

        return pd.DataFrame(rows).sort_values("importancia_absoluta", ascending=False)

    if df_detl.empty or df_comb.empty:
        st.info("Sem **pokemon_details.csv** e **combats.csv**, não dá para analisar atributos.")
    else:
        df_inf = attribute_influence(df_detl, df_comb)
        if df_inf.empty:
            st.info("Nenhum atributo numérico encontrado — verifique o schema de `pokemon_details.csv`.")
        else:
            top_k = st.slider("Mostrar Top K atributos", 3, min(30, len(df_inf)), 10)
            st.dataframe(df_inf.head(top_k), use_container_width=True, hide_index=True)
            st.download_button(
                "⬇️ Baixar correlações (CSV)",
                data=df_inf.to_csv(index=False).encode("utf-8"),
                file_name="atributos_influencia.csv",
                mime="text/csv",
                use_container_width=True,
            )

            # Scatter: escolher um atributo para ver relação com win rate
            st.markdown("##### Relação atributo × win rate (dispersão + tendência)")
            # montar base perf + detalhes numéricos
            num_cols = df_detl.select_dtypes(include=[np.number]).columns.tolist()
            num_cols = [c for c in num_cols if c != "id"]
            if num_cols:
                x_attr = st.selectbox("Atributo (eixo X)", options=num_cols, index=min(1, len(num_cols)-1))
                # juntar
                merged = perf.merge(df_detl[["id", x_attr]], on="id", how="left").dropna(subset=[x_attr, "win_rate"]).copy()
                if not merged.empty:
                    # Regressão linear simples para linha de tendência
                    x = merged[x_attr].astype(float).values
                    y = merged["win_rate"].astype(float).values
                    if x.size >= 2 and np.nanstd(x) > 0:
                        coef = np.polyfit(x, y, 1)
                        a, b = float(coef[0]), float(coef[1])
                        merged["trend"] = a * merged[x_attr] + b
                        # Pearson r
                        r = np.corrcoef(x, y)[0, 1]
                        st.caption(f"Correlação de Pearson (r) = {r:.3f}")

                        base_chart = alt.Chart(merged).mark_circle(size=60, opacity=0.6).encode(
                            x=alt.X(f"{x_attr}:Q", title=x_attr.replace("_", " ").title()),
                            y=alt.Y("win_rate:Q", axis=alt.Axis(format="%"), title="Win rate"),
                            tooltip=["name", x_attr, alt.Tooltip("win_rate:Q", title="Win rate", format=".1%")],
                        )
                        trend_chart = alt.Chart(merged).transform_regression(x_attr, "win_rate").mark_line().encode(
                            x=alt.X(f"{x_attr}:Q"),
                            y=alt.Y("win_rate:Q"),
                        )
                        st.altair_chart((base_chart + trend_chart).interactive().properties(height=420), use_container_width=True)
                    else:
                        st.info("Amostra insuficiente para regressão.")
                else:
                    st.info("Sem dados suficientes após junção para exibir o gráfico.")

# ========== Tab 5: Team Builder ==========
with t5:
    st.subheader("4) Sugestão de time (heurística simples)")
    if df_comb.empty or df_poke.empty:
        st.info("Precisamos de **combats.csv** e **pokemons.csv** para sugerir um time.")
    else:
        base = apply_global_filters(perf)
        optimize_by = st.radio("Otimizar por", ["win_rate", "wins"], horizontal=True, index=0)
        team_size = st.selectbox("Tamanho do time", [3, 4, 5, 6], index=3)
        team = base.sort_values([optimize_by, "appearances"], ascending=[False, False]).head(team_size)
        st.dataframe(
            team.assign(win_rate=lambda d: (d["win_rate"] * 100).round(1)).rename(columns={"win_rate": "win_rate_%"}),
            use_container_width=True,
            hide_index=True,
        )
        st.caption("Heurística: escolhe os melhores pelo critério selecionado, respeitando filtros globais.")

# ========== Tab 6: Explore ==========
with t6:
    st.subheader("Explorar dados brutos")
    with st.expander("pokemons.csv"):
        st.dataframe(df_poke, use_container_width=True)
    with st.expander("combats.csv"):
        st.dataframe(df_comb, use_container_width=True)
    with st.expander("pokemon_details.csv"):
        st.dataframe(df_detl, use_container_width=True)

st.caption("App pronto para demo • filtros persistentes • gráficos interativos Altair • integração com Power BI.")
