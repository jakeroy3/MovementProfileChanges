import requests
from datetime import date
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pybaseball import statcast, statcast_pitcher
import numpy as np
import re

BASE_URL = "https://statsapi.mlb.com/api"

# -----------------------------
# MLB Stats API helpers
# -----------------------------
def get_schedule_for_date(dt: date) -> pd.DataFrame:
    date_str = dt.strftime("%Y-%m-%d")
    url = f"{BASE_URL}/v1/schedule"
    params = {"sportId": 1, "date": date_str}
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    games = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            games.append(
                {
                    "game_pk": g["gamePk"],
                    "away_team": g["teams"]["away"]["team"]["name"],
                    "home_team": g["teams"]["home"]["team"]["name"],
                    "status": g["status"]["detailedState"],
                }
            )

    return pd.DataFrame(games, columns=["game_pk", "away_team", "home_team", "status"])


def get_pitchers_for_game(game_pk: int) -> pd.DataFrame:
    url = f"{BASE_URL}/v1/game/{game_pk}/boxscore"
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    pitchers = []
    for team_side in ["home", "away"]:
        team = data["teams"][team_side]
        team_name = team["team"]["name"]

        for pid, player in team["players"].items():
            mlbam_id = int(pid.replace("ID", ""))
            pos_code = player.get("position", {}).get("code")
            primary_pos_code = player.get("person", {}).get("primaryPosition", {}).get("code")
            if pos_code == "1" or primary_pos_code == "1":
                pitchers.append(
                    {
                        "mlbam_id": mlbam_id,
                        "name": player["person"]["fullName"],
                        "team": team_name,
                        "side": team_side,
                    }
                )

    return pd.DataFrame(pitchers, columns=["mlbam_id", "name", "team", "side"])


def get_pitchers_who_threw_fast(game_pk: int) -> pd.DataFrame:
    url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    pitcher_ids = set()
    for play in data["liveData"]["plays"]["allPlays"]:
        for ev in play.get("playEvents", []):
            if ev.get("isPitch"):
                pitcher_ids.add(play["matchup"]["pitcher"]["id"])

    box_pitchers = get_pitchers_for_game(game_pk)
    if box_pitchers.empty:
        return box_pitchers

    return box_pitchers[box_pitchers["mlbam_id"].isin(pitcher_ids)].reset_index(drop=True)


# -----------------------------
# Savant /gf helpers (extension)
# -----------------------------
@st.cache_data(ttl=300)
def get_savant_gf_extension_by_type(game_pk: int, pitcher_id: int) -> pd.DataFrame:
    """
    Baseball Savant /gf returns pitch_type codes (FF/FC/SI/...) and extension.
    Return mean extension by pitch_type code for this pitcher in this game.
    """
    url = f"https://baseballsavant.mlb.com/gf?game_pk={game_pk}"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    data = r.json()

    rows = []
    for team_data in (data.get("team_home", []), data.get("team_away", [])):
        for p in (team_data or []):
            if p.get("pitcher") != pitcher_id:
                continue
            rows.append({"pitch_type": p.get("pitch_type"), "extension": p.get("extension")})

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["pitch_type", "extension"])

    df["extension"] = pd.to_numeric(df["extension"], errors="coerce")
    df = df.dropna(subset=["pitch_type", "extension"])
    if df.empty:
        return pd.DataFrame(columns=["pitch_type", "extension"])

    return df.groupby("pitch_type", as_index=False)["extension"].mean()


# -----------------------------
# Description normalization + pitch outcome definitions
# -----------------------------
def normalize_desc_to_statcast_token(desc: str) -> str:
    """
    Convert MLB Stats 'details.description' into Statcast-style 'description' tokens
    used by pybaseball (e.g. 'called_strike', 'swinging_strike', 'in_play', ...).
    """
    if desc is None:
        return ""

    s = str(desc).strip().lower()

    # Common MLB Stats wording -> Statcast tokens
    if s == "called strike":
        return "called_strike"
    if s == "ball":
        return "ball"
    if s == "blocked ball":
        return "blocked_ball"
    if s == "hit by pitch":
        return "hit_by_pitch"

    if "swinging strike" in s:
        # handles "Swinging Strike", "Swinging Strike (Blocked)"
        return "swinging_strike_blocked" if "blocked" in s else "swinging_strike"

    if s == "swinging pitchout":
        return "swinging_pitchout"

    if s.startswith("foul tip bunt"):
        return "foul_tip_bunt"
    if s.startswith("foul bunt"):
        return "foul_bunt"
    if s.startswith("foul pitchout"):
        return "foul_pitchout"
    if s.startswith("foul tip"):
        return "foul_tip"
    if s.startswith("foul"):
        return "foul"

    if s.startswith("in play"):
        return "in_play"

    # Fallback: slugify
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


STRIKES = {
    "called_strike",
    "foul",
    "foul_bunt",
    "foul_tip_bunt",
    "foul_pitchout",
    "hit_into_play",
    "foul_tip",
    "swinging_pitchout",
    "swinging_strike",
    "swinging_strike_blocked",
    "in_play"
}

SWINGING_STRIKES = {
    "swinging_pitchout",
    "swinging_strike",
    "swinging_strike_blocked",
    "foul_tip",
    "foul_tip_bunt",
}

CALLED_STRIKES = {"called_strike"}

SWINGS = {
    "swinging_pitchout",
    "swinging_strike",
    "swinging_strike_blocked",
    "foul",
    "foul_tip",
    "foul_bunt",
    "foul_tip_bunt",
    "foul_pitchout",
    "hit_into_play",
    "in_play"
}


def is_in_zone(z) -> bool:
    try:
        if pd.isna(z):
            return False
        zi = int(z)
        return 1 <= zi <= 9
    except Exception:
        return False


# -----------------------------
# Game data pull (Statcast preferred, MLB Stats fallback)
# -----------------------------
@st.cache_data(ttl=300)
def get_pitcher_game_data(game_pk: int, pitcher_id: int, game_date: date) -> pd.DataFrame:
    """
    Prefer Statcast (pybaseball) if available; fallback to MLB Stats API induced break.
    Ensures columns:
      game_pk, pitcher, pitch_type, pitch_name, release_speed, pfx_x, pfx_z,
      extension, description, zone
    pfx_x/pfx_z are in FEET (Statcast convention). Extension is backfilled from Savant /gf if missing.
    """
    date_str = game_date.strftime("%Y-%m-%d")

    def backfill_extension_from_gf(df_in: pd.DataFrame) -> pd.DataFrame:
        df = df_in.copy()
        if "extension" not in df.columns or df["extension"].isna().all():
            ext_by_type = get_savant_gf_extension_by_type(game_pk, pitcher_id)
            if not ext_by_type.empty:
                df = df.merge(ext_by_type, on="pitch_type", how="left", suffixes=("", "_gf"))
                if "extension_gf" in df.columns:
                    if "extension" not in df.columns:
                        df["extension"] = np.nan
                    df["extension"] = df["extension"].fillna(df["extension_gf"])
                    df = df.drop(columns=["extension_gf"])
        if "extension" not in df.columns:
            df["extension"] = np.nan
        return df

    # Method 1: statcast(day) -> filter
    try:
        sc_all = statcast(date_str, date_str)
        if sc_all is not None and not sc_all.empty:
            sc_filtered = sc_all[(sc_all["game_pk"] == game_pk) & (sc_all["pitcher"] == pitcher_id)]
            if not sc_filtered.empty:
                keep = [
                    "game_pk",
                    "pitcher",
                    "game_date",
                    "pitch_type",
                    "pitch_name",
                    "pfx_x",
                    "pfx_z",
                    "release_speed",
                    "extension",
                    "description",
                    "zone",
                ]
                sc_filtered = sc_filtered[[c for c in keep if c in sc_filtered.columns]].copy()
                sc_filtered = sc_filtered.dropna(subset=["pfx_x", "pfx_z"])
                if not sc_filtered.empty:
                    if "description" not in sc_filtered.columns:
                        sc_filtered["description"] = ""
                    if "zone" not in sc_filtered.columns:
                        sc_filtered["zone"] = np.nan
                    sc_filtered = backfill_extension_from_gf(sc_filtered)
                    return sc_filtered
    except Exception:
        pass

    # Method 2: statcast_pitcher(day) -> filter by game
    try:
        sc_pitcher = statcast_pitcher(date_str, date_str, pitcher_id)
        if sc_pitcher is not None and not sc_pitcher.empty:
            sc_filtered = sc_pitcher[sc_pitcher["game_pk"] == game_pk]
            if not sc_filtered.empty:
                keep = [
                    "game_pk",
                    "pitcher",
                    "game_date",
                    "pitch_type",
                    "pitch_name",
                    "pfx_x",
                    "pfx_z",
                    "release_speed",
                    "extension",
                    "description",
                    "zone",
                ]
                sc_filtered = sc_filtered[[c for c in keep if c in sc_filtered.columns]].copy()
                sc_filtered = sc_filtered.dropna(subset=["pfx_x", "pfx_z"])
                if not sc_filtered.empty:
                    if "description" not in sc_filtered.columns:
                        sc_filtered["description"] = ""
                    if "zone" not in sc_filtered.columns:
                        sc_filtered["zone"] = np.nan
                    sc_filtered = backfill_extension_from_gf(sc_filtered)
                    return sc_filtered
    except Exception:
        pass

    # Method 3: MLB Stats live feed induced movement
    url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    pitches = []
    for play in data["liveData"]["plays"]["allPlays"]:
        if play["matchup"]["pitcher"]["id"] != pitcher_id:
            continue

        for ev in play.get("playEvents", []):
            if not ev.get("isPitch"):
                continue

            pd_data = ev.get("pitchData", {}) or {}
            breaks = pd_data.get("breaks", {}) or {}
            details = ev.get("details", {}) or {}
            call = details.get("call", {}) or {}
            _coords = pd_data.get("coordinates", {}) or {}

            ihb = breaks.get("breakHorizontal")
            ivb = breaks.get("breakVerticalInduced")
            if ihb is None or ivb is None:
                continue

            zone = pd_data.get("zone")
            if zone is None:
                zone = _coords.get("zone")

            desc_token = normalize_desc_to_statcast_token(details.get("description"))

            pitches.append(
                {
                    "game_pk": game_pk,
                    "pitcher": pitcher_id,
                    "pitch_type": details.get("type", {}).get("code"),
                    "pitch_name": details.get("type", {}).get("description"),
                    "release_speed": pd_data.get("startSpeed"),
                    # movement: inches -> feet; invert horizontal to match your Statcast orientation
                    "pfx_x": (-ihb) / 12.0,
                    "pfx_z": (ivb) / 12.0,
                    "description": desc_token,
                    "zone": zone,
                }
            )

    df = pd.DataFrame(pitches)
    if df.empty:
        return df

    df = backfill_extension_from_gf(df)
    return df


# -----------------------------
# Comparison data (Statcast pitcher)
# -----------------------------
@st.cache_data(ttl=600)
def get_comparison_movement(pitcher_id: int, start_date: date, end_date: date) -> pd.DataFrame:
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    df_comp = statcast_pitcher(start_str, end_str, pitcher_id)
    if df_comp is None or df_comp.empty:
        return pd.DataFrame(
            columns=[
                "pitch_type",
                "avg_pfx_x",
                "avg_pfx_z",
                "n_pitches",
                "avg_pfx_x_in",
                "avg_pfx_z_in",
                "start_date",
                "end_date",
            ]
        )

    if "pitch_type" in df_comp.columns:
        df_comp = df_comp[~df_comp["pitch_type"].isin(["PO", "CS"])]

    df_comp = df_comp.dropna(subset=["pfx_x", "pfx_z"])
    if df_comp.empty:
        return pd.DataFrame(
            columns=[
                "pitch_type",
                "avg_pfx_x",
                "avg_pfx_z",
                "n_pitches",
                "avg_pfx_x_in",
                "avg_pfx_z_in",
                "start_date",
                "end_date",
            ]
        )

    mov = (
        df_comp.groupby("pitch_type", as_index=False)
        .agg(
            avg_pfx_x=("pfx_x", "mean"),
            avg_pfx_z=("pfx_z", "mean"),
            n_pitches=("pitch_type", "size"),
        )
    )

    mov["avg_pfx_x_in"] = mov["avg_pfx_x"] * 12.0
    mov["avg_pfx_z_in"] = mov["avg_pfx_z"] * 12.0
    mov["start_date"] = start_date
    mov["end_date"] = end_date
    return mov


# -----------------------------
# Plot styling
# -----------------------------
pitch_colors = {
    "FF": "#d62728",
    "SI": "#ff7f0e",
    "SL": "#f1c40f",
    "CU": "#17becf",
    "CH": "#2ca02c",
    "FC": "#8c564b",
    "KN": "#9467bd",
    "KC": "#1f77b4",
    "FS": "#20b2aa",
    "ST": "#9C7314",
    "SC": "#7fff00",
    "FO": "#00ced1",
}

pitch_row_colors = pitch_colors.copy()

def shade_row_by_pitch_type(row):
    pt = row["Pitch Type"]
    base = pitch_row_colors.get(pt, "#ffffff")
    return [f"background-color: {base}22; color: #E6E6E6;"] * len(row)


def add_concentric_circles(fig, radii=(12, 24), dashed=(12,)):
    theta = np.linspace(0, 2 * np.pi, 200)
    for r in radii:
        fig.add_trace(
            go.Scatter(
                x=r * np.cos(theta),
                y=r * np.sin(theta),
                mode="lines",
                line=dict(
                    color="rgba(180,180,180,0.7)",
                    width=1,
                    dash="dash" if r in dashed else "solid",
                ),
                showlegend=False,
                hoverinfo="skip",
            )
        )


def plot_game_vs_comparison(sc_df: pd.DataFrame, comparison_mov: pd.DataFrame, pitcher_name: str, game_date: date) -> go.Figure:
    sc_df = sc_df.copy()
    sc_df["pfx_x_in"] = sc_df["pfx_x"] * 12.0
    sc_df["pfx_z_in"] = sc_df["pfx_z"] * 12.0

    comparison_label = "Comparison"
    if comparison_mov is not None and not comparison_mov.empty:
        s = comparison_mov["start_date"].iloc[0]
        e = comparison_mov["end_date"].iloc[0]
        comparison_label = s.strftime("%Y-%m-%d") if s == e else f"{s:%Y-%m-%d} to {e:%Y-%m-%d}"

    mlbam_id = int(sc_df["pitcher"].dropna().astype(int).unique()[0])
    headshot_url = (
        "https://img.mlbstatic.com/mlb-photos/image/upload/"
        f"w_180,q_auto:best/v1/people/{mlbam_id}/headshot/67/current"
    )

    fig = go.Figure()
    add_concentric_circles(fig, radii=(12, 24), dashed=(12,))

    fig.add_trace(go.Scatter(
        x=[0, 0], y=[-24, 24], mode="lines",
        line=dict(color="rgba(200,200,200,0.7)", width=1, dash="dot"),
        showlegend=False, hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=[-24, 24], y=[0, 0], mode="lines",
        line=dict(color="rgba(200,200,200,0.7)", width=1),
        showlegend=False, hoverinfo="skip"
    ))

    for pitch_type, sub in sc_df.groupby("pitch_type"):
        avg_velo = sub["release_speed"].mean() if "release_speed" in sub.columns else np.nan
        fig.add_trace(go.Scatter(
            x=sub["pfx_x_in"],
            y=sub["pfx_z_in"],
            mode="markers",
            name=f"{pitch_type} ({avg_velo:.1f})" if pd.notna(avg_velo) else f"{pitch_type}",
            marker=dict(size=12, symbol="diamond", color=pitch_colors.get(pitch_type, "#ffffff")),
            opacity=0.75,
        ))

    if comparison_mov is not None and not comparison_mov.empty:
        for _, row in comparison_mov.iterrows():
            pt = row["pitch_type"]
            fig.add_trace(go.Scatter(
                x=[row["avg_pfx_x_in"]],
                y=[row["avg_pfx_z_in"]],
                mode="markers+text",
                marker=dict(size=15, symbol="star-open", color="white", line=dict(color="white", width=2)),
                text=[pt],
                textposition="top center",
                showlegend=False,
                hovertemplate="Horiz: %{x:.1f} in<br>Vert: %{y:.1f} in<extra></extra>",
            ))

    fig.add_layout_image(dict(
        source=headshot_url, xref="paper", yref="paper",
        x=1.02, y=1.16, sizex=0.18, sizey=0.18,
        xanchor="right", yanchor="top", opacity=1, layer="above"
    ))

    fig.update_layout(
        title=dict(
            text=f"{pitcher_name}<br>Game Pitch Movement {game_date}",
            x=0.5, xanchor="center", y=0.96, yanchor="top",
            font=dict(size=16, color="rgb(200,200,200)"),
        ),
        plot_bgcolor="#0f1419",
        paper_bgcolor="#0f1419",
        font=dict(color="#d7dce2", size=16, family="Arial"),
        legend=dict(
            title_text="Pitchtype (MPH)",
            orientation="v",
            yanchor="top",
            y=1.15,
            xanchor="left",
            x=-0.03,
            bgcolor="rgba(0,0,0,0)",
            tracegroupgap=4,
            itemsizing="constant",
            itemwidth=30,
            font=dict(color="rgba(224,224,224,0.7)"),
            title_font=dict(color="rgba(224,224,224,0.7)"),
        ),
        xaxis=dict(
            range=[24, -24],
            gridcolor="rgba(215,220,226,0.12)",
            zerolinecolor="rgba(215,220,226,0.35)",
            tickfont=dict(color="#d7dce2"),
            constrain="domain",
            scaleanchor="y",
            scaleratio=1,
        ),
        yaxis=dict(
            range=[-24, 24],
            gridcolor="rgba(215,220,226,0.12)",
            zerolinecolor="rgba(215,220,226,0.35)",
            tickfont=dict(color="#d7dce2"),
            scaleanchor="x",
            scaleratio=1,
            constrain="domain",
        ),
        width=800,
        height=800,
        margin=dict(t=70, b=25, l=20, r=20),
    )

    fig.add_annotation(
        text=f"Stars Represent Average Movement From {comparison_label}",
        xref="paper", yref="paper", x=0.5, y=1.081,
        showarrow=False, font=dict(size=12)
    )
    fig.add_annotation(
        text="@Jake3Roy",
        xref="paper", yref="paper", xanchor="left", x=0.90, y=-0.045,
        showarrow=False
    )
    return fig


# -----------------------------
# Streamlit app
# -----------------------------
st.set_page_config(page_title="Pitch Movement", layout="wide")

if st.sidebar.button("Clear cache"):
    st.cache_data.clear()
    st.rerun()

st.title("Pitch Movement by Game (Statcast / Live Fallback)")

st.sidebar.header("Step 1: Select Date")
selected_date = st.sidebar.date_input("Game date", value=date.today())

with st.spinner("Loading schedule..."):
    games_df = get_schedule_for_date(selected_date)

if games_df.empty:
    st.sidebar.write("No MLB games found on this date.")
    st.stop()

games_df["label"] = games_df.apply(
    lambda r: f"{r.away_team} @ {r.home_team} (#{r.game_pk}, {r.status})",
    axis=1,
)

st.sidebar.header("Step 2: Select Game")
selected_game_label = st.sidebar.selectbox("Game", games_df["label"].tolist())
selected_game_pk = int(games_df.loc[games_df["label"] == selected_game_label, "game_pk"].iloc[0])

with st.spinner("Loading pitchers..."):
    pitchers_df = get_pitchers_who_threw_fast(selected_game_pk)

if pitchers_df.empty:
    st.sidebar.write("No pitchers found for this game.")
    st.stop()

pitchers_df["label"] = pitchers_df.apply(lambda r: f"{r['name']} ({r['team']})", axis=1)

st.sidebar.header("Step 3: Select Pitcher")
selected_pitcher_label = st.sidebar.selectbox("Pitcher", pitchers_df["label"].tolist())
selected_pitcher_id = int(pitchers_df.loc[pitchers_df["label"] == selected_pitcher_label, "mlbam_id"].iloc[0])

st.sidebar.header("Step 4: Comparison Range")
default_start = date(2025, 4, 1) if selected_date.year >= 2025 else date(selected_date.year - 1, 1, 1)
default_start = min(default_start, selected_date)

comparison_start = st.sidebar.date_input(
    "Comparison start date",
    value=default_start,
    min_value=date(2015, 1, 1),
    max_value=selected_date,
)

default_end = min(date(comparison_start.year, 12, 31), selected_date)
default_end = max(default_end, comparison_start)

comparison_end = st.sidebar.date_input(
    "Comparison end date",
    value=default_end,
    min_value=comparison_start,
    max_value=selected_date,
)

st.write(f"**Selected date:** {selected_date}")
st.write(f"**Selected pitcher:** {selected_pitcher_label}")

if st.button("Load Statcast and Plot"):
    sc_df = get_pitcher_game_data(selected_game_pk, selected_pitcher_id, selected_date)
    if sc_df is None or sc_df.empty:
        st.error("No pitch data returned for this pitcher/game yet. Try again in a few minutes.")
        st.stop()

    # ensure columns for plot/table
    sc_df = sc_df.copy()
    if "description" not in sc_df.columns:
        sc_df["description"] = ""
    if "zone" not in sc_df.columns:
        sc_df["zone"] = np.nan
    if "extension" not in sc_df.columns:
        sc_df["extension"] = np.nan

    sc_df["pfx_x_in"] = sc_df["pfx_x"] * 12.0
    sc_df["pfx_z_in"] = sc_df["pfx_z"] * 12.0

    with st.spinner("Fetching comparison movement..."):
        comparison_mov = get_comparison_movement(selected_pitcher_id, comparison_start, comparison_end)

    pitcher_name = pitchers_df.loc[pitchers_df["mlbam_id"] == selected_pitcher_id, "name"].iloc[0]
    fig = plot_game_vs_comparison(sc_df, comparison_mov, pitcher_name, selected_date)

    st.plotly_chart(
        fig,
        use_container_width=False,
        key=f"pitch_movement_chart_{selected_game_pk}_{selected_pitcher_id}",
    )

    # -----------------------------
    # Pitch type summary + requested rates
    # -----------------------------
    st.subheader("Pitch Type Summary")

    df = sc_df.copy()
    desc = df["description"].fillna("").astype(str).str.lower()

    df["is_strike"] = desc.isin(STRIKES)
    df["is_swstr"] = desc.isin(SWINGING_STRIKES)
    df["is_called_strike"] = desc.isin(CALLED_STRIKES)
    df["is_swing"] = desc.isin(SWINGS)

    df["is_in_zone"] = df["zone"].apply(is_in_zone)
    df["is_out_zone"] = df["zone"].apply(lambda z: pd.notna(z) and not is_in_zone(z))
    df["is_chase_swing"] = df["is_swing"] & df["is_out_zone"]

    base = df.groupby("pitch_type").agg(
        Pitch_Type=("pitch_type", "first"),
        Count=("pitch_type", "size"),
        MPH=("release_speed", "mean"),
        Extension=("extension", "mean"),
        HB=("pfx_x_in", "mean"),
        iVB=("pfx_z_in", "mean"),
        Strikes=("is_strike", "sum"),
        SwStr=("is_swstr", "sum"),
        CalledStr=("is_called_strike", "sum"),
        InZone=("is_in_zone", "sum"),
        OutZone=("is_out_zone", "sum"),
        ChaseSwings=("is_chase_swing", "sum"),
    ).reset_index(drop=True)

    base["Strike%"] = base["Strikes"] / base["Count"]
    base["SwStr%"] = base["SwStr"] / base["Count"]
    base["CalledStr%"] = base["CalledStr"] / base["Count"]
    base["Zone%"] = base["InZone"] / base["Count"]
    base["Chase%"] = base["ChaseSwings"] / base["OutZone"].replace({0: np.nan})

    # Keep/display columns (rename as you like)
    summary = base[
        [
            "Pitch_Type",
            "Count",
            "MPH",
            "Extension",
            "HB",
            "iVB",
            "Strike%",
            "SwStr%",
            "CalledStr%",
            "Zone%",
            "Chase%",
        ]
    ].rename(
        columns={
            "Pitch_Type": "Pitch Type",
            "Strike%": "Strike Rate",
            "SwStr%": "SwStr Rate",
            "CalledStr%": "Called Strike Rate",
            "Zone%": "Zone Rate",
            "Chase%": "Chase Rate",
        }
    )

    # Convert rates to percent
    for c in ["Strike Rate", "SwStr Rate", "Called Strike Rate", "Zone Rate", "Chase Rate"]:
        summary[c] = summary[c] * 100.0

    # Round for display
    summary = summary.round(
        {
            "MPH": 1,
            "Extension": 2,
            "HB": 2,
            "iVB": 2,
            "Strike Rate": 1,
            "SwStr Rate": 1,
            "Called Strike Rate": 1,
            "Zone Rate": 1,
            "Chase Rate": 1,
        }
    )

    # Style + format
    styled = (
        summary.style
        .format(
            {
                "MPH": "{:.1f}",
                "Extension": "{:.2f}",
                "HB": "{:.2f}",
                "iVB": "{:.2f}",
                "Strike Rate": "{:.1f}%",
                "SwStr Rate": "{:.1f}%",
                "Called Strike Rate": "{:.1f}%",
                "Zone Rate": "{:.1f}%",
                "Chase Rate": "{:.1f}%",
            }
        )
        .apply(shade_row_by_pitch_type, axis=1)
        .set_table_styles([{"selector": "th", "props": [("color", "#E6E6E6")]}])
    )

    st.dataframe(styled, use_container_width=False, hide_index=True)