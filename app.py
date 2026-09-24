"""
BGMI 1-Day / 3-Match Tournament Points Tracker
===============================================
Maps   : M1 = Rondo, M2 = Erangel, M3 = Miramar
Scoring: Placement (1st=10, 2nd=6, 3rd=5, 4th=4, 5th=3, 6th=2, 7th=1, 8th=1)
         + 1 point per kill.
Tiebreak: Total Pts -> Total Kills -> WWCDs -> Match 3 Placement Rank.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pandas.io.formats.style import Styler


# --------------------------------------------------------------------------- #
#  CONSTANTS
# --------------------------------------------------------------------------- #

PLACEMENT_POINTS: Dict[int, int] = {
    1: 10,
    2: 6,
    3: 5,
    4: 4,
    5: 3,
    6: 2,
    7: 1,
    8: 1,
}

MAX_RANK: int = 16
MIN_RANK: int = 1
DNP_RANK: int = 0

# Match order matters: M3 is the tie-breaker match.
MATCHES: List[Tuple[str, str]] = [
    ("M1", "Rondo"),
    ("M2", "Erangel"),
    ("M3", "Miramar"),
]

MATCH_LABELS: Dict[str, str] = {prefix: f"{prefix} · {name}" for prefix, name in MATCHES}

DATA_COLUMNS: List[str] = [
    "Team",
    "M1 Rank",
    "M1 Kills",
    "M2 Rank",
    "M2 Kills",
    "M3 Rank",
    "M3 Kills",
]

NUMERIC_COLUMNS: List[str] = DATA_COLUMNS[1:]

LEADERBOARD_COLUMNS: List[str] = [
    "Rank",
    "Team",
    "M1 Pts",
    "M2 Pts",
    "M3 Pts",
    "WWCDs",
    "Total Kills",
    "Placement Pts",
    "Grand Total",
]

PODIUM_COLORS: Dict[int, str] = {1: "#FFD700", 2: "#C0C0C0", 3: "#CD7F32"}
MEDALS: Dict[int, str] = {1: "🥇", 2: "🥈", 3: "🥉"}

COLUMN_ALIASES: Dict[str, str] = {
    "team": "Team",
    "team name": "Team",
    "teamname": "Team",
    "name": "Team",
    "m1 rank": "M1 Rank",
    "match 1 rank": "M1 Rank",
    "match1 rank": "M1 Rank",
    "rondo rank": "M1 Rank",
    "m1 placement": "M1 Rank",
    "m1 place": "M1 Rank",
    "m1 kills": "M1 Kills",
    "match 1 kills": "M1 Kills",
    "match1 kills": "M1 Kills",
    "rondo kills": "M1 Kills",
    "m2 rank": "M2 Rank",
    "match 2 rank": "M2 Rank",
    "match2 rank": "M2 Rank",
    "erangel rank": "M2 Rank",
    "m2 placement": "M2 Rank",
    "m2 place": "M2 Rank",
    "m2 kills": "M2 Kills",
    "match 2 kills": "M2 Kills",
    "match2 kills": "M2 Kills",
    "erangel kills": "M2 Kills",
    "m3 rank": "M3 Rank",
    "match 3 rank": "M3 Rank",
    "match3 rank": "M3 Rank",
    "miramar rank": "M3 Rank",
    "m3 placement": "M3 Rank",
    "m3 place": "M3 Rank",
    "m3 kills": "M3 Kills",
    "match 3 kills": "M3 Kills",
    "match3 kills": "M3 Kills",
    "miramar kills": "M3 Kills",
}

DEMO_TEAMS: List[Dict[str, Any]] = [
    {"Team": "Team Soul", "M1 Rank": 1, "M1 Kills": 12, "M2 Rank": 3, "M2 Kills": 8, "M3 Rank": 2, "M3 Kills": 10},
    {"Team": "GodLike Esports", "M1 Rank": 2, "M1 Kills": 9, "M2 Rank": 1, "M2 Kills": 14, "M3 Rank": 5, "M3 Kills": 4},
    {"Team": "OR Esports", "M1 Rank": 5, "M1 Kills": 7, "M2 Rank": 2, "M2 Kills": 11, "M3 Rank": 1, "M3 Kills": 9},
    {"Team": "TSM Entity", "M1 Rank": 3, "M1 Kills": 10, "M2 Rank": 7, "M2 Kills": 5, "M3 Rank": 4, "M3 Kills": 7},
    {"Team": "Blind Esports", "M1 Rank": 4, "M1 Kills": 6, "M2 Rank": 6, "M2 Kills": 6, "M3 Rank": 3, "M3 Kills": 8},
    {"Team": "Revenant Xspark", "M1 Rank": 7, "M1 Kills": 4, "M2 Rank": 4, "M2 Kills": 9, "M3 Rank": 6, "M3 Kills": 5},
    {"Team": "Hyderabad Hydras", "M1 Rank": 6, "M1 Kills": 5, "M2 Rank": 8, "M2 Kills": 3, "M3 Rank": 7, "M3 Kills": 6},
    {"Team": "8Bit Esports", "M1 Rank": 8, "M1 Kills": 3, "M2 Rank": 5, "M2 Kills": 7, "M3 Rank": 8, "M3 Kills": 2},
    {"Team": "Enigma Gaming", "M1 Rank": 10, "M1 Kills": 2, "M2 Rank": 9, "M2 Kills": 4, "M3 Rank": 9, "M3 Kills": 3},
    {"Team": "Velocity Gaming", "M1 Rank": 9, "M1 Kills": 1, "M2 Rank": 10, "M2 Kills": 2, "M3 Rank": 10, "M3 Kills": 1},
]


# --------------------------------------------------------------------------- #
#  CORE LOGIC
# --------------------------------------------------------------------------- #

def safe_int(value: Any, default: int = 0) -> int:
    """Coerce an arbitrary value to int without ever raising."""
    if value is None:
        return default
    if isinstance(value, str):
        value = value.strip().replace(",", "")
        if value == "":
            return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def placement_points(rank: Any) -> int:
    """Return placement points for a finishing rank (0/None => 0 points)."""
    r = safe_int(rank, DNP_RANK)
    return PLACEMENT_POINTS.get(r, 0)


def match_points(rank: Any, kills: Any) -> int:
    """Total points for a single match = placement points + finish points."""
    return placement_points(rank) + max(0, safe_int(kills, 0))


def empty_teams_df() -> pd.DataFrame:
    """A fresh, empty dataframe with the canonical schema."""
    data: Dict[str, pd.Series] = {}
    for column in DATA_COLUMNS:
        dtype = "object" if column == "Team" else "int64"
        data[column] = pd.Series(dtype=dtype)
    return pd.DataFrame(data)


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Force an arbitrary dataframe into the canonical schema + dtypes."""
    if df is None or not isinstance(df, pd.DataFrame):
        return empty_teams_df()

    work = df.copy()
    for column in DATA_COLUMNS:
        if column not in work.columns:
            work[column] = "" if column == "Team" else 0

    work = work[DATA_COLUMNS].copy()
    work["Team"] = (
        work["Team"]
        .fillna("")
        .astype(str)
        .str.strip()
        .replace({"nan": "", "None": ""})
    )
    for column in NUMERIC_COLUMNS:
        work[column] = work[column].apply(lambda v: safe_int(v, 0))
    return work.reset_index(drop=True)


def build_leaderboard(teams_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-match data into a ranked leaderboard.

    Tie-breakers applied in strict order:
        1. Grand Total (desc)
        2. Total Kills (desc)
        3. WWCDs (desc)
        4. Match-3 placement rank (asc - lowest rank number wins)
    """
    records: List[Dict[str, Any]] = []

    for _, row in teams_df.iterrows():
        team = str(row.get("Team", "")).strip()
        if not team or team.lower() in {"nan", "none"}:
            continue

        per_match: Dict[str, int] = {}
        total_kills = 0
        placement_total = 0
        wwcds = 0

        for prefix, _map_name in MATCHES:
            rank = safe_int(row.get(f"{prefix} Rank"), DNP_RANK)
            kills = max(0, safe_int(row.get(f"{prefix} Kills"), 0))
            pp = placement_points(rank)
            per_match[prefix] = pp + kills
            placement_total += pp
            total_kills += kills
            if rank == 1:
                wwcds += 1

        m3_rank = safe_int(row.get("M3 Rank"), DNP_RANK)
        records.append(
            {
                "Team": team,
                "M1 Pts": per_match.get("M1", 0),
                "M2 Pts": per_match.get("M2", 0),
                "M3 Pts": per_match.get("M3", 0),
                "WWCDs": wwcds,
                "Total Kills": total_kills,
                "Placement Pts": placement_total,
                "Grand Total": placement_total + total_kills,
                "_m3_tiebreak": m3_rank if m3_rank >= MIN_RANK else 99,
            }
        )

    if not records:
        return pd.DataFrame(columns=LEADERBOARD_COLUMNS)

    lb = pd.DataFrame(records)
    lb = lb.sort_values(
        by=["Grand Total", "Total Kills", "WWCDs", "_m3_tiebreak"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    lb.insert(0, "Rank", range(1, len(lb) + 1))
    lb = lb.drop(columns=["_m3_tiebreak"])
    return lb[LEADERBOARD_COLUMNS]


def match_summary(teams_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Per-match breakdown table (rank / kills / points) sorted by points."""
    rows: List[Dict[str, Any]] = []
    for _, row in teams_df.iterrows():
        team = str(row.get("Team", "")).strip()
        if not team or team.lower() in {"nan", "none"}:
            continue
        rank = safe_int(row.get(f"{prefix} Rank"), DNP_RANK)
        kills = max(0, safe_int(row.get(f"{prefix} Kills"), 0))
        rows.append(
            {
                "Team": team,
                "Placement": rank if rank >= MIN_RANK else "—",
                "Kills": kills,
                "Placement Pts": placement_points(rank),
                "Match Pts": placement_points(rank) + kills,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=["Team", "Placement", "Kills", "Placement Pts", "Match Pts"]
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["Match Pts", "Kills"], ascending=[False, False])
        .reset_index(drop=True)
    )


def validate_team_name(team_name: str, existing_names: Sequence[str]) -> List[str]:
    """Validate a new team name (uniqueness, non-empty, length)."""
    errors: List[str] = []
    cleaned = team_name.strip()

    if not cleaned:
        errors.append("❌ Team name cannot be empty.")
    elif len(cleaned) > 40:
        errors.append("❌ Team name must be 40 characters or fewer.")
    elif cleaned.lower() in {n.strip().lower() for n in existing_names}:
        errors.append(
            f"❌ A team named **{cleaned}** already exists. "
            "Use the editor below to change it."
        )
    return errors


# --------------------------------------------------------------------------- #
#  STATE MANAGEMENT
# --------------------------------------------------------------------------- #

def init_state() -> None:
    """Initialise every session-state key the app relies on."""
    if "teams_df" not in st.session_state:
        st.session_state.teams_df = empty_teams_df()
    if "editor_version" not in st.session_state:
        st.session_state.editor_version = 0
    st.session_state.teams_df = normalize_dataframe(st.session_state.teams_df)


def bump_editor() -> None:
    """Force the data-editor widget to re-mount with fresh data."""
    st.session_state.editor_version += 1


def load_teams(df: pd.DataFrame) -> None:
    st.session_state.teams_df = normalize_dataframe(df)
    bump_editor()


def apply_match_edits(
    teams_df: pd.DataFrame,
    prefix: str,
    edited_match: pd.DataFrame,
) -> pd.DataFrame:
    """Merge edits from a single-match editor back into the master frame."""
    work = teams_df.copy().reset_index(drop=True)
    edited_match = edited_match.copy().reset_index(drop=True)

    # Match rows by Team name (case-insensitive, stripped).
    work["_key"] = work["Team"].astype(str).str.strip().str.lower()
    edited_match["_key"] = edited_match["Team"].astype(str).str.strip().str.lower()

    rank_col = f"{prefix} Rank"
    kills_col = f"{prefix} Kills"

    lookup = edited_match.set_index("_key")[[rank_col, kills_col]]

    for idx, key in work["_key"].items():
        if key in lookup.index:
            work.at[idx, rank_col] = safe_int(lookup.at[key, rank_col], 0)
            work.at[idx, kills_col] = safe_int(lookup.at[key, kills_col], 0)

    # Append any brand-new teams created inside the match editor.
    existing_keys = set(work["_key"].tolist())
    new_rows: List[Dict[str, Any]] = []
    for _, row in edited_match.iterrows():
        key = row["_key"]
        team_name = str(row.get("Team", "")).strip()
        if not team_name or key in existing_keys:
            continue
        new_rows.append(
            {
                "Team": team_name,
                "M1 Rank": 0,
                "M1 Kills": 0,
                "M2 Rank": 0,
                "M2 Kills": 0,
                "M3 Rank": 0,
                "M3 Kills": 0,
            }
        )
        existing_keys.add(key)

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        new_df.loc[:, rank_col] = [
            safe_int(r.get(rank_col, 0), 0) for r in new_rows
        ]
        new_df.loc[:, kills_col] = [
            safe_int(r.get(kills_col, 0), 0) for r in new_rows
        ]
        work = pd.concat([work, new_df], ignore_index=True)

    return work.drop(columns=["_key"])


# --------------------------------------------------------------------------- #
#  CSV IMPORT / EXPORT
# --------------------------------------------------------------------------- #

def _canonical_column_name(raw: str) -> Optional[str]:
    key = str(raw).strip().lower().replace("_", " ").replace("#", "")
    key = " ".join(key.split())
    if key in COLUMN_ALIASES:
        return COLUMN_ALIASES[key]
    compact = key.replace(" ", "")
    for alias, target in COLUMN_ALIASES.items():
        if alias.replace(" ", "") == compact:
            return target
    return None


def read_uploaded_csv(file: Any) -> Optional[pd.DataFrame]:
    """Parse an uploaded CSV into the canonical schema, or None on failure."""
    try:
        raw = pd.read_csv(file)
    except Exception as exc:
        st.error(f"Could not read the CSV file: {exc}")
        return None

    if raw.empty:
        st.error("The uploaded CSV contains no rows.")
        return None

    rename_map: Dict[str, str] = {}
    for column in raw.columns:
        target = _canonical_column_name(column)
        if target and target not in rename_map.values():
            rename_map[column] = target

    renamed = raw.rename(columns=rename_map)
    if "Team" not in renamed.columns:
        st.error(
            "The CSV must contain a team-name column (e.g. `Team`, `Team Name`). "
            f"Found: {list(raw.columns)}"
        )
        return None

    return normalize_dataframe(renamed)


def leaderboard_to_csv(lb: pd.DataFrame) -> bytes:
    return lb.to_csv(index=False).encode("utf-8")


def teams_to_csv(teams_df: pd.DataFrame) -> bytes:
    return teams_df.to_csv(index=False).encode("utf-8")


# --------------------------------------------------------------------------- #
#  RENDERING HELPERS
# --------------------------------------------------------------------------- #

def render_dataframe(obj: Any, **kwargs: Any) -> None:
    try:
        st.dataframe(obj, width="stretch", hide_index=True, **kwargs)
    except Exception:
        st.dataframe(obj, use_container_width=True, hide_index=True, **kwargs)


def render_editor(obj: pd.DataFrame, **kwargs: Any) -> Any:
    try:
        return st.data_editor(obj, width="stretch", **kwargs)
    except Exception:
        return st.data_editor(obj, use_container_width=True, **kwargs)


def render_chart(fig: go.Figure) -> None:
    try:
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    except Exception:
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def style_leaderboard(lb: pd.DataFrame) -> Styler:
    def _row_style(row: pd.Series) -> List[str]:
        color = PODIUM_COLORS.get(int(row["Rank"]))
        if color:
            style = (
                f"background-color: {color}22; "
                f"border-left: 4px solid {color}; "
                f"font-weight: 700;"
            )
            return [style] * len(row)
        return [""] * len(row)

    return lb.style.apply(_row_style, axis=1)


def render_podium(lb: pd.DataFrame) -> None:
    top = lb.head(3)
    if top.empty:
        return

    columns = st.columns(3)
    for position, (_, row) in enumerate(top.iterrows()):
        rank = int(row["Rank"])
        color = PODIUM_COLORS[rank]
        medal = MEDALS[rank]
        team_name = row["Team"]
        grand_total = int(row["Grand Total"])
        kills = int(row["Total Kills"])
        wwcds = int(row["WWCDs"])

        card_html = (
            f'<div style="'
            f'border: 2px solid {color};'
            f'border-radius: 16px;'
            f'padding: 18px 12px;'
            f'text-align: center;'
            f'background: linear-gradient(180deg, {color}26 0%, rgba(0,0,0,0) 100%);'
            f'box-shadow: 0 4px 18px {color}33;'
            f'margin-bottom: 12px;">'
            f'<div style="font-size: 2rem; line-height: 1;">{medal}</div>'
            f'<div style="font-size: 1.05rem; font-weight: 700; color: {color};'
            f'margin-top: 6px; word-break: break-word;">{team_name}</div>'
            f'<div style="font-size: 1.9rem; font-weight: 800; margin-top: 4px;">'
            f'{grand_total}'
            f'<span style="font-size: 0.9rem; font-weight: 600; opacity: 0.7;"> PTS</span>'
            f'</div>'
            f'<div style="font-size: 0.85rem; opacity: 0.85; margin-top: 4px;">'
            f'🎯 {kills} kills &nbsp;•&nbsp; 🍗 {wwcds} WWCD'
            f'</div>'
            f'</div>'
        )

        with columns[position]:
            st.markdown(card_html, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
#  PAGE CONFIG + STYLES
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title="BGMI Tournament Points Tracker",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container { padding-top: 2rem; padding-bottom: 3rem; }
        div[data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 700; }
        div[data-testid="stMetricLabel"] { font-size: 0.85rem; opacity: 0.75; }
        .stTabs [data-baseweb="tab-list"] { gap: 8px; }
        .stTabs [data-baseweb="tab"] {
            height: 48px;
            border-radius: 10px 10px 0 0;
            padding: 0 20px;
            font-weight: 600;
        }
        footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
#  BOOTSTRAP
# --------------------------------------------------------------------------- #

init_state()
teams_df: pd.DataFrame = st.session_state.teams_df
leaderboard: pd.DataFrame = build_leaderboard(teams_df)


# --------------------------------------------------------------------------- #
#  SIDEBAR
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.markdown("## 🎮 BGMI Points Tracker")
    st.caption("1-Day • 3 Matches • Rondo → Erangel → Miramar")
    st.divider()

    st.markdown("### 📥 Import")
    uploaded_file = st.file_uploader(
        "Upload tournament CSV",
        type=["csv"],
        help="Columns: Team, M1 Rank, M1 Kills, M2 Rank, M2 Kills, M3 Rank, M3 Kills",
    )

    if uploaded_file is not None:
        if st.button("Load CSV into tracker", use_container_width=True):
            parsed = read_uploaded_csv(uploaded_file)
            if parsed is not None:
                load_teams(parsed)
                st.success(f"Loaded {len(parsed)} team(s).")
                st.rerun()

    st.divider()

    st.markdown("### 📤 Export")
    st.download_button(
        "⬇️ Download Leaderboard (CSV)",
        data=leaderboard_to_csv(leaderboard),
        file_name="bgmi_leaderboard.csv",
        mime="text/csv",
        disabled=leaderboard.empty,
        use_container_width=True,
    )
    st.download_button(
        "⬇️ Download Raw Match Data (CSV)",
        data=teams_to_csv(teams_df),
        file_name="bgmi_raw_data.csv",
        mime="text/csv",
        disabled=teams_df.empty,
        use_container_width=True,
    )

    st.divider()

    with st.expander("ℹ️ Scoring & Tie-breakers", expanded=False):
        st.markdown(
            """
            **Maps**
            - M1 · Rondo
            - M2 · Erangel
            - M3 · Miramar

            **Placement Points**

            | Rank | Pts |
            |:----:|:---:|
            | 1st  | 10  |
            | 2nd  | 6   |
            | 3rd  | 5   |
            | 4th  | 4   |
            | 5th  | 3   |
            | 6th  | 2   |
            | 7th  | 1   |
            | 8th  | 1   |
            | 9–16 | 0   |

            **Finish Points:** 1 point per kill

            **Tie-breakers (in order)**
            1. Grand Total (highest)
            2. Total Kills (highest)
            3. WWCDs (highest)
            4. Match 3 placement (lowest rank)
            """
        )

    with st.expander("🗑️ Danger Zone", expanded=False):
        st.caption("These actions cannot be undone.")
        if st.button(
            "Reset tournament data",
            type="secondary",
            use_container_width=True,
        ):
            st.session_state.teams_df = empty_teams_df()
            bump_editor()
            st.success("All data cleared.")
            st.rerun()


# --------------------------------------------------------------------------- #
#  HEADER + KPI STRIP
# --------------------------------------------------------------------------- #

st.title("🏆 BGMI Tournament — Live Points Tracker")
st.caption(
    "Enter each team's **placement rank** and **kills** one match at a time. "
    "Points, WWCDs and the leaderboard update instantly."
)

if not leaderboard.empty:
    total_kills = int(leaderboard["Total Kills"].sum())
    total_wwcds = int(leaderboard["WWCDs"].sum())
    top_row = leaderboard.iloc[0]

    kpi_cols = st.columns(4)
    kpi_cols[0].metric("Teams", len(leaderboard))
    kpi_cols[1].metric("Total Kills", total_kills)
    kpi_cols[2].metric("WWCDs Awarded", total_wwcds)
    kpi_cols[3].metric(
        "Current Leader",
        f"{top_row['Team']} ({top_row['Grand Total']} pts)",
    )

st.divider()


# --------------------------------------------------------------------------- #
#  TABS
# --------------------------------------------------------------------------- #

tab_entry, tab_leaderboard, tab_analytics = st.tabs(
    ["📝  Data Entry", "🏅  Live Leaderboard", "📊  Analytics"]
)


# =========================================================================== #
#  TAB 1 - DATA ENTRY  (per-match)
# =========================================================================== #

with tab_entry:
    st.subheader("1) Teams")
    st.caption("Add team names first, then enter each match's results below.")

    with st.form("add_team_form", clear_on_submit=True):
        name_col, button_col = st.columns([3, 1])
        with name_col:
            team_name = st.text_input(
                "Team Name",
                placeholder="e.g. Team Soul",
                max_chars=40,
                help="Must be unique across the tournament.",
                label_visibility="collapsed",
            )
        with button_col:
            st.write("")
            submitted = st.form_submit_button(
                "➕ Add Team",
                type="primary",
                use_container_width=True,
            )

    if submitted:
        existing = st.session_state.teams_df["Team"].astype(str).tolist()
        problems = validate_team_name(team_name, existing)
        if problems:
            for problem in problems:
                st.error(problem)
        else:
            new_row = {
                "Team": team_name.strip(),
                "M1 Rank": 0,
                "M1 Kills": 0,
                "M2 Rank": 0,
                "M2 Kills": 0,
                "M3 Rank": 0,
                "M3 Kills": 0,
            }
            updated = pd.concat(
                [st.session_state.teams_df, pd.DataFrame([new_row])],
                ignore_index=True,
            )
            load_teams(updated)
            st.success(f"✅ **{team_name.strip()}** added.")
            st.rerun()

    header_left, header_right = st.columns([3, 2])
    with header_left:
        st.caption("Manage the master team list below.")
    with header_right:
        if st.button("🎲 Load demo data", use_container_width=True):
            load_teams(pd.DataFrame(DEMO_TEAMS))
            st.success("Demo tournament loaded.")
            st.rerun()

    # Master team roster (names only, editable)
    roster = st.session_state.teams_df[["Team"]].copy()
    edited_roster = render_editor(
        roster,
        key=f"roster_editor_{st.session_state.editor_version}",
        num_rows="dynamic",
        hide_index=True,
        column_config={
            "Team": st.column_config.TextColumn(
                "Team",
                required=True,
                max_chars=40,
                width="large",
            ),
        },
    )

    # Merge roster changes back while preserving per-match data
    new_roster = normalize_dataframe(
        pd.concat(
            [
                edited_roster,
                st.session_state.teams_df.drop(columns=["Team"]),
            ],
            axis=1,
        )
    ) if not edited_roster.empty else empty_teams_df()

    # If rows were removed, drop them from the master
    kept_names = set(
        edited_roster["Team"].astype(str).str.strip().str.lower().tolist()
    )
    if kept_names:
        filtered = st.session_state.teams_df[
            st.session_state.teams_df["Team"].astype(str).str.strip().str.lower().isin(kept_names)
        ].reset_index(drop=True)
    else:
        filtered = empty_teams_df()

    # Preserve any brand-new team names added via the roster editor
    existing_keys = set(
        filtered["Team"].astype(str).str.strip().str.lower().tolist()
    )
    new_rows: List[Dict[str, Any]] = []
    for _, row in edited_roster.iterrows():
        name = str(row.get("Team", "")).strip()
        key = name.lower()
        if not name or key in existing_keys:
            continue
        new_rows.append(
            {
                "Team": name,
                "M1 Rank": 0,
                "M1 Kills": 0,
                "M2 Rank": 0,
                "M2 Kills": 0,
                "M3 Rank": 0,
                "M3 Kills": 0,
            }
        )
        existing_keys.add(key)

    if new_rows:
        filtered = pd.concat([filtered, pd.DataFrame(new_rows)], ignore_index=True)

    st.session_state.teams_df = normalize_dataframe(filtered)
    teams_df = st.session_state.teams_df

    duplicates = (
        teams_df.loc[teams_df["Team"] != "", "Team"]
        .str.lower()
        .value_counts()
        .loc[lambda s: s > 1]
    )
    if not duplicates.empty:
        duplicate_list = ", ".join(f"**{name}**" for name in duplicates.index)
        st.warning(f"⚠️ Duplicate team names: {duplicate_list}. Please resolve them.")

    st.divider()

    # ------------------------------------------------------------------ #
    #  Match-specific entry
    # ------------------------------------------------------------------ #
    st.subheader("2) Match Results")
    st.caption(
        "Pick a match, then enter **Rank** and **Kills** for every team. "
        "Use **0** in a rank field to mark a match the team did not play."
    )

    selected_match = st.radio(
        "Select match to enter / edit",
        options=[prefix for prefix, _ in MATCHES],
        format_func=lambda p: MATCH_LABELS[p],
        horizontal=True,
        key="selected_match_radio",
    )

    rank_col = f"{selected_match} Rank"
    kills_col = f"{selected_match} Kills"

    if teams_df.empty:
        st.info("📭 Add teams above first, then come back to enter match results.")
    else:
        match_view = teams_df[["Team", rank_col, kills_col]].copy()

        edited_match = render_editor(
            match_view,
            key=f"match_editor_{selected_match}_{st.session_state.editor_version}",
            num_rows="dynamic",
            hide_index=True,
            column_config={
                "Team": st.column_config.TextColumn(
                    "Team",
                    required=True,
                    max_chars=40,
                    width="large",
                ),
                rank_col: st.column_config.NumberColumn(
                    f"{MATCH_LABELS[selected_match]} — Rank",
                    min_value=DNP_RANK,
                    max_value=MAX_RANK,
                    step=1,
                    help="0 = did not play",
                    width="small",
                ),
                kills_col: st.column_config.NumberColumn(
                    f"{MATCH_LABELS[selected_match]} — Kills",
                    min_value=0,
                    max_value=100,
                    step=1,
                    width="small",
                ),
            },
        )

        merged = apply_match_edits(teams_df, selected_match, edited_match)
        st.session_state.teams_df = normalize_dataframe(merged)
        teams_df = st.session_state.teams_df

        entered = int(
            (teams_df[rank_col] > 0).sum()
        )
        st.caption(
            f"✅ **{entered}** team(s) have a placement recorded for "
            f"**{MATCH_LABELS[selected_match]}**."
        )


# =========================================================================== #
#  TAB 2 - LIVE LEADERBOARD
# =========================================================================== #

with tab_leaderboard:
    if leaderboard.empty:
        st.info(
            "📭 No teams yet. Add teams in the **Data Entry** tab "
            "to see the leaderboard."
        )
    else:
        st.subheader("Podium")
        render_podium(leaderboard)

        st.divider()

        st.subheader("Full Standings")
        st.caption(
            "M1 = Rondo · M2 = Erangel · M3 = Miramar. "
            "Sorted by **Total Points → Total Kills → WWCDs → Match 3 Placement**."
        )
        render_dataframe(style_leaderboard(leaderboard))

        st.divider()

        st.subheader("Match-by-Match Breakdown")
        for prefix, map_name in MATCHES:
            with st.expander(f"**{prefix} — {map_name}**", expanded=False):
                summary = match_summary(teams_df, prefix)
                if summary.empty:
                    st.caption("No data recorded for this match.")
                else:
                    render_dataframe(summary)

        st.divider()
        st.caption(
            "ℹ️ Placement points: 1st=10, 2nd=6, 3rd=5, 4th=4, 5th=3, 6th=2, "
            "7th=1, 8th=1, 9th–16th=0. Each kill is worth 1 point."
        )


# =========================================================================== #
#  TAB 3 - ANALYTICS
# =========================================================================== #

with tab_analytics:
    if leaderboard.empty:
        st.info("📭 No data to analyse yet. Add teams in the **Data Entry** tab.")
    else:
        st.subheader("Tournament Analytics")

        chart_col_left, chart_col_right = st.columns(2)

        with chart_col_left:
            top10 = leaderboard.head(10).copy()
            fig_bar = px.bar(
                top10,
                x="Grand Total",
                y="Team",
                orientation="h",
                text="Grand Total",
                color="Grand Total",
                color_continuous_scale="Turbo",
                title="Top 10 Teams — Total Points",
            )
            fig_bar.update_traces(
                textposition="outside",
                marker_line_width=0,
                hovertemplate="<b>%{y}</b><br>Total Points: %{x}<extra></extra>",
            )
            fig_bar.update_layout(
                yaxis=dict(autorange="reversed", title=""),
                xaxis=dict(title="Total Points"),
                coloraxis_showscale=False,
                height=460,
                margin=dict(l=10, r=30, t=60, b=20),
                plot_bgcolor="rgba(0,0,0,0)",
            )
            render_chart(fig_bar)

        with chart_col_right:
            fig_scatter = px.scatter(
                leaderboard,
                x="Placement Pts",
                y="Total Kills",
                size="Grand Total",
                color="Team",
                text="Team",
                size_max=48,
                title="Placement Points vs Total Kills",
                hover_data={
                    "Grand Total": True,
                    "WWCDs": True,
                    "Placement Pts": True,
                    "Total Kills": True,
                },
            )
            fig_scatter.update_traces(
                textposition="top center",
                marker=dict(
                    opacity=0.85,
                    line=dict(width=1, color="rgba(255,255,255,0.5)"),
                ),
            )
            mean_placement = float(leaderboard["Placement Pts"].mean())
            mean_kills = float(leaderboard["Total Kills"].mean())
            fig_scatter.add_vline(
                x=mean_placement,
                line_dash="dash",
                line_color="grey",
                opacity=0.6,
            )
            fig_scatter.add_hline(
                y=mean_kills,
                line_dash="dash",
                line_color="grey",
                opacity=0.6,
            )
            fig_scatter.update_layout(
                xaxis=dict(title="Total Placement Points"),
                yaxis=dict(title="Total Kills"),
                height=460,
                showlegend=False,
                margin=dict(l=10, r=30, t=60, b=20),
                plot_bgcolor="rgba(0,0,0,0)",
            )
            render_chart(fig_scatter)

        st.caption(
            "Dashed reference lines mark the tournament average for each axis — "
            "teams in the top-right quadrant excel at both survival and fragging."
        )

        st.divider()

        stacked = leaderboard.head(10).melt(
            id_vars=["Team"],
            value_vars=["M1 Pts", "M2 Pts", "M3 Pts"],
            var_name="Match",
            value_name="Points",
        )
        map_lookup = {
            "M1 Pts": "M1 · Rondo",
            "M2 Pts": "M2 · Erangel",
            "M3 Pts": "M3 · Miramar",
        }
        stacked["Match"] = stacked["Match"].map(map_lookup)

        fig_stack = px.bar(
            stacked,
            x="Team",
            y="Points",
            color="Match",
            title="Top 10 Teams — Points Contribution by Match",
            barmode="stack",
            color_discrete_sequence=px.colors.qualitative.Bold,
            text="Points",
        )
        fig_stack.update_traces(textposition="inside", insidetextanchor="middle")
        fig_stack.update_layout(
            xaxis=dict(title="", tickangle=-30),
            yaxis=dict(title="Points"),
            height=470,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
            margin=dict(l=10, r=30, t=80, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
        )
        render_chart(fig_stack)

        st.divider()

        with st.expander("🔍 Team Performance Summary", expanded=False):
            summary = leaderboard.copy()
            summary["Avg Pts / Match"] = (
                summary["Grand Total"] / len(MATCHES)
            ).round(2)
            summary["Kills / Match"] = (
                summary["Total Kills"] / len(MATCHES)
            ).round(2)
            summary["Best Match Pts"] = summary[
                ["M1 Pts", "M2 Pts", "M3 Pts"]
            ].max(axis=1)
            render_dataframe(
                summary[
                    [
                        "Team",
                        "Grand Total",
                        "Avg Pts / Match",
                        "Kills / Match",
                        "Best Match Pts",
                        "WWCDs",
                    ]
                ]
                .sort_values("Grand Total", ascending=False)
                .reset_index(drop=True)
            )