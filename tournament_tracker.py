"""
BGMI & Free Fire Tournament Points Tracker
==========================================
• SQLite persistence — data survives browser close/reopen
• Two games: BGMI (Rondo→Erangel→Miramar) / Free Fire (Bermuda→Kalahari→Purgatory)
• Multiple named tournaments per game
• Team registration: name + captain phone
• Member registration: Game UID (required), player name (optional), email, captain flag
• Match entry: placement rank per team + individual kills per member (auto-summed)
• Live leaderboard, podium, tiebreakers, match breakdown, top fraggers
• Analytics charts
• Delete at every level (tournament / team / member / match data)
• CSV export of standings

Requirements
------------
    pip install streamlit pandas plotly

Run
---
    streamlit run tournament_tracker.py

Note
----
Data is stored in tournament_tracker.db in the same folder where you run the app.
It persists across browser sessions. For cloud deployments with ephemeral filesystems,
swap the SQLite backend for a hosted database.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DB_PATH = "tournament_tracker.db"

PLACEMENT_POINTS: Dict[int, int] = {1: 10, 2: 6, 3: 5, 4: 4, 5: 3, 6: 2, 7: 1, 8: 1}

GAMES: Dict[str, Dict[str, Any]] = {
    "BGMI": {
        "icon": "🎮",
        "color": "#FF6B35",
        "maps": [("M1", "Rondo"), ("M2", "Erangel"), ("M3", "Miramar")],
    },
    "Free Fire": {
        "icon": "🔥",
        "color": "#FF3E3E",
        "maps": [("M1", "Bermuda"), ("M2", "Kalahari"), ("M3", "Purgatory")],
    },
}

PODIUM_COLORS: Dict[int, str] = {1: "#FFD700", 2: "#C0C0C0", 3: "#CD7F32"}
MEDALS: Dict[int, str] = {1: "🥇", 2: "🥈", 3: "🥉"}

# Demo data per game
_BGMI_DEMO = [
    {"team": "Team Soul",       "phone": "+91 98765 00001", "results": [(1,12),(3,8),(2,10)]},
    {"team": "GodLike Esports", "phone": "+91 98765 00002", "results": [(2,9),(1,14),(5,4)]},
    {"team": "OR Esports",      "phone": "+91 98765 00003", "results": [(5,7),(2,11),(1,9)]},
    {"team": "TSM Entity",      "phone": "+91 98765 00004", "results": [(3,10),(7,5),(4,7)]},
    {"team": "Blind Esports",   "phone": "+91 98765 00005", "results": [(4,6),(6,6),(3,8)]},
    {"team": "Revenant Xspark", "phone": "+91 98765 00006", "results": [(7,4),(4,9),(6,5)]},
    {"team": "Hydra Official",  "phone": "+91 98765 00007", "results": [(6,5),(8,3),(7,6)]},
    {"team": "8Bit Esports",    "phone": "+91 98765 00008", "results": [(8,3),(5,7),(8,2)]},
]
_FF_DEMO = [
    {"team": "Total Gaming",    "phone": "+91 97000 00001", "results": [(1,11),(3,7),(2,9)]},
    {"team": "Desi Gamers",     "phone": "+91 97000 00002", "results": [(2,8),(1,13),(4,5)]},
    {"team": "Gaming Tamizhan", "phone": "+91 97000 00003", "results": [(4,6),(2,10),(1,8)]},
    {"team": "Two Side Gamers", "phone": "+91 97000 00004", "results": [(3,9),(6,4),(5,6)]},
    {"team": "Raistar Squad",   "phone": "+91 97000 00005", "results": [(5,5),(5,6),(3,7)]},
    {"team": "Pahadi Gaming",   "phone": "+91 97000 00006", "results": [(6,4),(4,8),(6,4)]},
    {"team": "Ranked Gaming",   "phone": "+91 97000 00007", "results": [(7,3),(7,3),(7,5)]},
    {"team": "FF Noobs",        "phone": "+91 97000 00008", "results": [(8,2),(8,2),(8,1)]},
]

# ─────────────────────────────────────────────────────────────────────────────
# 2. DATABASE
# ─────────────────────────────────────────────────────────────────────────────

@contextmanager
def _db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with _db() as c:
        c.executescript("""
            CREATE TABLE IF NOT EXISTS tournaments (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                game    TEXT NOT NULL,
                name    TEXT NOT NULL,
                created TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS teams (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                tournament_id INTEGER NOT NULL
                              REFERENCES tournaments(id) ON DELETE CASCADE,
                team_name     TEXT NOT NULL,
                captain_phone TEXT DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS members (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                team_id      INTEGER NOT NULL
                             REFERENCES teams(id) ON DELETE CASCADE,
                display_name TEXT,
                game_uid     TEXT NOT NULL,
                email        TEXT DEFAULT '',
                is_captain   INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS match_results (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                team_id        INTEGER NOT NULL
                               REFERENCES teams(id) ON DELETE CASCADE,
                match_num      INTEGER NOT NULL,
                placement_rank INTEGER DEFAULT 0,
                UNIQUE(team_id, match_num)
            );
            CREATE TABLE IF NOT EXISTS member_kills (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                member_id INTEGER NOT NULL
                          REFERENCES members(id) ON DELETE CASCADE,
                match_num INTEGER NOT NULL,
                kills     INTEGER DEFAULT 0,
                UNIQUE(member_id, match_num)
            );
        """)

# ── Tournaments ──────────────────────────────────────────────────────────────

def db_tournaments(game: str) -> List[sqlite3.Row]:
    with _db() as c:
        return c.execute(
            "SELECT * FROM tournaments WHERE game=? ORDER BY created DESC", (game,)
        ).fetchall()

def db_create_tournament(game: str, name: str) -> int:
    with _db() as c:
        return c.execute(
            "INSERT INTO tournaments(game,name,created) VALUES(?,?,?)",
            (game, name, datetime.now().strftime("%Y-%m-%d %H:%M"))
        ).lastrowid

def db_delete_tournament(tid: int) -> None:
    with _db() as c:
        c.execute("DELETE FROM tournaments WHERE id=?", (tid,))

# ── Teams ────────────────────────────────────────────────────────────────────

def db_teams(tournament_id: int) -> List[sqlite3.Row]:
    with _db() as c:
        return c.execute(
            "SELECT * FROM teams WHERE tournament_id=? ORDER BY id", (tournament_id,)
        ).fetchall()

def db_add_team(tournament_id: int, name: str, phone: str = "") -> int:
    with _db() as c:
        return c.execute(
            "INSERT INTO teams(tournament_id,team_name,captain_phone) VALUES(?,?,?)",
            (tournament_id, name.strip(), phone.strip())
        ).lastrowid

def db_update_team(team_id: int, name: str, phone: str) -> None:
    with _db() as c:
        c.execute(
            "UPDATE teams SET team_name=?,captain_phone=? WHERE id=?",
            (name.strip(), phone.strip(), team_id)
        )

def db_delete_team(team_id: int) -> None:
    with _db() as c:
        c.execute("DELETE FROM teams WHERE id=?", (team_id,))

# ── Members ──────────────────────────────────────────────────────────────────

def db_members(team_id: int) -> List[sqlite3.Row]:
    with _db() as c:
        return c.execute(
            "SELECT * FROM members WHERE team_id=? ORDER BY is_captain DESC, id",
            (team_id,)
        ).fetchall()

def db_add_member(
    team_id: int, game_uid: str,
    display_name: str = "", email: str = "", is_captain: int = 0,
) -> int:
    with _db() as c:
        return c.execute(
            "INSERT INTO members(team_id,display_name,game_uid,email,is_captain)"
            " VALUES(?,?,?,?,?)",
            (team_id, display_name.strip() or None,
             game_uid.strip(), email.strip(), is_captain)
        ).lastrowid

def db_update_member(mid: int, uid: str, name: str, email: str, is_cap: int) -> None:
    with _db() as c:
        c.execute(
            "UPDATE members SET game_uid=?,display_name=?,email=?,is_captain=? WHERE id=?",
            (uid.strip(), name.strip() or None, email.strip(), is_cap, mid)
        )

def db_delete_member(mid: int) -> None:
    with _db() as c:
        c.execute("DELETE FROM members WHERE id=?", (mid,))

# ── Match data ───────────────────────────────────────────────────────────────

def db_upsert_result(team_id: int, match_num: int, rank: int) -> None:
    with _db() as c:
        c.execute("""
            INSERT INTO match_results(team_id,match_num,placement_rank) VALUES(?,?,?)
            ON CONFLICT(team_id,match_num)
            DO UPDATE SET placement_rank=excluded.placement_rank
        """, (team_id, match_num, rank))

def db_upsert_kills(member_id: int, match_num: int, kills: int) -> None:
    with _db() as c:
        c.execute("""
            INSERT INTO member_kills(member_id,match_num,kills) VALUES(?,?,?)
            ON CONFLICT(member_id,match_num)
            DO UPDATE SET kills=excluded.kills
        """, (member_id, match_num, kills))

def db_match_result(team_id: int, match_num: int) -> int:
    with _db() as c:
        row = c.execute(
            "SELECT placement_rank FROM match_results WHERE team_id=? AND match_num=?",
            (team_id, match_num)
        ).fetchone()
    return row["placement_rank"] if row else 0

def db_team_kills(team_id: int, match_num: int) -> int:
    with _db() as c:
        row = c.execute("""
            SELECT COALESCE(SUM(mk.kills), 0) AS total
            FROM members m
            LEFT JOIN member_kills mk
                   ON mk.member_id=m.id AND mk.match_num=?
            WHERE m.team_id=?
        """, (match_num, team_id)).fetchone()
    return row["total"] if row else 0

def db_member_kills_val(member_id: int, match_num: int) -> int:
    with _db() as c:
        row = c.execute(
            "SELECT kills FROM member_kills WHERE member_id=? AND match_num=?",
            (member_id, match_num)
        ).fetchone()
    return row["kills"] if row else 0

def db_clear_match(team_id: int, match_num: int) -> None:
    with _db() as c:
        c.execute(
            "DELETE FROM match_results WHERE team_id=? AND match_num=?",
            (team_id, match_num)
        )
        c.execute("""
            DELETE FROM member_kills
            WHERE member_id IN (SELECT id FROM members WHERE team_id=?)
              AND match_num=?
        """, (team_id, match_num))

# ─────────────────────────────────────────────────────────────────────────────
# 3. DEMO DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_demo_data(tournament_id: int, game: str) -> None:
    demo_list = _BGMI_DEMO if game == "BGMI" else _FF_DEMO
    for idx, d in enumerate(demo_list, 1):
        tid = db_add_team(tournament_id, d["team"], d["phone"])
        # Add 4 members (one captain, three players)
        members_data = [
            (f"UID{idx}0001", f"Captain_{idx}", True),
            (f"UID{idx}0002", f"Player_{idx}A", False),
            (f"UID{idx}0003", f"Player_{idx}B", False),
            (f"UID{idx}0004", f"Player_{idx}C", False),
        ]
        member_ids = []
        for uid, name, cap in members_data:
            mid = db_add_member(tid, uid, name, f"{name.lower()}@demo.com", int(cap))
            member_ids.append(mid)

        # Fill match results (rank, total_kills)
        for mn, (rank, total_kills) in enumerate(d["results"], 1):
            db_upsert_result(tid, mn, rank)
            # Distribute kills across 4 members
            splits = [
                total_kills // 4 + (1 if i < total_kills % 4 else 0)
                for i in range(4)
            ]
            for mid, k in zip(member_ids, splits):
                if k:
                    db_upsert_kills(mid, mn, k)

# ─────────────────────────────────────────────────────────────────────────────
# 4. SCORING
# ─────────────────────────────────────────────────────────────────────────────

def placement_pts(rank: int) -> int:
    return PLACEMENT_POINTS.get(rank, 0)


def build_leaderboard(
    tournament_id: int, maps: List[Tuple[str, str]]
) -> pd.DataFrame:
    teams = db_teams(tournament_id)
    if not teams:
        return pd.DataFrame()

    records: List[Dict[str, Any]] = []
    for t in teams:
        tid = t["id"]
        row: Dict[str, Any] = {"team_id": tid, "Team": t["team_name"]}
        pp_total = kills_total = wwcds = 0

        for mn, (prefix, _) in enumerate(maps, 1):
            rank  = db_match_result(tid, mn)
            kills = db_team_kills(tid, mn)
            pp    = placement_pts(rank)
            row[f"{prefix} Rank"]  = rank
            row[f"{prefix} Kills"] = kills
            row[f"{prefix} Pts"]   = pp + kills
            pp_total    += pp
            kills_total += kills
            if rank == 1:
                wwcds += 1

        m3_rank = db_match_result(tid, 3)
        row.update({
            "Placement Pts": pp_total,
            "Total Kills":   kills_total,
            "Grand Total":   pp_total + kills_total,
            "WWCDs":         wwcds,
            "_m3tb":         m3_rank if m3_rank >= 1 else 99,
        })
        records.append(row)

    lb = pd.DataFrame(records).sort_values(
        ["Grand Total", "Total Kills", "WWCDs", "_m3tb"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    lb.insert(0, "Rank", range(1, len(lb) + 1))
    return lb.drop(columns=["_m3tb"])

# ─────────────────────────────────────────────────────────────────────────────
# 5. UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def styled_lb(lb: pd.DataFrame):
    def _row(row):
        c = PODIUM_COLORS.get(int(row["Rank"]))
        if c:
            return [f"background:{c}22;border-left:4px solid {c};font-weight:700"] * len(row)
        return [""] * len(row)
    return lb.style.apply(_row, axis=1)


def render_podium(lb: pd.DataFrame) -> None:
    if lb.empty or len(lb) < 1:
        return
    cols = st.columns(min(3, len(lb)))
    for i, (_, row) in enumerate(lb.head(3).iterrows()):
        rank  = int(row["Rank"])
        color = PODIUM_COLORS[rank]
        medal = MEDALS[rank]
        with cols[i]:
            st.markdown(f"""
            <div style="border:2px solid {color};border-radius:16px;
                        padding:18px 12px;text-align:center;
                        background:linear-gradient(180deg,{color}26 0%,rgba(0,0,0,0) 100%);
                        box-shadow:0 4px 18px {color}33;margin-bottom:12px">
              <div style="font-size:2rem">{medal}</div>
              <div style="font-size:1.05rem;font-weight:700;color:{color};
                          word-break:break-word">{row['Team']}</div>
              <div style="font-size:1.9rem;font-weight:800;margin-top:4px">
                  {int(row['Grand Total'])}
                  <span style="font-size:.9rem;font-weight:600;opacity:.7"> PTS</span>
              </div>
              <div style="font-size:.85rem;opacity:.85;margin-top:4px">
                  🎯 {int(row['Total Kills'])} kills &nbsp;•&nbsp; 🍗 {int(row['WWCDs'])} WWCD
              </div>
            </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 6. on_change CALLBACKS  (write to DB before Streamlit reruns)
# ─────────────────────────────────────────────────────────────────────────────

def _cb_rank(team_id: int, match_num: int, key: str) -> None:
    db_upsert_result(team_id, match_num, int(st.session_state.get(key, 0)))

def _cb_kills(member_id: int, match_num: int, key: str) -> None:
    db_upsert_kills(member_id, match_num, int(st.session_state.get(key, 0)))

# ─────────────────────────────────────────────────────────────────────────────
# 7. SECTION: REGISTRATION
# ─────────────────────────────────────────────────────────────────────────────

def section_registration(tournament_id: int) -> None:
    st.subheader("🛡️ Team & Member Registration")
    st.caption(
        "Register teams with their captain's phone number, then add each player's "
        "Game UID (required) and name (optional — useful when UID is hard to read)."
    )

    # ── Add new team ─────────────────────────────────────────────────────────
    with st.form("form_add_team", clear_on_submit=True):
        st.markdown("**➕ Register a New Team**")
        fc1, fc2 = st.columns([3, 2])
        t_name  = fc1.text_input("Team Name *", placeholder="e.g. Team Soul", max_chars=40)
        t_phone = fc2.text_input("Captain Phone", placeholder="+91 98765 43210")
        if st.form_submit_button("Add Team", type="primary"):
            name = t_name.strip()
            if not name:
                st.error("Team name is required.")
            elif name.lower() in [t["team_name"].lower() for t in db_teams(tournament_id)]:
                st.error(f"'{name}' already exists.")
            else:
                db_add_team(tournament_id, name, t_phone)
                st.success(f"✅ **{name}** registered!")
                st.rerun()

    st.divider()

    teams = db_teams(tournament_id)
    if not teams:
        st.info("📭 No teams yet. Register one above.")
        return

    st.markdown(f"**{len(teams)} team{'s' if len(teams)>1 else ''} registered**")

    for team in teams:
        tid     = team["id"]
        members = db_members(tid)
        cap     = next((m for m in members if m["is_captain"]), None)

        expander_label = (
            f"{'👑 ' if cap else ''}"
            f"**{team['team_name']}** "
            f" ·  {len(members)} member{'s' if len(members)!=1 else ''}"
            + (f"  ·  📞 {team['captain_phone']}" if team['captain_phone'] else "")
        )

        with st.expander(expander_label, expanded=False):

            # ── Edit / Delete team info ───────────────────────────────────
            with st.form(f"form_edit_team_{tid}"):
                st.markdown("**Team Info**")
                ec1, ec2 = st.columns([3, 2])
                e_name  = ec1.text_input(
                    "Team Name *", value=team["team_name"], max_chars=40, key=f"en_{tid}"
                )
                e_phone = ec2.text_input(
                    "Captain Phone", value=team["captain_phone"] or "", key=f"ep_{tid}"
                )
                sb, db_b = st.columns(2)
                do_save = sb.form_submit_button("💾 Save Changes", type="primary")
                do_del  = db_b.form_submit_button("🗑️ Delete Team")

            if do_save:
                if e_name.strip():
                    db_update_team(tid, e_name, e_phone)
                    st.success("Team info saved.")
                    st.rerun()
                else:
                    st.error("Name cannot be empty.")
            if do_del:
                db_delete_team(tid)
                st.warning(f"Team '{team['team_name']}' deleted.")
                st.rerun()

            st.divider()

            # ── Existing members ──────────────────────────────────────────
            st.markdown("**👥 Members**")

            if not members:
                st.caption("No members yet.")
            else:
                # Header row
                h1, h2, h3, h4, h5, h6 = st.columns([2, 2, 2, 1, 0.6, 0.6])
                h1.markdown("**Game UID \\***")
                h2.markdown("**Name** *(optional)*")
                h3.markdown("**Email**")
                h4.markdown("**Captain**")
                h5.markdown("**Save**")
                h6.markdown("**Del**")
                st.markdown("---")

                for m in members:
                    mid  = m["id"]
                    icon = "👑 " if m["is_captain"] else "▸ "
                    with st.form(f"form_member_{mid}"):
                        mc1, mc2, mc3, mc4, mc5, mc6 = st.columns([2, 2, 2, 1, 0.6, 0.6])
                        u_uid   = mc1.text_input(
                            "UID", value=m["game_uid"],
                            placeholder="UID123", label_visibility="collapsed",
                            key=f"u_uid_{mid}"
                        )
                        u_name  = mc2.text_input(
                            "Name", value=m["display_name"] or "",
                            placeholder=icon + "Name / UID",
                            label_visibility="collapsed", key=f"u_name_{mid}"
                        )
                        u_email = mc3.text_input(
                            "Email", value=m["email"] or "",
                            placeholder="player@email.com",
                            label_visibility="collapsed", key=f"u_email_{mid}"
                        )
                        u_cap   = mc4.checkbox(
                            "👑", value=bool(m["is_captain"]), key=f"u_cap_{mid}"
                        )
                        m_save  = mc5.form_submit_button("💾")
                        m_del   = mc6.form_submit_button("🗑️")

                    if m_save:
                        if u_uid.strip():
                            db_update_member(mid, u_uid, u_name, u_email, int(u_cap))
                            st.success(f"Member updated.")
                            st.rerun()
                        else:
                            st.error("Game UID is required.")
                    if m_del:
                        db_delete_member(mid)
                        st.rerun()

            # ── Add member ────────────────────────────────────────────────
            st.markdown("**➕ Add Member**")
            with st.form(f"form_add_member_{tid}", clear_on_submit=True):
                st.caption(
                    "Game UID is required. "
                    "Leave Name blank to display UID on leaderboard. "
                    "Email is optional."
                )
                ac1, ac2 = st.columns(2)
                ac3, ac4 = st.columns(2)
                a_uid   = ac1.text_input("Game UID *", placeholder="UID123456")
                a_name  = ac2.text_input("Name (optional)", placeholder="PlayerName")
                a_email = ac3.text_input("Email (optional)", placeholder="player@example.com")
                a_cap   = ac4.checkbox("Is Captain?")
                if st.form_submit_button("Add Member", type="primary"):
                    if not a_uid.strip():
                        st.error("Game UID is required.")
                    else:
                        db_add_member(tid, a_uid, a_name, a_email, int(a_cap))
                        st.success(f"Member **{a_name.strip() or a_uid.strip()}** added.")
                        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# 8. SECTION: MATCH RESULTS
# ─────────────────────────────────────────────────────────────────────────────

def section_match_results(tournament_id: int, maps: List[Tuple[str, str]]) -> None:
    teams = db_teams(tournament_id)
    if not teams:
        st.info("📭 No teams registered. Go to **Registration** first.")
        return

    match_labels = {p: f"{p} · {n}" for p, n in maps}
    sel = st.radio(
        "Select Match to Enter / Edit",
        [p for p, _ in maps],
        format_func=lambda p: match_labels[p],
        horizontal=True,
        key=f"match_radio_{tournament_id}",
    )
    mn = next(i for i, (p, _) in enumerate(maps, 1) if p == sel)
    st.caption(
        f"Expand a team to enter its **placement rank** and each member's **kills** "
        f"for **{match_labels[sel]}**. Changes save automatically."
    )
    st.divider()

    for team in teams:
        tid       = team["id"]
        members   = db_members(tid)
        cur_rank  = db_match_result(tid, mn)
        team_kills = db_team_kills(tid, mn)
        pp        = placement_pts(cur_rank)
        rank_key  = f"rank_{tid}_{mn}"

        status = "✅" if cur_rank > 0 else "⬜"
        header = (
            f"{status} **{team['team_name']}**"
            f"  |  Rank: {'—' if cur_rank == 0 else cur_rank}"
            f"  |  Kills: {team_kills}"
            f"  |  Pts: **{pp + team_kills}**"
        )
        with st.expander(header, expanded=False):
            col_r, col_info = st.columns([2, 3])

            with col_r:
                st.number_input(
                    "Placement Rank  (0 = Did Not Play)",
                    min_value=0, max_value=16,
                    value=cur_rank, step=1,
                    key=rank_key,
                    on_change=_cb_rank,
                    args=(tid, mn, rank_key),
                )

            with col_info:
                disp_rank = st.session_state.get(rank_key, cur_rank)
                st.markdown(
                    f"**Placement Pts:** {placement_pts(int(disp_rank))}  \n"
                    f"**Kill Pts:** {team_kills}  \n"
                    f"**Match Total:** {placement_pts(int(disp_rank)) + team_kills}"
                )
                if cur_rank != 0 or team_kills > 0:
                    if st.button(
                        "🗑️ Clear this match data", key=f"clear_{tid}_{mn}",
                        help="Removes rank + all member kills for this team in this match"
                    ):
                        db_clear_match(tid, mn)
                        st.rerun()

            # Per-member kills
            if members:
                st.markdown("**🔫 Member Kills**")
                n_cols = min(len(members), 4)
                mc = st.columns(n_cols)
                for i, m in enumerate(members):
                    mid      = m["id"]
                    kills_key = f"kills_{mid}_{mn}"
                    cur_k    = db_member_kills_val(mid, mn)
                    label    = ("👑 " if m["is_captain"] else "") + (
                        m["display_name"] or m["game_uid"]
                    )
                    with mc[i % n_cols]:
                        st.number_input(
                            label,
                            min_value=0, max_value=100,
                            value=cur_k, step=1,
                            key=kills_key,
                            on_change=_cb_kills,
                            args=(mid, mn, kills_key),
                        )
            else:
                st.warning(
                    "No members registered — kills cannot be tracked. "
                    "Add members in the **Registration** tab."
                )

    st.divider()
    ranked = sum(1 for t in teams if db_match_result(t["id"], mn) > 0)
    st.caption(
        f"✅ **{ranked} / {len(teams)}** teams have a placement recorded "
        f"for **{match_labels[sel]}**."
    )

# ─────────────────────────────────────────────────────────────────────────────
# 9. SECTION: LEADERBOARD
# ─────────────────────────────────────────────────────────────────────────────

def section_leaderboard(
    tournament_id: int, maps: List[Tuple[str, str]]
) -> None:
    lb = build_leaderboard(tournament_id, maps)
    if lb.empty:
        st.info("📭 No data yet. Register teams and enter match results.")
        return

    # ── Podium ───────────────────────────────────────────────────────────────
    st.subheader("🏆 Podium")
    render_podium(lb)
    st.divider()

    # ── Full standings ────────────────────────────────────────────────────────
    st.subheader("📋 Full Standings")
    st.caption(
        "Sorted by Grand Total → Total Kills → WWCDs → M3 Placement (tiebreakers)."
    )
    map_pts = [f"{p} Pts" for p, _ in maps]
    display_cols = ["Rank", "Team"] + map_pts + [
        "WWCDs", "Total Kills", "Placement Pts", "Grand Total"
    ]
    try:
        st.dataframe(
            styled_lb(lb[display_cols]), use_container_width=True, hide_index=True
        )
    except Exception:
        st.dataframe(lb[display_cols], use_container_width=True, hide_index=True)

    csv = lb[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Export Standings (CSV)", csv, "standings.csv", "text/csv")

    st.divider()

    # ── Match breakdown ───────────────────────────────────────────────────────
    st.subheader("📊 Match Breakdown")
    for mn, (prefix, map_name) in enumerate(maps, 1):
        with st.expander(f"**{prefix} — {map_name}**", expanded=False):
            rows = []
            for t in db_teams(tournament_id):
                tid   = t["id"]
                rank  = db_match_result(tid, mn)
                kills = db_team_kills(tid, mn)
                pp    = placement_pts(rank)
                rows.append({
                    "Team":          t["team_name"],
                    "Rank":          rank if rank >= 1 else "—",
                    "Kills":         kills,
                    "Placement Pts": pp,
                    "Match Pts":     pp + kills,
                })
            if rows:
                df = (
                    pd.DataFrame(rows)
                    .sort_values("Match Pts", ascending=False)
                    .reset_index(drop=True)
                )
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.caption("No data recorded.")

    st.divider()

    # ── Top fraggers per match ────────────────────────────────────────────────
    st.subheader("🔫 Top Fraggers per Match")
    for mn, (prefix, map_name) in enumerate(maps, 1):
        with st.expander(f"Top Fraggers — {prefix} · {map_name}", expanded=False):
            rows = []
            for t in db_teams(tournament_id):
                for m in db_members(t["id"]):
                    k = db_member_kills_val(m["id"], mn)
                    rows.append({
                        "Player": m["display_name"] or m["game_uid"],
                        "Team":   t["team_name"],
                        "Kills":  k,
                    })
            if rows:
                df = (
                    pd.DataFrame(rows)
                    .sort_values("Kills", ascending=False)
                    .head(10)
                    .reset_index(drop=True)
                )
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.caption("No kills data.")

    # ── Overall top fraggers ──────────────────────────────────────────────────
    st.divider()
    st.subheader("🏅 Overall Tournament Top Fraggers")
    rows = []
    for t in db_teams(tournament_id):
        for m in db_members(t["id"]):
            total_k = sum(db_member_kills_val(m["id"], mn) for mn in range(1, len(maps)+1))
            rows.append({
                "Player":       m["display_name"] or m["game_uid"],
                "Team":         t["team_name"],
                "Total Kills":  total_k,
                "Is Captain":   "👑" if m["is_captain"] else "",
            })
    if rows:
        df = (
            pd.DataFrame(rows)
            .sort_values("Total Kills", ascending=False)
            .head(15)
            .reset_index(drop=True)
        )
        st.dataframe(df, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# 10. SECTION: ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────

def section_analytics(
    tournament_id: int, maps: List[Tuple[str, str]]
) -> None:
    lb = build_leaderboard(tournament_id, maps)
    if lb.empty:
        st.info("📭 No data to analyse yet.")
        return

    top10 = lb.head(10).copy()

    # ── Bar + Scatter ─────────────────────────────────────────────────────────
    cl, cr = st.columns(2)

    with cl:
        fig_bar = px.bar(
            top10, x="Grand Total", y="Team", orientation="h",
            text="Grand Total", color="Grand Total",
            color_continuous_scale="Turbo",
            title="Top 10 Teams — Total Points",
        )
        fig_bar.update_traces(textposition="outside", marker_line_width=0)
        fig_bar.update_layout(
            yaxis=dict(autorange="reversed", title=""),
            coloraxis_showscale=False, height=400,
            margin=dict(l=10, r=30, t=50, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

    with cr:
        fig_sc = px.scatter(
            lb, x="Placement Pts", y="Total Kills",
            size="Grand Total", color="Team", text="Team",
            size_max=48, title="Placement Points vs Total Kills",
        )
        fig_sc.update_traces(textposition="top center")
        fig_sc.add_vline(
            x=float(lb["Placement Pts"].mean()),
            line_dash="dash", line_color="grey", opacity=0.6
        )
        fig_sc.add_hline(
            y=float(lb["Total Kills"].mean()),
            line_dash="dash", line_color="grey", opacity=0.6
        )
        fig_sc.update_layout(
            height=400, showlegend=False,
            margin=dict(l=10, r=30, t=50, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_sc, use_container_width=True, config={"displayModeBar": False})

    st.caption(
        "Dashed lines = tournament averages. "
        "Top-right quadrant = excels at both survival and fragging."
    )
    st.divider()

    # ── Stacked match-points chart ────────────────────────────────────────────
    pts_cols  = [f"{p} Pts" for p, _ in maps]
    map_names = {f"{p} Pts": f"{p} · {n}" for p, n in maps}
    stacked = top10.melt(
        id_vars=["Team"], value_vars=pts_cols, var_name="Match", value_name="Points"
    )
    stacked["Match"] = stacked["Match"].map(map_names)
    fig_s = px.bar(
        stacked, x="Team", y="Points", color="Match", barmode="stack",
        text="Points", title="Top 10 — Points Contribution by Match",
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig_s.update_traces(textposition="inside", insidetextanchor="middle")
    fig_s.update_layout(
        xaxis=dict(title="", tickangle=-30), height=420,
        margin=dict(l=10, r=30, t=60, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_s, use_container_width=True, config={"displayModeBar": False})
    st.divider()

    # ── Performance summary table ─────────────────────────────────────────────
    with st.expander("🔍 Full Performance Summary", expanded=False):
        s = lb.copy()
        n = len(maps)
        s["Avg Pts/Match"]  = (s["Grand Total"] / n).round(2)
        s["Kills/Match"]    = (s["Total Kills"]  / n).round(2)
        s["Best Match Pts"] = s[pts_cols].max(axis=1)
        show = ["Team", "Grand Total", "Avg Pts/Match", "Kills/Match", "Best Match Pts", "WWCDs"]
        st.dataframe(
            s[show].sort_values("Grand Total", ascending=False).reset_index(drop=True),
            use_container_width=True, hide_index=True,
        )

    # ── Individual player kills chart ─────────────────────────────────────────
    with st.expander("🔫 Player Kill Analysis", expanded=False):
        kill_rows = []
        for t in db_teams(tournament_id):
            for m in db_members(t["id"]):
                for mn, (prefix, _) in enumerate(maps, 1):
                    k = db_member_kills_val(m["id"], mn)
                    kill_rows.append({
                        "Player": m["display_name"] or m["game_uid"],
                        "Team":   t["team_name"],
                        "Match":  prefix,
                        "Kills":  k,
                    })
        if kill_rows:
            kdf = pd.DataFrame(kill_rows)
            top_players = (
                kdf.groupby("Player")["Kills"].sum()
                   .nlargest(12).reset_index()
            )
            fig_p = px.bar(
                top_players, x="Kills", y="Player", orientation="h",
                color="Kills", color_continuous_scale="Oranges",
                title="Top 12 Fraggers — Total Kills (All Matches)",
            )
            fig_p.update_layout(
                yaxis=dict(autorange="reversed"),
                coloraxis_showscale=False, height=420,
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(
                fig_p, use_container_width=True, config={"displayModeBar": False}
            )

# ─────────────────────────────────────────────────────────────────────────────
# 11. MAIN APP
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Tournament Points Tracker",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .block-container { padding-top: 2rem; padding-bottom: 3rem; }
  .stTabs [data-baseweb="tab-list"] { gap: 8px; }
  .stTabs [data-baseweb="tab"] {
    height: 48px; border-radius: 10px 10px 0 0;
    padding: 0 20px; font-weight: 600;
  }
  footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

init_db()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🏆 Tournament Tracker")
    st.caption("BGMI & Free Fire · SQLite · Persistent data")
    st.divider()

    # Game selector
    game = st.radio(
        "Select Game",
        list(GAMES.keys()),
        format_func=lambda g: f"{GAMES[g]['icon']} {g}",
        key="game_selector",
    )
    cfg = GAMES[game]
    st.divider()

    # Tournament management
    st.markdown(f"### {cfg['icon']} {game} Tournaments")
    tournaments   = db_tournaments(game)
    t_options     = {t["name"]: t["id"] for t in tournaments}
    tournament_id: Optional[int] = None
    t_name        = ""

    if t_options:
        sel_name    = st.selectbox("Active Tournament", list(t_options.keys()), key=f"sel_{game}")
        tournament_id = t_options[sel_name]
        t_name      = sel_name
    else:
        st.info("No tournaments yet — create one below.")

    with st.expander("➕ Create Tournament"):
        new_t_input = st.text_input(
            "Name", key=f"newt_{game}", placeholder="e.g. College LAN 2026"
        )
        if st.button("Create", key=f"create_{game}", type="primary"):
            if new_t_input.strip():
                db_create_tournament(game, new_t_input.strip())
                st.success(f"Created '{new_t_input.strip()}'")
                st.rerun()
            else:
                st.error("Name is required.")

    if tournament_id:
        with st.expander("🎲 Load Demo Data"):
            st.caption("Adds 8 demo teams with 4 members each and fills in all match data.")
            if st.button("Load Demo", key=f"demo_{game}_{tournament_id}"):
                existing = db_teams(tournament_id)
                if existing:
                    st.warning("Remove existing teams first.")
                else:
                    load_demo_data(tournament_id, game)
                    st.success("Demo data loaded!")
                    st.rerun()

        with st.expander("⚠️ Danger Zone"):
            st.caption("Delete this entire tournament and all its data. Cannot be undone.")
            if st.button(
                "🗑️ Delete Tournament",
                type="secondary",
                key=f"del_t_{tournament_id}",
                use_container_width=True,
            ):
                db_delete_tournament(tournament_id)
                st.warning("Tournament deleted.")
                st.rerun()

    st.divider()

    with st.expander("ℹ️ Scoring Rules"):
        st.markdown(f"""
        **{game} Maps**  
        {'  ›  '.join(f'{p}: {n}' for p, n in cfg['maps'])}

        **Placement Points**
        | Place | Pts |
        |:-----:|:---:|
        | 1st   | 10  |
        | 2nd   | 6   |
        | 3rd   | 5   |
        | 4th   | 4   |
        | 5th   | 3   |
        | 6th   | 2   |
        | 7th–8th | 1 |
        | 9–16  | 0   |

        **+1 pt per kill**

        **Tie-breakers (in order)**
        1. Grand Total ↑
        2. Total Kills ↑
        3. WWCDs ↑
        4. M3 Placement ↓
        """)

    st.divider()
    st.caption("💾 All data auto-saved to `tournament_tracker.db` in the app folder.")

# ── MAIN CONTENT ──────────────────────────────────────────────────────────────

if tournament_id is None:
    st.title(f"{cfg['icon']} {game} Tournament Tracker")
    st.markdown(
        "### 👈 Create or select a tournament in the sidebar to begin.\n\n"
        "Your data is stored locally in **tournament_tracker.db** and persists "
        "even after you close the browser."
    )
    st.stop()

teams = db_teams(tournament_id)
lb    = build_leaderboard(tournament_id, cfg["maps"])

# Header
st.title(f"{cfg['icon']} {t_name}")
st.caption(
    f"{game}  ·  "
    + "  →  ".join(f"{p} ({n})" for p, n in cfg["maps"])
)

# KPI strip
k1, k2, k3, k4 = st.columns(4)
k1.metric("Teams Registered", len(teams))
if not lb.empty:
    k2.metric("Total Tournament Kills", int(lb["Total Kills"].sum()))
    k3.metric("WWCDs Awarded",          int(lb["WWCDs"].sum()))
    top = lb.iloc[0]
    k4.metric("Current Leader", f"{top['Team']} ({top['Grand Total']} pts)")
else:
    k2.metric("Total Kills", "—")
    k3.metric("WWCDs",       "—")
    k4.metric("Leader",      "—")

st.divider()

# Tabs
tab_reg, tab_match, tab_lb, tab_anal = st.tabs([
    "📝 Registration",
    "🎯 Match Results",
    "🏅 Leaderboard",
    "📊 Analytics",
])

with tab_reg:
    section_registration(tournament_id)

with tab_match:
    section_match_results(tournament_id, cfg["maps"])

with tab_lb:
    section_leaderboard(tournament_id, cfg["maps"])

with tab_anal:
    section_analytics(tournament_id, cfg["maps"])
