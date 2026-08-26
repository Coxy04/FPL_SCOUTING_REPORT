import difflib
import json
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import pandas as pd
import requests
from understatapi import UnderstatClient


BASE_URL = "https://fantasy.premierleague.com/api"
ARCHIVE_BASE_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
OUTPUT_FILE = Path("fpl_ml_predictions.csv")
GEMS_OUTPUT_FILE = Path("fpl_ml_hidden_gems.csv")
META_OUTPUT_FILE = Path("fpl_ml_meta.json")
GEM_OWNERSHIP_MAX = 10.0
GEM_MIN_PREDICTED_POINTS = 3.0
POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
UNDERSTAT_SEASON = "2026"
PRIOR_SEASON_UNDERSTAT = "2025"
PRIOR_SEASON_ARCHIVE = "2025-26"
TEAM_FORM_WINDOW = 6
NAME_MATCH_CUTOFF = 0.85
PRIOR_SEASON_NAME_MATCH_CUTOFF = 0.9

UNDERSTAT_TEAM_MAP = {
    "Arsenal": "Arsenal",
    "Aston Villa": "Aston Villa",
    "Bournemouth": "Bournemouth",
    "Brentford": "Brentford",
    "Brighton": "Brighton",
    "Burnley": "Burnley",
    "Chelsea": "Chelsea",
    "Coventry City": "Coventry",
    "Crystal Palace": "Crystal Palace",
    "Everton": "Everton",
    "Fulham": "Fulham",
    "Hull City": "Hull",
    "Ipswich Town": "Ipswich",
    "Leeds": "Leeds",
    "Liverpool": "Liverpool",
    "Man City": "Manchester City",
    "Man Utd": "Manchester United",
    "Newcastle": "Newcastle United",
    "Nott'm Forest": "Nottingham Forest",
    "Spurs": "Tottenham",
    "Sunderland": "Sunderland",
    "West Ham": "West Ham",
    "Wolves": "Wolverhampton Wanderers",
}

FEATURES = [
    "minutes",
    "minutes_sd",
    "expected_goals",
    "expected_assists",
    "expected_goals_conceded",
    "bonus",
    "was_home",
    "difficulty",
    "now_cost",
    "team_xg_for_form",
    "team_xg_against_form",
    "opp_xg_for_form",
    "opp_xg_against_form",
    "npxg90",
    "xa90",
    "xgchain90",
    "position_GK",
    "position_DEF",
    "position_MID",
    "position_FWD",
]


def get_json(session, endpoint):
    response = session.get(f"{BASE_URL}/{endpoint}", timeout=30)
    response.raise_for_status()
    return response.json()


def make_fixture_lookup(fixtures):
    lookup = {}
    for fixture in fixtures:
        lookup[fixture["id"]] = (
            fixture["team_h_difficulty"],
            fixture["team_a_difficulty"],
        )
    return lookup


def normalize_name(name):
    stripped = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    stripped = stripped.replace("'", "")
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in stripped)
    return " ".join(cleaned.split())


def get_understat_team_matches(understat, season):
    league = understat.league("EPL")
    matches = league.get_match_data(season=season)
    rows = []
    for match in matches:
        if not match["isResult"]:
            continue
        date = match["datetime"][:10]
        home, away = match["h"]["title"], match["a"]["title"]
        xg_h, xg_a = float(match["xG"]["h"]), float(match["xG"]["a"])
        rows.append({"team": home, "date": date, "xg_for": xg_h, "xg_against": xg_a})
        rows.append({"team": away, "date": date, "xg_for": xg_a, "xg_against": xg_h})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(["team", "date"]).reset_index(drop=True)


def add_prematch_team_form(frame):
    frame = frame.copy()
    grouped = frame.groupby("team")
    frame["team_xg_for_form"] = grouped["xg_for"].transform(
        lambda s: s.shift(1).rolling(TEAM_FORM_WINDOW, min_periods=1).mean()
    )
    frame["team_xg_against_form"] = grouped["xg_against"].transform(
        lambda s: s.shift(1).rolling(TEAM_FORM_WINDOW, min_periods=1).mean()
    )
    return frame


def build_team_form_lookup(frame):
    lookup = {}
    for row in frame.itertuples():
        lookup[(row.team, row.date)] = (row.team_xg_for_form, row.team_xg_against_form)
    return lookup


def build_current_team_form(frame):
    current = {}
    for team, group in frame.groupby("team"):
        tail = group.sort_values("date").tail(TEAM_FORM_WINDOW)
        current[team] = (tail["xg_for"].mean(), tail["xg_against"].mean())
    return current


def get_understat_player_stats(understat, season):
    league = understat.league("EPL")
    players = league.get_player_data(season=season)
    stats = []
    for player in players:
        minutes = float(player.get("time") or 0)
        if minutes <= 0:
            continue
        per90 = 90.0 / minutes
        stats.append(
            {
                "understat_id": player["id"],
                "name": player["player_name"],
                "teams": [team.strip() for team in player["team_title"].split(",")],
                "npxg90": float(player.get("npxG") or 0) * per90,
                "xa90": float(player.get("xA") or 0) * per90,
                "xgchain90": float(player.get("xGChain") or 0) * per90,
            }
        )
    return stats


def match_understat_players(elements, understat_players):
    by_team = {}
    for entry in understat_players:
        for team in entry["teams"]:
            by_team.setdefault(team, []).append(entry)

    matched = {}
    unmatched = []
    for player in elements:
        understat_team = UNDERSTAT_TEAM_MAP.get(player["team_name"])
        candidates = by_team.get(understat_team, [])
        full_name = normalize_name(f"{player['first_name']} {player['second_name']}")
        web_name = normalize_name(player["web_name"])

        best = None
        for entry in candidates:
            candidate_name = normalize_name(entry["name"])
            if candidate_name == full_name or candidate_name == web_name:
                best = entry
                break

        if best is None and candidates:
            # FPL stores full legal names (e.g. "Bruno Borges Fernandes") while Understat
            # often uses a shorter common name (e.g. "Bruno Fernandes"), so exact matches
            # miss real players. web_name is almost always already just the player's common
            # surname, so try that alone first -- combining it with full_name tokens risks a
            # false collision (e.g. "Raya Martin" spuriously overlapping teammate "Martin
            # Odegaard" on the token "martin"). Only fall back to full_name tokens if that fails.
            def token_overlap_matches(tokens):
                matches = []
                for entry in candidates:
                    if tokens & set(normalize_name(entry["name"]).split()):
                        matches.append(entry)
                return matches

            web_tokens = {tok for tok in web_name.split() if len(tok) >= 3}
            token_matches = token_overlap_matches(web_tokens) if web_tokens else []

            if len(token_matches) != 1:
                full_tokens = {tok for tok in full_name.split() if len(tok) >= 3}
                token_matches = token_overlap_matches(full_tokens) if full_tokens else []

            if len(token_matches) == 1:
                best = token_matches[0]
            elif len(token_matches) > 1:
                names = [normalize_name(entry["name"]) for entry in token_matches]
                close = difflib.get_close_matches(full_name, names, n=1, cutoff=NAME_MATCH_CUTOFF)
                if close:
                    best = token_matches[names.index(close[0])]

        if best is None and candidates:
            candidate_names = [normalize_name(entry["name"]) for entry in candidates]
            close = difflib.get_close_matches(full_name, candidate_names, n=1, cutoff=NAME_MATCH_CUTOFF)
            if not close:
                close = difflib.get_close_matches(web_name, candidate_names, n=1, cutoff=NAME_MATCH_CUTOFF)
            if close:
                best = candidates[candidate_names.index(close[0])]

        if best is not None:
            matched[player["id"]] = best
        elif player.get("minutes", 0) > 0:
            unmatched.append(f"{player['web_name']} ({player['team_name']})")

    return matched, unmatched


def load_prior_season_archive():
    base = f"{ARCHIVE_BASE_URL}/{PRIOR_SEASON_ARCHIVE}"
    merged_gw = pd.read_csv(f"{base}/gws/merged_gw.csv", encoding="utf-8", encoding_errors="ignore")
    fixtures = pd.read_csv(f"{base}/fixtures.csv", encoding="utf-8", encoding_errors="ignore")
    teams = pd.read_csv(f"{base}/teams.csv", encoding="utf-8", encoding_errors="ignore")
    player_idlist = pd.read_csv(f"{base}/player_idlist.csv", encoding="utf-8", encoding_errors="ignore")

    difficulty_lookup = {
        row.id: (row.team_h_difficulty, row.team_a_difficulty) for row in fixtures.itertuples()
    }
    team_names = {row.id: row.name for row in teams.itertuples()}
    return merged_gw, difficulty_lookup, team_names, player_idlist


def match_prior_season_players(elements, player_idlist):
    full_name_to_id = {}
    web_name_to_id = {}
    for player in elements:
        full_name_to_id[normalize_name(f"{player['first_name']} {player['second_name']}")] = player["id"]
        web_name_to_id.setdefault(normalize_name(player["web_name"]), player["id"])

    all_full_names = list(full_name_to_id)

    id_map = {}
    unmatched = []
    for row in player_idlist.itertuples():
        full_name = normalize_name(f"{row.first_name} {row.second_name}")
        current_id = full_name_to_id.get(full_name) or web_name_to_id.get(full_name)
        if current_id is None:
            close = difflib.get_close_matches(full_name, all_full_names, n=1, cutoff=PRIOR_SEASON_NAME_MATCH_CUTOFF)
            if close:
                current_id = full_name_to_id[close[0]]
        if current_id is not None:
            id_map[row.id] = current_id
        else:
            unmatched.append(f"{row.first_name} {row.second_name}")

    return id_map, unmatched


def build_prior_season_rows(merged_gw, difficulty_lookup, team_names, id_map, team_form_lookup, understat_matches, prior_per90_by_understat_id):
    gw = merged_gw.copy()
    for column in LAGGED_MATCH_STATS:
        gw[column] = pd.to_numeric(gw[column], errors="coerce").fillna(0)
    gw = gw.sort_values(["element", "GW"]).reset_index(drop=True)
    gw["actual_minutes"] = gw["minutes"]
    gw = add_lagged_match_form(gw, group_col="element")

    rows = []
    for row in gw.itertuples():
        current_id = id_map.get(row.element)
        if current_id is None or row.actual_minutes <= 0:
            continue

        was_home = bool(row.was_home)
        difficulty_pair = difficulty_lookup.get(row.fixture, (3, 3))
        date = str(row.kickoff_time)[:10]
        own_understat_team = UNDERSTAT_TEAM_MAP.get(row.team)
        opponent_name = team_names.get(row.opponent_team, "")
        opp_understat_team = UNDERSTAT_TEAM_MAP.get(opponent_name)
        own_form = team_form_lookup.get((own_understat_team, date), (0, 0))
        opp_form = team_form_lookup.get((opp_understat_team, date), (0, 0))

        entry = understat_matches.get(current_id)
        prior_stats = prior_per90_by_understat_id.get(entry["understat_id"], {}) if entry else {}

        rows.append(
            {
                "minutes": row.minutes,
                "minutes_sd": row.minutes_sd,
                "expected_goals": row.expected_goals,
                "expected_assists": row.expected_assists,
                "expected_goals_conceded": row.expected_goals_conceded,
                "bonus": row.bonus,
                "was_home": was_home,
                "difficulty": difficulty_pair[0] if was_home else difficulty_pair[1],
                "now_cost": row.value,
                "team_xg_for_form": own_form[0],
                "team_xg_against_form": own_form[1],
                "opp_xg_for_form": opp_form[0],
                "opp_xg_against_form": opp_form[1],
                "npxg90": prior_stats.get("npxg90", 0),
                "xa90": prior_stats.get("xa90", 0),
                "xgchain90": prior_stats.get("xgchain90", 0),
                "position": row.position,
                "total_points": row.total_points,
                "source": f"fpl_archive_{PRIOR_SEASON_ARCHIVE}",
            }
        )
    return pd.DataFrame(rows)


def add_features(frame):
    frame = frame.copy()
    numeric = [
        "minutes",
        "minutes_sd",
        "expected_goals",
        "expected_assists",
        "expected_goals_conceded",
        "bonus",
        "difficulty",
        "now_cost",
        "team_xg_for_form",
        "team_xg_against_form",
        "opp_xg_for_form",
        "opp_xg_against_form",
        "npxg90",
        "xa90",
        "xgchain90",
    ]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)
    frame["was_home"] = frame["was_home"].astype(int)
    positions = pd.get_dummies(frame["position"], prefix="position")
    frame = pd.concat([frame, positions], axis=1)
    for column in FEATURES:
        if column not in frame:
            frame[column] = 0
    return frame


LAGGED_MATCH_STATS = ["minutes", "expected_goals", "expected_assists", "expected_goals_conceded", "bonus"]


def add_lagged_match_form(frame, group_col=None):
    # Match-level stats (minutes, xG, xA, xGC, bonus) must be lagged to a pre-match rolling
    # average, not the actual value from that same match -- using the same match's own actual
    # bonus/minutes/etc. to predict that match's own total_points is target leakage (bonus
    # especially, since it's literally summed into total_points). Predictions already use a
    # rolling average of past matches as a form estimate, since the future is unknown; training
    # must use the identical definition or the feature means something different in each context.
    frame = frame.copy()

    def rolling_form(series):
        return series.shift(1).rolling(TEAM_FORM_WINDOW, min_periods=1).mean()

    def rolling_sd(series):
        return series.shift(1).rolling(TEAM_FORM_WINDOW, min_periods=1).std()

    if group_col:
        rolling_mean = frame.groupby(group_col)[LAGGED_MATCH_STATS].transform(rolling_form)
        minutes_sd = frame.groupby(group_col)["minutes"].transform(rolling_sd)
    else:
        rolling_mean = frame[LAGGED_MATCH_STATS].apply(rolling_form)
        minutes_sd = rolling_sd(frame["minutes"])

    for column in LAGGED_MATCH_STATS:
        frame[column] = rolling_mean[column]
    frame["minutes_sd"] = minutes_sd.fillna(0)
    return frame


def build_training_rows(elements, fixtures, teams, session, team_form_lookup, understat_matches):
    difficulty_lookup = make_fixture_lookup(fixtures)
    rows = []
    for player in elements:
        position = POSITION_MAP.get(player["element_type"])
        player_stats = understat_matches.get(player["id"], {})
        history = get_json(session, f"element-summary/{player['id']}/").get("history", [])
        time.sleep(0.05)
        if not history:
            continue

        hist_frame = pd.DataFrame(history)
        for column in LAGGED_MATCH_STATS:
            if column not in hist_frame:
                hist_frame[column] = 0
            hist_frame[column] = pd.to_numeric(hist_frame[column], errors="coerce").fillna(0)
        hist_frame = add_lagged_match_form(hist_frame)

        for idx, match in enumerate(history):
            fixture_id = match.get("fixture")
            difficulty_pair = difficulty_lookup.get(fixture_id)
            if difficulty_pair is None:
                continue
            if not match.get("minutes", 0):
                continue
            was_home = bool(match.get("was_home"))
            date = (match.get("kickoff_time") or "")[:10]
            own_understat_team = UNDERSTAT_TEAM_MAP.get(player["team_name"])
            opponent_name = teams.get(match.get("opponent_team"), "")
            opp_understat_team = UNDERSTAT_TEAM_MAP.get(opponent_name)
            own_form = team_form_lookup.get((own_understat_team, date), (0, 0))
            opp_form = team_form_lookup.get((opp_understat_team, date), (0, 0))
            lagged = hist_frame.iloc[idx]
            rows.append(
                {
                    "minutes": lagged["minutes"],
                    "minutes_sd": lagged["minutes_sd"],
                    "expected_goals": lagged["expected_goals"],
                    "expected_assists": lagged["expected_assists"],
                    "expected_goals_conceded": lagged["expected_goals_conceded"],
                    "bonus": lagged["bonus"],
                    "was_home": was_home,
                    "difficulty": difficulty_pair[0] if was_home else difficulty_pair[1],
                    "now_cost": player.get("now_cost", 0),
                    "team_xg_for_form": own_form[0],
                    "team_xg_against_form": own_form[1],
                    "opp_xg_for_form": opp_form[0],
                    "opp_xg_against_form": opp_form[1],
                    "npxg90": player_stats.get("npxg90", 0),
                    "xa90": player_stats.get("xa90", 0),
                    "xgchain90": player_stats.get("xgchain90", 0),
                    "position": position,
                    "total_points": match.get("total_points", 0),
                }
            )
    return pd.DataFrame(rows)


def recent_player_features(elements, fixtures, teams, session, current_team_form, understat_matches):
    difficulty_lookup = make_fixture_lookup(fixtures)
    rows = []
    for player in elements:
        player_stats = understat_matches.get(player["id"], {})
        history = get_json(session, f"element-summary/{player['id']}/").get("history", [])
        history = history[-6:]
        if not history:
            continue
        history_frame = pd.DataFrame(history)
        for column in ("expected_goals", "expected_assists", "expected_goals_conceded"):
            if column in history_frame:
                history_frame[column] = pd.to_numeric(history_frame[column], errors="coerce")
        averages = history_frame.mean(numeric_only=True).to_dict()
        minutes_sd = pd.to_numeric(history_frame.get("minutes"), errors="coerce").std()
        minutes_sd = 0 if pd.isna(minutes_sd) else minutes_sd

        season_minutes = player.get("minutes", 0) or 0
        season_goals = player.get("goals_scored", 0) or 0
        actual_goals_per90 = (season_goals / season_minutes * 90) if season_minutes > 0 else 0

        own_understat_team = UNDERSTAT_TEAM_MAP.get(player["team_name"])
        own_form = current_team_form.get(own_understat_team, (0, 0))
        for fixture in fixtures:
            if fixture["finished"]:
                continue
            if player["team"] not in (fixture["team_h"], fixture["team_a"]):
                continue
            was_home = player["team"] == fixture["team_h"]
            pair = difficulty_lookup[fixture["id"]]
            opponent_id = fixture["team_a"] if was_home else fixture["team_h"]
            opp_understat_team = UNDERSTAT_TEAM_MAP.get(teams.get(opponent_id, ""))
            opp_form = current_team_form.get(opp_understat_team, (0, 0))
            row = {
                "id": player["id"],
                "web_name": player["web_name"],
                "team_name": player["team_name"],
                "position": POSITION_MAP.get(player["element_type"]),
                "now_cost": player.get("now_cost", 0),
                "event": fixture.get("event", 0),
                "was_home": was_home,
                "difficulty": pair[0] if was_home else pair[1],
                "minutes": averages.get("minutes", 0),
                "minutes_sd": minutes_sd,
                "expected_goals": averages.get("expected_goals", 0),
                "expected_assists": averages.get("expected_assists", 0),
                "expected_goals_conceded": averages.get("expected_goals_conceded", 0),
                "bonus": averages.get("bonus", 0),
                "team_xg_for_form": own_form[0],
                "team_xg_against_form": own_form[1],
                "opp_xg_for_form": opp_form[0],
                "opp_xg_against_form": opp_form[1],
                "npxg90": player_stats.get("npxg90", 0),
                "xa90": player_stats.get("xa90", 0),
                "xgchain90": player_stats.get("xgchain90", 0),
                "recent_points_avg": averages.get("total_points", 0),
                "ownership_pct": float(player.get("selected_by_percent", 0) or 0),
                "goals_vs_npxg90": actual_goals_per90 - player_stats.get("npxg90", 0),
            }
            rows.append(row)
            break
        time.sleep(0.05)
    return pd.DataFrame(rows)


def main():
    print("Loading FPL data...")
    session = requests.Session()
    bootstrap = get_json(session, "bootstrap-static/")
    fixtures = get_json(session, "fixtures/")
    teams = {team["id"]: team["name"] for team in bootstrap["teams"]}
    elements = []
    for player in bootstrap["elements"]:
        player = player.copy()
        player["team_name"] = teams.get(player["team"], "Unknown")
        elements.append(player)

    print("Loading Understat data...")
    understat = UnderstatClient()
    team_matches = get_understat_team_matches(understat, UNDERSTAT_SEASON)
    team_matches = add_prematch_team_form(team_matches)
    team_form_lookup = build_team_form_lookup(team_matches)
    current_team_form = build_current_team_form(team_matches)

    understat_players = get_understat_player_stats(understat, UNDERSTAT_SEASON)
    understat_matches, unmatched = match_understat_players(elements, understat_players)
    print(f"Matched {len(understat_matches)} players to Understat data.")
    if unmatched:
        print(f"Could not match {len(unmatched)} active players to Understat (using 0s for their npxg90/xa90/xgchain90):")
        for name in unmatched:
            print(f"  - {name}")

    print(f"Loading {PRIOR_SEASON_ARCHIVE} archived FPL data to bulk out training...")
    prior_team_matches = get_understat_team_matches(understat, PRIOR_SEASON_UNDERSTAT)
    prior_team_matches = add_prematch_team_form(prior_team_matches)
    prior_team_form_lookup = build_team_form_lookup(prior_team_matches)
    prior_per90_by_understat_id = {
        p["understat_id"]: p for p in get_understat_player_stats(understat, PRIOR_SEASON_UNDERSTAT)
    }

    merged_gw, prior_difficulty_lookup, prior_team_names, player_idlist = load_prior_season_archive()
    prior_id_map, prior_unmatched = match_prior_season_players(elements, player_idlist)
    print(f"Matched {len(prior_id_map)} of {len(player_idlist)} {PRIOR_SEASON_ARCHIVE} players to the current squad")
    print(f"  ({len(prior_unmatched)} unmatched -- mostly players who left the Premier League, which is expected).")

    prior_rows = build_prior_season_rows(
        merged_gw,
        prior_difficulty_lookup,
        prior_team_names,
        prior_id_map,
        prior_team_form_lookup,
        understat_matches,
        prior_per90_by_understat_id,
    )

    print("Building training data...")
    training = build_training_rows(elements, fixtures, teams, session, team_form_lookup, understat_matches)
    training["source"] = "fpl_live_current_season"
    training = pd.concat([training, prior_rows], ignore_index=True)
    training = add_features(training)
    if len(training) < 100:
        raise RuntimeError("Not enough historical rows to train the model.")
    print(f"Training rows: {len(training)} ({(training['source'] == 'fpl_live_current_season').sum()} current season, {(training['source'] != 'fpl_live_current_season').sum()} from {PRIOR_SEASON_ARCHIVE} archive)")

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=250,
        learning_rate=0.04,
        num_leaves=15,
        max_depth=5,
        min_child_samples=30,
        reg_lambda=2.0,
        random_state=42,
        verbosity=-1,
    )
    model.fit(training[FEATURES], training["total_points"])

    importance = pd.DataFrame(
        {
            "feature": FEATURES,
            "importance_gain": model.booster_.feature_importance(importance_type="gain"),
            "importance_split": model.booster_.feature_importance(importance_type="split"),
        }
    ).sort_values("importance_gain", ascending=False)
    importance["importance_gain_pct"] = (
        100 * importance["importance_gain"] / importance["importance_gain"].sum()
    ).round(1)
    print("\nFeature importance (gain = share of total prediction-error reduction; split = times used to split):")
    print(importance.to_string(index=False))
    importance.to_csv("fpl_ml_feature_importance.csv", index=False)

    print("\nPredicting upcoming fixtures...")
    upcoming = recent_player_features(elements, fixtures, teams, session, current_team_form, understat_matches)
    featured_upcoming = add_features(upcoming)
    upcoming["predicted_points"] = model.predict(featured_upcoming[FEATURES]).clip(0, 15)
    upcoming["points_per_million"] = upcoming["predicted_points"] / (upcoming["now_cost"] / 10)

    # "Hidden gems": the model rates their underlying process (xG/xA/team form) well above what
    # they've actually returned in points recently -- a signal they may be about to click, before
    # the price/ownership catches up. goals_vs_npxg90 flags pure finishing-luck separately.
    upcoming["underperformance_gap"] = upcoming["predicted_points"] - upcoming["recent_points_avg"]

    output = upcoming.sort_values(["event", "predicted_points"], ascending=[True, False])
    output.to_csv(OUTPUT_FILE, index=False)

    gems = upcoming[
        (upcoming["ownership_pct"] < GEM_OWNERSHIP_MAX)
        & (upcoming["predicted_points"] >= GEM_MIN_PREDICTED_POINTS)
        & (upcoming["underperformance_gap"] > 0)
    ].sort_values("underperformance_gap", ascending=False)
    gems.to_csv(GEMS_OUTPUT_FILE, index=False)
    print(f"\nTop hidden gems (rated highly, under {GEM_OWNERSHIP_MAX}% owned, underperforming recent points):")
    print(
        gems[
            ["web_name", "team_name", "position", "now_cost", "ownership_pct", "predicted_points", "recent_points_avg", "underperformance_gap", "goals_vs_npxg90"]
        ].head(15).to_string(index=False)
    )

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "training_rows_total": int(len(training)),
        "training_rows_current_season": int((training["source"] == "fpl_live_current_season").sum()),
        "training_rows_archive": int((training["source"] != "fpl_live_current_season").sum()),
        "understat_players_matched": int(len(understat_matches)),
        "prior_season_players_matched": int(len(prior_id_map)),
        "predicted_players": int(len(upcoming)),
        "gems_found": int(len(gems)),
        "next_event": int(upcoming["event"].min()) if len(upcoming) else None,
        "top_feature": importance.iloc[0]["feature"],
        "top_feature_gain_pct": float(importance.iloc[0]["importance_gain_pct"]),
    }
    with open(META_OUTPUT_FILE, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
