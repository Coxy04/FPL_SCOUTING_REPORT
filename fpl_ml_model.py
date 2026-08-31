import difflib
import json
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
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
MAX_FUTURE_GAMEWEEKS = 5
RECENCY_HALF_LIFE_DAYS = 180
DEFAULT_PLAYING_TIME_DENOMINATOR = 60
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
    "clearances_blocks_interceptions",
    "recoveries",
    "tackles",
    "defensive_contribution",
    "own_days_rest",
    "opp_days_rest",
    "is_penalty_taker",
    "is_freekick_taker",
    "is_corner_taker",
    "position_GK",
    "position_DEF",
    "position_MID",
    "position_FWD",
]

# One model per position instead of a single pooled model with position one-hot columns: a
# defender's clean-sheet/goals-conceded signals and a forward's attacking signals don't compete
# for the same tree splits this way, and per-position sample sizes (roughly 1,000-4,200 rows
# each) are large enough that specializing doesn't starve any of them.
POSITION_FEATURES = [f for f in FEATURES if not f.startswith("position_")]

DEFAULT_MODEL_PARAMS = {
    "n_estimators": 250,
    "learning_rate": 0.04,
    "num_leaves": 15,
    "max_depth": 5,
    "min_child_samples": 30,
    "reg_lambda": 2.0,
    "half_life_days": RECENCY_HALF_LIFE_DAYS,
    "playing_time_denominator": DEFAULT_PLAYING_TIME_DENOMINATOR,
}

# LGBMRegressor doesn't accept half_life_days or playing_time_denominator -- they're ours,
# consumed by compute_sample_weights and the playing-time multiplier respectively, not the model
# itself. Every params dict (defaults, per-position overrides, and search candidates) carries
# them alongside the real LightGBM hyperparameters; strip them out right before constructing the
# model.
NON_LGB_PARAM_KEYS = ("half_life_days", "playing_time_denominator")

# Per-position hyperparameters found by tune_model.py's random search (scored across two
# time-based holdout folds, so these beat the defaults on more than one window, not just one
# lucky split). MID's default settings already won, so it's left out and falls back to
# DEFAULT_MODEL_PARAMS. Re-run tune_model.py periodically and update this as the season evolves.
POSITION_MODEL_PARAMS = {
    "GK": {"n_estimators": 250, "learning_rate": 0.03, "num_leaves": 7, "max_depth": 7, "min_child_samples": 50, "reg_lambda": 4.0, "half_life_days": 60, "playing_time_denominator": 90},
    "DEF": {"n_estimators": 150, "learning_rate": 0.03, "num_leaves": 23, "max_depth": 3, "min_child_samples": 10, "reg_lambda": 8.0, "half_life_days": 365, "playing_time_denominator": 75},
    "MID": {"n_estimators": 150, "learning_rate": 0.02, "num_leaves": 31, "max_depth": 4, "min_child_samples": 80, "reg_lambda": 1.0, "half_life_days": 100000, "playing_time_denominator": 75},
    "FWD": {"n_estimators": 100, "learning_rate": 0.04, "num_leaves": 7, "max_depth": 7, "min_child_samples": 30, "reg_lambda": 8.0, "half_life_days": 100000, "playing_time_denominator": 90},
}


def get_position_params(position):
    return POSITION_MODEL_PARAMS.get(position, DEFAULT_MODEL_PARAMS)


def get_half_life_days(position):
    return get_position_params(position).get("half_life_days", RECENCY_HALF_LIFE_DAYS)


def get_playing_time_denominator(position):
    return get_position_params(position).get("playing_time_denominator", DEFAULT_PLAYING_TIME_DENOMINATOR)


def make_model(position):
    params = {k: v for k, v in get_position_params(position).items() if k not in NON_LGB_PARAM_KEYS}
    return lgb.LGBMRegressor(objective="regression", random_state=42, verbosity=-1, **params)


# An 80% interval (10th-90th percentile) rather than a tighter or wider one -- narrow enough to be
# useful ("this is a nailed-on 6, that's anywhere from 1 to 9"), wide enough that it isn't
# constantly missed by variance the model can't see coming (a red card, a 90th-minute change).
QUANTILE_LOW_ALPHA = 0.1
QUANTILE_HIGH_ALPHA = 0.9


def make_quantile_model(position, alpha):
    """A separate LightGBM model trained to predict a specific percentile of the score
    distribution (quantile regression) rather than the mean -- this is how the dashboard gets a
    real uncertainty range ("could be anywhere from 2 to 9") instead of a single point estimate
    that hides how spiky or reliable a player's returns actually are."""
    params = {k: v for k, v in get_position_params(position).items() if k not in NON_LGB_PARAM_KEYS}
    return lgb.LGBMRegressor(objective="quantile", alpha=alpha, random_state=42, verbosity=-1, **params)


# Raw quantile-model output is NOT well calibrated out of the box -- backtest_model.py found the
# nominal 80% interval only actually contained the real outcome 57-66% of the time on held-out
# data, i.e. it was systematically overconfident. This margin is a conformal-prediction correction
# (widen both bounds by the (1 - miscoverage) quantile of held-out |actual - bound| residuals) that
# backtest_model.py recomputes and rewrites here every run, so the interval's claimed 80% keeps
# being backed by a real, current measurement rather than trusting the model's raw quantile output.
QUANTILE_MARGIN = {"GK": 1.029, "DEF": 0.967, "MID": 1.0, "FWD": 1.0}


def get_quantile_margin(position):
    return QUANTILE_MARGIN.get(position, 0.0)


def explain_predictions(model, X, feature_names, top_n=5):
    """Per-row SHAP-style feature contributions via LightGBM's own pred_contrib -- decomposes
    each prediction into how much every feature pushed it up or down from the model's base value.
    Global feature_importance (gain) can only say what a position's model weighs on average; it
    can't say why THIS player is rated above THAT one this week. This can."""
    contributions = model.booster_.predict(X, pred_contrib=True)
    feature_contribs = contributions[:, :-1]
    explanations = []
    for row in feature_contribs:
        pairs = sorted(zip(feature_names, row), key=lambda p: abs(p[1]), reverse=True)[:top_n]
        explanations.append([{"feature": f, "contribution": round(float(c), 3)} for f, c in pairs])
    return explanations


def compute_sample_weights(dates, half_life_days=RECENCY_HALF_LIFE_DAYS, as_of=None):
    """Recency weighting: a row from last season should count for less than one from this
    week, since team/player form drifts over time. Exponential decay by calendar days, not
    gameweeks, so it blends current-season and archive rows on one consistent scale. A very
    large half_life_days effectively switches this off (weights ~1.0 for any realistic date) --
    tune_model.py searches that as a candidate, not just shorter decay rates."""
    as_of = as_of or pd.Timestamp.now(tz="UTC")
    parsed = pd.to_datetime(dates, utc=True, errors="coerce")
    days_ago = (as_of - parsed).dt.days.clip(lower=0)
    weights = 0.5 ** (days_ago / half_life_days)
    return weights.fillna(weights.median())


def ensure_utf8_stdout():
    """Windows' console/file-redirect encoding (cp1252) can't represent many player names
    (accents, etc.) -- a print() of one partway through a script can crash it after some outputs
    are already saved but before others are written, leaving those silently stale with no error
    surfaced anywhere obvious (this has now bitten three separate scripts the same way before
    being centralized here). Every script's entry point should call this first."""
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


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


DEFAULT_REST_DAYS = 7


def build_rest_days_lookup(fixtures):
    """Days between a team's previous fixture and each fixture, purely from kickoff_time gaps in
    the team's own full fixture list (finished and upcoming together, so an upcoming fixture's
    rest is measured against its most recent COMPLETED match). Captures fixture congestion --
    cup replays, rearranged midweek games, European away trips -- that a plain gameweek count
    can't. A team's very first fixture in the list has no prior match to diff against, so falls
    back to a typical week's rest rather than a misleading 0."""
    by_team = {}
    for fixture in fixtures:
        kickoff = fixture.get("kickoff_time")
        fixture_id = fixture.get("id")
        if not kickoff or fixture_id is None:
            continue
        date = pd.Timestamp(kickoff)
        for side, key in (("home", "team_h"), ("away", "team_a")):
            team_id = fixture.get(key)
            if team_id is not None:
                by_team.setdefault(team_id, []).append((date, fixture_id, side))

    rest_days_by_fixture = {}
    for entries in by_team.values():
        entries.sort(key=lambda e: e[0])
        prev_date = None
        for date, fixture_id, side in entries:
            days = (date - prev_date).days if prev_date is not None else DEFAULT_REST_DAYS
            rest_days_by_fixture.setdefault(fixture_id, {})[side] = days
            prev_date = date
    return rest_days_by_fixture


def get_rest_days(rest_days_lookup, fixture_id, was_home):
    pair = rest_days_lookup.get(fixture_id, {})
    own_side, opp_side = ("home", "away") if was_home else ("away", "home")
    return pair.get(own_side, DEFAULT_REST_DAYS), pair.get(opp_side, DEFAULT_REST_DAYS)


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
        rows.append({"team": home, "date": date, "xg_for": xg_h, "xg_against": xg_a, "was_home": True})
        rows.append({"team": away, "date": date, "xg_for": xg_a, "xg_against": xg_h, "was_home": False})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(["team", "date"]).reset_index(drop=True)


def add_prematch_team_form(frame):
    # Grouped by (team, was_home), not just team: a team's home form and away form are rolled
    # separately, so "how strong is this team" reflects the venue they're actually playing at in
    # the fixture being featured, not a blend that dilutes a fortress-at-home team's home number
    # with their away results (or vice versa).
    frame = frame.copy()
    grouped = frame.groupby(["team", "was_home"])
    frame["team_xg_for_form"] = grouped["xg_for"].transform(
        lambda s: s.shift(1).rolling(TEAM_FORM_WINDOW, min_periods=1).mean()
    )
    frame["team_xg_against_form"] = grouped["xg_against"].transform(
        lambda s: s.shift(1).rolling(TEAM_FORM_WINDOW, min_periods=1).mean()
    )
    return frame


def build_team_form_lookup(frame):
    # Keyed by (team, date), same as before -- but since each row is already tagged with the
    # venue that team actually played that match, the stored form value is automatically the
    # correct home-or-away-specific one for both the player's own team AND their opponent's row
    # on that same date (who were, by definition, at the opposite venue). No was_home lookup key
    # needed here; the historical record already encodes it.
    lookup = {}
    for row in frame.itertuples():
        lookup[(row.team, row.date)] = (row.team_xg_for_form, row.team_xg_against_form)
    return lookup


def build_current_team_form(frame):
    # Unlike build_team_form_lookup, this isn't tied to a specific historical match record, so
    # the venue has to be part of the key explicitly: "current form" means something different
    # depending on whether the upcoming fixture is home or away.
    current = {}
    for (team, was_home), group in frame.groupby(["team", "was_home"]):
        tail = group.sort_values("date").tail(TEAM_FORM_WINDOW)
        current[(team, was_home)] = (tail["xg_for"].mean(), tail["xg_against"].mean())

    # Early in a season (or for a team that just hasn't played one venue yet), the split above
    # can be empty on one side -- falling back to 0 there would misread as "this team never
    # scores" rather than "no data yet". Fall back to the team's overall blended form instead,
    # which is less precise but not misleading.
    for team, group in frame.groupby("team"):
        tail = group.sort_values("date").tail(TEAM_FORM_WINDOW)
        overall = (tail["xg_for"].mean(), tail["xg_against"].mean())
        for was_home in (True, False):
            current.setdefault((team, was_home), overall)

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
    # players_raw.csv is a bootstrap-static snapshot archived alongside that season's own data --
    # unlike gws/merged_gw.csv (match-level, no set-piece fields at all), this has the same
    # penalties_order/direct_freekicks_order/corners_and_indirect_freekicks_order fields the live
    # API has, keyed by that season's own player ids (same id space as merged_gw's "element"
    # column -- confirmed full overlap). Season-accurate, not "today's snapshot applied to old
    # rows" -- an initial version of this feature did exactly that as a stopgap before this file
    # was found to have the real thing.
    players_raw = pd.read_csv(f"{base}/players_raw.csv", encoding="utf-8", encoding_errors="ignore")
    prior_set_piece_lookup = {
        row["id"]: set_piece_flags(row) for row in players_raw.to_dict("records")
    }

    difficulty_lookup = {
        row.id: (row.team_h_difficulty, row.team_a_difficulty) for row in fixtures.itertuples()
    }
    rest_days_lookup = build_rest_days_lookup(fixtures.to_dict("records"))
    team_names = {row.id: row.name for row in teams.itertuples()}
    return merged_gw, difficulty_lookup, team_names, player_idlist, rest_days_lookup, prior_set_piece_lookup


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


DEFAULT_SET_PIECE_FLAGS = {"is_penalty_taker": 0, "is_freekick_taker": 0, "is_corner_taker": 0}


def build_prior_season_rows(merged_gw, difficulty_lookup, team_names, id_map, team_form_lookup, understat_matches, prior_per90_by_understat_id, rest_days_lookup=None, prior_set_piece_lookup=None):
    gw = merged_gw.copy()
    for column in LAGGED_MATCH_STATS:
        gw[column] = pd.to_numeric(gw[column], errors="coerce").fillna(0)
    gw = gw.sort_values(["element", "GW"]).reset_index(drop=True)
    gw["actual_minutes"] = gw["minutes"]
    gw = add_lagged_match_form(gw, group_col="element")
    rest_days_lookup = rest_days_lookup or {}
    # Keyed by THAT SEASON's own player id (row.element below), not the current-day id -- sourced
    # from that season's own players_raw.csv snapshot via load_prior_season_archive, so this is
    # season-accurate set-piece duty, not today's assignment retroactively applied to old rows.
    prior_set_piece_lookup = prior_set_piece_lookup or {}

    rows = []
    for row in gw.itertuples():
        current_id = id_map.get(row.element)
        if current_id is None or row.actual_minutes <= 0:
            continue

        was_home = bool(row.was_home)
        difficulty_pair = difficulty_lookup.get(row.fixture, (3, 3))
        own_rest, opp_rest = get_rest_days(rest_days_lookup, row.fixture, was_home)
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
                "clearances_blocks_interceptions": row.clearances_blocks_interceptions,
                "recoveries": row.recoveries,
                "tackles": row.tackles,
                "defensive_contribution": row.defensive_contribution,
                "own_days_rest": own_rest,
                "opp_days_rest": opp_rest,
                **prior_set_piece_lookup.get(row.element, DEFAULT_SET_PIECE_FLAGS),
                "position": row.position,
                "total_points": row.total_points,
                "source": f"fpl_archive_{PRIOR_SEASON_ARCHIVE}",
                "gw": row.GW,
                "date": date,
                "current_id": current_id,
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
        "clearances_blocks_interceptions",
        "recoveries",
        "tackles",
        "defensive_contribution",
        "own_days_rest",
        "opp_days_rest",
        "is_penalty_taker",
        "is_freekick_taker",
        "is_corner_taker",
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


# defensive_contribution is a same-match derived flag (FPL awards bonus points once a player's
# CBIT/recoveries/tackles cross a position-specific threshold that match) -- using its own-match
# value to predict that match's own total_points would be the exact same leakage bug bonus had
# before it was lagged here. Lagging it (and its raw components) to a pre-match rolling average
# keeps it a genuine "recent involvement" signal instead of a peek at the answer.
LAGGED_MATCH_STATS = [
    "minutes", "expected_goals", "expected_assists", "expected_goals_conceded", "bonus",
    "clearances_blocks_interceptions", "recoveries", "tackles", "defensive_contribution",
]


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


def set_piece_flags(player):
    """Primary-taker-only flags (order == 1), not backup/third-choice -- a backup rarely actually
    takes a kick unless the primary is unavailable, so the designated taker is the signal that
    matters. penalties_order specifically fills a real gap the model otherwise has no way to see:
    npxg90 is NON-penalty xG by definition, so a designated penalty taker gets zero credit there
    for that role, even though a penalty is a genuinely high-probability scoring chance.

    Applied from FPL's CURRENT bootstrap-static snapshot to every row for that player, including
    prior-season and earlier-this-season training rows -- there's no historical archive of who
    held set-piece duty in past seasons, so this trades some noise (a player's role can change
    over time) for having the signal at all, same trade-off already made for Understat's
    season-aggregate npxg90/xa90/xgchain90 being applied across a player's whole current season."""
    return {
        "is_penalty_taker": 1 if player.get("penalties_order") == 1 else 0,
        "is_freekick_taker": 1 if player.get("direct_freekicks_order") == 1 else 0,
        "is_corner_taker": 1 if player.get("corners_and_indirect_freekicks_order") == 1 else 0,
    }


def build_training_rows(elements, fixtures, teams, session, team_form_lookup, understat_matches):
    difficulty_lookup = make_fixture_lookup(fixtures)
    rest_days_lookup = build_rest_days_lookup(fixtures)
    rows = []
    for player in elements:
        position = POSITION_MAP.get(player["element_type"])
        player_stats = understat_matches.get(player["id"], {})
        set_piece = set_piece_flags(player)
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
            own_rest, opp_rest = get_rest_days(rest_days_lookup, fixture_id, was_home)
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
                    "now_cost": match.get("value") or player.get("now_cost", 0),
                    "team_xg_for_form": own_form[0],
                    "team_xg_against_form": own_form[1],
                    "opp_xg_for_form": opp_form[0],
                    "opp_xg_against_form": opp_form[1],
                    "npxg90": player_stats.get("npxg90", 0),
                    "xa90": player_stats.get("xa90", 0),
                    "xgchain90": player_stats.get("xgchain90", 0),
                    "clearances_blocks_interceptions": lagged["clearances_blocks_interceptions"],
                    "recoveries": lagged["recoveries"],
                    "tackles": lagged["tackles"],
                    "defensive_contribution": lagged["defensive_contribution"],
                    "own_days_rest": own_rest,
                    "opp_days_rest": opp_rest,
                    **set_piece,
                    "position": position,
                    "total_points": match.get("total_points", 0),
                    "date": date,
                }
            )
    return pd.DataFrame(rows)


def upcoming_event_numbers(fixtures, count):
    """The next `count` real gameweek numbers with at least one unfinished fixture anywhere in
    the league -- computed once for the whole league, not per team. Walking a team's own next-N
    fixtures independently (the previous approach) silently misaligns "weeks_ahead" across teams
    the moment any team blanks a gameweek (European/cup clash) or doubles one up (a rearranged
    fixture): one team's "week 3" would stop meaning the same calendar gameweek as another's."""
    events = sorted({f["event"] for f in fixtures if not f["finished"] and f.get("event")})
    return events[:count]


def recent_player_features(elements, fixtures, teams, session, current_team_form, understat_matches, team_codes=None, team_short_names=None):
    difficulty_lookup = make_fixture_lookup(fixtures)
    rest_days_lookup = build_rest_days_lookup(fixtures)
    target_events = upcoming_event_numbers(fixtures, MAX_FUTURE_GAMEWEEKS)
    team_codes = team_codes or {}
    team_short_names = team_short_names or {}
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
        season_points = player.get("total_points", 0)

        # FPL leaves chance_of_playing_next_round null only when a player is fully fit; injured/
        # suspended/unavailable/doubtful players all get a real 0-100 value, so this discounts
        # predictions for anyone with a fitness doubt instead of blindly trusting their form stats.
        chance = player.get("chance_of_playing_next_round")
        availability_multiplier = 1.0 if chance is None else chance / 100.0

        # A fully fit third-choice keeper (or any fringe player) still gets availability_multiplier
        # = 1.0 -- there's no injury, they're just not selected. Team-level features (clean sheet
        # form etc.) don't know that, so a benched player behind a strong defense can otherwise
        # predict a real score despite having ~0 chance of actually playing. Recent actual minutes
        # closes that gap: capped at 1.0 once averaging a full match, tapering to 0 for the unused.
        # The denominator is per-position and tuned (see tune_model.py) -- ablation testing showed
        # a graduated discount actively hurts GK/MID accuracy while helping DEF/FWD, so a single
        # hand-picked value for every position was wrong.
        position = POSITION_MAP.get(player["element_type"])
        playing_time_multiplier = min(1.0, averages.get("minutes", 0) / get_playing_time_denominator(position))

        base_row = {
            "id": player["id"],
            "web_name": player["web_name"],
            "team_name": player["team_name"],
            "team_code": team_codes.get(player["team"]),
            "position": position,
            "now_cost": player.get("now_cost", 0),
            "minutes": averages.get("minutes", 0),
            "minutes_sd": minutes_sd,
            "expected_goals": averages.get("expected_goals", 0),
            "expected_assists": averages.get("expected_assists", 0),
            "expected_goals_conceded": averages.get("expected_goals_conceded", 0),
            "bonus": averages.get("bonus", 0),
            "npxg90": player_stats.get("npxg90", 0),
            "clearances_blocks_interceptions": averages.get("clearances_blocks_interceptions", 0),
            "recoveries": averages.get("recoveries", 0),
            "tackles": averages.get("tackles", 0),
            "defensive_contribution": averages.get("defensive_contribution", 0),
            "xa90": player_stats.get("xa90", 0),
            "xgchain90": player_stats.get("xgchain90", 0),
            **set_piece_flags(player),
            "recent_points_avg": averages.get("total_points", 0),
            "ownership_pct": float(player.get("selected_by_percent", 0) or 0),
            "goals_vs_npxg90": actual_goals_per90 - player_stats.get("npxg90", 0),
            "season_points": season_points,
            "status": player.get("status", "a"),
            "chance_of_playing_next_round": chance,
            "availability_multiplier": availability_multiplier,
            "playing_time_multiplier": playing_time_multiplier,
        }

        for weeks_ahead, event_number in enumerate(target_events, start=1):
            team_matches_this_event = [
                f for f in fixtures
                if f.get("event") == event_number and player["team"] in (f["team_h"], f["team_a"])
            ]
            team_matches_this_event.sort(key=lambda f: f.get("kickoff_time") or "")

            if not team_matches_this_event:
                # Blank gameweek for this team (postponed for a cup replay, European fixture
                # rearrangement, etc.) -- emit a zero-prediction placeholder rather than silently
                # skipping to the next fixture, which is what let weeks_ahead drift out of sync
                # with the real calendar gameweek in the first place.
                rows.append({
                    **base_row, "weeks_ahead": weeks_ahead, "event": event_number,
                    # was_home/difficulty are placeholders, not real values -- is_blank=True is
                    # what actually excludes this row from the model (see main()), so these just
                    # need to be types add_features() can process, not meaningful predictions.
                    "is_blank": True, "fixture_index": 0, "was_home": False, "difficulty": 0,
                    "opponent_name": None, "opponent_short_name": None, "opponent_code": None,
                    "team_xg_for_form": 0, "team_xg_against_form": 0,
                    "opp_xg_for_form": 0, "opp_xg_against_form": 0,
                    "own_days_rest": DEFAULT_REST_DAYS, "opp_days_rest": DEFAULT_REST_DAYS,
                })
                continue

            for fixture_index, fixture in enumerate(team_matches_this_event):
                was_home = player["team"] == fixture["team_h"]
                pair = difficulty_lookup[fixture["id"]]
                opponent_id = fixture["team_a"] if was_home else fixture["team_h"]
                opp_understat_team = UNDERSTAT_TEAM_MAP.get(teams.get(opponent_id, ""))
                # Own form uses this fixture's actual venue; the opponent is necessarily at the
                # opposite venue in this same match, so their relevant form is the other side's.
                own_form = current_team_form.get((own_understat_team, was_home), (0, 0))
                opp_form = current_team_form.get((opp_understat_team, not was_home), (0, 0))
                own_rest, opp_rest = get_rest_days(rest_days_lookup, fixture["id"], was_home)
                rows.append({
                    **base_row,
                    "weeks_ahead": weeks_ahead,
                    "event": event_number,
                    "is_blank": False,
                    # 0 for the only fixture in a normal gameweek, or 0/1 for a double gameweek's
                    # two fixtures -- distinguishes them in the UI without disturbing weeks_ahead.
                    "fixture_index": fixture_index,
                    "was_home": was_home,
                    "difficulty": pair[0] if was_home else pair[1],
                    "opponent_name": teams.get(opponent_id, ""),
                    "opponent_short_name": team_short_names.get(opponent_id, ""),
                    "opponent_code": team_codes.get(opponent_id),
                    "team_xg_for_form": own_form[0],
                    "team_xg_against_form": own_form[1],
                    "opp_xg_for_form": opp_form[0],
                    "opp_xg_against_form": opp_form[1],
                    "own_days_rest": own_rest,
                    "opp_days_rest": opp_rest,
                })
        time.sleep(0.05)
    return pd.DataFrame(rows)


def get_target_event(events):
    """The gameweek fresh predictions should target -- the next one whose transfer deadline
    hasn't passed yet. Returns None if we're currently mid-gameweek (deadline passed, but the
    gameweek's matches aren't all finished): predictions for that gameweek aren't actionable any
    more (the deadline's gone, no team change can use them), and regenerating them from FPL's own
    API mid-gameweek is unreliable -- element-summary pre-creates a zeroed placeholder row for a
    player's current fixture before it's even kicked off, which would corrupt recent-form
    averages for anyone whose match hasn't happened yet. Simplest fix: don't predict into that
    window at all -- hold whatever was generated before the deadline until the gameweek finishes."""
    for event in sorted(events, key=lambda e: e["id"]):
        if event["finished"]:
            continue
        deadline = datetime.fromisoformat(event["deadline_time"].replace("Z", "+00:00"))
        return event["id"] if datetime.now(timezone.utc) < deadline else None
    return None


def main():
    print("Loading FPL data...")
    session = requests.Session()
    bootstrap = get_json(session, "bootstrap-static/")
    target_event = get_target_event(bootstrap["events"])
    if target_event is None:
        print(
            "The current gameweek's deadline has passed but it hasn't finished yet -- "
            "predictions would only cover a gameweek you can no longer change your team for, "
            "and FPL's own API returns zeroed placeholder data for fixtures that haven't been "
            "played yet mid-gameweek, which corrupts recent-form averages if pulled now. "
            "Skipping this refresh entirely -- nothing is touched. Re-run once the gameweek "
            "finishes (or before the next deadline, whichever comes first)."
        )
        return
    print(f"Targeting GW{target_event} (deadline not yet passed).")
    fixtures = get_json(session, "fixtures/")
    teams = {team["id"]: team["name"] for team in bootstrap["teams"]}
    team_codes = {team["id"]: team["code"] for team in bootstrap["teams"]}
    team_short_names = {team["id"]: team["short_name"] for team in bootstrap["teams"]}
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

    merged_gw, prior_difficulty_lookup, prior_team_names, player_idlist, prior_rest_days_lookup, prior_set_piece_lookup = load_prior_season_archive()
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
        prior_rest_days_lookup,
        prior_set_piece_lookup,
    )

    print("Building training data...")
    training = build_training_rows(elements, fixtures, teams, session, team_form_lookup, understat_matches)
    training["source"] = "fpl_live_current_season"
    training = pd.concat([training, prior_rows], ignore_index=True)
    training = add_features(training)
    if len(training) < 100:
        raise RuntimeError("Not enough historical rows to train the model.")
    print(f"Training rows: {len(training)} ({(training['source'] == 'fpl_live_current_season').sum()} current season, {(training['source'] != 'fpl_live_current_season').sum()} from {PRIOR_SEASON_ARCHIVE} archive)")

    models = {}
    quantile_models = {}
    importance_tables = []
    for position in POSITION_MAP.values():
        subset = training[training["position"] == position]
        weights = compute_sample_weights(subset["date"], half_life_days=get_half_life_days(position))
        model = make_model(position)
        model.fit(subset[POSITION_FEATURES], subset["total_points"], sample_weight=weights)
        models[position] = model

        low_model = make_quantile_model(position, QUANTILE_LOW_ALPHA)
        low_model.fit(subset[POSITION_FEATURES], subset["total_points"], sample_weight=weights)
        high_model = make_quantile_model(position, QUANTILE_HIGH_ALPHA)
        high_model.fit(subset[POSITION_FEATURES], subset["total_points"], sample_weight=weights)
        quantile_models[position] = (low_model, high_model)

        position_importance = pd.DataFrame(
            {
                "position": position,
                "feature": POSITION_FEATURES,
                "importance_gain": model.booster_.feature_importance(importance_type="gain"),
                "importance_split": model.booster_.feature_importance(importance_type="split"),
            }
        ).sort_values("importance_gain", ascending=False)
        position_importance["importance_gain_pct"] = (
            100 * position_importance["importance_gain"] / position_importance["importance_gain"].sum()
        ).round(1)
        importance_tables.append(position_importance)

        print(f"\n{position} model -- trained on {len(subset)} rows. Top features by gain:")
        print(position_importance.head(8).to_string(index=False))

    importance = pd.concat(importance_tables, ignore_index=True)
    importance.to_csv("fpl_ml_feature_importance.csv", index=False)

    print("\nPredicting upcoming fixtures...")
    upcoming = recent_player_features(elements, fixtures, teams, session, current_team_form, understat_matches, team_codes, team_short_names)
    featured_upcoming = add_features(upcoming)
    upcoming["predicted_points"] = 0.0
    upcoming["predicted_points_low"] = 0.0
    upcoming["predicted_points_high"] = 0.0
    upcoming["explanation"] = "[]"
    for position, model in models.items():
        # Blank-gameweek rows (no fixture at all this week) are excluded here rather than left to
        # naturally predict ~0 -- their was_home/difficulty are placeholder values, not real
        # inputs, so running them through the model would just be a meaningless prediction that
        # happens to often be small, not a genuine "nothing happening" zero.
        mask = ((upcoming["position"] == position) & (~upcoming["is_blank"])).to_numpy()
        if mask.any():
            X = featured_upcoming.loc[mask, POSITION_FEATURES]
            upcoming.loc[mask, "predicted_points"] = model.predict(X)
            explanations = explain_predictions(model, X, POSITION_FEATURES)
            upcoming.loc[mask, "explanation"] = [json.dumps(e) for e in explanations]

            low_model, high_model = quantile_models[position]
            low_raw = low_model.predict(X)
            high_raw = high_model.predict(X)
            # Two independently-trained quantile models aren't guaranteed monotonic ("quantile
            # crossing") -- pointwise min/max keeps low <= high always, rather than trusting alpha
            # order to hold on every single row.
            margin = get_quantile_margin(position)
            upcoming.loc[mask, "predicted_points_low"] = np.minimum(low_raw, high_raw) - margin
            upcoming.loc[mask, "predicted_points_high"] = np.maximum(low_raw, high_raw) + margin

    for column in ("predicted_points", "predicted_points_low", "predicted_points_high"):
        upcoming[column] = (
            upcoming[column] * upcoming["availability_multiplier"] * upcoming["playing_time_multiplier"]
        ).clip(0, 15)
    # The interval must contain the point estimate -- it's the same 80% central interval, just
    # from a separately-trained model, so nothing guarantees that ordering on its own.
    upcoming["predicted_points_low"] = np.minimum(upcoming["predicted_points_low"], upcoming["predicted_points"])
    upcoming["predicted_points_high"] = np.maximum(upcoming["predicted_points_high"], upcoming["predicted_points"])
    upcoming["points_per_million"] = upcoming["predicted_points"] / (upcoming["now_cost"] / 10)
    upcoming["predicted_points_5gw"] = upcoming.groupby("id")["predicted_points"].transform("sum")

    # "Hidden gems": the model rates their underlying process (xG/xA/team form) well above what
    # they've actually returned in points recently -- a signal they may be about to click, before
    # the price/ownership catches up. goals_vs_npxg90 flags pure finishing-luck separately.
    # Anchored to each player's nearest fixture only, not diluted across all 5 gameweeks.
    upcoming["underperformance_gap"] = upcoming["predicted_points"] - upcoming["recent_points_avg"]

    output = upcoming.sort_values(["weeks_ahead", "predicted_points"], ascending=[True, False])
    output.to_csv(OUTPUT_FILE, index=False)

    nearest = upcoming[upcoming["weeks_ahead"] == 1]
    gems = nearest[
        (nearest["ownership_pct"] < GEM_OWNERSHIP_MAX)
        & (nearest["predicted_points"] >= GEM_MIN_PREDICTED_POINTS)
        & (nearest["underperformance_gap"] > 0)
    ].sort_values("underperformance_gap", ascending=False)
    gems.to_csv(GEMS_OUTPUT_FILE, index=False)

    # Written before the display prints below on purpose: a crash in a print() (player names,
    # console encoding, whatever) must never leave meta.json stale after predictions/gems have
    # already saved successfully -- the file the dashboard build actually depends on shouldn't be
    # gated behind a display step that has nothing to do with the data itself.
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "training_rows_total": int(len(training)),
        "training_rows_current_season": int((training["source"] == "fpl_live_current_season").sum()),
        "training_rows_archive": int((training["source"] != "fpl_live_current_season").sum()),
        "understat_players_matched": int(len(understat_matches)),
        "prior_season_players_matched": int(len(prior_id_map)),
        "predicted_players": int(len(nearest)),
        "gems_found": int(len(gems)),
        "next_event": int(nearest["event"].min()) if len(nearest) else None,
        "max_weeks_ahead": int(upcoming["weeks_ahead"].max()) if len(upcoming) else 0,
        "top_feature_by_position": {
            position: {
                "feature": group.iloc[0]["feature"],
                "gain_pct": float(group.iloc[0]["importance_gain_pct"]),
            }
            for position, group in importance.groupby("position")
        },
    }
    with open(META_OUTPUT_FILE, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved {OUTPUT_FILE}")

    print(f"\nTop hidden gems (rated highly, under {GEM_OWNERSHIP_MAX}% owned, underperforming recent points):")
    print(
        gems[
            ["web_name", "team_name", "position", "now_cost", "ownership_pct", "predicted_points", "recent_points_avg", "underperformance_gap", "goals_vs_npxg90"]
        ].head(15).to_string(index=False)
    )


if __name__ == "__main__":
    ensure_utf8_stdout()
    main()
