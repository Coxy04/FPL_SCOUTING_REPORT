"""Builds dashboard.html from the fpl_ml_model.py outputs (predictions/gems/feature-importance CSVs
and the meta.json summary). Re-run this after fpl_ml_model.py any time you want the dashboard's
embedded data refreshed -- the HTML has no live backend, so this bake step is the "refresh"."""
import json
import math
from pathlib import Path

import pandas as pd

from fpl_ml_model import ensure_utf8_stdout

PREDICTIONS_FILE = Path("fpl_ml_predictions.csv")
GEMS_FILE = Path("fpl_ml_hidden_gems.csv")
FEATURE_IMPORTANCE_FILE = Path("fpl_ml_feature_importance.csv")
META_FILE = Path("fpl_ml_meta.json")
TEAM_HISTORY_FILE = Path("fpl_ml_team_history.jsonl")
MY_TEAM_FILE = Path("my_fpl_team.json")
OUTPUT_FILE = Path("docs/index.html")

PLAYER_COLUMNS = [
    "id", "web_name", "team_name", "team_code", "position", "now_cost", "event", "weeks_ahead", "was_home",
    "opponent_name", "opponent_short_name", "opponent_code",
    "predicted_points", "predicted_points_low", "predicted_points_high",
    "predicted_points_5gw", "points_per_million",
    "season_points", "recent_points_avg",
    "underperformance_gap", "ownership_pct", "goals_vs_npxg90",
    "minutes", "minutes_sd", "expected_goals", "expected_assists",
    "expected_goals_conceded", "difficulty", "team_xg_for_form",
    "team_xg_against_form", "opp_xg_for_form", "opp_xg_against_form",
    "npxg90", "xa90", "xgchain90", "xgbuildup90", "shots90", "key_passes90",
    "clearances_blocks_interceptions", "recoveries", "tackles", "defensive_contribution", "saves",
    "own_days_rest", "opp_days_rest",
    "status", "chance_of_playing_next_round", "playing_time_multiplier",
    "explanation", "is_blank", "fixture_index",
]


def round_numeric(records, ndigits=3):
    for record in records:
        for key, value in record.items():
            if isinstance(value, float) and not math.isnan(value):
                record[key] = round(value, ndigits)
    return records


def safe_json(records):
    return json.dumps(records, separators=(",", ":")).replace("</", "<\\/")


def build():
    predictions = pd.read_csv(PREDICTIONS_FILE)
    gems = pd.read_csv(GEMS_FILE)
    feature_importance = pd.read_csv(FEATURE_IMPORTANCE_FILE)
    meta = json.loads(META_FILE.read_text())

    # fpl_ml_model.py refuses to touch predictions.csv while the gameweek deadline lock is active
    # (see get_target_event), which can leave it frozen at an older schema for days at a time --
    # a genuine new column added here while that lock is active shouldn't hard-crash the rest of
    # the pipeline against otherwise-perfectly-usable frozen data. Missing columns just read as 0
    # (or "[]" for the JSON explanation column, via the existing try/except below) until the next
    # real refresh catches up.
    for column in PLAYER_COLUMNS:
        if column not in predictions.columns:
            predictions[column] = 0

    players = round_numeric(predictions[PLAYER_COLUMNS].to_dict("records"))
    gem_ids = set(gems["id"].tolist())
    for player in players:
        player["is_gem"] = player["id"] in gem_ids
        try:
            player["explanation"] = json.loads(player["explanation"])
        except (TypeError, ValueError):
            player["explanation"] = []

    features = round_numeric(
        feature_importance[["position", "feature", "importance_gain_pct"]].to_dict("records")
    )

    team_history = []
    if TEAM_HISTORY_FILE.exists():
        team_history = [
            json.loads(line) for line in TEAM_HISTORY_FILE.read_text().splitlines() if line.strip()
        ]
        team_history.sort(key=lambda e: e["event"])

    my_team = json.loads(MY_TEAM_FILE.read_text()) if MY_TEAM_FILE.exists() else None

    template = Path("dashboard_template.html").read_text(encoding="utf-8")
    html = (
        template
        .replace("__PLAYERS_JSON__", safe_json(players))
        .replace("__FEATURES_JSON__", safe_json(features))
        .replace("__META_JSON__", safe_json(meta))
        .replace("__TEAM_HISTORY_JSON__", safe_json(team_history))
        .replace("__MY_TEAM_JSON__", safe_json(my_team))
    )
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(html, encoding="utf-8")
    print(f"Built {OUTPUT_FILE} ({len(players)} players, {len(features)} features, {len(team_history)} tracked picks)")


if __name__ == "__main__":
    ensure_utf8_stdout()
    build()
