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
OUTPUT_FILE = Path("docs/index.html")

PLAYER_COLUMNS = [
    "id", "web_name", "team_name", "position", "now_cost", "event", "weeks_ahead", "was_home",
    "predicted_points", "predicted_points_5gw", "points_per_million",
    "season_points", "recent_points_avg",
    "underperformance_gap", "ownership_pct", "goals_vs_npxg90",
    "minutes", "minutes_sd", "expected_goals", "expected_assists",
    "expected_goals_conceded", "difficulty", "team_xg_for_form",
    "team_xg_against_form", "opp_xg_for_form", "opp_xg_against_form",
    "npxg90", "xa90", "xgchain90",
    "status", "chance_of_playing_next_round", "playing_time_multiplier",
    "explanation",
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

    template = Path("dashboard_template.html").read_text(encoding="utf-8")
    html = (
        template
        .replace("__PLAYERS_JSON__", safe_json(players))
        .replace("__FEATURES_JSON__", safe_json(features))
        .replace("__META_JSON__", safe_json(meta))
        .replace("__TEAM_HISTORY_JSON__", safe_json(team_history))
    )
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(html, encoding="utf-8")
    print(f"Built {OUTPUT_FILE} ({len(players)} players, {len(features)} features, {len(team_history)} tracked picks)")


if __name__ == "__main__":
    ensure_utf8_stdout()
    build()
