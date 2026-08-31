"""Picks an FPL-legal 15-man squad that maximizes the model's predicted points, and tracks how
each picked team actually performs once its gameweek finishes.

Two things happen on every run:
1. Score any past pick whose gameweek has finished but hasn't been scored yet (real actual points
   from the FPL API, not predictions).
2. Pick a new squad for the upcoming gameweek, if one hasn't been picked already.

Squad selection is a proper MILP (via PuLP + the bundled CBC solver), not a greedy heuristic --
budget, position quotas, and the max-3-per-club rule all interact, so "highest predicted points
first" style greedy picking can paint itself into an infeasible corner. Two decision variables per
player (in the 15-man squad? in the starting XI?) let the same solve handle both the squad and a
formation-legal starting XI (1 GK, 3-5 DEF, 2-5 MID, 1-3 FWD, 11 total) at once. Captain is chosen
post-hoc as the highest-predicted starter, which is trivially optimal once the XI is fixed.

Known simplification: actual-score tracking sums the 11 starters' real points (captain doubled)
with no autosubstitution logic -- if a starter registers 0 minutes, the real FPL app would
auto-sub them for a bench player in formation order; this doesn't. Flagged in the output so it's
never silently wrong.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pulp
import requests

from fpl_ml_model import ensure_utf8_stdout

BASE_URL = "https://fantasy.premierleague.com/api"
PREDICTIONS_FILE = Path("fpl_ml_predictions.csv")
HISTORY_FILE = Path("fpl_ml_team_history.jsonl")
BUDGET = 1000  # now_cost units (tenths of a million) -- FPL's real £100.0m budget
MAX_PER_CLUB = 3
SQUAD_QUOTAS = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
STARTER_MIN = {"GK": 1, "DEF": 3, "MID": 2, "FWD": 1}
STARTER_MAX = {"GK": 1, "DEF": 5, "MID": 5, "FWD": 3}
BENCH_WEIGHT = 0.001  # tie-break only; must never compete with starter value


def load_history():
    if not HISTORY_FILE.exists():
        return []
    return [json.loads(line) for line in HISTORY_FILE.read_text().splitlines() if line.strip()]


def save_history(entries):
    with open(HISTORY_FILE, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def pick_squad(players):
    """players: list of dicts with id, web_name, team_name, position, now_cost, predicted_points."""
    prob = pulp.LpProblem("fpl_squad", pulp.LpMaximize)
    squad = {p["id"]: pulp.LpVariable(f"squad_{p['id']}", cat="Binary") for p in players}
    starter = {p["id"]: pulp.LpVariable(f"start_{p['id']}", cat="Binary") for p in players}
    by_id = {p["id"]: p for p in players}

    prob += pulp.lpSum(
        starter[i] * by_id[i]["predicted_points"] + BENCH_WEIGHT * squad[i] * by_id[i]["predicted_points"]
        for i in squad
    )

    prob += pulp.lpSum(squad.values()) == 15
    prob += pulp.lpSum(by_id[i]["now_cost"] * squad[i] for i in squad) <= BUDGET
    prob += pulp.lpSum(starter.values()) == 11
    for i in squad:
        prob += starter[i] <= squad[i]

    for position, quota in SQUAD_QUOTAS.items():
        prob += pulp.lpSum(squad[i] for i in squad if by_id[i]["position"] == position) == quota
    for position in SQUAD_QUOTAS:
        starters_in_pos = pulp.lpSum(starter[i] for i in squad if by_id[i]["position"] == position)
        prob += starters_in_pos >= STARTER_MIN[position]
        prob += starters_in_pos <= STARTER_MAX[position]

    for team_name in {p["team_name"] for p in players}:
        prob += pulp.lpSum(squad[i] for i in squad if by_id[i]["team_name"] == team_name) <= MAX_PER_CLUB

    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"Squad optimization failed: {pulp.LpStatus[status]}")

    squad_ids = [i for i in squad if squad[i].value() == 1]
    starter_ids = {i for i in starter if starter[i].value() == 1}
    captain_id = max(starter_ids, key=lambda i: by_id[i]["predicted_points"])

    return [
        {
            **by_id[i],
            "is_starter": i in starter_ids,
            "is_captain": i == captain_id,
        }
        for i in squad_ids
    ]


DISPLAY_COLUMNS = ["team_code", "opponent_name", "opponent_short_name", "opponent_code", "was_home", "difficulty"]


def load_nearest_players(predictions, horizon=1):
    # A double gameweek gives a player two rows in the same week (one per fixture), and horizon>1
    # spans several weeks -- predicted points are summed across all of them so squad selection
    # reflects total value over the window, while the display-only fields (shirt, next opponent)
    # take the WEEKS_AHEAD==1 fixture specifically, since a compact card can only show one game and
    # "what's next" is more useful there than an ambiguous multi-week aggregate. Sorting by
    # weeks_ahead before the groupby is what makes pandas's "first" aggregation reliably pick that
    # nearest fixture's row rather than depending on incidental CSV row order.
    nearest = predictions[predictions["weeks_ahead"] <= horizon].sort_values("weeks_ahead")
    agg = {"predicted_points": "sum", **{c: "first" for c in DISPLAY_COLUMNS}}
    grouped = nearest.groupby(
        ["id", "web_name", "team_name", "position", "now_cost"], as_index=False
    ).agg(agg)
    return grouped


def pick_new_team(event):
    predictions = pd.read_csv(PREDICTIONS_FILE)
    nearest = load_nearest_players(predictions)
    columns = ["id", "web_name", "team_name", "position", "now_cost", "predicted_points"] + DISPLAY_COLUMNS
    players = nearest[columns].to_dict("records")
    for p in players:
        # A player with no weeks_ahead==1 row at all (shouldn't normally happen, but a squad
        # player outside the model's tracked pool) leaves these as NaN -- normalize so
        # json.dumps in save_history doesn't choke and the dashboard gets a real value or null.
        for key in ("team_code", "opponent_code"):
            p[key] = int(p[key]) if pd.notna(p[key]) else None
        for key in ("opponent_name", "opponent_short_name"):
            p[key] = p[key] if pd.notna(p[key]) else None
        p["was_home"] = bool(p["was_home"]) if pd.notna(p["was_home"]) else None
        p["difficulty"] = int(p["difficulty"]) if pd.notna(p["difficulty"]) else None

    squad = pick_squad(players)
    predicted_total = sum(
        p["predicted_points"] * (2 if p["is_captain"] else 1) for p in squad if p["is_starter"]
    )

    entry = {
        "event": int(event),
        "picked_at": datetime.now(timezone.utc).isoformat(),
        "squad": squad,
        "predicted_total": round(predicted_total, 2),
        "actual_total": None,
        "scored_at": None,
    }
    return entry


def score_pick(entry, session):
    starters = [p for p in entry["squad"] if p["is_starter"]]
    total = 0
    breakdown = []
    for player in starters:
        history = session.get(f"{BASE_URL}/element-summary/{player['id']}/", timeout=30).json().get("history", [])
        match = next((m for m in history if m.get("round") == entry["event"]), None)
        points = match.get("total_points", 0) if match else 0
        multiplier = 2 if player["is_captain"] else 1
        total += points * multiplier
        breakdown.append({"web_name": player["web_name"], "actual_points": points, "captain": player["is_captain"]})

    entry["actual_total"] = total
    entry["scored_at"] = datetime.now(timezone.utc).isoformat()
    entry["breakdown"] = breakdown
    return entry


def main():
    session = requests.Session()
    bootstrap = session.get(f"{BASE_URL}/bootstrap-static/", timeout=30).json()
    events = {e["id"]: e for e in bootstrap["events"]}

    history = load_history()

    # 1. Score any finished-but-unscored past pick.
    for entry in history:
        if entry["actual_total"] is not None:
            continue
        event = events.get(entry["event"])
        if event and event["finished"] and event["data_checked"]:
            score_pick(entry, session)
            print(f"Scored GW{entry['event']}: predicted {entry['predicted_total']}, actual {entry['actual_total']}"
                  f" (no autosubs applied -- see module docstring)")

    # 2. Pick a new team for the upcoming gameweek, if not already picked.
    predictions = pd.read_csv(PREDICTIONS_FILE)
    nearest = predictions[predictions["weeks_ahead"] == 1]
    if nearest.empty:
        print("No upcoming-gameweek predictions found -- run fpl_ml_model.py first.")
    else:
        upcoming_event = int(nearest["event"].min())
        already_picked = any(e["event"] == upcoming_event for e in history)
        if already_picked:
            print(f"Already have a pick for GW{upcoming_event}, skipping.")
        else:
            entry = pick_new_team(upcoming_event)
            history.append(entry)
            starters = [p for p in entry["squad"] if p["is_starter"]]
            captain = next(p for p in starters if p["is_captain"])
            print(f"Picked GW{upcoming_event} squad. Predicted starting XI total: {entry['predicted_total']} "
                  f"(captain: {captain['web_name']})")
            for p in sorted(starters, key=lambda x: -x["predicted_points"]):
                tag = " (C)" if p["is_captain"] else ""
                print(f"  {p['position']:<4} {p['web_name']:<16} {p['team_name']:<16} "
                      f"£{p['now_cost']/10:.1f}m  {p['predicted_points']:.2f}{tag}")
            bench = [p for p in entry["squad"] if not p["is_starter"]]
            print("  Bench: " + ", ".join(f"{p['web_name']} ({p['position']})" for p in bench))

    save_history(history)
    print(f"Saved {HISTORY_FILE}")


if __name__ == "__main__":
    ensure_utf8_stdout()
    main()
