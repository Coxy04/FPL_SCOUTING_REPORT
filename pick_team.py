"""Picks an FPL-legal 15-man squad that maximizes the model's predicted points, and tracks how
each picked team actually performs once its gameweek finishes.

Two things happen on every run:
1. Score any past pick whose gameweek has finished but hasn't been scored yet (real actual points
   from the FPL API, not predictions). A "template" squad picked purely by ownership is scored
   alongside it, and the gameweek's game-wide average recorded, so the model's picks are measured
   against what a no-model manager following the crowd would have scored -- the GW1-6 review found
   weekly captain hit-rate is the wrong yardstick (weekly points are mostly noise) and cumulative
   points vs template/average is the right one.
2. Pick a new squad for the upcoming gameweek, if one hasn't been picked already.

Squad selection is a proper MILP (via PuLP + the bundled CBC solver), not a greedy heuristic --
budget, position quotas, and the max-3-per-club rule all interact, so "highest predicted points
first" style greedy picking can paint itself into an infeasible corner. Three decision variables
per player (in the 15-man squad? in the starting XI? captain?) let the same solve handle the squad,
a formation-legal starting XI (1 GK, 3-5 DEF, 2-5 MID, 1-3 FWD, 11 total) and the armband at once.

The captain term matters: without it (how this worked for GW1-6) a GBP 15.5m player predicted 6.5
was valued at 6.5, not 13, so the optimiser preferred three mid-priced 5-pointers and structurally
undervalued the one asset the whole mini-league was winning with (SEASON_REVIEW_GW1-6.md, 4E).

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


def pick_squad(players, budget=BUDGET, value_key="predicted_points"):
    """players: list of dicts with id, web_name, team_name, position, now_cost, predicted_points.

    `budget` defaults to the flat £100.0m assumption everything else here uses, but takes a real
    total (selling value + bank) when the caller has one -- a wildcard or free hit genuinely
    resets you to THAT number, not a round £100m, and the two differ by up to a few tenths.

    `value_key` is what the squad is optimised for. predicted_points normally; ownership_pct
    builds the "template" squad the tracker scores the model against."""
    prob = pulp.LpProblem("fpl_squad", pulp.LpMaximize)
    squad = {p["id"]: pulp.LpVariable(f"squad_{p['id']}", cat="Binary") for p in players}
    starter = {p["id"]: pulp.LpVariable(f"start_{p['id']}", cat="Binary") for p in players}
    captain = {p["id"]: pulp.LpVariable(f"cap_{p['id']}", cat="Binary") for p in players}
    by_id = {p["id"]: p for p in players}

    def value(i):
        return by_id[i].get(value_key) or 0.0

    # starter + captain: the captain's value counts twice, which is literally what the armband
    # does. Linear, so the MILP stays a MILP; one captain, and only a starter can wear it.
    prob += pulp.lpSum(
        (starter[i] + captain[i]) * value(i) + BENCH_WEIGHT * squad[i] * value(i)
        for i in squad
    )

    prob += pulp.lpSum(squad.values()) == 15
    prob += pulp.lpSum(by_id[i]["now_cost"] * squad[i] for i in squad) <= budget
    prob += pulp.lpSum(starter.values()) == 11
    prob += pulp.lpSum(captain.values()) == 1
    for i in squad:
        prob += starter[i] <= squad[i]
        prob += captain[i] <= starter[i]

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
    captain_id = max(starter_ids, key=value)

    return [
        {
            **by_id[i],
            "is_starter": i in starter_ids,
            "is_captain": i == captain_id,
        }
        for i in squad_ids
    ]


# predicted_points_low/high and haul_prob are SINGLE-FIXTURE quantities (calibrated 80% interval
# and P(8+ points) -- see fpl_ml_model.py) and, like the rest of DISPLAY_COLUMNS, are aggregated
# with "first" rather than summed. That is only ever valid at horizon=1 (one real gameweek) -- a
# multi-week caller gets whichever week's value happened to be nearest, silently mislabelled as
# covering the whole window, the exact trap the dashboard's own rangeCell() already guards against
# for the main player table. Anything reading these fields off a >1-week horizon call is a bug.
DISPLAY_COLUMNS = ["team_code", "opponent_name", "opponent_short_name", "opponent_code", "was_home",
                   "difficulty", "predicted_points_low", "predicted_points_high", "haul_prob", "ownership_pct"]


def load_nearest_players(predictions, horizon=1, start_week=1):
    # A double gameweek gives a player two rows in the same week (one per fixture), and horizon>1
    # spans several weeks -- predicted points are summed across all of them so squad selection
    # reflects total value over the window, while the display-only fields (shirt, next opponent)
    # take the WEEKS_AHEAD==1 fixture specifically, since a compact card can only show one game and
    # "what's next" is more useful there than an ambiguous multi-week aggregate. Sorting by
    # weeks_ahead before the groupby is what makes pandas's "first" aggregation reliably pick that
    # nearest fixture's row rather than depending on incidental CSV row order.
    for column in DISPLAY_COLUMNS:
        if column not in predictions.columns:
            predictions[column] = 0
    nearest = predictions[(predictions["weeks_ahead"] <= horizon)
                          & (predictions["weeks_ahead"] >= start_week)].sort_values("weeks_ahead")
    agg = {"predicted_points": "sum", **{c: "first" for c in DISPLAY_COLUMNS}}
    grouped = nearest.groupby(
        ["id", "web_name", "team_name", "position", "now_cost"], as_index=False
    ).agg(agg)
    return grouped


def slim(player):
    return {k: player.get(k) for k in ("id", "web_name", "team_name", "position", "now_cost",
                                       "predicted_points", "ownership_pct", "is_starter", "is_captain")}


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
        for key in ("haul_prob", "ownership_pct"):
            p[key] = float(p[key]) if pd.notna(p[key]) else 0.0

    squad = pick_squad(players)
    predicted_total = sum(
        p["predicted_points"] * (2 if p["is_captain"] else 1) for p in squad if p["is_starter"]
    )

    # The crowd's squad: same rules, same budget, optimised on ownership instead of the model.
    # Scored alongside the model's pick every week so "is the model adding anything over just
    # following the template" is a number, not an impression.
    template = pick_squad(players, value_key="ownership_pct")

    entry = {
        "event": int(event),
        "picked_at": datetime.now(timezone.utc).isoformat(),
        "squad": squad,
        "predicted_total": round(predicted_total, 2),
        "actual_total": None,
        "scored_at": None,
        "template_squad": [slim(p) for p in template],
        "template_total": None,
        "game_average": None,
    }
    return entry


def fetch_points(session, player_id, event):
    history = session.get(f"{BASE_URL}/element-summary/{player_id}/", timeout=30).json().get("history", [])
    match = next((m for m in history if m.get("round") == event), None)
    return match.get("total_points", 0) if match else 0


def score_squad(session, squad, event):
    total = 0
    breakdown = []
    for player in (p for p in squad if p["is_starter"]):
        points = fetch_points(session, player["id"], event)
        total += points * (2 if player["is_captain"] else 1)
        breakdown.append({"web_name": player["web_name"], "actual_points": points, "captain": player["is_captain"]})
    return total, breakdown


def score_pick(entry, session, event_info=None):
    entry["actual_total"], entry["breakdown"] = score_squad(session, entry["squad"], entry["event"])
    if entry.get("template_squad"):
        entry["template_total"], _ = score_squad(session, entry["template_squad"], entry["event"])
    if event_info:
        entry["game_average"] = event_info.get("average_entry_score")
    entry["scored_at"] = datetime.now(timezone.utc).isoformat()
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
            score_pick(entry, session, event)
            template_note = (f", template {entry['template_total']}" if entry.get("template_total") is not None else "")
            print(f"Scored GW{entry['event']}: predicted {entry['predicted_total']}, actual {entry['actual_total']}"
                  f"{template_note}, game average {entry.get('game_average')}"
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
                      f"£{p['now_cost']/10:.1f}m  {p['predicted_points']:.2f}  {100 * (p.get('haul_prob') or 0):.0f}% haul{tag}")
            bench = [p for p in entry["squad"] if not p["is_starter"]]
            print("  Bench: " + ", ".join(f"{p['web_name']} ({p['position']})" for p in bench))
            template_captain = next(p for p in entry["template_squad"] if p["is_captain"])
            print(f"  Template squad (by ownership) captained by {template_captain['web_name']} -- scored alongside once the GW finishes.")

    scored = [e for e in history if e.get("actual_total") is not None and e.get("template_total") is not None]
    if scored:
        model_pts = sum(e["actual_total"] for e in scored)
        template_pts = sum(e["template_total"] for e in scored)
        avg_pts = sum(e["game_average"] or 0 for e in scored)
        print(f"Running total over {len(scored)} scored GW(s): model {model_pts}, template {template_pts}, game average {avg_pts}")

    save_history(history)
    print(f"Saved {HISTORY_FILE}")


if __name__ == "__main__":
    ensure_utf8_stdout()
    main()
