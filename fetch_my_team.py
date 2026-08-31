"""Pulls Nathan's real FPL squad (not the model's own auto-picked tracking team -- see
pick_team.py/fpl_ml_team_history.jsonl for that) via FPL's public entry API, rates it as a
percentage of the model's own best-possible squad for the same budget, and searches for the best
combination of 1 and 2 transfers within his actual bank.

The transfer search is a proper joint MILP (same PuLP/CBC approach as pick_squad), not an ad-hoc
check of a handful of hand-picked candidates -- an earlier manual check of Anderson vs. Gomez as a
Tzolis replacement missed Gomez entirely because the price range searched by hand was too narrow.
Searching the full affordable player pool at once removes that whole class of mistake.

Runs server-side during the manual refresh. FPL's API sends no CORS headers, so this couldn't run
as browser JS on the GitHub Pages site directly -- that's a separate, deferred piece of work for
letting other people check their own team; this script only ever looks up Nathan's own team ID.
"""
import json
from pathlib import Path

import pandas as pd
import pulp
import requests

from fpl_ml_model import ensure_utf8_stdout
from pick_team import (
    BASE_URL,
    BENCH_WEIGHT,
    DISPLAY_COLUMNS,
    MAX_PER_CLUB,
    SQUAD_QUOTAS,
    STARTER_MAX,
    STARTER_MIN,
    load_nearest_players,
    pick_squad,
)

MY_TEAM_ID = 4340534
PREDICTIONS_FILE = Path("fpl_ml_predictions.csv")
OUTPUT_FILE = Path("my_fpl_team.json")
TRANSFER_OPTIONS = [1, 2]
# Tie-break only, same role as pick_squad's BENCH_WEIGHT -- makes the solver prefer fewer changes
# when two squads predict equally well, instead of recommending a pointless swap on a coin flip.
TRANSFER_PENALTY = 0.001


def fetch_current_squad(session, team_id):
    entry = session.get(f"{BASE_URL}/entry/{team_id}/", timeout=30).json()
    event = entry["current_event"]
    picks_data = session.get(f"{BASE_URL}/entry/{team_id}/event/{event}/picks/", timeout=30).json()
    return entry, event, picks_data


def pick_best_lineup(players):
    """Given a FIXED set of players (not choosing which 15 -- just which 11 start and who
    captains), finds the FPL-legal starting XI that maximizes predicted points."""
    prob = pulp.LpProblem("fpl_lineup", pulp.LpMaximize)
    starter = {p["id"]: pulp.LpVariable(f"start_{p['id']}", cat="Binary") for p in players}
    by_id = {p["id"]: p for p in players}

    prob += pulp.lpSum(starter[i] * by_id[i]["predicted_points"] for i in starter)
    prob += pulp.lpSum(starter.values()) == 11
    for position in SQUAD_QUOTAS:
        starters_in_pos = pulp.lpSum(starter[i] for i in starter if by_id[i]["position"] == position)
        prob += starters_in_pos >= STARTER_MIN[position]
        prob += starters_in_pos <= STARTER_MAX[position]

    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"Lineup optimization failed: {pulp.LpStatus[status]}")

    starter_ids = {i for i in starter if starter[i].value() == 1}
    captain_id = max(starter_ids, key=lambda i: by_id[i]["predicted_points"])
    return [{**by_id[i], "is_starter": i in starter_ids, "is_captain": i == captain_id} for i in by_id]


def pick_with_transfers(current_ids, current_value, bank, all_players, max_transfers):
    """Same MILP family as pick_squad, plus one extra piece: at most `max_transfers` players in
    the new squad may be ones not already owned. Budget is bank + current squad value (selling any
    owned player refunds its current listed price) rather than a flat budget, so keeping a player
    costs nothing and swapping X for Y costs exactly Y's price minus X's against the bank -- this
    is what correctly prices "free to keep, real cost only on genuine changes."""
    prob = pulp.LpProblem("fpl_transfers", pulp.LpMaximize)
    squad = {p["id"]: pulp.LpVariable(f"squad_{p['id']}", cat="Binary") for p in all_players}
    starter = {p["id"]: pulp.LpVariable(f"start_{p['id']}", cat="Binary") for p in all_players}
    by_id = {p["id"]: p for p in all_players}

    new_players_in = pulp.lpSum(squad[i] for i in squad if i not in current_ids)
    prob += (
        pulp.lpSum(
            starter[i] * by_id[i]["predicted_points"] + BENCH_WEIGHT * squad[i] * by_id[i]["predicted_points"]
            for i in squad
        )
        - TRANSFER_PENALTY * new_players_in
    )

    prob += pulp.lpSum(squad.values()) == 15
    prob += pulp.lpSum(by_id[i]["now_cost"] * squad[i] for i in squad) <= bank + current_value
    prob += pulp.lpSum(starter.values()) == 11
    for i in squad:
        prob += starter[i] <= squad[i]

    for position, quota in SQUAD_QUOTAS.items():
        prob += pulp.lpSum(squad[i] for i in squad if by_id[i]["position"] == position) == quota
    for position in SQUAD_QUOTAS:
        starters_in_pos = pulp.lpSum(starter[i] for i in squad if by_id[i]["position"] == position)
        prob += starters_in_pos >= STARTER_MIN[position]
        prob += starters_in_pos <= STARTER_MAX[position]

    for team_name in {p["team_name"] for p in all_players}:
        prob += pulp.lpSum(squad[i] for i in squad if by_id[i]["team_name"] == team_name) <= MAX_PER_CLUB

    prob += new_players_in <= max_transfers

    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"Transfer optimization failed: {pulp.LpStatus[status]}")

    squad_ids = [i for i in squad if squad[i].value() == 1]
    starter_ids = {i for i in starter if starter[i].value() == 1}
    captain_id = max(starter_ids, key=lambda i: by_id[i]["predicted_points"])
    return [{**by_id[i], "is_starter": i in starter_ids, "is_captain": i == captain_id} for i in squad_ids]


def starting_total(squad):
    return round(sum(p["predicted_points"] * (2 if p["is_captain"] else 1) for p in squad if p["is_starter"]), 2)


def load_all_players():
    predictions = pd.read_csv(PREDICTIONS_FILE)
    nearest = load_nearest_players(predictions)
    columns = ["id", "web_name", "team_name", "position", "now_cost", "predicted_points"] + DISPLAY_COLUMNS
    players = nearest[columns].to_dict("records")
    for p in players:
        for key in ("team_code", "opponent_code"):
            p[key] = int(p[key]) if pd.notna(p[key]) else None
        for key in ("opponent_name", "opponent_short_name"):
            p[key] = p[key] if pd.notna(p[key]) else None
        p["was_home"] = bool(p["was_home"]) if pd.notna(p["was_home"]) else None
        p["difficulty"] = int(p["difficulty"]) if pd.notna(p["difficulty"]) else None
    return players


def main():
    session = requests.Session()
    entry, event, picks_data = fetch_current_squad(session, MY_TEAM_ID)
    bank_raw = picks_data["entry_history"]["bank"]
    team_value_raw = picks_data["entry_history"]["value"]

    all_players = load_all_players()
    by_id = {p["id"]: p for p in all_players}

    fpl_elements = {e["id"]: e for e in session.get(f"{BASE_URL}/bootstrap-static/", timeout=30).json()["elements"]}
    current_ids_raw = [pick["element"] for pick in picks_data["picks"]]
    missing = [pid for pid in current_ids_raw if pid not in by_id]
    if missing:
        names = [fpl_elements[pid]["web_name"] for pid in missing if pid in fpl_elements]
        print(f"Warning: {len(missing)} owned player(s) have no current-gameweek prediction row "
              f"(likely a blank gameweek or unmatched fixture) -- excluded from optimization: {names}")
    current_ids = {pid for pid in current_ids_raw if pid in by_id}
    current_players = [by_id[pid] for pid in current_ids]
    current_value = sum(p["now_cost"] for p in current_players)

    current_lineup = pick_best_lineup(current_players)
    current_total = starting_total(current_lineup)

    top_team = pick_squad(all_players)
    top_total = starting_total(top_team)
    rating_pct = round(100 * current_total / top_total, 1) if top_total else None

    transfer_scenarios = {}
    for n in TRANSFER_OPTIONS:
        squad = pick_with_transfers(current_ids, current_value, bank_raw, all_players, n)
        total = starting_total(squad)
        new_ids = {p["id"] for p in squad}
        transferred_out = sorted((by_id[pid] for pid in current_ids if pid not in new_ids), key=lambda p: p["position"])
        transferred_in = sorted((p for p in squad if p["id"] not in current_ids), key=lambda p: p["position"])
        transfer_scenarios[str(n)] = {
            "squad": squad,
            "predicted_total": total,
            "rating_pct": round(100 * total / top_total, 1) if top_total else None,
            "transfers_out": transferred_out,
            "transfers_in": transferred_in,
        }

    # Deliberately NOT saving manager_name/team_name into the output file -- this JSON gets baked
    # into a public GitHub Pages dashboard, and a real name in a public git history is effectively
    # permanent. Fine to print to the console for a local sanity check, not fine to publish.
    manager_name = f"{entry.get('player_first_name', '')} {entry.get('player_last_name', '')}".strip()
    output = {
        "team_id": MY_TEAM_ID,
        "event": event,
        "bank": bank_raw / 10,
        "team_value": team_value_raw / 10,
        "current_squad": current_lineup,
        "current_predicted_total": current_total,
        "top_team_predicted_total": top_total,
        "rating_pct": rating_pct,
        "transfer_scenarios": transfer_scenarios,
    }
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"{manager_name}'s team \"{entry.get('name')}\" (GW{event}): {current_total} pts predicted "
          f"({rating_pct}% of the model's own best-possible {top_total}-pt squad)")
    for n in TRANSFER_OPTIONS:
        s = transfer_scenarios[str(n)]
        if s["transfers_out"]:
            moves = ", ".join(f"{o['web_name']} -> {i['web_name']}" for o, i in zip(s["transfers_out"], s["transfers_in"]))
        else:
            moves = "no changes beat your current squad"
        print(f"  With {n} transfer(s): {s['predicted_total']} pts ({s['rating_pct']}%) -- {moves}")
    print(f"Saved {OUTPUT_FILE}")


if __name__ == "__main__":
    ensure_utf8_stdout()
    main()
