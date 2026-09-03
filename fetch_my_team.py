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
TOP_N_OPTIONS = 3
# This is a multi-week planning decision (use transfers now vs. bank them), not a single-gameweek
# call -- 5 gameweeks is deliberately the outlook here (wider than "My Team"'s own rating, which
# stayed at 3): long enough to judge whether a transfer's benefit holds up, short enough that the
# fixtures being weighed haven't drifted into guesswork.
HORIZON_GAMEWEEKS = 5
# Tie-break only, same role as pick_squad's BENCH_WEIGHT -- makes the solver prefer fewer changes
# when two squads predict equally well, instead of recommending a pointless swap on a coin flip.
TRANSFER_PENALTY = 0.001
# The real cost of a transfer beyond your free allowance -- FPL's own rule, not a modeling choice.
HIT_COST_PER_TRANSFER = 4
FREE_TRANSFER_CAP = 5
FIRST_TRANSFER_EVENT = 2  # GW1 is the initial squad -- no transfers possible, nothing to bank yet
ROLLOVER_CHIPS = {"wildcard", "freehit"}


def fetch_current_squad(session, team_id):
    entry = session.get(f"{BASE_URL}/entry/{team_id}/", timeout=30).json()
    event = entry["current_event"]
    picks_data = session.get(f"{BASE_URL}/entry/{team_id}/event/{event}/picks/", timeout=30).json()
    return entry, event, picks_data


def compute_free_transfers(session, team_id, current_event):
    """Free transfers available going into `current_event`. FPL's public API has no field for
    this directly -- confirmed by checking /entry/{id}/ and /entry/{id}/history/, neither exposes
    it -- only the raw per-gameweek transfer counts this reconstructs it from. FPL's own rule: +1
    free transfer each gameweek, rolling over up to a cap of 5 (so you can bank up to 4 beyond the
    1 you'd get anyway); a gameweek played under Wildcard or Free Hit doesn't touch the running
    count at all, since those chips give unlimited free transfers for that week only."""
    history = session.get(f"{BASE_URL}/entry/{team_id}/history/", timeout=30).json()
    chip_events = {c["event"] for c in history.get("chips", []) if c.get("name") in ROLLOVER_CHIPS}

    free_transfers = 1
    for gw in sorted(history["current"], key=lambda e: e["event"]):
        event = gw["event"]
        if event < FIRST_TRANSFER_EVENT or event >= current_event or event in chip_events:
            continue
        remaining = max(0, free_transfers - gw["event_transfers"])
        free_transfers = min(FREE_TRANSFER_CAP, remaining + 1)
    return free_transfers


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


def pick_with_transfers(current_ids, current_value, bank, all_players, max_transfers, exclude_combos=()):
    """Same MILP family as pick_squad, plus one extra piece: at most `max_transfers` players in
    the new squad may be ones not already owned. Budget is bank + current squad value (selling any
    owned player refunds its current listed price) rather than a flat budget, so keeping a player
    costs nothing and swapping X for Y costs exactly Y's price minus X's against the bank -- this
    is what correctly prices "free to keep, real cost only on genuine changes."

    exclude_combos: previously-found sets of newly-bought player ids -- each one gets a cutting-
    plane constraint ("don't buy ALL of these same new players again") so a repeated call is
    forced to surface a genuinely different combination instead of re-finding the same optimum,
    which is how pick_top_transfer_scenarios below builds a ranked list of distinct options."""
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
    for combo in exclude_combos:
        prob += pulp.lpSum(squad[i] for i in combo) <= len(combo) - 1

    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[status] != "Optimal":
        return None

    squad_ids = [i for i in squad if squad[i].value() == 1]
    starter_ids = {i for i in starter if starter[i].value() == 1}
    captain_id = max(starter_ids, key=lambda i: by_id[i]["predicted_points"])
    return [{**by_id[i], "is_starter": i in starter_ids, "is_captain": i == captain_id} for i in squad_ids]


def pick_top_transfer_scenarios(current_ids, current_value, bank, all_players, max_transfers, current_total, top_n=3):
    """The single best combination of transfers only shows you one option -- this repeats the
    search, excluding each previously-found combination of new players, to surface up to `top_n`
    genuinely distinct alternatives ranked by predicted gain. Stops early (rather than padding
    with weaker filler) once a candidate no longer actually beats the current squad, or once no
    further distinct feasible combination exists at all."""
    scenarios = []
    exclude_combos = []
    for _ in range(top_n):
        squad = pick_with_transfers(current_ids, current_value, bank, all_players, max_transfers, exclude_combos)
        if squad is None:
            break
        total = starting_total(squad)
        if total <= current_total:
            break
        new_ids = frozenset(p["id"] for p in squad if p["id"] not in current_ids)
        if not new_ids:
            break
        scenarios.append(squad)
        exclude_combos.append(new_ids)
    return scenarios


def starting_total(squad):
    return round(sum(p["predicted_points"] * (2 if p["is_captain"] else 1) for p in squad if p["is_starter"]), 2)


def load_all_players():
    predictions = pd.read_csv(PREDICTIONS_FILE)
    nearest = load_nearest_players(predictions, horizon=HORIZON_GAMEWEEKS)
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

    # entry["current_event"] (used for `event` above) is Nathan's last-LOCKED squad snapshot --
    # it doesn't advance until a new gameweek's deadline actually passes, so right up until then
    # it still points at the gameweek that just finished. Free-transfer planning needs the next
    # gameweek transfers would actually apply to, which is exactly what predictions.csv's own
    # weeks_ahead==1 already resolves to (fpl_ml_model.py's own deadline-aware target event).
    raw_predictions = pd.read_csv(PREDICTIONS_FILE)
    upcoming_event = int(raw_predictions.loc[raw_predictions["weeks_ahead"] == 1, "event"].iloc[0])
    free_transfers = compute_free_transfers(session, MY_TEAM_ID, upcoming_event)
    next_week_free_transfers_if_hold = min(FREE_TRANSFER_CAP, free_transfers + 1)

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

    # Check using 1 transfer through one hit beyond the free allowance -- e.g. with 2 free
    # transfers, that's 1, 2 (both free) and 3 (1 hit), so the -4 cost of overspending is visible
    # right next to the free options rather than assumed away.
    transfer_counts = list(range(1, free_transfers + 2))
    transfer_scenarios = {}
    best_net_gain = 0.0  # 0 transfers ("hold") is always a valid baseline with zero net gain
    recommended_transfers = 0
    for n in transfer_counts:
        hit_cost = max(0, n - free_transfers) * HIT_COST_PER_TRANSFER
        ranked = pick_top_transfer_scenarios(current_ids, current_value, bank_raw, all_players, n, current_total, TOP_N_OPTIONS)
        scenario_list = []
        for rank, squad in enumerate(ranked):
            total = starting_total(squad)
            net_gain = round(total - current_total - hit_cost, 2)
            new_ids = {p["id"] for p in squad}
            transferred_out = sorted((by_id[pid] for pid in current_ids if pid not in new_ids), key=lambda p: p["position"])
            transferred_in = sorted((p for p in squad if p["id"] not in current_ids), key=lambda p: p["position"])
            scenario_list.append({
                "squad": squad,
                "predicted_total": total,
                "rating_pct": round(100 * total / top_total, 1) if top_total else None,
                "hit_cost": hit_cost,
                "net_gain": net_gain,
                "transfers_out": transferred_out,
                "transfers_in": transferred_in,
            })
            if rank == 0 and net_gain > best_net_gain:
                best_net_gain = net_gain
                recommended_transfers = n
        transfer_scenarios[str(n)] = scenario_list

    # Deliberately NOT saving manager_name/team_name into the output file -- this JSON gets baked
    # into a public GitHub Pages dashboard, and a real name in a public git history is effectively
    # permanent. Fine to print to the console for a local sanity check, not fine to publish.
    manager_name = f"{entry.get('player_first_name', '')} {entry.get('player_last_name', '')}".strip()
    output = {
        "team_id": MY_TEAM_ID,
        "event": event,
        "upcoming_event": upcoming_event,
        "horizon_gameweeks": HORIZON_GAMEWEEKS,
        "bank": bank_raw / 10,
        "team_value": team_value_raw / 10,
        "free_transfers": free_transfers,
        "next_week_free_transfers_if_hold": next_week_free_transfers_if_hold,
        "recommended_transfers": recommended_transfers,
        "recommended_net_gain": best_net_gain,
        "current_squad": current_lineup,
        "current_predicted_total": current_total,
        "top_team_predicted_total": top_total,
        "rating_pct": rating_pct,
        "transfer_scenarios": transfer_scenarios,
    }
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"{manager_name}'s team \"{entry.get('name')}\" going into GW{upcoming_event} "
          f"({HORIZON_GAMEWEEKS}-GW outlook, {free_transfers} free transfer(s) available): "
          f"{current_total} pts predicted ({rating_pct}% of the model's own best-possible {top_total}-pt squad)")
    for n in transfer_counts:
        options = transfer_scenarios[str(n)]
        hit_note = "" if n <= free_transfers else f", includes a {(n - free_transfers) * HIT_COST_PER_TRANSFER}-pt hit"
        if not options:
            print(f"  With {n} transfer(s){hit_note}: no changes beat your current squad")
            continue
        print(f"  With {n} transfer(s){hit_note}, top {len(options)} option(s):")
        for rank, s in enumerate(options, start=1):
            moves = ", ".join(f"{o['web_name']} -> {i['web_name']}" for o, i in zip(s["transfers_out"], s["transfers_in"]))
            print(f"    {rank}. {s['predicted_total']} pts, net {s['net_gain']:+.2f} after any hit ({s['rating_pct']}%) -- {moves}")

    if recommended_transfers == 0:
        print(f"\nRecommendation: HOLD. No transfer combination found nets a real gain over your "
              f"current squad -- banking would take you to {next_week_free_transfers_if_hold} free "
              f"transfer(s) next week.")
    else:
        print(f"\nRecommendation: use {recommended_transfers} transfer(s) now for a net gain of "
              f"+{best_net_gain:.2f} pts over {HORIZON_GAMEWEEKS} GWs.")
    print(
        "(This only measures the cost of waiting -- the predicted points forgone by not "
        "transferring now -- not the value of waiting, which depends on information the model "
        "doesn't have yet: a price rise, a fitness update, a fixture swing further out.)"
    )
    print(f"Saved {OUTPUT_FILE}")


if __name__ == "__main__":
    ensure_utf8_stdout()
    main()
