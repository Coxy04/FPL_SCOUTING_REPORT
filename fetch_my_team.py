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
TOP_N_OPTIONS = 5
# This is a multi-week planning decision (use transfers now vs. bank them), not a single-gameweek
# call -- 5 gameweeks is deliberately the outlook here (wider than "My Team"'s own rating, which
# stayed at 3): long enough to judge whether a transfer's benefit holds up, short enough that the
# fixtures being weighed haven't drifted into guesswork.
HORIZON_GAMEWEEKS = 5
# Picking a starting XI and bench order is a THIS WEEK decision -- you set it again next gameweek,
# so a 5-gameweek total is the wrong number to plan a bench on. The best XI over five weeks and
# the best XI for the next match are genuinely different teams when fixtures diverge.
HORIZON_LINEUP = 1
# Tie-break only, same role as pick_squad's BENCH_WEIGHT -- makes the solver prefer fewer changes
# when two squads predict equally well, instead of recommending a pointless swap on a coin flip.
TRANSFER_PENALTY = 0.001
# The real cost of a transfer beyond your free allowance -- FPL's own rule, not a modeling choice.
HIT_COST_PER_TRANSFER = 4
# Predicted transfer gains are systematically too optimistic and must be discounted before being
# weighed against a hit. Two structural reasons, neither about football: pick_with_transfers takes
# the max over ~700 players, so it selects for players whose predictions are noisily HIGH, while
# the incumbents it sells are a fixed 15 with no such selection applied; and a hit costs -4 once
# while the gain is summed over the whole horizon, so any per-week bias gets multiplied.
#
# calibrate_transfer_gains.py measures the size of that gap by replaying the decision on last
# season's archive: 288 scenarios, incumbent squads at 84.7% of optimal (matching a real managed
# squad), predicted +19.72 -> realised +12.97. Re-run it to refresh this number; without it the
# tool recommended a -12 hit as its BEST option, which is what prompted measuring this at all.
TRANSFER_GAIN_SHRINKAGE = 0.66
# Shrinkage varies a lot by position, and the flat rate above is known wrong in both directions.
# Measured (90% CI): GK 0.55 [-0.13, 1.30], DEF 0.89 [0.66, 1.12], MID 0.57 [0.35, 0.82],
# FWD 0.08 [-0.33, 0.48]. DEF and FWD don't overlap at all -- defender swaps deliver roughly what
# the model predicts, forward swaps essentially don't (51% of them gained anything, a coin flip).
# Plausible reading: defender points lean on clean sheets, which are a team-level property the
# model's team-form features genuinely capture, while the gap between two decent forwards is
# mostly finishing variance the model can't see.
#
# These are PARTIALLY POOLED 50/50 with the overall rate rather than used raw: n<100 per position
# and a ratio-of-means estimate, so the extremes (especially FWD's 0.08) are not worth taking at
# face value yet. As live snapshots accumulate and the samples grow, the pooling weight should
# move toward the per-position numbers. GK stays effectively at the pooled rate -- its interval is
# too wide to say anything, though GK gains are so small it rarely changes a decision.
_MEASURED_BY_POSITION = {"GK": 0.55, "DEF": 0.89, "MID": 0.57, "FWD": 0.08}
POSITION_POOLING_WEIGHT = 0.5
TRANSFER_GAIN_SHRINKAGE_BY_POSITION = {
    position: round(
        POSITION_POOLING_WEIGHT * measured + (1 - POSITION_POOLING_WEIGHT) * TRANSFER_GAIN_SHRINKAGE, 3
    )
    for position, measured in _MEASURED_BY_POSITION.items()
}
# A hit has to beat the best hit-free option by this margin before it's worth recommending.
# The calibration's own sub-slices disagree by roughly this much: on a live example the 3-transfer
# option came out at +13.2 using the overall 0.66, +12.4 using per-transfer-count shrinkages, and
# +14.2 using per-gain-size ones -- while the 2-transfer (free) option sat at +11.7 to +12.9. Those
# slices are noisy (n=96 each, non-monotonic, and gain-size is confounded with transfer count), so
# a sub-3-point edge for a hit is not a real edge. A hit is also irreversible and spends a transfer
# that would otherwise bank, so ties should break towards not taking it.
HIT_DECISION_MARGIN = 3.0
FREE_TRANSFER_CAP = 5
FIRST_TRANSFER_EVENT = 2  # GW1 is the initial squad -- no transfers possible, nothing to bank yet
ROLLOVER_CHIPS = {"wildcard", "freehit"}

# Flag the most- and least-backed decile of THIS week's players, rather than an absolute
# net-transfer rate. See compute_price_pressure for why the absolute version had to be abandoned.
PRICE_PRESSURE_DECILE = 0.10
PRICE_MIN_OWNERS = 1000

# How many weeks of banking to evaluate. Capped at 2 because it leans on predictions further out,
# and the horizon stops at 5 precisely because they stop being trustworthy past that.
MAX_BANK_WEEKS = 2
# EXTRA discount on a deferred gain, on top of the per-position shrinkage every move already gets.
# A banked move carries the same optimiser's curse as an immediate one PLUS an assumption an
# immediate move never makes: that today's view of next week survives to next week.
#
# calibrate_hold_value.py measured it on 440 paired scenarios where both arms spend the SAME number
# of transfers over the SAME weeks and end with none banked, so only the timing differs. Deferred
# gains realised 0.53 of their projection against 0.71 for immediate ones -- a relative 0.75.
#
# But the headline is not the number to ship, because the calibration mostly measures a regime this
# tool is rarely in. Its mean raw gain is +19.6 and only 17% of pairs sit below +11, while a typical
# live week here offers +5 to +11. In that low-gain slice the relative rate is far harsher (0.35,
# n=76) and banking actually LOST by 1.4 pts. So this is partially pooled 50/50 between the two,
# exactly as TRANSFER_GAIN_SHRINKAGE_BY_POSITION is, rather than trusting either a noisy 76-scenario
# slice or a headline measured somewhere else: 0.5*0.35 + 0.5*0.75 = 0.55.
HOLD_GAIN_SHRINKAGE = 0.55
# Banking must clear the best act-now option by this margin before it is recommended. The measured
# hold edge is +0.70 pts with a 90% CI of [-0.99, +2.24] -- i.e. statistically indistinguishable
# from zero, and holding won only 49% of pairs. A sub-2-point edge for banking is therefore not an
# edge at all, it is the width of the measurement. Same reasoning as HIT_DECISION_MARGIN.
HOLD_DECISION_MARGIN = 2.0

# Rolling window (in weeks) used to judge Wildcard timing -- see evaluate_chip_timing. Set equal
# to the model's own horizon: a wildcard's whole case rests on a gap that persists across the
# window the model can actually see, and there is no visibility to judge a longer one anyway.
WILDCARD_WINDOW = HORIZON_GAMEWEEKS

# Nathan's mini-league, for gauging how contested a transfer target is among the managers he's
# actually competing with -- see fetch_league_ownership. Same convention as MY_TEAM_ID: a personal
# account fact, hardcoded rather than passed around, since this tool only ever runs for Nathan.
MY_LEAGUE_ID = 1026520
# None = every rival in the league (his is 11 entries, small enough that "top 5" was an arbitrary
# cut with no reason behind it -- caught when he pointed out the league has more members than that).
# The cap exists only as a safety net against hammering the API with picks requests for a genuinely
# huge public league, not as a considered choice of how many rivals matter.
LEAGUE_TOP_N = None
LEAGUE_MAX_RIVALS = 20


def fetch_current_squad(session, team_id):
    entry = session.get(f"{BASE_URL}/entry/{team_id}/", timeout=30).json()
    event = entry["current_event"]
    picks_data = session.get(f"{BASE_URL}/entry/{team_id}/event/{event}/picks/", timeout=30).json()
    return entry, event, picks_data


def compute_selling_prices(session, team_id, picks, elements_by_id):
    """What each owned player would ACTUALLY sell for, which is not their listed price.

    FPL returns the purchase price plus half of any subsequent rise, rounded down to 0.1m; a price
    fall is absorbed in full. Valuing a squad at listed prices therefore overstates the budget and
    lets the optimiser propose transfers that don't actually fit -- the failure this was added to
    fix, after a recommended move turned out to be unaffordable in the app.

    `selling_price` exists only on FPL's authenticated /my-team/ endpoint, but it's derivable from
    public data: cost_change_start gives the move since the season began, so a player never
    transferred in was bought at now_cost - cost_change_start, and /entry/{id}/transfers/ carries
    element_in_cost for anyone bought since (latest purchase wins if bought more than once)."""
    transfers = session.get(f"{BASE_URL}/entry/{team_id}/transfers/", timeout=30).json()
    purchase_by_element = {}
    for transfer in sorted(transfers, key=lambda t: t.get("event", 0)):
        purchase_by_element[transfer["element_in"]] = transfer["element_in_cost"]

    selling = {}
    for pick in picks:
        element_id = pick["element"]
        element = elements_by_id.get(element_id)
        if element is None:
            continue
        now_cost = element["now_cost"]
        purchase = purchase_by_element.get(element_id, now_cost - element.get("cost_change_start", 0))
        profit = max(0, now_cost - purchase)
        # The min() is the loss half of the rule and is not redundant: a player whose price has
        # FALLEN has zero profit, so `purchase + 0` would hand back the original (higher) price and
        # invent money. Le Fee, bought at 6.0 and now 5.9, sells at 5.9.
        selling[element_id] = min(now_cost, purchase + profit // 2)
    return selling


def compute_price_pressure(bootstrap):
    """Flags who the market is about to reprice, so a transfer can be timed rather than just chosen.

    FPL moves a player's price off net transfers, so net transfer flow LEADS the price. Measured on
    the 2025-26 archive (ablation_price_trajectory.py), rate = transfers_balance / selected predicts
    the NEXT gameweek's price move with correlation 0.31 and a cleanly monotonic decile curve.

    Ranks within the current gameweek rather than against an absolute rate, which is not a detail.
    An absolute cut was tried first and had to be thrown out: transfer churn collapses as a season
    settles, so the same +/-0.06 threshold that flags a sane 8% of players as falling in GW20 flags
    68% of them in GW3. A flag that fires on two thirds of the league says nothing. Ranking within
    the week is self-normalising, and the predictive power survives it in every block of the season:

        top decile of the week    -> 12-16% rise next GW vs a ~2% base rate, 0.4% fall
        bottom decile of the week -> 15-26% fall next GW vs a 2.5-11.5% base rate, 0.3% rise

    Wrong-direction rates under 0.5% are what make this worth surfacing: the flag is rarely
    backwards, it just won't catch every real move.

    Emphatically NOT a points signal. The same ablation found price momentum worth +0.1% MAE --
    i.e. nothing -- as a model feature, and a plain ownership control beat both trajectory variants.
    The market predicts its own prices, not the football, so this belongs on the timing of a move
    that was already chosen on merit, never on the choice itself.

    Two caveats. (1) transfers_*_event is the CURRENT window still filling up, while the calibration
    used complete windows; ranking makes this far less damaging than it would be for an absolute cut,
    since every player is equally under-counted, but the ordering can still be rough right after a
    deadline. (2) FPL reprices daily, so a weekly-resolution rate is a proxy for a faster process."""
    total_managers = bootstrap["total_players"]
    rates = {}
    for element in bootstrap["elements"]:
        owners = (float(element["selected_by_percent"]) / 100) * total_managers
        if owners < PRICE_MIN_OWNERS:
            # A near-unowned player's ratio is dominated by its tiny denominator -- the calibration
            # excluded these for the same reason rather than emitting confident noise.
            continue
        rates[element["id"]] = (element["transfers_in_event"] - element["transfers_out_event"]) / owners

    ordered = sorted(rates.values())
    if ordered:
        rise_cut = ordered[int(len(ordered) * (1 - PRICE_PRESSURE_DECILE))]
        fall_cut = ordered[int(len(ordered) * PRICE_PRESSURE_DECILE)]
    else:
        rise_cut = fall_cut = 0.0

    pressure = {}
    for element in bootstrap["elements"]:
        rate = rates.get(element["id"])
        direction = None
        if rate is not None:
            if rate >= rise_cut:
                direction = "rising"
            elif rate <= fall_cut:
                direction = "falling"
        pressure[element["id"]] = {
            "rate": round(rate, 4) if rate is not None else None,
            "direction": direction,
            "change_this_gw": element["cost_change_event"],
        }
    return pressure


def fetch_league_ownership(session, league_id, my_team_id, event, top_n=LEAGUE_TOP_N):
    """What fraction of the managers actually being competed against already own each player.

    `top_n=None` (the default) means every rival in the league, capped only at LEAGUE_MAX_RIVALS
    as a safety net against a genuinely huge public league, not as a considered "top N" cutoff --
    a small private league has no natural reason to look at only some of its rivals.

    Deliberately excludes Nathan's own entry -- the question this answers is about rivals, and
    including yourself would trivially show 100% on anything you already own, which tells you
    nothing about how contested a transfer target is.

    `event` should be the SAME locked gameweek already resolved for Nathan's own squad
    (fetch_current_squad's `event`) -- everyone in a league shares one deadline clock, so there's
    no need to re-resolve it per rival, and picks for that event exist the moment it locks, not
    only once it's been played.

    Both endpoints (league standings, another entry's picks) are public with no auth needed, same
    as Nathan's own team fetch -- confirmed against this actual league before building this.

    Returns None on any failure (bad league id, FPL API hiccup, an unexpected response shape)
    rather than raising -- an external league lookup going wrong should never take down the whole
    refresh over what is, after all, a nice-to-have."""
    try:
        standings = session.get(
            f"{BASE_URL}/leagues-classic/{league_id}/standings/", timeout=30
        ).json()
        league_name = standings.get("league", {}).get("name", "your league")
        rivals = [r for r in standings["standings"]["results"] if r["entry"] != my_team_id]
        rivals = rivals[:top_n or LEAGUE_MAX_RIVALS]
        if not rivals:
            return None

        counts = {}
        managers = []
        for rival in rivals:
            picks = session.get(
                f"{BASE_URL}/entry/{rival['entry']}/event/{event}/picks/", timeout=30
            ).json()["picks"]
            managers.append({"entry_name": rival["entry_name"], "rank": rival["rank"]})
            for pick in picks:
                counts[pick["element"]] = counts.get(pick["element"], 0) + 1

        n = len(rivals)
        ownership_pct = {pid: round(100 * count / n, 0) for pid, count in counts.items()}
        return {"league_name": league_name, "top_n": n, "managers": managers,
                "ownership_pct": ownership_pct}
    except (requests.RequestException, KeyError, ValueError) as exc:
        print(f"Warning: couldn't fetch league ownership for league {league_id} ({exc}); "
              f"skipping league ownership context.")
        return None


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


def evaluate_chip_timing(current_ids, selling_prices, bank_raw):
    """When a Free Hit or Wildcard would be worth most, within the horizon the model can currently
    see (weeks_ahead 1-5 -- there is no visibility past that, so a gap that opens up later, e.g. a
    Christmas blank, will not show here until it enters the window).

    The two chips are fundamentally different shapes of decision, and conflating them would answer
    neither question honestly:

    FREE HIT reverts after one gameweek, so its whole value is a SINGLE-WEEK spike: is there a week
    where an unconstrained squad clears your saved squad by a lot more than usual? That is exactly
    what a blank (your players have no fixture) or a double (an alternative squad's players have
    two) looks like. Measured per week in isolation -- summing it into a multi-week total would
    mean nothing, since the whole point is the squad is thrown away again after one week.

    WILDCARD is permanent, so it needs its own squad, held fixed across every week it covers --
    NOT one squad re-optimized fresh per week, which is what a series of Free Hits would be, and
    would overstate a wildcard's true value by handing it a advantage no wildcard actually has. So
    each candidate wildcard week gets ONE squad chosen to maximise the SUM of predicted points
    from that week to the end of the visible horizon (same method as the headline rating_pct
    already on this dashboard -- wildcarding today is exactly that number, top_total minus
    current_total, and this just re-runs it from later possible starts to see whether the gap
    driving it is real and persistent or already fading by the time it would resolve).

    The window necessarily SHRINKS as the candidate start moves later (5 weeks visible from now,
    4 from next week, ...), so raw point totals are not comparable across starts -- a smaller gap
    later could just mean less remaining time to accumulate one, not a smaller problem. Comparing
    gap PER REMAINING WEEK is what actually answers "is this getting better or worse", and even
    that is on a shrinking, noisier sample as the window narrows.

    Both chips compare against the SAME reachable optimum: pick_squad on your real budget (selling
    value plus bank), not a flat GBP100m -- a wildcard or free hit resets you to what you actually
    have, not a round number."""
    budget = sum(selling_prices.values()) + bank_raw
    predictions = pd.read_csv(PREDICTIONS_FILE)
    event_by_week = predictions.drop_duplicates("weeks_ahead").set_index("weeks_ahead")["event"].to_dict()

    free_hit = []
    free_hit_squads = {}
    for week in range(1, HORIZON_GAMEWEEKS + 1):
        players = load_all_players(horizon=week, start_week=week)
        by_id = {p["id"]: p for p in players}
        ids = {pid for pid in current_ids if pid in by_id}
        if len(ids) < len(current_ids):
            continue
        current = starting_total(pick_best_lineup([by_id[pid] for pid in ids]))
        optimal_squad = pick_squad(players, budget=budget)
        optimal = starting_total(pick_best_lineup(optimal_squad))
        free_hit.append({"weeks_ahead": week, "event": int(event_by_week.get(week, 0)),
                         "current": round(current, 2), "optimal": round(optimal, 2),
                         "gap": round(optimal - current, 2)})
        # Kept for every week, not just the best one -- if that week's gap is too small to be
        # worth a Free Hit, seeing the alternative squad explains WHY (it barely differs from
        # your own), which a bare number can't.
        free_hit_squads[week] = pick_best_lineup(optimal_squad)
    if not free_hit:
        return {"free_hit": [], "wildcard": [], "best_free_hit_week": None, "best_wildcard_week": None}

    free_hit_ranked = sorted(free_hit, key=lambda w: -w["gap"])
    best_fh = free_hit_ranked[0]
    best_fh_squad = free_hit_squads.get(best_fh["weeks_ahead"])

    wildcard = []
    wildcard_now_squad = None
    for start in range(1, HORIZON_GAMEWEEKS + 1):
        players = load_all_players(horizon=HORIZON_GAMEWEEKS, start_week=start)
        by_id = {p["id"]: p for p in players}
        ids = {pid for pid in current_ids if pid in by_id}
        if len(ids) < len(current_ids):
            continue
        current = starting_total(pick_best_lineup([by_id[pid] for pid in ids]))
        optimal_squad = pick_squad(players, budget=budget)
        optimal = starting_total(pick_best_lineup(optimal_squad))
        weeks_remaining = HORIZON_GAMEWEEKS - start + 1
        gap = optimal - current
        wildcard.append({
            "weeks_ahead": start, "event": int(event_by_week.get(start, 0)),
            "weeks_remaining": weeks_remaining, "current": round(current, 2),
            "optimal": round(optimal, 2), "gap": round(gap, 2),
            "gap_per_week": round(gap / weeks_remaining, 2),
        })
        if start == 1:
            # "Wildcard now" is the only one worth showing in full -- later candidate starts exist
            # to judge whether the CURRENT gap persists, not to browse alternative squads for a
            # decision that (if taken) would be made with next week's information anyway.
            wildcard_now_squad = pick_best_lineup(optimal_squad)
    now = wildcard[0] if wildcard else None
    later = wildcard[-1] if len(wildcard) > 1 else None
    fading = bool(now and later and later["gap_per_week"] < 0.6 * now["gap_per_week"])

    return {
        "free_hit": free_hit_ranked,
        "wildcard": wildcard,
        "best_free_hit_week": best_fh,
        "best_free_hit_squad": best_fh_squad,
        "wildcard_now": now,
        "wildcard_now_squad": wildcard_now_squad,
        "wildcard_gap_fading": fading,
        "horizon_weeks": len(free_hit),
        "budget": budget,
    }


def evaluate_bank_plans(current_ids, selling_prices, bank_raw, free_transfers):
    """Every way of using transfers over the next few weeks, on ONE comparable scale.

    This exists because the tool previously scored holding at exactly zero, so it could say "hold,
    nothing is worth doing" but never "hold, next week is worth more" -- a move made now was
    credited with the full horizon while the same move made a week later was credited with nothing.
    Banking pays through two mechanisms that a zero can't represent: a week more information, and
    BUNDLING, since some moves need several simultaneous transfers to express at all (downgrade two
    players to fund a premium) and are unreachable with one free transfer at any price.

    The trick that makes the arms comparable is the window. A move deferred by w weeks is valued
    over weeks w+1..HORIZON only -- the weeks you would actually own the player. The weeks before
    that are identical under either choice (you field the same squad), so they cancel, and both
    numbers mean the same thing: extra points over the next HORIZON gameweeks versus doing nothing.
    That cancellation is why the earlier horizon asymmetry disappears rather than being patched
    over: waiting is now charged for the week of gain it actually costs."""
    plans = []
    for wait in range(MAX_BANK_WEEKS + 1):
        players = load_all_players(start_week=wait + 1)
        by_id = {p["id"]: p for p in players}
        ids = {pid for pid in current_ids if pid in by_id}
        if len(ids) < len(current_ids):
            continue
        base = starting_total(pick_best_lineup([by_id[pid] for pid in ids]))
        value = sum(selling_prices.get(pid, by_id[pid]["now_cost"]) for pid in ids)
        free_then = min(FREE_TRANSFER_CAP, free_transfers + wait)
        for k in range(1, free_then + 2):
            # Top-N then re-rank by NET, not the single raw optimum. Taking the MILP's highest-raw
            # squad here silently biased the whole comparison: position shrinkage varies enough
            # (FWD 0.37 vs DEF 0.78) that the highest-raw move is routinely not the highest-net one,
            # and act-now was being under-reported at +2.35 against its true +3.60 -- an error
            # pointing squarely in favour of banking, the exact thing this function must not do.
            candidates = pick_top_transfer_scenarios(
                ids, value, bank_raw, players, k, base, TOP_N_OPTIONS
            )
            scored = []
            for squad in candidates:
                new_ids = {p["id"] for p in squad}
                incoming = sorted((p for p in squad if p["id"] not in ids), key=lambda p: p["position"])
                if not incoming:
                    continue
                outgoing = sorted((by_id[pid] for pid in ids if pid not in new_ids),
                                  key=lambda p: p["position"])
                raw_gain = starting_total(pick_best_lineup([by_id[p["id"]] for p in squad])) - base
                rates = [TRANSFER_GAIN_SHRINKAGE_BY_POSITION.get(p["position"], TRANSFER_GAIN_SHRINKAGE)
                         for p in incoming]
                shrink = sum(rates) / len(rates)
                scored.append((raw_gain * shrink, raw_gain, shrink, incoming, outgoing))
            if not scored:
                continue
            expected, raw_gain, shrink, incoming, outgoing = max(scored, key=lambda x: x[0])
            # A deferred gain gets a SECOND discount, because it rests on an assumption an
            # immediate move doesn't make: that today's view of next week survives to next week.
            # Until that is measured the plan is still shown, but flagged uncalibrated and kept out
            # of the recommendation.
            if wait:
                expected *= HOLD_GAIN_SHRINKAGE if HOLD_GAIN_SHRINKAGE is not None else 1.0
            hit_cost = max(0, k - free_then) * HIT_COST_PER_TRANSFER
            plans.append({
                "wait_weeks": wait,
                "transfers": len(incoming),
                "free_transfers_then": free_then,
                "raw_gain": round(raw_gain, 2),
                "shrinkage_applied": round(shrink, 3),
                "hold_shrinkage_applied": HOLD_GAIN_SHRINKAGE if wait else None,
                "expected_gain": round(expected, 2),
                "hit_cost": hit_cost,
                "net_gain": round(expected - hit_cost, 2),
                "calibrated": (not wait) or HOLD_GAIN_SHRINKAGE is not None,
                "transfers_in": [{"id": p["id"], "web_name": p["web_name"], "position": p["position"],
                                  "now_cost": p["now_cost"]} for p in incoming],
                "transfers_out": [{"id": p["id"], "web_name": p["web_name"], "position": p["position"],
                                   "now_cost": p["now_cost"]} for p in outgoing],
            })
    plans.sort(key=lambda p: p["net_gain"], reverse=True)
    return plans


def load_all_players(horizon=HORIZON_GAMEWEEKS, start_week=1):
    predictions = pd.read_csv(PREDICTIONS_FILE)
    nearest = load_nearest_players(predictions, horizon=horizon, start_week=start_week)
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

    bootstrap = session.get(f"{BASE_URL}/bootstrap-static/", timeout=30).json()
    fpl_elements = {e["id"]: e for e in bootstrap["elements"]}
    price_pressure = compute_price_pressure(bootstrap)
    current_ids_raw = [pick["element"] for pick in picks_data["picks"]]
    missing = [pid for pid in current_ids_raw if pid not in by_id]
    if missing:
        names = [fpl_elements[pid]["web_name"] for pid in missing if pid in fpl_elements]
        print(f"Warning: {len(missing)} owned player(s) have no current-gameweek prediction row "
              f"(likely a blank gameweek or unmatched fixture) -- excluded from optimization: {names}")
    current_ids = {pid for pid in current_ids_raw if pid in by_id}
    current_players = [by_id[pid] for pid in current_ids]
    # Budget must be built from what these players would actually SELL for, not their listed
    # prices -- see compute_selling_prices. Using listed prices overstates the budget and lets the
    # optimiser propose transfers that don't fit in the app.
    selling_prices = compute_selling_prices(session, MY_TEAM_ID, picks_data["picks"], fpl_elements)
    current_value = sum(selling_prices.get(pid, by_id[pid]["now_cost"]) for pid in current_ids)
    listed_value = sum(p["now_cost"] for p in current_players)
    selling_shortfall = listed_value - current_value

    current_lineup = pick_best_lineup(current_players)
    current_total = starting_total(current_lineup)

    # A second, separate lineup solved on NEXT GAMEWEEK ONLY -- this is what the pitch view shows,
    # because that's the decision it supports (who starts, who benches, who takes the armband this
    # week). The 5-GW numbers above stay as the basis for rating and transfer planning.
    next_gw_by_id = {p["id"]: p for p in load_all_players(horizon=HORIZON_LINEUP)}
    next_gw_players = [next_gw_by_id[pid] for pid in current_ids if pid in next_gw_by_id]
    next_gw_lineup = pick_best_lineup(next_gw_players) if next_gw_players else []
    next_gw_total = starting_total(next_gw_lineup) if next_gw_lineup else 0.0

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
    best_free_gain = 0.0  # best option that doesn't cost a hit -- what a hit has to beat
    best_free_transfers = 0
    for n in transfer_counts:
        hit_cost = max(0, n - free_transfers) * HIT_COST_PER_TRANSFER
        ranked = pick_top_transfer_scenarios(current_ids, current_value, bank_raw, all_players, n, current_total, TOP_N_OPTIONS)
        scenario_list = []
        for squad in ranked:
            total = starting_total(squad)
            raw_gain = total - current_total
            new_ids = {p["id"] for p in squad}
            transferred_out = sorted((by_id[pid] for pid in current_ids if pid not in new_ids), key=lambda p: p["position"])
            transferred_in = sorted((p for p in squad if p["id"] not in current_ids), key=lambda p: p["position"])

            # Discount the predicted gain before netting off the hit -- the hit is a real, certain
            # -4, the gain is an optimistic estimate, so comparing them undiscounted is comparing
            # a hard cost against a soft benefit. Squad quotas force every transfer to be
            # like-for-like by position, so the incoming players' positions are exactly the
            # positions being changed, and averaging their rates weights the mix correctly.
            rates = [
                TRANSFER_GAIN_SHRINKAGE_BY_POSITION.get(p["position"], TRANSFER_GAIN_SHRINKAGE)
                for p in transferred_in
            ]
            shrink = sum(rates) / len(rates) if rates else TRANSFER_GAIN_SHRINKAGE
            expected_gain = raw_gain * shrink
            net_gain = round(expected_gain - hit_cost, 2)
            scenario_list.append({
                "squad": squad,
                "predicted_total": total,
                "rating_pct": round(100 * total / top_total, 1) if top_total else None,
                "hit_cost": hit_cost,
                "raw_gain": round(raw_gain, 2),
                "shrinkage_applied": round(shrink, 3),
                "expected_gain": round(expected_gain, 2),
                "net_gain": net_gain,
                # Price pressure is a TIMING note on an already-chosen move, never a reason to
                # choose it -- it predicts the market, not the football (see compute_price_pressure).
                "transfers_out": [{**p, "price_pressure": price_pressure.get(p["id"])} for p in transferred_out],
                "transfers_in": [{**p, "price_pressure": price_pressure.get(p["id"])} for p in transferred_in],
            })
        # Re-rank by net gain, not by raw predicted total. Position-aware shrinkage means a
        # slightly-lower-raw option with a defender-heavy mix can genuinely beat a higher-raw one,
        # so the MILP's raw ordering is no longer the ordering that matters.
        scenario_list.sort(key=lambda s: s["net_gain"], reverse=True)
        if scenario_list:
            best = scenario_list[0]["net_gain"]
            if hit_cost == 0 and best > best_free_gain:
                best_free_gain = best
                best_free_transfers = n
            if best > best_net_gain:
                best_net_gain = best
                recommended_transfers = n
        transfer_scenarios[str(n)] = scenario_list

    # A hit only gets recommended if it clears the best hit-free option by a real margin -- see
    # HIT_DECISION_MARGIN. Without this the tool would call a sub-1-point edge for paying 4 points,
    # which is well inside what the calibration itself can resolve.
    hit_rejected = False
    if recommended_transfers > free_transfers and best_net_gain - best_free_gain < HIT_DECISION_MARGIN:
        hit_rejected = True
        recommended_transfers = best_free_transfers
        best_net_gain = best_free_gain

    # Banking as a first-class option, valued on the same scale as acting now rather than assumed
    # to be worth zero. See evaluate_bank_plans.
    bank_plans = evaluate_bank_plans(current_ids, selling_prices, bank_raw, free_transfers)
    chip_timing = evaluate_chip_timing(current_ids, selling_prices, bank_raw)

    # How contested each candidate transfer target is among the managers Nathan is actually
    # competing with -- attached to every transfers_in entry across both scenarios and bank plans,
    # since both are "transfer suggestions" in exactly the sense the feature is for.
    league = fetch_league_ownership(session, MY_LEAGUE_ID, MY_TEAM_ID, event)
    if league:
        for opts in transfer_scenarios.values():
            for s in opts:
                for p in s["transfers_in"]:
                    p["league_ownership_pct"] = league["ownership_pct"].get(p["id"], 0.0)
        for plan in bank_plans:
            for p in plan["transfers_in"]:
                p["league_ownership_pct"] = league["ownership_pct"].get(p["id"], 0.0)
    act_now_best = next((p for p in bank_plans if p["wait_weeks"] == 0), None)
    hold_best = next((p for p in bank_plans if p["wait_weeks"] > 0), None)
    # Only let banking change the recommendation once its discount has actually been measured.
    # Until then it is reported alongside, clearly marked, but the recommendation stays where the
    # evidence is -- the same standard the -4 hit was held to.
    recommended_plan = act_now_best
    bank_rejected_as_marginal = False
    if hold_best and act_now_best and hold_best["net_gain"] > act_now_best["net_gain"]:
        # Ranking first is not enough. The measured edge for banking is indistinguishable from zero,
        # so it has to clear acting by more than the measurement itself can resolve before the
        # recommendation moves -- otherwise the tool would just be reading its own noise.
        if hold_best["net_gain"] - act_now_best["net_gain"] >= HOLD_DECISION_MARGIN:
            recommended_plan = hold_best
        else:
            bank_rejected_as_marginal = True
    elif hold_best and not act_now_best:
        recommended_plan = hold_best
    hold_uncalibrated = HOLD_GAIN_SHRINKAGE is None and hold_best is not None

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
        "squad_selling_value": current_value / 10,
        "selling_price_shortfall": selling_shortfall / 10,
        "free_transfers": free_transfers,
        "transfer_gain_shrinkage": TRANSFER_GAIN_SHRINKAGE,
        "transfer_gain_shrinkage_by_position": TRANSFER_GAIN_SHRINKAGE_BY_POSITION,
        "next_week_free_transfers_if_hold": next_week_free_transfers_if_hold,
        "recommended_transfers": recommended_transfers,
        "recommended_net_gain": best_net_gain,
        "hit_rejected_as_marginal": hit_rejected,
        "bank_plans": bank_plans,
        "chip_timing": chip_timing,
        "league": league,
        "recommended_plan": recommended_plan,
        "hold_shrinkage": HOLD_GAIN_SHRINKAGE,
        "hold_uncalibrated": hold_uncalibrated,
        "bank_rejected_as_marginal": bank_rejected_as_marginal,
        "hold_decision_margin": HOLD_DECISION_MARGIN,
        # current_squad drives the pitch view and is deliberately the NEXT-GAMEWEEK lineup, since
        # that's the decision it informs. horizon_lineup records which it is so the dashboard can
        # label it without hardcoding an assumption.
        "current_squad": [{**p, "price_pressure": price_pressure.get(p["id"])}
                          for p in (next_gw_lineup or current_lineup)],
        "horizon_lineup": HORIZON_LINEUP,
        "next_gw_predicted_total": next_gw_total,
        "current_predicted_total": current_total,
        "top_team_predicted_total": top_total,
        "rating_pct": rating_pct,
        "transfer_scenarios": transfer_scenarios,
    }
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"{manager_name}'s team \"{entry.get('name')}\" going into GW{upcoming_event} "
          f"({free_transfers} free transfer(s) available)")
    print(f"  Budget: £{bank_raw / 10:.1f}m bank + £{current_value / 10:.1f}m squad selling value"
          + (f" (£{selling_shortfall / 10:.1f}m less than listed prices, per FPL's sell-at-half-the-rise rule)"
             if selling_shortfall else " (no player has risen yet, so selling = listed)"))
    print(f"  GW{upcoming_event} only: {next_gw_total} pts predicted from the best XI this week")
    print(f"  {HORIZON_GAMEWEEKS}-GW outlook: {current_total} pts "
          f"({rating_pct}% of the model's own best-possible {top_total}-pt squad)")
    if next_gw_lineup:
        captain = next((p for p in next_gw_lineup if p["is_captain"]), None)
        bench = [p for p in next_gw_lineup if not p["is_starter"]]
        if captain:
            print(f"  Captain this week: {captain['web_name']} ({captain['predicted_points']:.2f} pts)")
        if bench:
            order = ", ".join(f"{p['web_name']} {p['predicted_points']:.2f}"
                              for p in sorted(bench, key=lambda p: -p["predicted_points"]))
            print(f"  Bench (best first): {order}")
    def league_note(player):
        pct = player.get("league_ownership_pct")
        return f" ({pct:.0f}% of top {league['top_n']})" if league and pct is not None else ""

    if league:
        print(f"  League context: {league['top_n']} rivals ahead of you in \"{league['league_name']}\" "
              f"-- the ownership % next to each incoming player below is how many of THEM already own "
              f"him, not overall FPL ownership.")
    for n in transfer_counts:
        options = transfer_scenarios[str(n)]
        hit_note = "" if n <= free_transfers else f", includes a {(n - free_transfers) * HIT_COST_PER_TRANSFER}-pt hit"
        if not options:
            print(f"  With {n} transfer(s){hit_note}: no changes beat your current squad")
            continue
        print(f"  With {n} transfer(s){hit_note}, top {len(options)} option(s):")
        for rank, s in enumerate(options, start=1):
            moves = ", ".join(
                f"{o['web_name']} -> {i['web_name']}{league_note(i)}"
                for o, i in zip(s["transfers_out"], s["transfers_in"])
            )
            hit_part = f" - {s['hit_cost']} hit" if s["hit_cost"] else ""
            print(f"    {rank}. raw {s['raw_gain']:+.2f} x{s['shrinkage_applied']:.2f} -> "
                  f"{s['expected_gain']:+.2f}{hit_part} = net {s['net_gain']:+.2f}  -- {moves}")

    if recommended_transfers == 0:
        print(f"\nRecommendation: HOLD. No transfer combination found nets a real gain over your "
              f"current squad -- banking would take you to {next_week_free_transfers_if_hold} free "
              f"transfer(s) next week.")
    else:
        print(f"\nRecommendation: use {recommended_transfers} transfer(s) now for a net gain of "
              f"+{best_net_gain:.2f} pts over {HORIZON_GAMEWEEKS} GWs.")
    print()
    print("USE THEM OR BANK THEM -- every option scored as extra points over the SAME next "
          f"{HORIZON_GAMEWEEKS} gameweeks, so waiting is charged for the week of gain it costs:")
    print(f"  {'Plan':<34}{'Raw':<9}{'Discounted':<12}{'Hit':<6}{'Net':<9}Move")
    for plan in bank_plans[:8]:
        when = "act now" if not plan["wait_weeks"] else f"bank {plan['wait_weeks']}wk"
        label = f"{when}, {plan['transfers']} transfer(s) w/ {plan['free_transfers_then']} free"
        moves = ", ".join(f"{o['web_name']}->{i['web_name']}{league_note(i)}"
                          for o, i in zip(plan["transfers_out"], plan["transfers_in"]))
        flag = "" if plan["calibrated"] else " *"
        print(f"  {label:<34}{plan['raw_gain']:<+9.2f}{plan['expected_gain']:<+12.2f}"
              f"{plan['hit_cost']:<6}{plan['net_gain']:<+9.2f}{moves}{flag}")
    if not hold_uncalibrated:
        print(f"  * Banked gains carry an extra x{HOLD_GAIN_SHRINKAGE} deferral discount on top of "
              f"position shrinkage -- measured over 440 paired replays where both arms spend the "
              f"same transfers over the same weeks.")
    if bank_rejected_as_marginal and hold_best and act_now_best:
        print(f"  Banking ranked highest ({hold_best['net_gain']:+.2f} vs {act_now_best['net_gain']:+.2f}) "
              f"but by less than {HOLD_DECISION_MARGIN:.0f} pts, which is inside what the hold "
              f"calibration can resolve (measured edge +0.70, 90% CI [-0.99, +2.24], banking won "
              f"49% of pairs) -- so it is not recommended.")

    if chip_timing["best_free_hit_week"]:
        print()
        print(f"CHIP TIMING (visible only within the model's {chip_timing['horizon_weeks']}-week "
              f"horizon -- a later blank/double will not show until it enters this window):")
        fh = chip_timing["best_free_hit_week"]
        print(f"  Free Hit: best week so far is GW{fh['event']} ({fh['optimal']:.1f} unconstrained "
              f"vs {fh['current']:.1f} for your saved squad, a {fh['gap']:+.1f} gap). Free Hit "
              f"reverts after one week, so only a real single-week spike (a blank or a double) "
              f"makes it worth it -- a small, similar gap every week is not a signal to act on.")
        gaps = [w["gap"] for w in chip_timing["free_hit"]]
        if gaps and (max(gaps) - min(gaps)) < 2.0:
            print(f"  Free Hit gaps are flat across the window right now ({min(gaps):+.1f} to "
                  f"{max(gaps):+.1f}) -- no standout week yet. Expected early season, before "
                  f"blanks/doubles from cup exits and fixture pile-ups appear in the fixtures.")
        wc = chip_timing["wildcard_now"]
        if wc:
            print(f"  Wildcard now would net {wc['gap']:+.1f} pts over the {wc['weeks_remaining']} "
                  f"visible weeks ({wc['gap_per_week']:+.1f}/week) -- this is the same gap as your "
                  f"rating vs the model's best-possible squad, just isolated as a number.")
            if chip_timing["wildcard_gap_fading"]:
                print(f"  But that gap is FADING across the visible window (down to "
                      f"{chip_timing['wildcard'][-1]['gap_per_week']:+.1f}/week by the last week "
                      f"visible) -- some of it may be a short-term blip (an injury, a rough run of "
                      f"fixtures) rather than a structural problem, so it is worth checking why "
                      f"before spending a wildcard on it.")
            else:
                print(f"  That gap holds up across the visible window rather than fading -- a "
                      f"steadier signal that it reflects a real, structural squad problem.")
        wc_squad = chip_timing.get("wildcard_now_squad")
        if wc_squad:
            wc_starters = sorted((p for p in wc_squad if p["is_starter"]),
                                 key=lambda p: (["GK", "DEF", "MID", "FWD"].index(p["position"]), -p["predicted_points"]))
            wc_bench = sorted((p for p in wc_squad if not p["is_starter"]), key=lambda p: -p["predicted_points"])
            print(f"  Wildcard-now XI: " + ", ".join(f"{p['web_name']} ({p['position']})" for p in wc_starters))
            print(f"  Wildcard-now bench: " + ", ".join(p["web_name"] for p in wc_bench))

    rising = sorted((p for p in all_players
                     if (price_pressure.get(p["id"]) or {}).get("direction") == "rising"),
                    key=lambda p: -price_pressure[p["id"]]["rate"])[:8]
    falling_mine = [p for p in current_players
                    if (price_pressure.get(p["id"]) or {}).get("direction") == "falling"]
    print("\nPrice timing -- bottom/top decile of this week's net transfer flow. Predicts the "
          "market, NOT points (see compute_price_pressure):")
    if falling_mine:
        names = ", ".join(f"{p['web_name']} ({price_pressure[p['id']]['rate']:+.2f})" for p in falling_mine)
        print(f"  Yours being sold off -- ~21% fall next GW vs ~6% base: {names}")
    else:
        print("  None of your players are in the week's most-sold decile.")
    if rising:
        names = ", ".join(f"{p['web_name']} ({price_pressure[p['id']]['rate']:+.2f})" for p in rising)
        print(f"  Most bought -- ~14% rise next GW vs ~2% base: {names}")

    if hit_rejected:
        print(f"A hit-paying option scored higher on the raw numbers but by less than "
              f"{HIT_DECISION_MARGIN:.0f} pts, which is inside what the gain calibration can "
              f"actually resolve -- not recommending it. A hit is also irreversible and spends a "
              f"transfer that would otherwise bank.")
    rates = ", ".join(f"{pos} {rate:.2f}" for pos, rate in TRANSFER_GAIN_SHRINKAGE_BY_POSITION.items())
    print(
        f"(Gains are discounted by position before netting off any hit -- {rates}. "
        "calibrate_transfer_gains.py measured how much of a predicted transfer gain actually "
        "materialises: forward swaps deliver least, defender swaps most. This also only measures "
        "the cost of waiting, not the value of waiting, which depends on information the model "
        "doesn't have yet.)"
    )
    print(f"Saved {OUTPUT_FILE}")


if __name__ == "__main__":
    ensure_utf8_stdout()
    main()
