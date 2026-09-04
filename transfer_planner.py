"""Transfer planner with beam-search multi-GW lookahead.

This is a lightweight planner that reads historical per-gameweek prediction CSVs in
prediction_history/gw{n}.csv.gz (or falls back to fpl_ml_predictions.csv), constructs
predicted points for a short horizon, and performs a beam-search over single-week
transfer actions (one transfer per GW) to choose the best first-week transfers that
maximize discounted expected points minus transfer hits.

The planner is intentionally conservative and lightweight so it can run in CI (GitHub
Actions) without heavy dependencies beyond pandas.
"""

import copy
import csv
import gzip
import glob
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

# Config
HORIZON = int(os.environ.get("PLANNER_HORIZON", 3))
BEAM_WIDTH = int(os.environ.get("PLANNER_BEAM_WIDTH", 200))
CANDIDATE_WIDTH = int(os.environ.get("PLANNER_CAND_WIDTH", 50))
DISCOUNT = float(os.environ.get("PLANNER_GAMMA", 0.95))
TRANSFER_HIT = int(os.environ.get("PLANNER_HIT", 4))

REPO_ROOT = Path(".")
PRED_HISTORY_DIR = REPO_ROOT / "prediction_history"
TEAM_HISTORY_FILE = REPO_ROOT / "fpl_ml_team_history.jsonl"
OUTPUT_FILE = REPO_ROOT / "planner_output.json"

POSITION_ORDER = ["GK", "DEF", "MID", "FWD"]
SQUAD_QUOTAS = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
STARTER_MIN = {"GK": 1, "DEF": 3, "MID": 2, "FWD": 1}
STARTER_MAX = {"GK": 1, "DEF": 5, "MID": 5, "FWD": 3}


def load_predictions_from_history() -> Tuple[Dict[int, Dict[int, float]], Dict[int, Dict]]:
    """Load prediction_history/gw{n}.csv.gz files. Returns:
    - preds[player_id][event] = predicted_points
    - players_meta[player_id] = {web_name, position, now_cost, team_name}
    """
    preds: Dict[int, Dict[int, float]] = {}
    meta: Dict[int, Dict] = {}
    files = sorted(glob.glob(str(PRED_HISTORY_DIR / "gw*.csv.gz")))
    if not files:
        # Fallback: try fpl_ml_predictions.csv
        fallback = REPO_ROOT / "fpl_ml_predictions.csv"
        if fallback.exists():
            df = pd.read_csv(fallback)
            # Expect columns: id, event, predicted_points, position, now_cost, web_name, team_name
            for _, r in df.iterrows():
                pid = int(r["id"])
                event = int(r.get("event", 0))
                preds.setdefault(pid, {})[event] = float(r.get("predicted_points", 0.0))
                meta.setdefault(pid, {})["web_name"] = r.get("web_name")
                meta[pid]["position"] = r.get("position")
                meta[pid]["now_cost"] = int(r.get("now_cost", 0))
                meta[pid]["team_name"] = r.get("team_name")
            return preds, meta
        else:
            raise FileNotFoundError("No prediction_history files or fpl_ml_predictions.csv found.")

    events = []
    for p in files:
        # filename like prediction_history/gw03.csv.gz
        name = Path(p).stem  # gw03.csv -> gw03
        if name.startswith("gw"):
            try:
                ev = int(name[2:])
            except Exception:
                continue
        else:
            continue
        events.append(ev)
        # read gz csv
        with gzip.open(p, "rt", newline="") as fh:
            df = pd.read_csv(fh)
            # expect id, predicted_points, web_name, position, now_cost, team_name
            for _, r in df.iterrows():
                pid = int(r["id"])
                preds.setdefault(pid, {})[ev] = float(r.get("predicted_points", 0.0))
                meta.setdefault(pid, {})["web_name"] = r.get("web_name")
                meta[pid]["position"] = r.get("position")
                meta[pid]["now_cost"] = int(r.get("now_cost", 0))
                meta[pid]["team_name"] = r.get("team_name")

    return preds, meta


def get_upcoming_events(preds: Dict[int, Dict[int, float]], horizon: int) -> List[int]:
    all_events = set()
    for pid in preds:
        all_events.update(preds[pid].keys())
    if not all_events:
        return []
    start = min(all_events)
    # build contiguous horizon from start
    return [e for e in sorted(all_events) if e >= start][:horizon]


def load_current_squad() -> List[Dict]:
    if not TEAM_HISTORY_FILE.exists():
        return []
    entries = [json.loads(l) for l in TEAM_HISTORY_FILE.read_text().splitlines() if l.strip()]
    if not entries:
        return []
    # pick most recent (by event or picked_at)
    entries_sorted = sorted(entries, key=lambda e: (e.get("event", 0), e.get("picked_at") or ""))
    latest = entries_sorted[-1]
    return latest.get("squad", [])


def select_starting_xi(squad_players: List[Dict], predicted_by_pid: Dict[int, float]) -> List[int]:
    """Select a starting XI given squad players and predicted points for a single event.
    Returns list of player ids selected as starters.

    Simple greedy algorithm that enforces STARTER_MIN/STARTER_MAX and GK=1.
    """
    by_pos = {pos: [] for pos in POSITION_ORDER}
    for p in squad_players:
        pid = int(p["id"])
        pos = p["position"]
        score = predicted_by_pid.get(pid, 0.0)
        by_pos[pos].append((score, pid))
    # ensure min
    starters = set()
    for pos in POSITION_ORDER:
        lst = sorted(by_pos[pos], reverse=True)
        needed = STARTER_MIN[pos]
        for _, pid in lst[:needed]:
            starters.add(pid)
    # fill remaining slots until 11 respecting max
    remaining_slots = 11 - len(starters)
    # flatten all candidates sorted
    all_candidates = []
    for pos in POSITION_ORDER:
        for score, pid in by_pos[pos]:
            if pid in starters:
                continue
            all_candidates.append((score, pid, pos))
    all_candidates.sort(reverse=True)
    pos_count = {pos: sum(1 for p in squad_players if int(p["id"]) in starters and p["position"] == pos) for pos in POSITION_ORDER}
    idx = 0
    while remaining_slots > 0 and idx < len(all_candidates):
        score, pid, pos = all_candidates[idx]
        if pos_count[pos] < STARTER_MAX[pos]:
            starters.add(pid)
            pos_count[pos] += 1
            remaining_slots -= 1
        idx += 1
    return list(starters)


class State:
    def __init__(self, squad: Dict[int, Dict], free_transfers=1, bank=0, path=None, score=0.0):
        # squad: dict pid -> player dict
        self.squad = squad
        self.free_transfers = free_transfers
        self.bank = bank
        self.path = path or []  # list of transfer actions per week
        self.score = score

    def clone(self):
        return State(copy.deepcopy(self.squad), self.free_transfers, self.bank, list(self.path), self.score)


def evaluate_state(state: State, events: List[int], preds: Dict[int, Dict[int, float]]) -> float:
    """Compute discounted expected points for this state's squad across the events list.
    Use greedy starting XI selection per event.
    """
    total = 0.0
    for t, ev in enumerate(events):
        # predicted points mapping for event ev
        predicted_by_pid = {pid: preds.get(pid, {}).get(ev, 0.0) for pid in state.squad}
        starters = select_starting_xi(list(state.squad.values()), predicted_by_pid)
        week_points = sum(predicted_by_pid.get(pid, 0.0) * (2 if state.squad[pid].get("is_captain") else 1) for pid in starters)
        total += (DISCOUNT ** t) * week_points
    return total


def generate_candidate_swaps(state: State, all_players_meta: Dict[int, Dict], preds: Dict[int, Dict[int, float]], events0: List[int], M: int) -> List[Tuple[Tuple[int, int], float]]:
    """Generate candidate single-player swaps (out_pid, in_pid) and score them by immediate+future gain estimate.
    Returns top M candidates sorted by estimated gain (descending). Include (None,None) as no-op.
    """
    # Compute baseline expected for current squad
    baseline = 0.0
    for t, ev in enumerate(events0):
        for pid, p in state.squad.items():
            baseline += (DISCOUNT ** t) * preds.get(pid, {}).get(ev, 0.0)
    # Candidate pool: players not in squad
    in_pool = [pid for pid in all_players_meta.keys() if pid not in state.squad]
    out_pool = list(state.squad.keys())
    candidates = []
    # simple heuristic: consider in candidates that have high sum(predicted over events)
    in_scores = []
    for pid in in_pool:
        s = sum(preds.get(pid, {}).get(ev, 0.0) * (DISCOUNT ** t) for t, ev in enumerate(events0))
        in_scores.append((s, pid))
    in_scores.sort(reverse=True)
    top_in = [pid for _, pid in in_scores[: max(200, M*3)]]

    for out in out_pool:
        out_pos = state.squad[out]["position"]
        # consider some top_in that share same position
        for in_pid in top_in:
            if all_players_meta[in_pid]["position"] != out_pos:
                continue
            # enforce max-per-club roughly: count clubs
            # crude check: don't exceed 3 per club
            club_counts = {}
            for pid in state.squad:
                if pid == out:
                    continue
                club_counts[state.squad[pid]["team_name"]] = club_counts.get(state.squad[pid]["team_name"], 0) + 1
            club_counts[all_players_meta[in_pid]["team_name"]] = club_counts.get(all_players_meta[in_pid]["team_name"], 0) + 1
            if club_counts.get(all_players_meta[in_pid]["team_name"], 0) > 3:
                continue
            # estimate gain
            new_total = 0.0
            for t, ev in enumerate(events0):
                # sum predicted for squad after swap
                for pid in state.squad:
                    if pid == out:
                        continue
                    new_total += (DISCOUNT ** t) * preds.get(pid, {}).get(ev, 0.0)
                new_total += (DISCOUNT ** t) * preds.get(in_pid, {}).get(ev, 0.0)
            gain = new_total - baseline
            # cost: if using more than free_transfers (we simulate one transfer), assume transfer consumes free transfer unless free_transfers==0
            hit = 0
            if state.free_transfers <= 0:
                hit = TRANSFER_HIT
            net = gain - hit
            candidates.append(((out, in_pid), net))
    # include no-op
    candidates.append(((None, None), 0.0))
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:M]


def beam_search(initial_state: State, events: List[int], preds: Dict[int, Dict[int, float]], all_players_meta: Dict[int, Dict]) -> State:
    beam = [initial_state]
    for depth in range(len(events)):
        next_beam = []
        for state in beam:
            # evaluate no-op (carry squad forward)
            s0 = state.clone()
            s0.path.append(None)
            # Update score incrementally by event t predicted for starters
            # For speed, we compute full evaluate at the end; here keep state and let evaluation happen
            next_beam.append(s0)
            # generate candidate swaps
            cands = generate_candidate_swaps(state, all_players_meta, preds, events[depth:], M=CANDIDATE_WIDTH)
            for (out, inn), est in cands:
                if out is None:
                    continue
                new_state = state.clone()
                # perform swap
                player_in_meta = all_players_meta[inn]
                # copy player meta into squad entry
                new_state.squad.pop(out, None)
                new_state.squad[inn] = {
                    "id": inn,
                    "web_name": player_in_meta.get("web_name"),
                    "position": player_in_meta.get("position"),
                    "now_cost": player_in_meta.get("now_cost", 0),
                    "team_name": player_in_meta.get("team_name"),
                }
                # update free_transfers and bank simplistically
                if new_state.free_transfers > 0:
                    new_state.free_transfers -= 1
                else:
                    # applied a hit
                    new_state.score -= TRANSFER_HIT
                new_state.path.append((out, inn))
                next_beam.append(new_state)
        # prune beam by evaluating full expected points for remaining horizon
        scored = []
        for s in next_beam:
            s.score = evaluate_state(s, events[depth:], preds)
            scored.append((s.score, s))
        scored.sort(key=lambda x: x[0], reverse=True)
        beam = [s for _, s in scored[:BEAM_WIDTH]]
    # after full horizon, pick best
    best = max(beam, key=lambda s: s.score)
    return best


def main():
    print("Transfer planner starting...")
    preds, meta = load_predictions_from_history()
    events = get_upcoming_events(preds, HORIZON)
    if not events:
        raise RuntimeError("No upcoming events found in predictions.")
    print(f"Found events: {events}")
    # build initial squad
    current = load_current_squad()
    if not current:
        # fallback: pick top 15 by sum predicted for first event
        first_ev = events[0]
        scores = []
        for pid in preds:
            scores.append((sum(preds.get(pid, {}).get(ev, 0.0) for ev in events), pid))
        scores.sort(reverse=True)
        top_ids = [pid for _, pid in scores[:15]]
        squad = {}
        for pid in top_ids:
            m = meta.get(pid, {})
            squad[pid] = {"id": pid, "web_name": m.get("web_name"), "position": m.get("position"), "now_cost": m.get("now_cost", 0), "team_name": m.get("team_name")}
    else:
        squad = {int(p["id"]): p for p in current}
    initial_state = State(squad, free_transfers=1, bank=0, path=[], score=0.0)
    print(f"Initial squad size: {len(initial_state.squad)}")
    best = beam_search(initial_state, events, preds, meta)
    # determine first-week action
    if not best.path:
        first_action = None
    else:
        first_action = best.path[0]
    out = {
        "events": events,
        "horizon": HORIZON,
        "discount": DISCOUNT,
        "initial_squad_ids": list(initial_state.squad.keys()),
        "best_score": best.score,
        "first_week_action": first_action,
        "best_path": best.path,
    }
    with open(OUTPUT_FILE, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"Wrote planner output to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
