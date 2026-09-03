"""Measures how much of a recommended transfer's PREDICTED gain actually materialises.

fetch_my_team.py will happily recommend a -12 point hit, and its net-gain numbers are what justify
that. But those numbers carry a structural upward bias that has nothing to do with football:

1. The optimiser's curse. pick_with_transfers scans ~700 players and takes the highest predicted.
   Players whose predictions are noisily HIGH are disproportionately likely to be selected -- it is
   selecting for positive prediction error. The incumbents it sells are a fixed 15 with no such
   selection applied. Max-of-700-noisy-estimates versus a fixed benchmark is biased upward before
   a ball is kicked.
2. Horizon asymmetry. A hit costs -4 once; the gain is summed over 5 gameweeks, so any per-week
   bias is multiplied by 5 while the cost stays fixed.

And the sanity check that should have caught it earlier: backtest MAE is ~1.96 pts per player per
gameweek, while these recommendations claim edges of roughly 1.2-2 pts per player per week. The
claimed edge is the same size as the model's own average error.

So this walks the prior-season archive: at each decision gameweek, train only on what was knowable
before it, build random FPL-legal incumbent squads (random rather than model-picked on purpose --
a model-picked incumbent would carry the same upward bias on both sides and hide the very effect
being measured), ask the optimiser for its transfers, then score what those players ACTUALLY did
over the following 5 gameweeks. Starting XI and captain are chosen on predictions and then scored
on actuals, exactly as a real manager is forced to do.

The output is a shrinkage factor -- realised gain divided by predicted gain -- and the implied
break-even, i.e. how large a predicted gain has to be before a -4 hit is genuinely worth taking.
Same pattern as the quantile calibration in backtest_model.py: don't trust the nominal number,
measure what it's actually worth, correct for the gap.
"""
import numpy as np
import pandas as pd
import pulp
from understatapi import UnderstatClient

from backtest_model import build_backtest_elements
from fetch_my_team import pick_best_lineup, pick_with_transfers
from fpl_ml_model import (
    POSITION_FEATURES,
    POSITION_MAP,
    PRIOR_SEASON_UNDERSTAT,
    add_features,
    add_prematch_team_form,
    build_prior_season_rows,
    build_team_form_lookup,
    compute_sample_weights,
    ensure_utf8_stdout,
    get_half_life_days,
    get_playing_time_denominator,
    get_understat_player_stats,
    get_understat_team_matches,
    load_prior_season_archive,
    make_model,
    match_understat_players,
)
from pick_team import BUDGET, MAX_PER_CLUB, SQUAD_QUOTAS, STARTER_MAX, STARTER_MIN, pick_squad

SCENARIOS_FILE = "transfer_calibration_scenarios.csv"
HORIZON = 5
# Each decision point needs enough history behind it to train on and HORIZON gameweeks of results
# after it to score against, which is what bounds this range rather than an arbitrary choice.
DECISION_GAMEWEEKS = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
SQUADS_PER_GAMEWEEK = 8
TRANSFER_COUNTS = [1, 2, 3]
HIT_COST_PER_TRANSFER = 4
RANDOM_SEED = 42
# Spread of the "imperfect judgement" noise added to past form when building incumbent squads.
# Tuned so simulated squads land near the ~86%-of-optimal a real managed squad sits at: too little
# and every simulated manager owns the same near-optimal squad, too much and they're random again.
JUDGEMENT_NOISE = 1.0


def build_archive():
    understat = UnderstatClient()
    team_matches = get_understat_team_matches(understat, PRIOR_SEASON_UNDERSTAT)
    team_matches = add_prematch_team_form(team_matches)
    team_form_lookup = build_team_form_lookup(team_matches)

    merged_gw, difficulty_lookup, team_names, player_idlist, rest_days_lookup, prior_set_piece_lookup = load_prior_season_archive()
    elements = build_backtest_elements(merged_gw, player_idlist)

    understat_players = get_understat_player_stats(understat, PRIOR_SEASON_UNDERSTAT)
    understat_matches, _ = match_understat_players(elements, understat_players)
    prior_per90 = {p["understat_id"]: p for p in understat_players}
    id_map = {e: e for e in merged_gw["element"].unique()}

    rows = build_prior_season_rows(
        merged_gw, difficulty_lookup, team_names, id_map, team_form_lookup,
        understat_matches, prior_per90, rest_days_lookup, prior_set_piece_lookup,
    )
    rows = add_features(rows)
    name_by_id = {e["id"]: e["web_name"] for e in elements}
    rows["web_name"] = rows["current_id"].map(name_by_id)
    # build_prior_season_rows doesn't carry the club through, but the max-3-per-club constraint
    # needs it -- without a real club per player every squad would look like one giant club and
    # the MILP would be infeasible. merged_gw has it; take each player's last known club.
    team_by_element = merged_gw.groupby("element")["team"].last().to_dict()
    return rows, merged_gw, name_by_id, team_by_element


def actual_points_in_window(merged_gw, gameweeks):
    """Real points scored per player across a gameweek window, counting a gameweek a player
    didn't appear in as the zero it actually was -- build_prior_season_rows drops those rows
    (it only keeps matches actually played), so reading actuals from there would quietly credit
    an injured transfer target only for the weeks he managed to play."""
    window = merged_gw[merged_gw["GW"].isin(gameweeks)]
    return window.groupby("element")["total_points"].sum().to_dict()


def realistic_incumbent_squad(players, rng):
    """An FPL-legal 15 built the way a human builds one: from PAST points-per-game plus imperfect
    judgement, not from this model's forward predictions.

    Two earlier attempts were both wrong in instructive ways. A uniformly random squad is full of
    cheap non-starters, so the optimiser was just replacing obvious dead weight -- gains of ~18 pts
    per single transfer, nearly all realised, which measures something real but useless. Ownership
    weighting was better but still only reached ~76% of optimal against a live squad at ~86%.

    Selecting on past points-per-game gets the quality right AND keeps the incumbent statistically
    independent of the model's forward-looking prediction noise, which matters: if incumbents were
    chosen by maximising the same predictions the replacements are chosen from, both sides would
    carry the same upward selection bias, it would partly cancel, and the effect being measured
    would be hidden. A real manager picks on form and reputation, which is what this imitates."""
    prob = pulp.LpProblem("incumbent_squad", pulp.LpMaximize)
    squad = {p["id"]: pulp.LpVariable(f"sq_{p['id']}", cat="Binary") for p in players}
    by_id = {p["id"]: p for p in players}

    # Past form plus noise -- the noise is the "imperfect judgement" that stops every simulated
    # manager owning an identical squad and keeps them short of optimal, like real ones are.
    prob += pulp.lpSum((by_id[i]["past_ppg"] + rng.normal(0, JUDGEMENT_NOISE)) * squad[i] for i in squad)
    prob += pulp.lpSum(squad.values()) == 15
    prob += pulp.lpSum(by_id[i]["now_cost"] * squad[i] for i in squad) <= BUDGET
    for position, quota in SQUAD_QUOTAS.items():
        prob += pulp.lpSum(squad[i] for i in squad if by_id[i]["position"] == position) == quota
    for team_name in {p["team_name"] for p in players}:
        prob += pulp.lpSum(squad[i] for i in squad if by_id[i]["team_name"] == team_name) <= MAX_PER_CLUB

    if pulp.LpStatus[prob.solve(pulp.PULP_CBC_CMD(msg=0))] != "Optimal":
        return None
    return [by_id[i] for i in squad if squad[i].value() == 1]


def bootstrap_shrinkage_ci(group, iterations=2000, seed=7):
    """90% CI for the shrinkage ratio, by resampling scenarios with replacement. A ratio of two
    means has no tidy closed-form standard error, and these per-position samples are small (~75-95
    scenarios), so a point estimate on its own invites reading noise as signal -- which is the
    exact failure mode this whole calibration exists to correct."""
    rng = np.random.default_rng(seed)
    predicted = group["predicted_gain"].to_numpy()
    realised = group["realised_gain"].to_numpy()
    ratios = []
    for _ in range(iterations):
        idx = rng.integers(0, len(predicted), len(predicted))
        pred_mean = predicted[idx].mean()
        if pred_mean:
            ratios.append(realised[idx].mean() / pred_mean)
    if not ratios:
        return float("nan"), float("nan")
    return float(np.percentile(ratios, 5)), float(np.percentile(ratios, 95))


def score_squad(squad_players, actuals):
    """Pick the XI and captain on PREDICTIONS, then score that same XI on ACTUALS -- a manager
    commits to a lineup before knowing outcomes, so scoring an in-hindsight-optimal XI would
    flatter every scenario equally and measure nothing."""
    lineup = pick_best_lineup(squad_players)
    predicted = sum(p["predicted_points"] * (2 if p["is_captain"] else 1) for p in lineup if p["is_starter"])
    realised = sum(
        actuals.get(p["id"], 0) * (2 if p["is_captain"] else 1) for p in lineup if p["is_starter"]
    )
    return predicted, realised


def main():
    print("Building archive feature rows...")
    rows, merged_gw, name_by_id, team_by_element = build_archive()
    max_gw = int(rows["gw"].max())
    rng = np.random.default_rng(RANDOM_SEED)

    records = []
    for decision_gw in DECISION_GAMEWEEKS:
        window = list(range(decision_gw, decision_gw + HORIZON))
        if window[-1] > max_gw:
            print(f"GW{decision_gw}: skipped, needs results through GW{window[-1]} but archive ends at GW{max_gw}")
            continue

        train = rows[rows["gw"] < decision_gw]
        future = rows[rows["gw"].isin(window)]
        if len(train) < 500 or future.empty:
            print(f"GW{decision_gw}: skipped, not enough data")
            continue

        # Train only on what was knowable before this decision point, then predict the window.
        future = future.copy()
        future["pred"] = 0.0
        for position in POSITION_MAP.values():
            pos_train = train[train["position"] == position]
            pos_mask = (future["position"] == position).to_numpy()
            if len(pos_train) < 30 or not pos_mask.any():
                continue
            model = make_model(position)
            weights = compute_sample_weights(
                pos_train["date"], half_life_days=get_half_life_days(position),
                as_of=pd.Timestamp(pos_train["date"].max(), tz="UTC"),
            )
            model.fit(pos_train[POSITION_FEATURES], pos_train["total_points"], sample_weight=weights)
            raw = model.predict(future.loc[pos_mask, POSITION_FEATURES])
            multiplier = np.minimum(1.0, future.loc[pos_mask, "minutes"].to_numpy() / get_playing_time_denominator(position))
            future.loc[pos_mask, "pred"] = np.clip(raw * multiplier, 0, 15)

        # One row per player: predicted total across the window, and the price they'd cost now.
        predicted_totals = future.groupby("current_id")["pred"].sum()
        at_decision = rows[rows["gw"] == decision_gw].drop_duplicates("current_id").set_index("current_id")
        actuals = actual_points_in_window(merged_gw, window)
        # Points per gameweek elapsed BEFORE the decision point -- what a human would judge a
        # player on. Dividing by gameweeks rather than appearances deliberately penalises players
        # who miss matches, which is also what a manager building a squad actually wants.
        past = merged_gw[merged_gw["GW"] < decision_gw]
        past_ppg = (past.groupby("element")["total_points"].sum() / max(decision_gw - 1, 1)).to_dict()

        players = []
        for pid, pred in predicted_totals.items():
            if pid not in at_decision.index:
                continue
            info = at_decision.loc[pid]
            players.append({
                "id": int(pid),
                "web_name": name_by_id.get(pid, str(pid)),
                "team_name": team_by_element.get(pid, "unknown"),
                "position": info["position"],
                "now_cost": float(info["now_cost"]),
                "predicted_points": float(pred),
                "past_ppg": float(past_ppg.get(pid, 0.0) or 0.0),
            })
        if len(players) < 200:
            print(f"GW{decision_gw}: skipped, only {len(players)} players with predictions")
            continue

        by_id = {p["id"]: p for p in players}
        # The unconstrained optimum, to report how realistic the incumbent squads actually are --
        # if they don't land near a real manager's ~85% of optimal, this isn't measuring the
        # regime the live recommendations operate in.
        best_possible, _ = score_squad(pick_squad(players), actuals)

        for squad_no in range(SQUADS_PER_GAMEWEEK):
            incumbent = realistic_incumbent_squad(players, rng)
            if incumbent is None:
                continue
            current_ids = {p["id"] for p in incumbent}
            current_value = sum(p["now_cost"] for p in incumbent)
            base_pred, base_real = score_squad(incumbent, actuals)

            for k in TRANSFER_COUNTS:
                new_squad = pick_with_transfers(current_ids, current_value, 0, players, k)
                if new_squad is None:
                    continue
                changed = len([p for p in new_squad if p["id"] not in current_ids])
                if changed == 0:
                    continue
                new_pred, new_real = score_squad([by_id[p["id"]] for p in new_squad], actuals)
                records.append({
                    "decision_gw": decision_gw,
                    "squad_no": squad_no,
                    "transfers": changed,
                    "position": "any",
                    "squad_quality_pct": round(100 * base_pred / best_possible, 1) if best_possible else None,
                    "predicted_gain": new_pred - base_pred,
                    "realised_gain": new_real - base_real,
                })

            # Per-position single transfers, so each position gets a balanced sample rather than
            # whatever mix the unconstrained optimiser happened to pick. Restricting the candidate
            # pool to incumbents plus one position forces the swap to be within that position:
            # the exact-quota constraints mean bringing in a DEF can only drop a DEF.
            for position in POSITION_MAP.values():
                pool = [p for p in players if p["id"] in current_ids or p["position"] == position]
                constrained = pick_with_transfers(current_ids, current_value, 0, pool, 1)
                if constrained is None:
                    continue
                changed = [p for p in constrained if p["id"] not in current_ids]
                if len(changed) != 1:
                    continue
                pos_pred, pos_real = score_squad([by_id[p["id"]] for p in constrained], actuals)
                records.append({
                    "decision_gw": decision_gw,
                    "squad_no": squad_no,
                    "transfers": 1,
                    "position": position,
                    "squad_quality_pct": round(100 * base_pred / best_possible, 1) if best_possible else None,
                    "predicted_gain": pos_pred - base_pred,
                    "realised_gain": pos_real - base_real,
                })
        print(f"GW{decision_gw}: done ({len([r for r in records if r['decision_gw'] == decision_gw])} scenarios)")

    if not records:
        print("No scenarios evaluated.")
        return

    all_records = pd.DataFrame(records)
    # Saved so the scenarios can be re-analysed (different bins, extra cuts) without paying for
    # another full walk of the archive, which is the expensive part of this script by far.
    all_records.to_csv(SCENARIOS_FILE, index=False)
    print(f"\nSaved {len(all_records)} raw scenarios to {SCENARIOS_FILE}")
    # The headline shrinkage must come only from unconstrained scenarios -- the per-position runs
    # are a separate, deliberately balanced sample and would skew the overall figure towards
    # single transfers if pooled in.
    df = all_records[all_records["position"] == "any"].copy()
    by_position = all_records[all_records["position"] != "any"].copy()

    print(f"\n{len(df)} transfer scenarios across {df['decision_gw'].nunique()} decision gameweeks, "
          f"{HORIZON}-GW horizon.")
    print(f"Incumbent squad quality: {df['squad_quality_pct'].mean():.1f}% of optimal on average "
          f"(range {df['squad_quality_pct'].min():.0f}-{df['squad_quality_pct'].max():.0f}%) -- "
          f"for reference, the live squad this is meant to represent sits around 86%.\n")

    print(f"{'Transfers':<11}{'n':<6}{'Mean predicted':<17}{'Mean realised':<16}{'Shrinkage':<12}")
    for k, group in df.groupby("transfers"):
        pred_mean, real_mean = group["predicted_gain"].mean(), group["realised_gain"].mean()
        shrink = real_mean / pred_mean if pred_mean else float("nan")
        print(f"{k:<11}{len(group):<6}{pred_mean:<17.2f}{real_mean:<16.2f}{shrink:<12.2f}")

    # The decision that actually matters -- "is this worth -4" -- lives in the SMALL predicted-gain
    # range. A single average across all gains would be dominated by the big obvious upgrades and
    # say nothing useful about the marginal case, so bin it.
    print(f"\nBy size of predicted gain (the marginal band is what a hit decision turns on):")
    print(f"{'Predicted gain':<18}{'n':<6}{'Mean predicted':<17}{'Mean realised':<16}{'Shrinkage':<12}{'% that gained':<14}")
    bins = [(0, 5), (5, 10), (10, 20), (20, 40), (40, 1e9)]
    for low, high in bins:
        group = df[(df["predicted_gain"] >= low) & (df["predicted_gain"] < high)]
        if group.empty:
            continue
        pred_mean, real_mean = group["predicted_gain"].mean(), group["realised_gain"].mean()
        shrink = real_mean / pred_mean if pred_mean else float("nan")
        win = 100 * float((group["realised_gain"] > 0).mean())
        label = f"{low:.0f}-{high:.0f} pts" if high < 1e9 else f"{low:.0f}+ pts"
        print(f"{label:<18}{len(group):<6}{pred_mean:<17.2f}{real_mean:<16.2f}{shrink:<12.2f}{win:<14.0f}")

    if not by_position.empty:
        print("\nSingle transfers by position (balanced sample -- the swap is forced to be within "
              "each position). 90% CI is bootstrapped, because a ratio of means has no clean "
              "closed-form error and these per-position samples are small enough that the point "
              "estimate alone would be easy to over-read:")
        print(f"{'Position':<11}{'n':<6}{'Mean predicted':<17}{'Mean realised':<16}{'Shrinkage':<12}{'90% CI':<20}{'% gained':<10}")
        for position in POSITION_MAP.values():
            group = by_position[by_position["position"] == position]
            if group.empty:
                continue
            pred_mean, real_mean = group["predicted_gain"].mean(), group["realised_gain"].mean()
            shrink = real_mean / pred_mean if pred_mean else float("nan")
            low, high = bootstrap_shrinkage_ci(group)
            win = 100 * float((group["realised_gain"] > 0).mean())
            print(f"{position:<11}{len(group):<6}{pred_mean:<17.2f}{real_mean:<16.2f}{shrink:<12.2f}"
                  f"{f'[{low:.2f}, {high:.2f}]':<20}{win:<10.0f}")

    overall_pred, overall_real = df["predicted_gain"].mean(), df["realised_gain"].mean()
    shrinkage = overall_real / overall_pred if overall_pred else float("nan")
    hit_rate = float((df["realised_gain"] > 0).mean())

    print(f"\nOverall: predicted {overall_pred:+.2f} -> realised {overall_real:+.2f} "
          f"(shrinkage {shrinkage:.2f})")
    print(f"Recommended transfers that actually gained points: {100 * hit_rate:.0f}%")

    # Break-even straight from the shrinkage ratio, NOT from a regression fit. An earlier version
    # fitted realised ~ predicted and solved for where realised crosses the hit cost, which
    # produced a break-even BELOW the hit cost -- nonsense, and an artifact: the optimiser never
    # proposes near-zero-gain swaps, so there's no data anywhere near the intercept and the line
    # was being extrapolated into empty space. Ratio math needs no such extrapolation.
    single = df[df["transfers"] == 1]
    single_shrink = single["realised_gain"].mean() / single["predicted_gain"].mean() if len(single) else float("nan")
    print("\nFor a hit to pay, the REALISED gain must clear the hit cost, so the predicted gain "
          f"has to clear (hit / shrinkage):")
    for label, s in (("all transfer counts", shrinkage), ("single transfers only", single_shrink)):
        if s and s > 0:
            print(f"  {label:<24} shrinkage {s:.2f} -> need predicted gain > {HIT_COST_PER_TRANSFER / s:.1f} pts "
                  f"to justify -{HIT_COST_PER_TRANSFER}")
        else:
            print(f"  {label:<24} shrinkage non-positive -- no hit justifiable on these numbers")
    print("\nSingle transfers are the number to trust for a marginal hit decision: they're the "
          "tightest margin, where selection-on-noise bites hardest.")


if __name__ == "__main__":
    ensure_utf8_stdout()
    main()
