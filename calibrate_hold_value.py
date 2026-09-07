"""Measures whether BANKING a free transfer beats spending it now, and by how much.

fetch_my_team.py could not answer this. Its hold option was hardcoded to a net gain of exactly
zero, so holding could only ever win when every available move had negative gain -- the tool could
say "hold, nothing is worth doing" but never "hold, next week is worth more". Every move was scored
over the full horizon while its alternative, the same move made a week later, was scored at nothing.

Holding genuinely pays through three mechanisms, and only the first is obvious:

1. Bundling. Some moves are unreachable with one free transfer at any price -- downgrading two
   players to fund a premium needs two simultaneous transfers. Banking buys the ABILITY to express
   a move, not just a cheaper version of it.
2. Better information. A week later you know who is injured, who is rotating, whose price moved.
3. Insurance. A spare transfer covers next week's injury without a hit.

Against exactly one week of forgone gain from the move not made.

The trap this script exists to avoid: simply crediting the hold arm with a shifted-window number
would swap one bias for its mirror image. The deferred gain is computed by the same
max-over-~700-noisy-estimates optimiser as the immediate one, so it carries the same optimiser's
curse (see calibrate_transfer_gains.py), PLUS an assumption the immediate arm doesn't make -- that
today's view of next week survives to next week. Three things move it: prediction churn (the target
stops being the best target), price drift (the target becomes unaffordable), and better information
(you find something better). Whether that nets out positive is an empirical question, not one to
assert.

So this replays the prior season the same way the transfer-gain calibration does, with matched arms
scored over the SAME window of gameweeks:

  A(k)      spend k transfers at GW t.                    New squad plays weeks t..t+H-1.
  C1(k+1)   hold at t, spend k+1 at t+1 with FRESH info.  Old squad week t, new squad after.
  C2(k+2)   hold twice, spend k+2 at t+2 with FRESH info. Old squad weeks t..t+1, new after.

The predicted gain for a hold arm is deliberately computed from the DECISION-TIME model on a
shifted window -- that is exactly what the live tool can see today -- while its realised gain comes
from a squad actually chosen later with later information. Predicting as the tool predicts and
realising as reality realises is what makes the resulting ratio a usable discount factor.

Starting XI and captain are re-picked every gameweek from the freshest model available at that
gameweek, identically in every arm. Only the SQUAD differs between arms, which is the thing being
measured; letting the hold arms also pick better line-ups would credit them for something a real
manager gets regardless of whether they transferred.
"""
import numpy as np
import pandas as pd

from calibrate_transfer_gains import (
    HORIZON,
    JUDGEMENT_NOISE,
    RANDOM_SEED,
    actual_points_in_window,
    bootstrap_shrinkage_ci,
    build_archive,
    realistic_incumbent_squad,
)
from fetch_my_team import pick_best_lineup, pick_with_transfers
from fpl_ml_model import (
    POSITION_FEATURES,
    POSITION_MAP,
    compute_sample_weights,
    ensure_utf8_stdout,
    get_half_life_days,
    get_playing_time_denominator,
    make_model,
)

SCENARIOS_FILE = "hold_calibration_scenarios.csv"
# Same decision points as the transfer-gain calibration, minus the last two: a hold arm has to be
# able to defer twice and still have results to score, so it needs more archive after it.
DECISION_GAMEWEEKS = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
SQUADS_PER_GAMEWEEK = 8
# Free transfers in hand at the decision point. 1 is the standard week; 2 is what Nathan usually
# has, and is where bundling starts to matter.
START_FREE_TRANSFERS = [1, 2]
MAX_WAIT = 2


def fit_vintage(rows, decision_gw, target_gws):
    """Predicted points per (player, gameweek) using only what was knowable before `decision_gw`.

    Re-fitting per vintage is the whole point -- a hold arm's advantage is that it decides with a
    week more information, so reusing the earlier model's predictions for a later decision would
    define that advantage away."""
    train = rows[rows["gw"] < decision_gw]
    future = rows[rows["gw"].isin(target_gws)].copy()
    if len(train) < 500 or future.empty:
        return None
    future["pred"] = 0.0
    for position in POSITION_MAP.values():
        pos_train = train[train["position"] == position]
        mask = (future["position"] == position).to_numpy()
        if len(pos_train) < 30 or not mask.any():
            continue
        model = make_model(position)
        weights = compute_sample_weights(
            pos_train["date"], half_life_days=get_half_life_days(position),
            as_of=pd.Timestamp(pos_train["date"].max(), tz="UTC"),
        )
        model.fit(pos_train[POSITION_FEATURES], pos_train["total_points"], sample_weight=weights)
        raw = model.predict(future.loc[mask, POSITION_FEATURES])
        multiplier = np.minimum(
            1.0, future.loc[mask, "minutes"].to_numpy() / get_playing_time_denominator(position)
        )
        future.loc[mask, "pred"] = np.clip(raw * multiplier, 0, 15)
    return future


def per_week_predictions(future):
    """{gameweek: {player_id: predicted points}} -- for weekly line-up selection."""
    out = {}
    for gw, group in future.groupby("gw"):
        out[int(gw)] = group.groupby("current_id")["pred"].sum().to_dict()
    return out


def identity_base(rows, decision_gw, name_by_id, team_by_element, past_ppg):
    """The fixed candidate set: who exists, what position, what club, what price.

    Deliberately built ONCE, at the decision gameweek, and shared by every arm. Rebuilding it per
    arm from that arm's own gameweek looked more realistic but silently biased the comparison: a
    player who missed GW t+2 has no row there, so any squad containing him became unrepresentable
    in the deferred MILP and the whole scenario was dropped -- which culled the hold arms far more
    often than the spend-now arm (n=4 against n=8 in testing). Arms must differ only in what they
    KNOW, not in which players exist for them.

    Prices are likewise frozen at the decision gameweek. A transfer deferred a week or two is
    really made at that week's price, but the drift is 0.1-0.2m and letting it vary would confound
    the information effect being measured with a budget effect. Price drift is handled live and
    separately, by compute_price_pressure in fetch_my_team.py."""
    at_decision = rows[rows["gw"] == decision_gw].drop_duplicates("current_id").set_index("current_id")
    base = {}
    for pid, info in at_decision.iterrows():
        base[int(pid)] = {
            "id": int(pid),
            "web_name": name_by_id.get(pid, str(pid)),
            "team_name": team_by_element.get(pid, "unknown"),
            "position": info["position"],
            "now_cost": float(info["now_cost"]),
            "predicted_points": 0.0,
            "past_ppg": float(past_ppg.get(pid, 0.0) or 0.0),
        }
    return base


def universe_from(base, future, window):
    """The shared candidate set re-scored with one model vintage's view of one window.

    A player with no rows in the window scores 0 rather than vanishing -- that is what a blank or
    an injury actually is, and keeping him in the pool is what holds the candidate set identical
    across arms."""
    totals = future[future["gw"].isin(window)].groupby("current_id")["pred"].sum().to_dict()
    return [{**player, "predicted_points": float(totals.get(pid, 0.0))} for pid, player in base.items()]


def bootstrap_mean_ci(values, iterations=2000, seed=11):
    """90% CI for a paired mean difference. bootstrap_shrinkage_ci next door does a RATIO of two
    means, which is the wrong statistic here -- the hold edge is already a difference taken within
    each pair, so it just needs its own mean resampled."""
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(iterations)]
    return float(np.percentile(means, 5)), float(np.percentile(means, 95))


def score_weeks(squad_ids, weeks, lineup_preds, actuals_by_week, static_by_id):
    """Actual points scored by a squad across `weeks`, re-picking XI and captain each week.

    A real manager re-picks a line-up every gameweek from current information whether or not they
    made a transfer, so doing it weekly here (rather than once per window, as the transfer-gain
    calibration does) removes an asymmetry that would otherwise flatter whichever arm happened to
    own more players with lumpy fixtures."""
    total = 0.0
    for week in weeks:
        preds = lineup_preds.get(week, {})
        squad = [
            {**static_by_id[pid], "predicted_points": float(preds.get(pid, 0.0))}
            for pid in squad_ids if pid in static_by_id
        ]
        if len(squad) < 15:
            raise ValueError(
                f"GW{week}: squad resolved to {len(squad)} players, not 15 -- scoring it would "
                f"silently credit zero for the week and quietly bias whichever arm owns the "
                f"missing player. Fix the identity lookup rather than skipping."
            )
        lineup = pick_best_lineup(squad)
        actuals = actuals_by_week.get(week, {})
        total += sum(
            actuals.get(p["id"], 0) * (2 if p["is_captain"] else 1)
            for p in lineup if p["is_starter"]
        )
    return total


def main():
    print("Building archive feature rows...")
    rows, merged_gw, name_by_id, team_by_element = build_archive()
    max_gw = int(rows["gw"].max())
    rng = np.random.default_rng(RANDOM_SEED)

    records = []
    for decision_gw in DECISION_GAMEWEEKS:
        window = list(range(decision_gw, decision_gw + HORIZON))
        if window[-1] > max_gw or decision_gw + MAX_WAIT > window[-1]:
            print(f"GW{decision_gw}: skipped, needs results through GW{window[-1]} (archive ends GW{max_gw})")
            continue

        # One model vintage per decision point in the ladder: t, t+1, t+2.
        vintages = {}
        for wait in range(MAX_WAIT + 1):
            gw = decision_gw + wait
            future = fit_vintage(rows, gw, [w for w in window if w >= gw])
            if future is None:
                break
            vintages[wait] = future
        if len(vintages) < MAX_WAIT + 1:
            print(f"GW{decision_gw}: skipped, not enough history for all vintages")
            continue

        # Line-up predictions for week w come from the freshest vintage that exists by then --
        # applied identically to every arm, so only the squad differs.
        lineup_preds = {}
        for wait in range(MAX_WAIT + 1):
            for gw, preds in per_week_predictions(vintages[wait]).items():
                if gw >= decision_gw + wait:
                    lineup_preds[gw] = preds

        actuals_by_week = {w: actual_points_in_window(merged_gw, [w]) for w in window}
        past = merged_gw[merged_gw["GW"] < decision_gw]
        past_ppg = (past.groupby("element")["total_points"].sum() / max(decision_gw - 1, 1)).to_dict()

        # Universes: [wait][for_decision_at_wait] -- and, separately, the DECISION-TIME view of a
        # deferred window, which is what the live tool would compute today.
        base = identity_base(rows, decision_gw, name_by_id, team_by_element, past_ppg)
        if len(base) < 200:
            print(f"GW{decision_gw}: skipped, only {len(base)} players in the candidate set")
            continue
        static_by_id = base

        universe_now = universe_from(base, vintages[0], window)
        # Each vintage's view of everything still to come from its own week onwards -- what the
        # spend-as-you-go chain re-optimises against at each step.
        universe_fresh_full = {
            step: universe_from(base, vintages[step], [w for w in window if w >= decision_gw + step])
            for step in range(MAX_WAIT + 1)
        }
        universe_fresh = {}   # decided later, with later information -- drives realised outcomes
        universe_stale = {}   # decided later, but foreseen with TODAY's model -- drives predictions
        for wait in range(1, MAX_WAIT + 1):
            sub_window = [w for w in window if w >= decision_gw + wait]
            universe_fresh[wait] = universe_from(base, vintages[wait], sub_window)
            universe_stale[wait] = universe_from(base, vintages[0], sub_window)

        for squad_no in range(SQUADS_PER_GAMEWEEK):
            incumbent = realistic_incumbent_squad(universe_now, rng)
            if incumbent is None:
                continue
            current_ids = {p["id"] for p in incumbent}
            base_real = score_weeks(current_ids, window, lineup_preds, actuals_by_week, static_by_id)

            def solve(universe, k, from_ids):
                pool = {p["id"]: p for p in universe}
                value = sum(pool[i]["now_cost"] for i in from_ids)
                squad = pick_with_transfers(from_ids, value, 0, universe, k)
                if squad is None:
                    return None
                ids = {p["id"] for p in squad}
                if not ids - from_ids:
                    return None
                return ids, pool

            # Both arms spend the SAME total number of transfers over the same weeks and both end
            # with nothing banked. An earlier version let the hold arm make more transfers overall
            # (it banked while the spend arm never used the transfers it went on to accrue), which
            # made holding look good for the trivial reason that it was doing more. With the count
            # held equal the only differences left are the two that actually matter: the hold arm
            # decides everything with later information, and it can BUNDLE -- express a move that
            # needs several simultaneous transfers, which the spend arm cannot at any price.
            for start_ft in START_FREE_TRANSFERS:
                for wait in range(1, MAX_WAIT + 1):
                    total_transfers = start_ft + wait

                    # SPEND AS YOU GO: start_ft transfers at t, then one more each week, each
                    # decided with only the information available in that week.
                    chain_ids, chain_ok = current_ids, True
                    segments, first_step = [], None
                    for step in range(wait + 1):
                        gw_step = decision_gw + step
                        k_step = start_ft if step == 0 else 1
                        pool_step = universe_fresh_full[step]
                        got = solve(pool_step, k_step, chain_ids)
                        if got is None:
                            chain_ok = False
                            break
                        chain_ids, pool = got
                        if first_step is None:
                            first_step = (chain_ids, pool)
                        weeks_held = [w for w in window if gw_step <= w < min(gw_step + 1, window[-1] + 1)]                             if step < wait else [w for w in window if w >= gw_step]
                        segments.append((chain_ids, weeks_held))
                    if not chain_ok or first_step is None:
                        continue
                    spend_real = sum(
                        score_weeks(ids, weeks, lineup_preds, actuals_by_week, static_by_id)
                        for ids, weeks in segments
                    ) - base_real
                    # What the LIVE tool would show for "spend now": the immediate move only, since
                    # it has no model of the follow-ups it would make in later weeks.
                    first_ids, first_pool = first_step
                    spend_pred = (sum(first_pool[i]["predicted_points"] for i in first_ids)
                                  - sum(first_pool[i]["predicted_points"] for i in current_ids))

                    # BANK AND BUNDLE: nothing until t+wait, then every transfer at once.
                    gw_later = decision_gw + wait
                    later_weeks = [w for w in window if w >= gw_later]
                    held_weeks = [w for w in window if w < gw_later]
                    stale = solve(universe_stale[wait], total_transfers, current_ids)
                    fresh = solve(universe_fresh[wait], total_transfers, current_ids)
                    if stale is None or fresh is None:
                        continue
                    stale_ids, stale_pool = stale
                    hold_pred = (sum(stale_pool[i]["predicted_points"] for i in stale_ids)
                                 - sum(stale_pool[i]["predicted_points"] for i in current_ids))
                    fresh_ids, _ = fresh
                    hold_real = (
                        score_weeks(current_ids, held_weeks, lineup_preds, actuals_by_week, static_by_id)
                        + score_weeks(fresh_ids, later_weeks, lineup_preds, actuals_by_week, static_by_id)
                    ) - base_real

                    records.append({
                        "decision_gw": decision_gw, "squad_no": squad_no,
                        "start_ft": start_ft, "wait": wait, "total_transfers": total_transfers,
                        "spend_predicted": spend_pred, "spend_realised": spend_real,
                        "hold_predicted": hold_pred, "hold_realised": hold_real,
                        "hold_edge": hold_real - spend_real,
                    })

        print(f"GW{decision_gw}: done ({len([r for r in records if r['decision_gw'] == decision_gw])} scenarios)")

    if not records:
        print("No scenarios evaluated.")
        return

    df = pd.DataFrame(records)
    df.to_csv(SCENARIOS_FILE, index=False)
    print()
    print(f"Saved {len(df)} paired scenarios to {SCENARIOS_FILE}")
    print()
    print(f"{len(df)} pairs across {df['decision_gw'].nunique()} decision gameweeks, "
          f"{HORIZON}-GW evaluation window. Within each pair both arms spend the same number of "
          f"transfers over the same weeks and end with none banked -- only the timing differs.")
    print()

    print("REALISATION -- how much of each arm's predicted gain actually materialised:")
    print(f"{'Arm':<26}{'n':<6}{'Mean predicted':<17}{'Mean realised':<16}{'Shrinkage':<12}")
    for label, pred_col, real_col in (("spend now", "spend_predicted", "spend_realised"),
                                      ("bank, then bundle", "hold_predicted", "hold_realised")):
        pred_mean, real_mean = df[pred_col].mean(), df[real_col].mean()
        shrink = real_mean / pred_mean if pred_mean else float("nan")
        print(f"{label:<26}{len(df):<6}{pred_mean:<17.2f}{real_mean:<16.2f}{shrink:<12.2f}")
    print()
    print("The hold shrinkage is the number the live tool needs: it is what a deferred gain "
          "computed from TODAY's model on a shifted window is actually worth once next week's "
          "information, churn and price drift have had their say.")
    print()

    print("HEAD TO HEAD (paired on squad and window, so most of the variance differences out):")
    print(f"{'Situation':<34}{'n':<6}{'Spend':<10}{'Hold':<10}{'Hold edge':<12}{'90% CI':<20}{'Hold wins':<10}")
    for (start_ft, wait), group in df.groupby(["start_ft", "wait"]):
        edge = group["hold_edge"]
        lo, hi = bootstrap_mean_ci(edge.to_numpy())
        label = f"{start_ft} FT in hand, bank {wait}wk -> {start_ft + wait}"
        print(f"{label:<34}{len(group):<6}{group['spend_realised'].mean():<10.2f}"
              f"{group['hold_realised'].mean():<10.2f}{edge.mean():<+12.2f}"
              f"{f'[{lo:+.1f}, {hi:+.1f}]':<20}{100 * float((edge > 0).mean()):<10.0f}%")

    overall = df["hold_edge"]
    lo, hi = bootstrap_mean_ci(overall.to_numpy())
    print()
    print(f"Overall hold edge: {overall.mean():+.2f} pts (90% CI [{lo:+.2f}, {hi:+.2f}], "
          f"holding won {100 * float((overall > 0).mean()):.0f}% of pairs)")
    print()

    # The rule the tool actually needs. Waiting costs a week of whatever the immediate move was
    # worth, so the edge should shrink as the immediate move gets better -- and the band where it
    # crosses zero is the threshold to switch the recommendation at. A single average across all
    # situations would hide exactly that.
    print("By how good the SPEND-NOW option looked at decision time (this is the decision rule):")
    print(f"{'Spend-now predicted':<22}{'n':<6}{'Hold edge':<12}{'90% CI':<20}{'Hold wins':<10}")
    for low_b, high_b in [(-1e9, 3), (3, 6), (6, 10), (10, 20), (20, 1e9)]:
        group = df[(df["spend_predicted"] >= low_b) & (df["spend_predicted"] < high_b)]
        if len(group) < 15:
            continue
        edge = group["hold_edge"]
        blo, bhi = bootstrap_mean_ci(edge.to_numpy())
        label = (f"< {high_b:.0f} pts" if low_b < -1e8 else
                 f"{low_b:.0f}+ pts" if high_b > 1e8 else f"{low_b:.0f}-{high_b:.0f} pts")
        print(f"{label:<22}{len(group):<6}{edge.mean():<+12.2f}{f'[{blo:+.1f}, {bhi:+.1f}]':<20}"
              f"{100 * float((edge > 0).mean()):<10.0f}%")

    print()
    print("A positive hold edge means banking beat spending, on real points, over the same "
          "gameweeks with the same number of transfers. Read the CIs before believing any single "
          "row: these are noisy paired differences and the point estimates move around a lot.")



if __name__ == "__main__":
    ensure_utf8_stdout()
    main()
