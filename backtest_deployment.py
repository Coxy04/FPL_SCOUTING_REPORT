"""Deployment-matching backtest: the situation the live model is actually in, replayed on history.

The live model trains on LAST season's archive plus the handful of gameweeks played so far THIS
season, and predicts the next one. tune_model.py and backtest_model.py only ever split one season
internally (train GW1-28, test GW29-33), which cannot answer the questions that bit in GW1-6:
how much should last season count once this season's data exists (recency weighting), and does
the prior season help at all? The GW1-6 review found 88% of training rows were last season with
recency weighting switched off for three of four positions, and the tuner had "found" that recency
weighting doesn't help -- on a within-season split where it could never have shown up.

This replays the real setup: train on 2024-25 (whole season) plus 2025-26 GW1..N, test on 2025-26
GW N+1..N+5, for several N, under several half-lives, and with the prior season included / excluded.
Judged on the same ranking metrics as backtest_model.py.

Caveat: the 2024-25 archive predates FPL's defensive-contribution stats, so those four features are
dropped for every configuration here (they are 1-5% of gain in the live models). Comparisons are
like-for-like within this script, but absolute numbers are not directly comparable to backtest_model.py.
"""
import json
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from understatapi import UnderstatClient

from backtest_model import build_archive_rows, ranking_metrics, selection_loss, cross_position_metrics
from fpl_ml_model import (
    NON_LGB_PARAM_KEYS,
    POSITION_FEATURES,
    POSITION_MAP,
    compute_sample_weights,
    ensure_utf8_stdout,
    get_playing_time_denominator,
    get_position_params,
)

OUTPUT_FILE = "fpl_ml_deployment_backtest.json"
PRIOR = ("2024-25", "2024")
CURRENT = ("2025-26", "2025")
DROP_FEATURES = ["clearances_blocks_interceptions", "recoveries", "tackles", "defensive_contribution"]
FEATURES = [f for f in POSITION_FEATURES if f not in DROP_FEATURES]
# How many current-season gameweeks are "played" at decision time. 3 and 6 are where this season is.
CUTOFFS = [3, 6, 9, 12, 16, 20, 24, 28, 32]
TEST_WINDOW = 5
HALF_LIVES = [60, 120, 180, 365, 100000]


def fit_predict(train, test, position, half_life):
    params = {k: v for k, v in get_position_params(position).items() if k not in NON_LGB_PARAM_KEYS}
    model = lgb.LGBMRegressor(objective="regression", random_state=42, verbosity=-1, **params)
    weights = compute_sample_weights(train["date"], half_life_days=half_life, as_of=pd.Timestamp(train["date"].max(), tz="UTC"))
    model.fit(train[FEATURES], train["total_points"], sample_weight=weights)
    pred = model.predict(test[FEATURES]) * np.minimum(1.0, test["minutes"].to_numpy() / get_playing_time_denominator(position))
    return np.clip(pred, 0, 15)


def run_config(prior_rows, current_rows, cutoff, half_life, use_prior=True, use_current=True):
    train_parts = []
    if use_prior:
        train_parts.append(prior_rows)
    if use_current:
        train_parts.append(current_rows[current_rows["gw"] <= cutoff])
    train_all = pd.concat(train_parts, ignore_index=True)
    test_all = current_rows[(current_rows["gw"] > cutoff) & (current_rows["gw"] <= cutoff + TEST_WINDOW)]
    frames = []
    for position in POSITION_MAP.values():
        train = train_all[train_all["position"] == position]
        test = test_all[test_all["position"] == position]
        if len(train) < 50 or len(test) < 5:
            continue
        pred = fit_predict(train, test, position, half_life)
        frames.append(pd.DataFrame({"gw": test["gw"].to_numpy(), "position": position, "pred": pred, "actual": test["total_points"].to_numpy()}))
    combined = pd.concat(frames, ignore_index=True)
    per_position = {f["position"].iloc[0]: ranking_metrics(f) for f in frames}
    return combined, per_position


def summarise(label, runs):
    """runs: list of (combined_frame, per_position_metrics) across cutoffs."""
    cross = [cross_position_metrics([c]) for c, _ in runs]
    pooled = pd.concat([c for c, _ in runs], ignore_index=True)
    # Give each cutoff its own gw key space so pools don't merge across replays of the same GW.
    pooled = pd.concat([c.assign(gw=c["gw"] + 1000 * i) for i, (c, _) in enumerate(runs)], ignore_index=True)
    m = ranking_metrics(pooled)
    by_pos = {}
    for position in POSITION_MAP.values():
        vals = [p[position] for _, p in runs if position in p]
        if vals:
            by_pos[position] = {
                "spearman_pool": round(float(np.mean([v["spearman_pool"] or 0 for v in vals])), 3),
                "pool_bias": round(float(np.mean([v["pool_bias"] for v in vals])), 2),
                "loss": round(float(np.mean([selection_loss(v) for v in vals])), 4),
            }
    summary = {
        "label": label,
        "spearman_pool_cross": round(float(np.mean([c["spearman_pool"] or 0 for c in cross])), 3),
        "top10_actual_cross": round(float(np.mean([c["top10_actual_mean"] for c in cross])), 2),
        "spearman_pool_all": round(m["spearman_pool"], 3) if m["spearman_pool"] is not None else None,
        "mae": round(m["mae"], 3),
        "by_position": by_pos,
    }
    print(f"[{label:<34}] cross-pos Spearman {summary['spearman_pool_cross']:.3f}  top-10 actual {summary['top10_actual_cross']:.2f}  "
          f"MAE {summary['mae']:.3f}  | " + "  ".join(f"{p}: rho {v['spearman_pool']:.3f} bias {v['pool_bias']:+.2f}" for p, v in by_pos.items()))
    return summary


def main():
    print("Building feature rows for both archive seasons (this takes a minute)...")
    understat = UnderstatClient()
    prior_rows = build_archive_rows(*PRIOR, understat=understat)
    current_rows = build_archive_rows(*CURRENT, understat=understat)
    print(f"  {PRIOR[0]}: {len(prior_rows)} rows   {CURRENT[0]}: {len(current_rows)} rows\n")

    summaries = []
    print("A. Prior season + current season to date, by recency half-life (days):")
    for half_life in HALF_LIVES:
        runs = [run_config(prior_rows, current_rows, cutoff, half_life) for cutoff in CUTOFFS]
        summaries.append(summarise(f"prior+current, half-life {half_life}", runs))

    print("\nB. Does the prior season help at all? (half-life off)")
    runs = [run_config(prior_rows, current_rows, cutoff, 100000, use_prior=False) for cutoff in CUTOFFS if cutoff >= 6]
    summaries.append(summarise("current season only (N>=6)", runs))
    runs = [run_config(prior_rows, current_rows, cutoff, 100000, use_current=False) for cutoff in CUTOFFS]
    summaries.append(summarise("prior season only", runs))

    print("\nC. Early-season view only (N = 3 and 6, where this season is now), by half-life:")
    early = {}
    for half_life in HALF_LIVES:
        runs = [run_config(prior_rows, current_rows, cutoff, half_life) for cutoff in (3, 6)]
        early[half_life] = summarise(f"early season, half-life {half_life}", runs)

    with open(OUTPUT_FILE, "w") as f:
        json.dump({"summaries": summaries, "early_season": early, "cutoffs": CUTOFFS, "half_lives": HALF_LIVES,
                   "dropped_features": DROP_FEATURES}, f, indent=2)
    print(f"\nSaved {OUTPUT_FILE}")


if __name__ == "__main__":
    ensure_utf8_stdout()
    main()
