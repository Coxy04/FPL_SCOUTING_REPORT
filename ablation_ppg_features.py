"""Does a points track record (points_form, season_ppg) earn its place in the model?

Why this exists: the GW8 fundamental review measured season points-per-game as the best single
predictor of the next five weeks (Spearman 0.36) with the fixture run nearly irrelevant (0.06), yet
the production model had no points feature at all -- it was built on process stats (xG, xA,
defensive actions) and fixtures. The one earlier test of PPG-as-feature was on single-week ranking
in a fixed pool, where it made no difference. Nobody had tested it on the horizon transfers are
actually judged on. De Cuyper (76 pts, best defender in the game by PPG) priced below Hill (56)
over five weeks was the symptom that prompted this.

Replay, deployment-matching (see backtest_deployment.py): train on 2024-25 plus 2025-26 GW<=N,
forecast GW N+1..N+5, for several N. Two arms, identical otherwise:
  base  -- production features minus the two new ones
  ppg   -- base + points_form + season_ppg
Judged two ways:
  1. next-week ranking, as backtest_model.py scores it (pool Spearman per position, cross-position
     top-100 pool, top-10 actual).
  2. FIVE-WEEK SUM, built the way the live pipeline builds it: every player-level feature frozen at
     the cutoff (the GW N+1 row, whose lagged stats only see GW<=N), only the fixture-level
     features vary week to week, playing-time multiplier from the cutoff minutes. Spearman of the
     forecast sum vs the actual 5-week points among regulars (recency-weighted minutes >= 45 at
     the cutoff), and the actual points of the top-10 by forecast. Reference rankings on the same
     pool: season_ppg alone, points_form alone.
Caveat: opponent form for week k is taken from that week's row (a slight look-ahead), identically
in both arms, so the comparison is fair even if the absolute numbers flatter both.
"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from backtest_model import build_archive_rows, cross_position_metrics, ranking_metrics, spearman
from fpl_ml_model import (
    NON_LGB_PARAM_KEYS,
    POSITION_FEATURES,
    POSITION_MAP,
    PPG_FEATURES,
    compute_sample_weights,
    ensure_utf8_stdout,
    get_half_life_days,
    get_playing_time_denominator,
    get_position_params,
)

OUTPUT_FILE = "fpl_ml_ppg_ablation.json"
PRIOR = ("2024-25", "2024")
CURRENT = ("2025-26", "2025")
# 2024-25 predates the defensive-contribution stats (see backtest_deployment.py).
DROP_FEATURES = ["clearances_blocks_interceptions", "recoveries", "tackles", "defensive_contribution"]
BASE = [f for f in POSITION_FEATURES if f not in DROP_FEATURES and f not in PPG_FEATURES]
ARMS = {"base": BASE, "ppg": BASE + PPG_FEATURES}
CUTOFFS = [6, 9, 12, 16, 20, 24, 28, 32]
HORIZON = 5
REGULAR_MINUTES = 45
FIXTURE_LEVEL = ["was_home", "difficulty", "own_days_rest", "opp_days_rest", "opp_xg_for_form", "opp_xg_against_form"]


def fit(train, position, features):
    params = {k: v for k, v in get_position_params(position).items() if k not in NON_LGB_PARAM_KEYS}
    model = lgb.LGBMRegressor(objective="regression", random_state=42, verbosity=-1, **params)
    weights = compute_sample_weights(train["date"], half_life_days=get_half_life_days(position),
                                     as_of=pd.Timestamp(train["date"].max(), tz="UTC"))
    model.fit(train[features], train["total_points"], sample_weight=weights)
    return model


def predict(model, rows, position, features):
    pred = model.predict(rows[features]) * np.minimum(1.0, rows["minutes"].to_numpy() / get_playing_time_denominator(position))
    return np.clip(pred, 0, 15)


def horizon_rows(current, cutoff):
    """Forecast rows for GW N+1..N+5 with player-level features frozen at the player's first row
    after the cutoff, fixture-level features from each week's own row."""
    window = current[(current["gw"] > cutoff) & (current["gw"] <= cutoff + HORIZON)].copy()
    first = window.sort_values("gw").groupby("current_id").head(1).set_index("current_id")
    first = first[first["gw"] == cutoff + 1]  # must have a fixture in the first week to be "known at the cutoff"
    window = window[window["current_id"].isin(first.index)]
    player_level = [c for c in POSITION_FEATURES if c not in FIXTURE_LEVEL]
    frozen = window[["current_id", "gw", "total_points", "position"] + FIXTURE_LEVEL].merge(
        first[player_level], left_on="current_id", right_index=True, how="left")
    return frozen


def five_week(train_all, current, cutoff, features_by_arm):
    rows = horizon_rows(current, cutoff)
    out = []
    for position in POSITION_MAP.values():
        pos_rows = rows[rows["position"] == position]
        train = train_all[train_all["position"] == position]
        if len(train) < 50 or pos_rows.empty:
            continue
        per_player = pos_rows.groupby("current_id").agg(actual=("total_points", "sum"), minutes=("minutes", "first"),
                                                       season_ppg=("season_ppg", "first"), points_form=("points_form", "first"))
        for arm, features in features_by_arm.items():
            model = fit(train, position, features)
            pos_rows = pos_rows.assign(**{f"pred_{arm}": predict(model, pos_rows, position, features)})
            per_player[f"pred_{arm}"] = pos_rows.groupby("current_id")[f"pred_{arm}"].sum()
        per_player["position"] = position
        out.append(per_player.reset_index())
    return pd.concat(out, ignore_index=True)


def score_five_week(frame, pred_col):
    pool = frame[frame["minutes"] >= REGULAR_MINUTES]
    by_pos = {}
    for position, g in pool.groupby("position"):
        top = g.nlargest(10, pred_col)
        by_pos[position] = {"rho": spearman(g[pred_col], g["actual"]), "top10_actual": float(top["actual"].mean()), "n": int(len(g))}
    top = pool.nlargest(15, pred_col)
    return {"rho_cross": spearman(pool[pred_col], pool["actual"]), "top15_actual_cross": float(top["actual"].mean()), "by_position": by_pos, "n": int(len(pool))}


def next_week(train_all, current, cutoff, features_by_arm):
    test_all = current[(current["gw"] > cutoff) & (current["gw"] <= cutoff + HORIZON) & (current["actual_minutes"] > 0)]
    results = {}
    for arm, features in features_by_arm.items():
        frames = []
        for position in POSITION_MAP.values():
            train = train_all[train_all["position"] == position]
            test = test_all[test_all["position"] == position]
            if len(train) < 50 or len(test) < 5:
                continue
            model = fit(train, position, features)
            frames.append(pd.DataFrame({"gw": test["gw"].to_numpy(), "position": position,
                                        "pred": predict(model, test, position, features), "actual": test["total_points"].to_numpy()}))
        cross = cross_position_metrics(frames)
        results[arm] = {"cross": cross, "by_position": {f["position"].iloc[0]: ranking_metrics(f) for f in frames}}
    return results


def mean_of(values):
    vals = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return round(float(np.mean(vals)), 3) if vals else None


def main():
    print("Building feature rows for 2024-25 and 2025-26 (unplayed rows kept for the 5-week sums)...")
    prior = build_archive_rows(*PRIOR, include_unplayed=True)
    current = build_archive_rows(*CURRENT, include_unplayed=True)
    for col in PPG_FEATURES:
        for f in (prior, current):
            if col not in f:
                raise SystemExit(f"{col} missing from archive rows -- feature plumbing is incomplete")
    print(f"  prior {len(prior)} rows, current {len(current)} rows; season_ppg non-zero on "
          f"{(current['season_ppg'] > 0).mean():.0%} of current rows\n")

    weekly, five = {arm: [] for arm in ARMS}, {k: [] for k in list(ARMS) + ["season_ppg", "points_form"]}
    for cutoff in CUTOFFS:
        train_all = pd.concat([prior[prior["actual_minutes"] > 0], current[(current["gw"] <= cutoff) & (current["actual_minutes"] > 0)]], ignore_index=True)
        nw = next_week(train_all, current, cutoff, ARMS)
        fw = five_week(train_all, current, cutoff, ARMS)
        line = f"N={cutoff:<2} next-week cross rho " + "  ".join(f"{arm} {nw[arm]['cross']['spearman_pool']:.3f}" for arm in ARMS)
        line += " | 5-wk cross rho " + "  ".join(f"{k} {score_five_week(fw, f'pred_{k}' if k in ARMS else k)['rho_cross']:.3f}" for k in five)
        print(line)
        for arm in ARMS:
            weekly[arm].append(nw[arm])
            five[arm].append(score_five_week(fw, f"pred_{arm}"))
        for ref in ("season_ppg", "points_form"):
            five[ref].append(score_five_week(fw, ref))

    summary = {"cutoffs": CUTOFFS, "next_week": {}, "five_week": {}}
    print("\nNEXT-WEEK RANKING (mean over cutoffs)")
    for arm in ARMS:
        s = {"cross_rho": mean_of(r["cross"]["spearman_pool"] for r in weekly[arm]),
             "top10_actual": mean_of(r["cross"]["top10_actual_mean"] for r in weekly[arm]),
             "by_position": {p: {"rho": mean_of(r["by_position"][p]["spearman_pool"] for r in weekly[arm] if p in r["by_position"]),
                                 "bias": mean_of(r["by_position"][p]["pool_bias"] for r in weekly[arm] if p in r["by_position"])}
                             for p in POSITION_MAP.values()}}
        summary["next_week"][arm] = s
        print(f"  {arm:<5} cross rho {s['cross_rho']:.3f}  top-10 actual {s['top10_actual']:.2f}  | "
              + "  ".join(f"{p} {v['rho']:.3f} (bias {v['bias']:+.2f})" for p, v in s["by_position"].items()))

    print("\nFIVE-WEEK SUM among regulars (mean over cutoffs)")
    for k in five:
        s = {"cross_rho": mean_of(r["rho_cross"] for r in five[k]),
             "top15_actual": mean_of(r["top15_actual_cross"] for r in five[k]),
             "by_position": {p: {"rho": mean_of(r["by_position"][p]["rho"] for r in five[k] if p in r["by_position"]),
                                 "top10_actual": mean_of(r["by_position"][p]["top10_actual"] for r in five[k] if p in r["by_position"])}
                             for p in POSITION_MAP.values()}}
        summary["five_week"][k] = s
        print(f"  {k:<12} cross rho {s['cross_rho']:.3f}  top-15 actual {s['top15_actual']:.1f}  | "
              + "  ".join(f"{p} {v['rho']:.3f} (top10 {v['top10_actual']:.1f})" for p, v in s["by_position"].items()))

    with open(OUTPUT_FILE, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {OUTPUT_FILE}")


if __name__ == "__main__":
    ensure_utf8_stdout()
    main()
