"""The decisive comparison: the production model vs season points-per-game vs a model that is allowed
to see points history and ownership, all scored on ONE fixed pool of regulars.

signal_study.py found season PPG alone ranks regulars at Spearman 0.22 for this week and 0.36 for the
next five, while the production model has never used points history or ownership at all. Earlier
ablations judged candidate features on a pool defined by the model's OWN predictions, which moves
when the features change and makes results incomparable. Here the pool is fixed before any model
runs: players with 60+ recency-weighted minutes and 3+ prior appearances, as in signal_study.py.

Deployment replay on 2025-26: train on 2024-25 + 2025-26 GW<=N, predict GW N+1..N+5.
"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from understatapi import UnderstatClient

from backtest_deployment import FEATURES as PROD_FEATURES, TEST_WINDOW
from backtest_model import build_archive_rows
from fpl_ml_model import NON_LGB_PARAM_KEYS, POSITION_MAP, compute_sample_weights, ensure_utf8_stdout, get_playing_time_denominator, get_position_params, load_prior_season_archive
from signal_study import add_lagged_signals, load_season, SIGNALS, NEGATIVE, spearman

OUT = "fpl_ml_fixed_pool_ablation.json"
CUTOFFS = [8, 12, 16, 20, 24, 28, 32]
EXTRA = ["ppg_season", "bps_season", "ownership", "ict_last6", "ppg_last6"]


def signal_frame(season):
    """Lagged signals from signal_study, keyed (current_id, gw), plus the fixed-pool flag."""
    s = add_lagged_signals(load_season(season))
    for c in NEGATIVE:
        s[c] = -s[c]
    s["in_pool"] = (s["min_last6"] >= 60) & (s["games_prior"] >= 3)
    keep = ["element", "GW", "in_pool"] + SIGNALS
    return s[keep].rename(columns={"element": "current_id", "GW": "gw"})


def fit_predict(train, test, feats, pos, params_from=None, big=False):
    params = {k: v for k, v in get_position_params(pos).items() if k not in NON_LGB_PARAM_KEYS}
    if big:
        params = dict(n_estimators=400, learning_rate=0.03, num_leaves=15, max_depth=5, min_child_samples=40, reg_lambda=2.0)
    m = lgb.LGBMRegressor(objective="regression", random_state=42, verbosity=-1, **params)
    w = compute_sample_weights(train["date"], half_life_days=100000, as_of=pd.Timestamp(train["date"].max(), tz="UTC"))
    m.fit(train[feats], train["total_points"], sample_weight=w)
    return m.predict(test[feats])


def main():
    u = UnderstatClient()
    prior = build_archive_rows("2024-25", "2024", understat=u).merge(signal_frame("2024-25"), on=["current_id", "gw"], how="left")
    cur = build_archive_rows("2025-26", "2025", understat=u).merge(signal_frame("2025-26"), on=["current_id", "gw"], how="left")
    for f in (prior, cur):
        f["in_pool"] = f["in_pool"].fillna(False).astype(bool)
    variants = {
        "production features (as live)": (PROD_FEATURES, False),
        "production + ppg/bps/ownership/ict": (PROD_FEATURES + EXTRA, False),
        "signal-study features only (kitchen sink)": ([s for s in SIGNALS] + ["was_home"], True),
        "production + ALL signal-study features": (list(dict.fromkeys(PROD_FEATURES + SIGNALS)), False),
    }
    rows = []
    for cutoff in CUTOFFS:
        train_all = pd.concat([prior, cur[cur["gw"] <= cutoff]], ignore_index=True)
        test_all = cur[(cur["gw"] > cutoff) & (cur["gw"] <= cutoff + TEST_WINDOW) & cur["in_pool"]].copy()
        for name, (feats, big) in variants.items():
            feats = [f for f in feats if f in train_all.columns]
            tr_ok = train_all.dropna(subset=[c for c in feats if c in EXTRA or c in SIGNALS])
            test_all[name] = np.nan
            for pos in POSITION_MAP.values():
                tr = tr_ok[tr_ok["position"] == pos]; te = test_all[test_all["position"] == pos]
                if te.empty or len(tr) < 100:
                    continue
                pred = fit_predict(tr, te, feats, pos, big=big)
                # Same playing-time discount as live, for a fair comparison.
                pred = pred * np.minimum(1.0, te["minutes"].to_numpy() / get_playing_time_denominator(pos))
                test_all.loc[te.index, name] = np.clip(pred, 0, 15)
        rows.append(test_all)
    t = pd.concat(rows, ignore_index=True)
    t["key"] = t["gw"].astype(str) + "@" + t.groupby(["gw"]).ngroup().astype(str)  # placeholder, replaced below
    # Score per (cutoff-replay, gameweek) so repeated gameweeks across cutoffs don't merge.
    t["rep"] = np.repeat(range(len(rows)), [len(r) for r in rows])
    t["key"] = t["rep"].astype(str) + "-" + t["gw"].astype(str)
    scorers = {name: name for name in variants} | {"season PPG alone": "ppg_season", "price alone": "now_cost", "ownership alone": "ownership", "BPS season alone": "bps_season"}
    summary = {}
    print(f"Fixed pool of regulars: {len(t)} test rows across {len(CUTOFFS)} cutoffs; mean pts {t['total_points'].mean():.2f}, haul rate {(t['total_points']>=8).mean():.3f}\n")
    print(f"{'ranker':<44} {'Spearman/GW':>12} {'top-10 actual':>14} {'captain pts':>12} {'captain haul%':>14}")
    for label, col in scorers.items():
        if col not in t.columns:
            continue
        per_gw = []
        top10 = []
        caps = []
        for _, g in t.groupby("key"):
            g = g.dropna(subset=[col])
            if len(g) < 10:
                continue
            per_gw.append(spearman(g[col], g["total_points"]))
            top10.append(g.nlargest(10, col)["total_points"].mean())
            caps.append(g.loc[g[col].idxmax(), "total_points"])
        summary[label] = {"spearman": float(np.nanmean(per_gw)), "top10_actual": float(np.mean(top10)), "captain_mean": float(np.mean(caps)), "captain_haul_rate": float(np.mean(np.array(caps) >= 8))}
        s = summary[label]
        print(f"{label:<44} {s['spearman']:>12.3f} {s['top10_actual']:>14.2f} {s['captain_mean']:>12.2f} {100*s['captain_haul_rate']:>13.0f}%")
    # Premium-attacker captaincy: argmax of each ranker restricted to MID/FWD priced 9.0+.
    print("\nCaptain restricted to MID/FWD priced >= 9.0 (what the haul-rate table says a captain should be):")
    prem = t[(t["position"].isin(["MID", "FWD"])) & (t["now_cost"] >= 90)]
    for label, col in scorers.items():
        if col not in t.columns:
            continue
        caps = [g.loc[g[col].idxmax(), "total_points"] for _, g in prem.groupby("key") if g[col].notna().sum() >= 3]
        if caps:
            summary[label]["captain_premium_mean"] = float(np.mean(caps))
            print(f"  {label:<42} mean {np.mean(caps):.2f}, hauls {100*np.mean(np.array(caps)>=8):.0f}%")
    with open(OUT, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    ensure_utf8_stdout()
    main()
