"""Chase mode: when you are far behind, should the XI and the captain be chosen for haul potential
rather than expected points -- and what does it cost?

Nathan's ask (GW12, 11th of 11, 36 behind): "go for players that could haul". Variance is worth
something when you need to catch up, but only if buying it doesn't cost too much mean. This
measures the trade on last season's replay (train on 2024-25 + 2025-26 GW<=N, predict N+1..N+5,
several N), with the production models:
  mean     -- the regression's expected points (what the optimiser maximises today)
  haul     -- the haul classifier's P(points >= 8)
  high     -- the 90th-percentile quantile model (the top of the dashboard's range)
  blend_k  -- mean + k * haul_prob * HAUL_THRESHOLD, i.e. expected points plus a bonus for tail mass

Each week an XI (1-4-4-2 among regulars: recency-weighted minutes >= 60) and a captain are picked
by each score and paid in REAL points. Reported per rule: mean weekly XI points, SD across weeks,
share of weeks at or above the 75th percentile of the mean rule's weeks ("big weeks"), hauls per
week in the XI, and captain points. A chase rule earns its place only if it raises the big-week
share without giving back more mean than the variance is worth.
"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from backtest_model import build_archive_rows
from fpl_ml_model import (
    HAUL_THRESHOLD,
    NON_LGB_PARAM_KEYS,
    POSITION_FEATURES,
    POSITION_MAP,
    QUANTILE_HIGH_ALPHA,
    compute_sample_weights,
    ensure_utf8_stdout,
    get_half_life_days,
    get_playing_time_denominator,
    get_position_params,
)

OUTPUT_FILE = "fpl_ml_chase_ablation.json"
PRIOR = ("2024-25", "2024")
CURRENT = ("2025-26", "2025")
DROP_FEATURES = ["clearances_blocks_interceptions", "recoveries", "tackles", "defensive_contribution"]
FEATURES = [f for f in POSITION_FEATURES if f not in DROP_FEATURES]
CUTOFFS = [6, 9, 12, 16, 20, 24, 28, 32]
HORIZON = 5
REGULAR_MINUTES = 60
FORMATION = {"GK": 1, "DEF": 4, "MID": 4, "FWD": 2}
BLENDS = [0.25, 0.5, 1.0]


def fit_all(train, position):
    params = {k: v for k, v in get_position_params(position).items() if k not in NON_LGB_PARAM_KEYS}
    weights = compute_sample_weights(train["date"], half_life_days=get_half_life_days(position),
                                     as_of=pd.Timestamp(train["date"].max(), tz="UTC"))
    mean = lgb.LGBMRegressor(objective="regression", random_state=42, verbosity=-1, **params).fit(train[FEATURES], train["total_points"], sample_weight=weights)
    high = lgb.LGBMRegressor(objective="quantile", alpha=QUANTILE_HIGH_ALPHA, random_state=42, verbosity=-1, **params).fit(train[FEATURES], train["total_points"], sample_weight=weights)
    haul = lgb.LGBMClassifier(random_state=42, verbosity=-1, **params).fit(train[FEATURES], (train["total_points"] >= HAUL_THRESHOLD).astype(int), sample_weight=weights)
    return mean, high, haul


def predict_week(train_all, test_all):
    frames = []
    for position in POSITION_MAP.values():
        train, test = train_all[train_all["position"] == position], test_all[test_all["position"] == position]
        if len(train) < 50 or test.empty:
            continue
        mean, high, haul = fit_all(train, position)
        mult = np.minimum(1.0, test["minutes"].to_numpy() / get_playing_time_denominator(position))
        f = test[["gw", "current_id", "position", "minutes", "total_points"]].copy()
        f["mean"] = np.clip(mean.predict(test[FEATURES]) * mult, 0, 15)
        f["high"] = np.clip(high.predict(test[FEATURES]) * mult, 0, 20)
        f["haul"] = haul.predict_proba(test[FEATURES])[:, 1] * mult
        for k in BLENDS:
            f[f"blend_{k}"] = f["mean"] + k * f["haul"] * HAUL_THRESHOLD
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def pick_xi(week, score):
    chosen = []
    for position, n in FORMATION.items():
        pool = week[week["position"] == position].nlargest(n, score)
        chosen.append(pool)
    xi = pd.concat(chosen)
    captain = xi.nlargest(1, score)
    return xi, captain


def main():
    print("Building feature rows (played rows only)...")
    prior, current = build_archive_rows(*PRIOR), build_archive_rows(*CURRENT)
    rules = ["mean", "high", "haul"] + [f"blend_{k}" for k in BLENDS]
    weeks = {r: [] for r in rules}
    captains = {r: [] for r in rules}
    hauls = {r: [] for r in rules}
    for cutoff in CUTOFFS:
        train_all = pd.concat([prior, current[current["gw"] <= cutoff]], ignore_index=True)
        test_all = current[(current["gw"] > cutoff) & (current["gw"] <= cutoff + HORIZON)]
        pred = predict_week(train_all, test_all)
        pred = pred[pred["minutes"] >= REGULAR_MINUTES]
        for gw, week in pred.groupby("gw"):
            for r in rules:
                xi, cap = pick_xi(week, r)
                weeks[r].append(float(xi["total_points"].sum()))
                captains[r].append(float(cap["total_points"].iloc[0]))
                hauls[r].append(int((xi["total_points"] >= HAUL_THRESHOLD).sum()))
        print(f"  N={cutoff}: " + "  ".join(f"{r} {np.mean(weeks[r][-5:]):.1f}" for r in rules))

    big_cut = float(np.percentile(weeks["mean"], 75))
    summary = {}
    print(f"\nXI (1-4-4-2 among regulars) paid in real points, {len(weeks['mean'])} weeks replayed. Big week = >= {big_cut:.0f} pts (75th pct of the mean rule).")
    print(f"  {'rule':<10} {'mean/wk':>8} {'sd':>6} {'big weeks':>10} {'hauls/wk':>9} {'captain/wk':>11} {'cap haul%':>10}")
    for r in rules:
        w, c, h = np.array(weeks[r]), np.array(captains[r]), np.array(hauls[r])
        summary[r] = {"xi_mean": round(float(w.mean()), 2), "xi_sd": round(float(w.std()), 2), "big_week_share": round(float((w >= big_cut).mean()), 3),
                      "hauls_per_week": round(float(h.mean()), 2), "captain_mean": round(float(c.mean()), 2), "captain_haul_rate": round(float((c >= HAUL_THRESHOLD).mean()), 3)}
        s = summary[r]
        print(f"  {r:<10} {s['xi_mean']:>8.1f} {s['xi_sd']:>6.1f} {100*s['big_week_share']:>9.0f}% {s['hauls_per_week']:>9.2f} {s['captain_mean']:>11.2f} {100*s['captain_haul_rate']:>9.0f}%")
    with open(OUTPUT_FILE, "w") as f:
        json.dump({"big_week_cut": big_cut, "weeks": len(weeks["mean"]), "rules": summary}, f, indent=2)
    print(f"\nSaved {OUTPUT_FILE}")


if __name__ == "__main__":
    ensure_utf8_stdout()
    main()
