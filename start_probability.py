"""A start-probability model, and the test that decides whether it replaces the playing-time discount.

The live model discounts predictions by min(1, recency-weighted minutes / denominator). That rule
treats one 0-minute match for a fit regular as a 28% haircut for ~3 weeks (Joao Pedro, GW7: raw 6.61,
4th in the game, shown as 4.78, 24th). ablation_playing_time_rule.py found the rule ranks well AMONG
PLAYERS WHO PLAYED -- but the discount's actual job is to price players who end up NOT playing, and
that test excluded them. This script does it properly:

1. Trains a LightGBM classifier on every squad row (including 0-minute rows) to predict
   P(minutes >= 60) and P(minutes > 0) from lagged minutes/starts history, position and price.
2. Replays the deployment setup on 2025-26 with the point model trained as live (played rows only),
   then prices EVERY squad player for the test weeks with each multiplier rule, scoring against real
   points where a non-player scores 0.

If P(start) wins on ranking skill including non-players, it should replace the rule in
fpl_ml_model.recent_player_features (with FPL's chance_of_playing applied on top as now).
"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from understatapi import UnderstatClient

from backtest_deployment import CUTOFFS, FEATURES, TEST_WINDOW
from backtest_model import build_archive_rows, cross_position_metrics, ranking_metrics
from fpl_ml_model import (
    NON_LGB_PARAM_KEYS, POSITION_MAP, compute_sample_weights, ensure_utf8_stdout, get_playing_time_denominator,
    get_position_params, load_prior_season_archive,
)

OUTPUT_FILE = "fpl_ml_start_probability.json"
START_FEATURES = ["minutes", "minutes_sd", "minutes_last", "minutes_last2", "played_last", "starts_share6",
                  "games_since_start", "now_cost", "gw", "position_GK", "position_DEF", "position_MID", "position_FWD"]
START_PARAMS = {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 15, "max_depth": 5, "min_child_samples": 50, "reg_lambda": 2.0}


def add_start_features(rows, archive_season):
    """Lagged availability history from the raw archive, keyed to the feature rows by (player, GW)."""
    merged_gw, *_ = load_prior_season_archive(archive_season)
    g = merged_gw[merged_gw["position"].isin(POSITION_MAP.values())].sort_values(["element", "GW"]).copy()
    g["minutes"] = pd.to_numeric(g["minutes"], errors="coerce").fillna(0)
    g["starts"] = pd.to_numeric(g.get("starts", (g["minutes"] >= 60).astype(int)), errors="coerce").fillna(0)
    grp = g.groupby("element")
    g["minutes_last"] = grp["minutes"].shift(1)
    g["minutes_last2"] = grp["minutes"].shift(2)
    g["played_last"] = (g["minutes_last"] > 0).astype(float)
    g["starts_share6"] = grp["starts"].transform(lambda s: s.shift(1).rolling(6, min_periods=1).mean())
    started = g["starts"] > 0
    g["games_since_start"] = grp.apply(lambda d: (~(d["starts"] > 0)).groupby((d["starts"] > 0).cumsum()).cumsum().shift(1), include_groups=False).reset_index(level=0, drop=True)
    extra = g[["element", "GW", "minutes_last", "minutes_last2", "played_last", "starts_share6", "games_since_start"]].rename(columns={"element": "current_id", "GW": "gw"})
    out = rows.merge(extra, on=["current_id", "gw"], how="left")
    for c in ("minutes_last", "minutes_last2", "played_last", "starts_share6", "games_since_start"):
        out[c] = out[c].fillna(0)
    return out


def fit_start_models(train):
    weights = compute_sample_weights(train["date"], half_life_days=100000, as_of=pd.Timestamp(train["date"].max(), tz="UTC"))
    models = {}
    for name, target in (("p60", train["actual_minutes"] >= 60), ("p_any", train["actual_minutes"] > 0)):
        m = lgb.LGBMClassifier(objective="binary", random_state=42, verbosity=-1, **START_PARAMS)
        m.fit(train[START_FEATURES], target.astype(int), sample_weight=weights)
        models[name] = m
    return models


def main():
    understat = UnderstatClient()
    prior = add_start_features(build_archive_rows("2024-25", "2024", understat=understat, include_unplayed=True), "2024-25")
    current = add_start_features(build_archive_rows("2025-26", "2025", understat=understat, include_unplayed=True), "2025-26")
    print(f"rows incl. unplayed: 2024-25 {len(prior)}, 2025-26 {len(current)}; "
          f"share unplayed {100 * (current['actual_minutes'] == 0).mean():.0f}%")

    # First run (2026-10-16): the raw probabilities halve the non-players in the top 30 (10.1% -> 6%)
    # but shave every nailed starter by 10-15% too (pool bias -0.6) and lose ranking skill (0.152 ->
    # 0.111). The two hybrids test whether the start model can be used only where it adds
    # information: to rescue a mis-discounted regular (max rule) or as a probability normalised so
    # a nailed starter is not shaved (P / 0.9, capped at 1).
    rules = {
        "current: ewm minutes / denominator": None,
        "P(minutes >= 60)": "p60",
        "P(played) blended: P60 + 0.5*(Pany - P60)": "blend",
        "max(current rule, blended P)": "max",
        "blended P normalised (/0.9, cap 1)": "norm",
        "no discount": "none",
    }
    frames = {k: [] for k in rules}
    aucs = []
    for cutoff in CUTOFFS:
        train_all = pd.concat([prior, current[current["gw"] <= cutoff]], ignore_index=True)
        test_all = current[(current["gw"] > cutoff) & (current["gw"] <= cutoff + TEST_WINDOW)].copy()
        start_models = fit_start_models(train_all)
        p60 = start_models["p60"].predict_proba(test_all[START_FEATURES])[:, 1]
        p_any = start_models["p_any"].predict_proba(test_all[START_FEATURES])[:, 1]
        aucs.append({"cutoff": cutoff,
                     "auc_p60": float(roc_auc_score((test_all["actual_minutes"] >= 60).astype(int), p60)),
                     "auc_any": float(roc_auc_score((test_all["actual_minutes"] > 0).astype(int), p_any))})
        test_all["p60"], test_all["p_any"] = p60, p_any
        # Point model exactly as live: trained on played rows only.
        played_train = train_all[train_all["actual_minutes"] > 0]
        raw = pd.Series(np.nan, index=test_all.index)
        for pos in POSITION_MAP.values():
            tr = played_train[played_train["position"] == pos]
            te = test_all[test_all["position"] == pos]
            if te.empty:
                continue
            params = {k: v for k, v in get_position_params(pos).items() if k not in NON_LGB_PARAM_KEYS}
            m = lgb.LGBMRegressor(objective="regression", random_state=42, verbosity=-1, **params)
            w = compute_sample_weights(tr["date"], half_life_days=100000, as_of=pd.Timestamp(tr["date"].max(), tz="UTC"))
            m.fit(tr[FEATURES], tr["total_points"], sample_weight=w)
            raw.loc[te.index] = m.predict(te[FEATURES])
        test_all["raw"] = raw
        den = test_all["position"].map(get_playing_time_denominator).astype(float)
        current_rule = np.minimum(1.0, test_all["minutes"] / den)
        blended = test_all["p60"] + 0.5 * (test_all["p_any"] - test_all["p60"])
        multipliers = {
            "current: ewm minutes / denominator": current_rule,
            "P(minutes >= 60)": test_all["p60"],
            "P(played) blended: P60 + 0.5*(Pany - P60)": blended,
            "max(current rule, blended P)": np.maximum(current_rule, blended),
            "blended P normalised (/0.9, cap 1)": np.minimum(1.0, blended / 0.9),
            "no discount": pd.Series(1.0, index=test_all.index),
        }
        for name, mult in multipliers.items():
            frames[name].append(pd.DataFrame({"gw": test_all["gw"].to_numpy(), "position": test_all["position"].to_numpy(),
                                              "pred": np.clip(test_all["raw"].to_numpy() * mult.to_numpy(), 0, 15),
                                              "actual": test_all["total_points"].to_numpy(),
                                              "played": (test_all["actual_minutes"] > 0).to_numpy()}))

    print("Start model AUC by cutoff:", [(a["cutoff"], round(a["auc_p60"], 3), round(a["auc_any"], 3)) for a in aucs])
    print("\nDeployment replay INCLUDING non-players (a non-player scores 0):")
    summary = {}
    for name, fs in frames.items():
        cross = [cross_position_metrics([f]) for f in fs]
        pooled = pd.concat([f.assign(gw=f["gw"] + 1000 * i) for i, f in enumerate(fs)], ignore_index=True)
        m = ranking_metrics(pooled)
        top = pooled.copy(); top["rank"] = top.groupby("gw")["pred"].rank(ascending=False, method="first")
        top30 = top[top["rank"] <= 30]
        summary[name] = {"spearman_pool_cross": float(np.mean([c["spearman_pool"] or 0 for c in cross])),
                         "top10_actual": float(np.mean([c["top10_actual_mean"] for c in cross])),
                         "top30_unplayed_pct": float(100 * (~top30["played"]).mean()),
                         "pool_bias": m["pool_bias"], "mae": m["mae"]}
        s = summary[name]
        print(f"  [{name:<44}] cross-pos pool Spearman {s['spearman_pool_cross']:.3f}  top-10 actual {s['top10_actual']:.2f}  "
              f"top-30 picks who didn't play {s['top30_unplayed_pct']:.1f}%  pool bias {s['pool_bias']:+.2f}  MAE {s['mae']:.3f}")
    with open(OUTPUT_FILE, "w") as f:
        json.dump({"aucs": aucs, "summary": summary}, f, indent=2)
    print(f"Saved {OUTPUT_FILE}")


if __name__ == "__main__":
    ensure_utf8_stdout()
    main()
