"""Does train/serve skew in the per-90 rate features explain the live collapse?

Live, the model is TRAINED on rows whose npxg90/xa90/etc. are full-season Understat aggregates (stable,
and including matches after the row's own date), and then SERVED current-season aggregates built from
whatever minutes exist so far -- 90-300 minutes in September, where a per-90 rate can be anything
(Chilwell: 2.05 npxG/90 from 14 minutes). On the archive replays the rates are full-season at both
train and test, so no ablation so far could see this.

Simulated here with FPL's own per-match xG/xA (available in every archive row) so the comparison is
clean. Three regimes for the "rate" features (xGI/90, xG/90, xA/90, xGC/90):
  A. full-season at train AND test (what every archive backtest has measured)
  B. full-season at train, point-in-time cumulative at test (what live actually does)
  C. point-in-time cumulative at train AND test, shrunk toward the position prior by minutes
     (the consistent, leak-free version a fix would ship)
Same deployment replay as elsewhere: 2024-25 + 2025-26 GW<=N train, N+1..N+5 test, fixed regulars pool.
"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from understatapi import UnderstatClient

from backtest_deployment import FEATURES as PROD_FEATURES, TEST_WINDOW
from backtest_model import build_archive_rows
from fpl_ml_model import NON_LGB_PARAM_KEYS, POSITION_MAP, compute_sample_weights, ensure_utf8_stdout, get_playing_time_denominator, get_position_params, load_prior_season_archive
from signal_study import spearman

OUT = "fpl_ml_train_serve_skew.json"
CUTOFFS = [3, 6, 9, 12, 16, 20, 24, 28, 32]
UNDERSTAT_RATES = ["npxg90", "xa90", "xgchain90", "xgbuildup90", "shots90", "key_passes90"]
RATE_FEATURES = ["r_xg90", "r_xa90", "r_xgi90", "r_xgc90"]
SHRINK_MINUTES = 600  # ~6.7 full matches of prior weight toward the position mean


def rate_frames(season):
    """Per (element, GW): full-season rates, point-in-time cumulative rates, and shrunk point-in-time rates."""
    mg, *_ = load_prior_season_archive(season)
    g = mg[mg["position"].isin(POSITION_MAP.values())].sort_values(["element", "GW"]).copy()
    for c in ("minutes", "expected_goals", "expected_assists", "expected_goals_conceded"):
        g[c] = pd.to_numeric(g[c], errors="coerce").fillna(0)
    g["xgi"] = g["expected_goals"] + g["expected_assists"]
    tot = g.groupby("element").agg(m=("minutes", "sum"), xg=("expected_goals", "sum"), xa=("expected_assists", "sum"), xgi=("xgi", "sum"), xgc=("expected_goals_conceded", "sum"))
    full = pd.DataFrame({"full_xg90": 90 * tot.xg / tot.m.replace(0, np.nan), "full_xa90": 90 * tot.xa / tot.m.replace(0, np.nan),
                         "full_xgi90": 90 * tot.xgi / tot.m.replace(0, np.nan), "full_xgc90": 90 * tot.xgc / tot.m.replace(0, np.nan)}).fillna(0)
    grp = g.groupby("element")
    cm = grp["minutes"].transform(lambda s: s.shift(1).cumsum()).fillna(0)
    out = g[["element", "GW", "position"]].copy()
    pos_prior = {}
    for col, src in (("xg90", "expected_goals"), ("xa90", "expected_assists"), ("xgi90", "xgi"), ("xgc90", "expected_goals_conceded")):
        cs = grp[src].transform(lambda s: s.shift(1).cumsum()).fillna(0)
        out[f"pit_{col}"] = np.where(cm > 0, 90 * cs / cm.replace(0, np.nan), 0.0)
        # Position prior = that position's full-season mean rate (would be last season's in live use).
        prior = full.join(g.groupby("element")["position"].first()).groupby("position")[f"full_{col}"].mean()
        pos_prior[col] = prior.to_dict()
        pr = out["position"].map(prior).fillna(0)
        out[f"shr_{col}"] = (90 * cs + SHRINK_MINUTES * pr) / (cm + SHRINK_MINUTES)
    out = out.join(full, on="element")
    return out.rename(columns={"element": "current_id", "GW": "gw"}).drop(columns=["position"])


def run(prior, cur, regime, label):
    base = [f for f in PROD_FEATURES if f not in UNDERSTAT_RATES]
    feats = base + RATE_FEATURES
    frames = []
    for cutoff in CUTOFFS:
        train_all = pd.concat([prior, cur[cur["gw"] <= cutoff]], ignore_index=True)
        test_all = cur[(cur["gw"] > cutoff) & (cur["gw"] <= cutoff + TEST_WINDOW) & cur["in_pool"]].copy()
        tr_src, te_src = {"A": ("full", "full"), "B": ("full", "pit"), "C": ("shr", "shr")}[regime]
        for col in ("xg90", "xa90", "xgi90", "xgc90"):
            train_all[f"r_{col}"] = train_all[f"{tr_src}_{col}"]
            test_all[f"r_{col}"] = test_all[f"{te_src}_{col}"]
        test_all["pred"] = np.nan
        for pos in POSITION_MAP.values():
            tr = train_all[train_all["position"] == pos]; te = test_all[test_all["position"] == pos]
            if te.empty:
                continue
            params = {k: v for k, v in get_position_params(pos).items() if k not in NON_LGB_PARAM_KEYS}
            m = lgb.LGBMRegressor(objective="regression", random_state=42, verbosity=-1, **params)
            w = compute_sample_weights(tr["date"], 100000, as_of=pd.Timestamp(tr["date"].max(), tz="UTC"))
            m.fit(tr[feats], tr["total_points"], sample_weight=w)
            pred = m.predict(te[feats]) * np.minimum(1.0, te["minutes"].to_numpy() / get_playing_time_denominator(pos))
            test_all.loc[te.index, "pred"] = np.clip(pred, 0, 15)
        test_all["rep"] = cutoff
        frames.append(test_all)
    t = pd.concat(frames, ignore_index=True)
    res = {}
    for name, sub in (("all cutoffs", t), ("early season (N=3,6)", t[t["rep"].isin([3, 6])]), ("late season (N>=20)", t[t["rep"] >= 20])):
        per = [spearman(g["pred"], g["total_points"]) for _, g in sub.groupby(["rep", "gw"]) if len(g) >= 10]
        caps = [g.loc[g["pred"].idxmax(), "total_points"] for _, g in sub.groupby(["rep", "gw"]) if len(g) >= 10]
        top10 = [g.nlargest(10, "pred")["total_points"].mean() for _, g in sub.groupby(["rep", "gw"]) if len(g) >= 10]
        res[name] = {"spearman": float(np.nanmean(per)), "captain": float(np.mean(caps)), "top10": float(np.mean(top10))}
    print(f"[{label:<58}] " + " | ".join(f"{k}: rho {v['spearman']:.3f} cap {v['captain']:.2f} top10 {v['top10']:.2f}" for k, v in res.items()))
    return res


def main():
    u = UnderstatClient()
    prior = build_archive_rows("2024-25", "2024", understat=u).merge(rate_frames("2024-25"), on=["current_id", "gw"], how="left")
    cur = build_archive_rows("2025-26", "2025", understat=u).merge(rate_frames("2025-26"), on=["current_id", "gw"], how="left")
    from signal_study import add_lagged_signals, load_season
    pool = add_lagged_signals(load_season("2025-26"))
    pool["in_pool"] = (pool["min_last6"] >= 60) & (pool["games_prior"] >= 3)
    cur = cur.merge(pool[["element", "GW", "in_pool"]].rename(columns={"element": "current_id", "GW": "gw"}), on=["current_id", "gw"], how="left")
    cur["in_pool"] = cur["in_pool"].fillna(False).astype(bool)
    results = {
        "A: full-season rates at train and test (every backtest so far)": run(prior, cur, "A", "A: full-season train+test (archive view)"),
        "B: full-season at train, point-in-time at test (what live does)": run(prior, cur, "B", "B: full-season train, point-in-time test (LIVE)"),
        "C: shrunk point-in-time at train and test (proposed fix)": run(prior, cur, "C", f"C: shrunk point-in-time both (fix, k={SHRINK_MINUTES} min)"),
    }
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    ensure_utf8_stdout()
    main()
