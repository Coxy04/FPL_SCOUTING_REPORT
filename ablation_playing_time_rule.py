"""Which playing-time discount ranks startable players best? Tests the live rule (recency-weighted
minutes / denominator, which let one 0-minute match cut Joao Pedro's GW7 prediction by 28% a week
after he played 71 minutes) against alternatives, on the deployment replay (2024-25 + 2025-26
GW<=N, test N+1..N+5). Only the multiplier changes; the fitted model is identical in every arm.
"""
import numpy as np, pandas as pd, lightgbm as lgb
from understatapi import UnderstatClient
from backtest_deployment import CUTOFFS, FEATURES, TEST_WINDOW
from backtest_model import build_archive_rows, ranking_metrics, cross_position_metrics
from fpl_ml_model import NON_LGB_PARAM_KEYS, POSITION_MAP, compute_sample_weights, ensure_utf8_stdout, get_playing_time_denominator, get_position_params, load_prior_season_archive
ensure_utf8_stdout()
u = UnderstatClient()
prior = build_archive_rows("2024-25", "2024", understat=u); cur = build_archive_rows("2025-26", "2025", understat=u)
mg, *_ = load_prior_season_archive("2025-26")
g = mg[mg.minutes.notna()].sort_values(["element", "GW"])
g["minutes_last"] = g.groupby("element")["minutes"].shift(1)
g["minutes_med3"] = g.groupby("element")["minutes"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).median())
cur = cur.merge(g[["element", "GW", "minutes_last", "minutes_med3"]].rename(columns={"element": "current_id", "GW": "gw"}), on=["current_id", "gw"], how="left")
RULES = {
    "current: ewm/den": lambda t, den: np.minimum(1, t["minutes"] / den),
    "max(ewm, last match)/den": lambda t, den: np.minimum(1, np.maximum(t["minutes"], t["minutes_last"].fillna(0)) / den),
    "median last 3/den": lambda t, den: np.minimum(1, t["minutes_med3"].fillna(t["minutes"]) / den),
    "last match only/den": lambda t, den: np.minimum(1, t["minutes_last"].fillna(t["minutes"]) / den),
    "no discount": lambda t, den: np.ones(len(t)),
}
results = {k: [] for k in RULES}
for cutoff in CUTOFFS:
    train_all = pd.concat([prior, cur[cur.gw <= cutoff]], ignore_index=True); test_all = cur[(cur.gw > cutoff) & (cur.gw <= cutoff + TEST_WINDOW)]
    raw_by_pos = {}
    for pos in POSITION_MAP.values():
        tr = train_all[train_all.position == pos]; te = test_all[test_all.position == pos]
        params = {k: v for k, v in get_position_params(pos).items() if k not in NON_LGB_PARAM_KEYS}
        m = lgb.LGBMRegressor(objective="regression", random_state=42, verbosity=-1, **params)
        m.fit(tr[FEATURES], tr["total_points"], sample_weight=compute_sample_weights(tr["date"], 100000, as_of=pd.Timestamp(tr["date"].max(), tz="UTC")))
        raw_by_pos[pos] = (te, m.predict(te[FEATURES]))
    for name, rule in RULES.items():
        frames = []
        for pos, (te, raw) in raw_by_pos.items():
            mult = np.asarray(rule(te, get_playing_time_denominator(pos)), dtype=float)
            frames.append(pd.DataFrame({"gw": te.gw.to_numpy(), "position": pos, "pred": np.clip(raw * mult, 0, 15), "actual": te.total_points.to_numpy()}))
        results[name].append(pd.concat(frames, ignore_index=True))
print("Deployment replay, 9 cutoffs x 5 GWs. Test rows are players who PLAYED (archive filter), so this measures ranking among players who did play:")
for name, frames in results.items():
    cross = [cross_position_metrics([f]) for f in frames]
    pooled = pd.concat([f.assign(gw=f.gw + 1000 * i) for i, f in enumerate(frames)], ignore_index=True); m = ranking_metrics(pooled)
    print(f"  [{name:<26}] cross-pos pool Spearman {np.mean([c['spearman_pool'] or 0 for c in cross]):.3f}  top-10 actual {np.mean([c['top10_actual_mean'] for c in cross]):.2f}  pool bias {m['pool_bias']:+.2f}  MAE {m['mae']:.3f}")
