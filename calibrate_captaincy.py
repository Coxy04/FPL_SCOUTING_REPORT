"""Calibrates the captaincy rule in fetch_my_team.py: how far must the model's captain pick beat
the most-owned player before following the model is actually worth it?

Replays last season week by week with the deployment setup (train on 2024-25 plus 2025-26 GW1..N,
predict GW N+1). Each week three captains are compared on REAL points: the model's argmax, the
most-owned player that week (the archive's `selected` column, i.e. the template captain), and the
rule "take the model's pick only if it beats the template by at least M points". The margin whose
rule scores most over the season is what CAPTAIN_TEMPLATE_MARGIN should be.

Motivation: over GW2-5 this season the model's top pick earned 23 captain points, the most-owned
player (Haaland, every week) 37. A ranking signal with Spearman ~0.1 in the startable pool should
only override the crowd when it disagrees by more than its own noise -- this measures how much.

Caveat: one season of decisions (~33 weeks) is a small sample for a captain-points total, so read
the curve's shape (where it flattens) rather than trusting one margin to the decimal.
"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from understatapi import UnderstatClient

from backtest_model import build_archive_rows
from fpl_ml_model import (
    NON_LGB_PARAM_KEYS,
    POSITION_FEATURES,
    POSITION_MAP,
    compute_sample_weights,
    ensure_utf8_stdout,
    get_half_life_days,
    get_playing_time_denominator,
    get_position_params,
)

OUTPUT_FILE = "fpl_ml_captaincy_calibration.json"
PRIOR = ("2024-25", "2024")
CURRENT = ("2025-26", "2025")
DROP_FEATURES = ["clearances_blocks_interceptions", "recoveries", "tackles", "defensive_contribution"]
FEATURES = [f for f in POSITION_FEATURES if f not in DROP_FEATURES]
FIRST_DECISION_GW = 4
MARGINS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 99.0]


def predict_week(train_all, test, position):
    train = train_all[train_all["position"] == position]
    params = {k: v for k, v in get_position_params(position).items() if k not in NON_LGB_PARAM_KEYS}
    model = lgb.LGBMRegressor(objective="regression", random_state=42, verbosity=-1, **params)
    weights = compute_sample_weights(train["date"], half_life_days=get_half_life_days(position), as_of=pd.Timestamp(train["date"].max(), tz="UTC"))
    model.fit(train[FEATURES], train["total_points"], sample_weight=weights)
    pred = model.predict(test[FEATURES]) * np.minimum(1.0, test["minutes"].to_numpy() / get_playing_time_denominator(position))
    return np.clip(pred, 0, 15)


def main():
    understat = UnderstatClient()
    prior_rows = build_archive_rows(*PRIOR, understat=understat)
    current_rows = build_archive_rows(*CURRENT, understat=understat)
    if "selected" not in current_rows:
        raise RuntimeError("archive has no `selected` column -- cannot build the template captain")
    current_rows["selected"] = pd.to_numeric(current_rows["selected"], errors="coerce").fillna(0)
    max_gw = int(current_rows["gw"].max())

    decisions = []
    for gw in range(FIRST_DECISION_GW, max_gw + 1):
        train_all = pd.concat([prior_rows, current_rows[current_rows["gw"] < gw]], ignore_index=True)
        week = current_rows[current_rows["gw"] == gw].copy()
        if week.empty:
            continue
        week["pred"] = 0.0
        for position in POSITION_MAP.values():
            mask = week["position"] == position
            if mask.any():
                week.loc[mask, "pred"] = predict_week(train_all, week[mask], position)
        # Note: rows only exist for players who actually played (minutes > 0 filter in
        # build_prior_season_rows), so a captain who ended up not playing can't appear here. That
        # flatters every strategy equally and doesn't change the comparison between them.
        model_pick = week.loc[week["pred"].idxmax()]
        template_pick = week.loc[week["selected"].idxmax()]
        best = week.loc[week["total_points"].idxmax()]
        decisions.append({
            "gw": gw,
            "model_pred": float(model_pick["pred"]), "model_actual": int(model_pick["total_points"]),
            "template_pred": float(template_pick["pred"]), "template_actual": int(template_pick["total_points"]),
            "same_player": bool(model_pick["current_id"] == template_pick["current_id"]),
            "edge": float(model_pick["pred"] - template_pick["pred"]),
            "best_actual": int(best["total_points"]),
        })
        print(f"GW{gw}: model {model_pick['pred']:.2f} -> {int(model_pick['total_points'])} pts | "
              f"template {template_pick['pred']:.2f} -> {int(template_pick['total_points'])} pts | "
              f"edge {model_pick['pred'] - template_pick['pred']:+.2f} | best {int(best['total_points'])}")

    frame = pd.DataFrame(decisions)
    curve = []
    print(f"\n{len(frame)} decisions. Captain points over the season by strategy:")
    print(f"  always model:    {frame['model_actual'].sum()}")
    print(f"  always template: {frame['template_actual'].sum()}")
    print(f"  hindsight best:  {frame['best_actual'].sum()}")
    for margin in MARGINS:
        follow_model = frame["edge"] >= margin
        pts = int(np.where(follow_model, frame["model_actual"], frame["template_actual"]).sum())
        curve.append({"margin": margin, "captain_points": pts, "weeks_following_model": int(follow_model.sum())})
        print(f"  rule, margin {margin:>5.2f}: {pts} pts  (followed the model in {int(follow_model.sum())} of {len(frame)} weeks)")

    best_margin = max(curve, key=lambda c: c["captain_points"])
    print(f"\nBest margin on this replay: {best_margin['margin']} ({best_margin['captain_points']} pts). "
          f"Read the whole curve before adopting it -- see module docstring.")
    with open(OUTPUT_FILE, "w") as f:
        json.dump({"decisions": decisions, "curve": curve, "best_margin": best_margin}, f, indent=2)
    print(f"Saved {OUTPUT_FILE}")


if __name__ == "__main__":
    ensure_utf8_stdout()
    main()
