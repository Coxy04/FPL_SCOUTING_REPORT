"""Backtests the prediction methodology against real historical results.

fpl_ml_model.py never validates its own accuracy -- it just trains on everything available and
predicts forward, so there's no way to know from a normal run whether the predictions are any
good. This script holds out the last few gameweeks of last season's archive, trains each position
model on everything before that, and checks how close the predictions land to what actually
happened -- against a naive baseline (each position's average score) so the numbers mean something.

Reuses fpl_ml_model.py's own feature-engineering functions rather than reimplementing them, so this
can't silently drift from what the real pipeline does.
"""
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from understatapi import UnderstatClient

from fpl_ml_model import (
    POSITION_FEATURES,
    POSITION_MAP,
    PRIOR_SEASON_UNDERSTAT,
    add_features,
    add_prematch_team_form,
    build_prior_season_rows,
    build_team_form_lookup,
    compute_sample_weights,
    get_understat_player_stats,
    get_understat_team_matches,
    load_prior_season_archive,
    make_model,
    match_understat_players,
)

HOLDOUT_GAMEWEEKS = 5
OUTPUT_FILE = "fpl_ml_backtest.json"
HISTORY_FILE = "fpl_ml_accuracy_history.jsonl"


def build_backtest_elements(merged_gw, player_idlist):
    name_by_id = {row.id: (row.first_name, row.second_name) for row in player_idlist.itertuples()}
    ordered = merged_gw.sort_values("GW")
    totals = ordered.groupby("element").agg(team_name=("team", "last"), minutes=("minutes", "sum")).reset_index()

    elements = []
    for row in totals.itertuples():
        first, second = name_by_id.get(row.element, ("", ""))
        elements.append(
            {
                "id": row.element,
                "first_name": first,
                "second_name": second,
                "web_name": second or first,
                "team_name": row.team_name,
                "minutes": row.minutes,
            }
        )
    return elements


def evaluate_position(train, test, position):
    model = make_model(position)
    weights = compute_sample_weights(train["date"], as_of=pd.Timestamp(train["date"].max(), tz="UTC"))
    model.fit(train[POSITION_FEATURES], train["total_points"], sample_weight=weights)
    predictions = np.clip(model.predict(test[POSITION_FEATURES]), 0, 15)
    actual = test["total_points"].to_numpy()

    model_mae = float(np.mean(np.abs(predictions - actual)))
    baseline_pred = train["total_points"].mean()
    baseline_mae = float(np.mean(np.abs(baseline_pred - actual)))
    correlation = float(np.corrcoef(predictions, actual)[0, 1]) if len(test) > 1 else None

    return {
        "position": position,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "model_mae": round(model_mae, 3),
        "baseline_mae": round(baseline_mae, 3),
        "improvement_vs_baseline_pct": round(100 * (1 - model_mae / baseline_mae), 1) if baseline_mae > 0 else None,
        "correlation": round(correlation, 3) if correlation is not None else None,
    }


def main():
    print(f"Backtesting against {PRIOR_SEASON_UNDERSTAT} archive: train on all but the last "
          f"{HOLDOUT_GAMEWEEKS} gameweeks, test on those held-out gameweeks.")

    understat = UnderstatClient()
    team_matches = get_understat_team_matches(understat, PRIOR_SEASON_UNDERSTAT)
    team_matches = add_prematch_team_form(team_matches)
    team_form_lookup = build_team_form_lookup(team_matches)

    merged_gw, difficulty_lookup, team_names, player_idlist = load_prior_season_archive()
    elements = build_backtest_elements(merged_gw, player_idlist)

    understat_players = get_understat_player_stats(understat, PRIOR_SEASON_UNDERSTAT)
    understat_matches, unmatched = match_understat_players(elements, understat_players)
    print(f"Matched {len(understat_matches)} of {len(elements)} archive players to Understat.")

    prior_per90_by_understat_id = {p["understat_id"]: p for p in understat_players}
    id_map = {e: e for e in merged_gw["element"].unique()}

    rows = build_prior_season_rows(
        merged_gw, difficulty_lookup, team_names, id_map, team_form_lookup, understat_matches, prior_per90_by_understat_id
    )
    rows = add_features(rows)

    max_gw = rows["gw"].max()
    split_point = max_gw - HOLDOUT_GAMEWEEKS
    train_all = rows[rows["gw"] <= split_point]
    test_all = rows[rows["gw"] > split_point]
    print(f"Split at GW{split_point}: {len(train_all)} train rows (GW1-{split_point}), "
          f"{len(test_all)} test rows (GW{split_point + 1}-{max_gw})\n")

    results = []
    for position in POSITION_MAP.values():
        train = train_all[train_all["position"] == position]
        test = test_all[test_all["position"] == position]
        if len(train) < 30 or len(test) < 5:
            print(f"{position}: skipped, not enough rows (train={len(train)}, test={len(test)})")
            continue
        result = evaluate_position(train, test, position)
        results.append(result)
        beat_baseline = "beats" if result["model_mae"] < result["baseline_mae"] else "WORSE than"
        print(
            f"{position:>3}  model MAE {result['model_mae']:.2f}  vs baseline MAE {result['baseline_mae']:.2f}  "
            f"({beat_baseline} baseline, {result['improvement_vs_baseline_pct']}% improvement)  "
            f"correlation {result['correlation']}  [{result['train_rows']} train / {result['test_rows']} test rows]"
        )

    overall_model_mae = float(np.mean([r["model_mae"] for r in results])) if results else None
    overall_baseline_mae = float(np.mean([r["baseline_mae"] for r in results])) if results else None
    print(f"\nOverall (mean across positions): model MAE {overall_model_mae:.3f} vs baseline {overall_baseline_mae:.3f}")

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "holdout_gameweeks": HOLDOUT_GAMEWEEKS,
        "split_gw": int(split_point),
        "max_gw": int(max_gw),
        "by_position": results,
        "overall_model_mae": round(overall_model_mae, 3) if overall_model_mae is not None else None,
        "overall_baseline_mae": round(overall_baseline_mae, 3) if overall_baseline_mae is not None else None,
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    with open(HISTORY_FILE, "a") as f:
        f.write(json.dumps(output) + "\n")
    print(f"Saved {OUTPUT_FILE} (and appended to {HISTORY_FILE})")


if __name__ == "__main__":
    main()
