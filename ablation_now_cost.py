"""Tests whether now_cost is pulling real independent weight as a model input, or whether it's
mostly proxying for signals already captured elsewhere (expected_goals, npxg90, set-piece duty,
etc.) -- and specifically whether including it works AGAINST the Hidden Gems thesis. That feature
is built entirely on finding players whose underlying process is better than their price/ownership
suggests; if the model itself leans on price as a shortcut, it's structurally less likely to rate a
genuinely underpriced player as highly as their process deserves, which cuts against the whole
point of the tool.

Two things get checked, not just headline MAE:
1. Does removing now_cost hurt accuracy? If barely, it was mostly redundant with other features.
2. Does removing it shift predictions for cheap players differently than expensive ones? A
   positive shift concentrated in the cheapest price band would confirm the suppression theory
   directly, rather than leaving it as speculation.

Same held-out split and per-position tuned hyperparameters as backtest_model.py, for a fair,
isolated A/B like ablation_recency.py and ablation_playing_time.py.
"""
import numpy as np
import pandas as pd
from understatapi import UnderstatClient

from backtest_model import build_backtest_elements
from fpl_ml_model import (
    POSITION_FEATURES,
    POSITION_MAP,
    PRIOR_SEASON_UNDERSTAT,
    add_features,
    add_prematch_team_form,
    build_prior_season_rows,
    build_team_form_lookup,
    compute_sample_weights,
    ensure_utf8_stdout,
    get_half_life_days,
    get_playing_time_denominator,
    get_understat_player_stats,
    get_understat_team_matches,
    load_prior_season_archive,
    make_model,
    match_understat_players,
)

HOLDOUT_GAMEWEEKS = 5
WITHOUT_NOW_COST = [f for f in POSITION_FEATURES if f != "now_cost"]


def predict_for(train, test, position, features):
    model = make_model(position)
    weights = compute_sample_weights(
        train["date"], half_life_days=get_half_life_days(position), as_of=pd.Timestamp(train["date"].max(), tz="UTC")
    )
    model.fit(train[features], train["total_points"], sample_weight=weights)
    predictions = model.predict(test[features])
    multiplier = np.minimum(1.0, test["minutes"].to_numpy() / get_playing_time_denominator(position))
    return np.clip(predictions * multiplier, 0, 15)


def mae(predictions, actual):
    return float(np.mean(np.abs(predictions - actual)))


def main():
    understat = UnderstatClient()
    team_matches = get_understat_team_matches(understat, PRIOR_SEASON_UNDERSTAT)
    team_matches = add_prematch_team_form(team_matches)
    team_form_lookup = build_team_form_lookup(team_matches)

    merged_gw, difficulty_lookup, team_names, player_idlist, rest_days_lookup, prior_set_piece_lookup = load_prior_season_archive()
    elements = build_backtest_elements(merged_gw, player_idlist)

    understat_players = get_understat_player_stats(understat, PRIOR_SEASON_UNDERSTAT)
    understat_matches, _ = match_understat_players(elements, understat_players)
    prior_per90_by_understat_id = {p["understat_id"]: p for p in understat_players}
    id_map = {e: e for e in merged_gw["element"].unique()}

    rows = build_prior_season_rows(
        merged_gw, difficulty_lookup, team_names, id_map, team_form_lookup, understat_matches, prior_per90_by_understat_id, rest_days_lookup, prior_set_piece_lookup
    )
    rows = add_features(rows)

    max_gw = rows["gw"].max()
    split_point = max_gw - HOLDOUT_GAMEWEEKS
    train_all = rows[rows["gw"] <= split_point]
    test_all = rows[rows["gw"] > split_point]

    print(f"Isolating now_cost only -- same tuned hyperparameters, same GW{split_point}/{max_gw} split, both runs.\n")
    print(f"{'Position':<10}{'With now_cost':<16}{'Without':<12}{'Change':<10}")
    totals = {"with": [], "without": []}
    price_shift_rows = []

    for position in POSITION_MAP.values():
        train = train_all[train_all["position"] == position]
        test = test_all[test_all["position"] == position]
        if len(train) < 30 or len(test) < 5:
            print(f"{position:<10}skipped, not enough rows")
            continue

        actual = test["total_points"].to_numpy()
        with_pred = predict_for(train, test, position, POSITION_FEATURES)
        without_pred = predict_for(train, test, position, WITHOUT_NOW_COST)

        with_mae = mae(with_pred, actual)
        without_mae = mae(without_pred, actual)
        change_pct = 100 * (without_mae - with_mae) / with_mae
        direction = "better without" if change_pct < 0 else "worse without" if change_pct > 0 else "no change"
        totals["with"].append(with_mae)
        totals["without"].append(without_mae)
        print(f"{position:<10}{with_mae:<16.3f}{without_mae:<12.3f}{change_pct:+.1f}% ({direction})")

        position_test = test.copy()
        position_test["shift"] = without_pred - with_pred
        price_shift_rows.append(position_test[["now_cost", "shift"]])

    overall_with = np.mean(totals["with"])
    overall_without = np.mean(totals["without"])
    overall_change = 100 * (overall_without - overall_with) / overall_with
    print(f"\n{'Overall':<10}{overall_with:<16.3f}{overall_without:<12.3f}{overall_change:+.1f}%")

    print("\nDoes removing now_cost shift predictions differently for cheap vs. expensive players?")
    print("(positive shift = prediction goes UP once now_cost is removed)")
    all_shifts = pd.concat(price_shift_rows, ignore_index=True)
    all_shifts["price_band"] = pd.qcut(all_shifts["now_cost"], 4, labels=["cheapest 25%", "2nd", "3rd", "priciest 25%"])
    band_summary = all_shifts.groupby("price_band", observed=True)["shift"].mean()
    for band, value in band_summary.items():
        print(f"  {band:<15} mean shift {value:+.3f} pts")


if __name__ == "__main__":
    ensure_utf8_stdout()
    main()
