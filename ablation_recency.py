"""Isolates the effect of recency weighting specifically, since backtest_model.py's headline
numbers combine it with the tuned hyperparameters and can't tell you which change did what.

Same holdout split as backtest_model.py, same per-position hyperparameters (POSITION_MODEL_PARAMS)
for both conditions -- the only thing that changes is whether sample_weight is passed to .fit().
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
    get_half_life_days,
    get_playing_time_denominator,
    get_understat_player_stats,
    get_understat_team_matches,
    load_prior_season_archive,
    make_model,
    match_understat_players,
)

HOLDOUT_GAMEWEEKS = 5


def mae_for(train, test, position, weighted):
    model = make_model(position)
    weights = None
    if weighted:
        weights = compute_sample_weights(
            train["date"], half_life_days=get_half_life_days(position), as_of=pd.Timestamp(train["date"].max(), tz="UTC")
        )
    model.fit(train[POSITION_FEATURES], train["total_points"], sample_weight=weights)
    predictions = model.predict(test[POSITION_FEATURES])
    multiplier = np.minimum(1.0, test["minutes"].to_numpy() / get_playing_time_denominator(position))
    predictions = np.clip(predictions * multiplier, 0, 15)
    return float(np.mean(np.abs(predictions - test["total_points"].to_numpy())))


def main():
    understat = UnderstatClient()
    team_matches = get_understat_team_matches(understat, PRIOR_SEASON_UNDERSTAT)
    team_matches = add_prematch_team_form(team_matches)
    team_form_lookup = build_team_form_lookup(team_matches)

    merged_gw, difficulty_lookup, team_names, player_idlist = load_prior_season_archive()
    elements = build_backtest_elements(merged_gw, player_idlist)

    understat_players = get_understat_player_stats(understat, PRIOR_SEASON_UNDERSTAT)
    understat_matches, _ = match_understat_players(elements, understat_players)
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

    print(f"Isolating recency weighting only -- same tuned hyperparameters, same GW{split_point}/{max_gw} split, both runs.\n")
    print(f"{'Position':<10}{'Unweighted MAE':<18}{'Weighted MAE':<16}{'Change':<10}")
    totals = {"unweighted": [], "weighted": []}
    for position in POSITION_MAP.values():
        train = train_all[train_all["position"] == position]
        test = test_all[test_all["position"] == position]
        if len(train) < 30 or len(test) < 5:
            print(f"{position:<10}skipped, not enough rows")
            continue
        unweighted_mae = mae_for(train, test, position, weighted=False)
        weighted_mae = mae_for(train, test, position, weighted=True)
        change_pct = 100 * (weighted_mae - unweighted_mae) / unweighted_mae
        direction = "better" if change_pct < 0 else "worse" if change_pct > 0 else "no change"
        totals["unweighted"].append(unweighted_mae)
        totals["weighted"].append(weighted_mae)
        print(f"{position:<10}{unweighted_mae:<18.3f}{weighted_mae:<16.3f}{change_pct:+.1f}% ({direction})")

    overall_unweighted = np.mean(totals["unweighted"])
    overall_weighted = np.mean(totals["weighted"])
    overall_change = 100 * (overall_weighted - overall_unweighted) / overall_unweighted
    print(f"\n{'Overall':<10}{overall_unweighted:<18.3f}{overall_weighted:<16.3f}{overall_change:+.1f}%")


if __name__ == "__main__":
    main()
