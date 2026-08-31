"""Tests whether a dedicated per-player model beats the pooled per-position model, for the
players with the MOST available data -- the best possible case for the per-player approach. If it
loses even here, that's strong evidence the data-volume problem (a season is ~30 usable rows,
versus 700-4,600 for the smallest/largest position models) is decisive rather than theoretical.

Picks the top 3 players by row count in each position from last season's archive, trains a
dedicated LightGBM model per player (conservative hyperparameters -- fair to the hypothesis,
since the position models' tuned settings assume thousands of rows, not ~30) on GW1-33, and
compares its holdout MAE (GW34-38) against the pooled position model's MAE on that same player's
rows. Same split as backtest_model.py, for a fair comparison.

Reuses fpl_ml_model.py's own data-building functions so this can't drift from the real pipeline.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
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
    get_understat_player_stats,
    get_understat_team_matches,
    load_prior_season_archive,
    make_model,
    match_understat_players,
)

HOLDOUT_GAMEWEEKS = 5
PLAYERS_PER_POSITION = 3
MIN_TEST_ROWS = 2

# Deliberately conservative -- appropriate for ~25-30 training rows, not the thousands the tuned
# position models assume. Giving the per-player approach a fair shot, not sabotaging it.
INDIVIDUAL_MODEL_PARAMS = {
    "n_estimators": 60,
    "learning_rate": 0.08,
    "num_leaves": 4,
    "max_depth": 2,
    "min_child_samples": 3,
    "reg_lambda": 2.0,
}


def build_dataset():
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
    name_by_id = {e["id"]: e["web_name"] for e in elements}
    rows["web_name"] = rows["current_id"].map(name_by_id)
    return rows


def mae(predictions, actual):
    return float(np.mean(np.abs(np.clip(predictions, 0, 15) - actual)))


def main():
    rows = build_dataset()
    max_gw = rows["gw"].max()
    split_point = max_gw - HOLDOUT_GAMEWEEKS

    print(f"Per-player vs. pooled-position model, GW{split_point + 1}-{max_gw} holdout, "
          f"top {PLAYERS_PER_POSITION} players by row count per position.\n")

    individual_maes, pooled_maes = [], []

    for position in POSITION_MAP.values():
        position_rows = rows[rows["position"] == position]
        pos_train_all = position_rows[position_rows["gw"] <= split_point]
        pooled_model = make_model(position)
        weights = compute_sample_weights(
            pos_train_all["date"], half_life_days=get_half_life_days(position),
            as_of=pd.Timestamp(pos_train_all["date"].max(), tz="UTC"),
        )
        pooled_model.fit(pos_train_all[POSITION_FEATURES], pos_train_all["total_points"], sample_weight=weights)

        counts = position_rows["current_id"].value_counts()
        candidates = counts.head(PLAYERS_PER_POSITION + 3).index  # a few spares in case of thin test windows

        picked = 0
        for player_id in candidates:
            if picked >= PLAYERS_PER_POSITION:
                break
            player_rows = position_rows[position_rows["current_id"] == player_id].sort_values("gw")
            train = player_rows[player_rows["gw"] <= split_point]
            test = player_rows[player_rows["gw"] > split_point]
            if len(test) < MIN_TEST_ROWS or len(train) < 10:
                continue
            picked += 1

            individual_model = lgb.LGBMRegressor(objective="regression", random_state=42, verbosity=-1, **INDIVIDUAL_MODEL_PARAMS)
            individual_model.fit(train[POSITION_FEATURES], train["total_points"])
            individual_pred = individual_model.predict(test[POSITION_FEATURES])
            pooled_pred = pooled_model.predict(test[POSITION_FEATURES])
            actual = test["total_points"].to_numpy()

            ind_mae = mae(individual_pred, actual)
            pool_mae = mae(pooled_pred, actual)
            individual_maes.append(ind_mae)
            pooled_maes.append(pool_mae)

            winner = "individual" if ind_mae < pool_mae else "pooled"
            name = train["web_name"].iloc[0]
            print(f"{position:<4} {name:<16} (train={len(train):>3} test={len(test)})  "
                  f"individual MAE {ind_mae:.2f}  pooled MAE {pool_mae:.2f}  -> {winner} wins")

    print(f"\nOverall: individual mean MAE {np.mean(individual_maes):.3f}  "
          f"pooled mean MAE {np.mean(pooled_maes):.3f}  "
          f"({sum(1 for i, p in zip(individual_maes, pooled_maes) if i < p)}/{len(individual_maes)} "
          f"players where individual model won)")


if __name__ == "__main__":
    ensure_utf8_stdout()
    main()
