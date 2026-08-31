"""Tests whether fpl_ml_model.py's playing_time_multiplier (min(1, minutes/60)) actually helps,
and whether 60 is a good denominator, rather than trusting a hand-picked shape.

Only tests the graduated "partial recent minutes" case: players who DID play in the held-out
test match (a genuinely unused player never appears in backtest data at all -- there's no
outcome to check a prediction against for a match where nothing happened, so the "should this be
~0" case isn't a prediction question, it's a rule of the game). What's actually testable is
whether their PRIOR recent-minutes trend (not this match's actual outcome, which isn't known at
prediction time) should discount the prediction, and by how much.

Same held-out split and per-position tuned models as backtest_model.py.
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
    get_understat_player_stats,
    get_understat_team_matches,
    load_prior_season_archive,
    make_model,
    match_understat_players,
)

HOLDOUT_GAMEWEEKS = 5
CANDIDATE_DENOMINATORS = [30, 45, 60, 75, 90]  # multiplier = min(1, recent_minutes / D)


def mae_with_multiplier(predictions, minutes, actual, denominator):
    adjusted = predictions if denominator is None else predictions * np.minimum(1.0, minutes / denominator)
    return float(np.mean(np.abs(np.clip(adjusted, 0, 15) - actual)))


def build_dataset():
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
    return add_features(rows)


def main():
    rows = build_dataset()
    max_gw = rows["gw"].max()
    split_point = max_gw - HOLDOUT_GAMEWEEKS
    train_all = rows[rows["gw"] <= split_point]
    test_all = rows[rows["gw"] > split_point]

    print(f"Testing playing-time multiplier shapes on GW{split_point + 1}-{max_gw} holdout "
          f"(players who actually played that match).\n")
    header = f"{'Position':<10}{'No mult.':<10}" + "".join(f"D={d:<8}" for d in CANDIDATE_DENOMINATORS)
    print(header)

    overall = {"none": []}
    for d in CANDIDATE_DENOMINATORS:
        overall[d] = []

    for position in POSITION_MAP.values():
        train = train_all[train_all["position"] == position]
        test = test_all[test_all["position"] == position]
        if len(train) < 30 or len(test) < 5:
            print(f"{position}: skipped, not enough rows")
            continue

        model = make_model(position)
        weights = compute_sample_weights(
            train["date"], half_life_days=get_half_life_days(position), as_of=pd.Timestamp(train["date"].max(), tz="UTC")
        )
        model.fit(train[POSITION_FEATURES], train["total_points"], sample_weight=weights)
        predictions = model.predict(test[POSITION_FEATURES])
        actual = test["total_points"].to_numpy()
        minutes = test["minutes"].to_numpy()  # pre-match rolling average, same as production

        none_mae = mae_with_multiplier(predictions, minutes, actual, None)
        overall["none"].append(none_mae)
        line = f"{position:<10}{none_mae:<10.3f}"
        for d in CANDIDATE_DENOMINATORS:
            mae = mae_with_multiplier(predictions, minutes, actual, d)
            overall[d].append(mae)
            line += f"{mae:<10.3f}"
        print(line)

    print()
    summary = f"{'Overall':<10}{np.mean(overall['none']):<10.3f}"
    for d in CANDIDATE_DENOMINATORS:
        summary += f"{np.mean(overall[d]):<10.3f}"
    print(summary)

    best_key = min(overall, key=lambda k: np.mean(overall[k]))
    best_label = "no multiplier" if best_key == "none" else f"D={best_key}"
    print(f"\nBest overall: {best_label} (current production uses D=60)")


if __name__ == "__main__":
    ensure_utf8_stdout()
    main()
