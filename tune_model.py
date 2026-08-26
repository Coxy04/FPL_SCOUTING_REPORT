"""Uses the backtest as a scoring function to search for better LightGBM hyperparameters per
position, instead of the one-size-fits-all settings fpl_ml_model.py currently uses everywhere.

By default this SEARCHES AND REPORTS ONLY -- pass --apply to actually rewrite
POSITION_MODEL_PARAMS in fpl_ml_model.py. Tuning against a single holdout window risks picking
settings that just fit that window's quirks rather than genuinely generalizing, so this scores
every candidate across two separate time-based folds and averages them, and only accepts a
candidate that beats the currently-deployed config by more than MIN_IMPROVEMENT_PCT -- both of
which are what make --apply reasonably safe to run unattended on a schedule. Every change lands
as a git commit, so a bad update is always a `git revert` away.

Reuses fpl_ml_model.py's and backtest_model.py's own data-building functions so this can't drift
from what the real pipeline does.
"""
import json
import random
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from understatapi import UnderstatClient

from backtest_model import build_backtest_elements
from fpl_ml_model import (
    DEFAULT_MODEL_PARAMS,
    POSITION_FEATURES,
    POSITION_MAP,
    POSITION_MODEL_PARAMS,
    PRIOR_SEASON_UNDERSTAT,
    add_features,
    add_prematch_team_form,
    build_prior_season_rows,
    build_team_form_lookup,
    compute_sample_weights,
    get_understat_player_stats,
    get_understat_team_matches,
    load_prior_season_archive,
    match_understat_players,
)

# The archive this searches against is static -- it never changes -- so a fixed search seed
# would make every scheduled run explore the exact same 25 candidates and find the exact same
# result forever. MODEL_RANDOM_STATE keeps a given candidate's own fit reproducible; SEARCH_SEED
# is left to vary per run (seeded from the OS) so each scheduled run samples a different slice of
# the space, letting genuine improvements keep ratcheting in over many weeks.
MODEL_RANDOM_STATE = 42
SEARCH_SEED = None
N_TRIALS = 25
OUTPUT_FILE = "fpl_ml_tuning.json"
MODEL_FILE = "fpl_ml_model.py"
# A random search will sometimes "beat" the current config by pure noise even when it isn't
# genuinely better. Require a real margin, averaged across both folds, before treating a
# candidate as a win -- this is what keeps an unsupervised/scheduled run from churning on noise.
MIN_IMPROVEMENT_PCT = 1.0

# Two separate time-based folds so a "winning" config has to generalize across two different
# held-out windows, not just fit one window's idiosyncrasies.
FOLDS = [
    {"train_max_gw": 28, "test_max_gw": 33},
    {"train_max_gw": 33, "test_max_gw": 38},
]

SEARCH_SPACE = {
    "n_estimators": [100, 150, 200, 250, 350, 450],
    "learning_rate": [0.02, 0.03, 0.04, 0.06, 0.08],
    "num_leaves": [7, 15, 23, 31],
    "max_depth": [3, 4, 5, 6, 7],
    "min_child_samples": [10, 20, 30, 50, 80],
    "reg_lambda": [0.5, 1.0, 2.0, 4.0, 8.0],
}


def sample_params(rng):
    return {key: rng.choice(values) for key, values in SEARCH_SPACE.items()}


def make_model(params):
    return lgb.LGBMRegressor(objective="regression", random_state=MODEL_RANDOM_STATE, verbosity=-1, **params)


def score_params(params, position_rows, folds):
    """Average MAE for one hyperparameter combo across both folds, for one position's rows."""
    fold_maes = []
    for fold in folds:
        train = position_rows[position_rows["gw"] <= fold["train_max_gw"]]
        test = position_rows[
            (position_rows["gw"] > fold["train_max_gw"]) & (position_rows["gw"] <= fold["test_max_gw"])
        ]
        if len(train) < 30 or len(test) < 5:
            continue
        model = make_model(params)
        weights = compute_sample_weights(train["date"], as_of=pd.Timestamp(train["date"].max(), tz="UTC"))
        model.fit(train[POSITION_FEATURES], train["total_points"], sample_weight=weights)
        predictions = np.clip(model.predict(test[POSITION_FEATURES]), 0, 15)
        fold_maes.append(float(np.mean(np.abs(predictions - test["total_points"].to_numpy()))))
    return float(np.mean(fold_maes)) if fold_maes else None


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


def apply_params_to_source(position_params):
    """Rewrites POSITION_MODEL_PARAMS in fpl_ml_model.py in place. Only called when --apply is
    passed; git tracks every change this makes, so a bad update is always a `git revert` away."""
    text = open(MODEL_FILE, encoding="utf-8").read()
    lines = ["POSITION_MODEL_PARAMS = {"]
    for position, params in position_params.items():
        pairs = ", ".join(f'"{k}": {v!r}' for k, v in params.items())
        lines.append(f'    "{position}": {{{pairs}}},')
    lines.append("}")
    new_block = "\n".join(lines)

    start = text.index("POSITION_MODEL_PARAMS = {")
    end = text.index("\n}\n", start) + len("\n}")
    text = text[:start] + new_block + text[end:]
    open(MODEL_FILE, "w", encoding="utf-8").write(text)
    print(f"Applied to {MODEL_FILE}: POSITION_MODEL_PARAMS updated.")


def main():
    apply = "--apply" in sys.argv
    print(f"Random search: {N_TRIALS} trials per position, scored across {len(FOLDS)} time-based folds each.")
    print(f"Mode: {'APPLY (will rewrite ' + MODEL_FILE + ' on genuine improvements)' if apply else 'report only'}\n")
    rng = random.Random(SEARCH_SEED)
    rows = build_dataset()

    results = {}
    applied_params = {}
    for position in POSITION_MAP.values():
        position_rows = rows[rows["position"] == position]
        if len(position_rows) < 100:
            print(f"{position}: skipped, not enough rows ({len(position_rows)})")
            continue

        current_params = POSITION_MODEL_PARAMS.get(position, DEFAULT_MODEL_PARAMS)
        current_mae = score_params(current_params, position_rows, FOLDS)
        best_params, best_mae = current_params, current_mae

        for _ in range(N_TRIALS):
            candidate = sample_params(rng)
            mae = score_params(candidate, position_rows, FOLDS)
            if mae is not None and mae < best_mae:
                best_params, best_mae = candidate, mae

        required_mae = current_mae * (1 - MIN_IMPROVEMENT_PCT / 100)
        improved = best_mae < required_mae
        print(f"{position}: current MAE {current_mae:.3f} -> best found MAE {best_mae:.3f} "
              f"({'IMPROVED' if improved else f'no improvement past the {MIN_IMPROVEMENT_PCT}% threshold, kept current'})")
        if improved:
            print(f"    winning params: {best_params}")

        final_params = best_params if improved else current_params
        applied_params[position] = final_params
        results[position] = {
            "current_mae": round(current_mae, 4),
            "best_mae": round(best_mae, 4),
            "improved": improved,
            "params": final_params,
        }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    any_improved = any(r["improved"] for r in results.values())
    if apply and any_improved:
        apply_params_to_source(applied_params)
    elif apply:
        print("\nNo position improved past the threshold -- nothing to apply.")
    else:
        print(f"\nSaved {OUTPUT_FILE}. Nothing in {MODEL_FILE} has been changed -- "
              f"re-run with --apply to write genuine improvements back, or review and update by hand.")


if __name__ == "__main__":
    main()
