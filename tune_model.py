"""Uses the backtest as a scoring function to search for better LightGBM hyperparameters per
position, instead of the one-size-fits-all settings fpl_ml_model.py would otherwise use.

By default this SEARCHES AND REPORTS ONLY -- pass --apply to actually rewrite
POSITION_MODEL_PARAMS in fpl_ml_model.py. Tuning against a single holdout window risks picking
settings that just fit that window's quirks rather than genuinely generalizing, so this scores
every candidate across two separate time-based folds and averages them, and only accepts a
candidate that beats the currently-deployed config by more than MIN_IMPROVEMENT_PCT -- both of
which are what make --apply reasonably safe to run unattended. Every change lands as a git commit,
so a bad update is always a `git revert` away.

WHAT IT SCORES ON changed after the GW1-6 review. It used to minimise MAE, and on a target where
63% of outcomes are 2 or less, MAE is minimised by predicting near the median for everyone. That
criterion had chosen a FWD model of 100 trees at learning rate 0.02 that could not predict above
4.92 for any forward in ten held-out gameweeks -- so Haaland could never be the captain pick -- and
rejected the configuration that fixed it because its MAE was 0.04 worse. The score is now
backtest_model.selection_loss: rank correlation in the startable pool plus a calibration penalty.
MAE is still printed so the trade-off is visible, but it decides nothing.

Reuses fpl_ml_model.py's and backtest_model.py's own data-building functions so this can't drift
from what the real pipeline does.
"""
import json
import random
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from backtest_model import build_archive_rows, ranking_metrics, selection_loss
from fpl_ml_model import (
    DEFAULT_MODEL_PARAMS,
    NON_LGB_PARAM_KEYS,
    POSITION_FEATURES,
    POSITION_MAP,
    POSITION_MODEL_PARAMS,
    compute_sample_weights,
    ensure_utf8_stdout,
)

# The archive this searches against is static, so the search is only as reproducible as its seed.
# An OS-seeded search re-run on identical data can "find" a different winner and reshuffle the
# whole ranking for no reason -- one of the churn sources in SEASON_REVIEW_GW1-6.md. The seed is
# now the calendar month: a re-run within the month reproduces exactly, a new month explores a new
# slice of the space. refresh_dashboard.ps1 only runs this every 28 days, so more trials per run.
from datetime import datetime, timezone

MODEL_RANDOM_STATE = 42
SEARCH_SEED = int(datetime.now(timezone.utc).strftime("%Y%m"))
N_TRIALS = 60
OUTPUT_FILE = "fpl_ml_tuning.json"
MODEL_FILE = "fpl_ml_model.py"
# A random search will sometimes "beat" the current config by pure noise even when it isn't
# genuinely better. Require a real margin, averaged across both folds, before treating a
# candidate as a win -- this is what keeps an unsupervised run from churning on noise.
MIN_IMPROVEMENT_PCT = 1.0

# Two separate time-based folds so a "winning" config has to generalize across two different
# held-out windows, not just fit one window's idiosyncrasies. NOTE these are both WITHIN one
# archived season, so they say nothing about cross-season drift (how much last season should count
# this season) -- that question belongs to backtest_deployment.py, which is why half_life_days is
# no longer searched here.
FOLDS = [
    {"train_max_gw": 28, "test_max_gw": 33},
    {"train_max_gw": 33, "test_max_gw": 38},
]

SEARCH_SPACE = {
    "n_estimators": [100, 150, 200, 250, 350, 450, 600],
    "learning_rate": [0.02, 0.03, 0.04, 0.06, 0.08],
    "num_leaves": [7, 15, 23, 31],
    "max_depth": [3, 4, 5, 6, 7],
    "min_child_samples": [10, 20, 30, 50, 80],
    "reg_lambda": [0.5, 1.0, 2.0, 4.0, 8.0],
    # Discounts predicted_points by min(1, recent_minutes / D). Ablation testing showed a single
    # D=60 for every position was wrong: it helped DEF/FWD but actively hurt GK/MID accuracy, so
    # this is searched per position like everything else here rather than hand-picked once.
    # 100000 is effectively "off" (multiplier ~1.0 for any realistic recent-minutes value).
    "playing_time_denominator": [30, 45, 60, 75, 90, 100000],
}


def sample_params(rng, current):
    params = {key: rng.choice(values) for key, values in SEARCH_SPACE.items()}
    # Recency weighting is decided by backtest_deployment.py, not searched here -- carry the
    # deployed value through unchanged so --apply can't silently reset it.
    params["half_life_days"] = current.get("half_life_days", DEFAULT_MODEL_PARAMS["half_life_days"])
    return params


def make_model(params):
    lgb_params = {k: v for k, v in params.items() if k not in NON_LGB_PARAM_KEYS}
    return lgb.LGBMRegressor(objective="regression", random_state=MODEL_RANDOM_STATE, verbosity=-1, **lgb_params)


def score_params(params, position_rows, folds):
    """Mean selection_loss (lower is better) for one hyperparameter combo across both folds, for
    one position's rows. Also returns the mean MAE purely for reporting."""
    losses = []
    maes = []
    for fold in folds:
        train = position_rows[position_rows["gw"] <= fold["train_max_gw"]]
        test = position_rows[
            (position_rows["gw"] > fold["train_max_gw"]) & (position_rows["gw"] <= fold["test_max_gw"])
        ]
        if len(train) < 30 or len(test) < 5:
            continue
        model = make_model(params)
        half_life = params.get("half_life_days", DEFAULT_MODEL_PARAMS["half_life_days"])
        weights = compute_sample_weights(train["date"], half_life_days=half_life, as_of=pd.Timestamp(train["date"].max(), tz="UTC"))
        model.fit(train[POSITION_FEATURES], train["total_points"], sample_weight=weights)
        predictions = model.predict(test[POSITION_FEATURES])
        denominator = params.get("playing_time_denominator", DEFAULT_MODEL_PARAMS["playing_time_denominator"])
        multiplier = np.minimum(1.0, test["minutes"].to_numpy() / denominator)
        predictions = np.clip(predictions * multiplier, 0, 15)
        frame = pd.DataFrame({"gw": test["gw"].to_numpy(), "pred": predictions, "actual": test["total_points"].to_numpy()})
        metrics = ranking_metrics(frame)
        losses.append(selection_loss(metrics))
        maes.append(metrics["mae"])
    if not losses:
        return None, None
    return float(np.mean(losses)), float(np.mean(maes))


def build_dataset():
    return build_archive_rows()


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
    print(f"Random search: {N_TRIALS} trials per position, scored across {len(FOLDS)} time-based folds each "
          f"on selection loss (1 - pool Spearman + bias penalty), NOT MAE.")
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
        current_loss, current_mae = score_params(current_params, position_rows, FOLDS)
        best_params, best_loss, best_mae = current_params, current_loss, current_mae

        for _ in range(N_TRIALS):
            candidate = sample_params(rng, current_params)
            loss, mae = score_params(candidate, position_rows, FOLDS)
            if loss is not None and loss < best_loss:
                best_params, best_loss, best_mae = candidate, loss, mae

        required_loss = current_loss * (1 - MIN_IMPROVEMENT_PCT / 100)
        improved = best_loss < required_loss
        print(f"{position}: current loss {current_loss:.4f} (MAE {current_mae:.3f}) -> best found {best_loss:.4f} (MAE {best_mae:.3f}) "
              f"({'IMPROVED' if improved else f'no improvement past the {MIN_IMPROVEMENT_PCT}% threshold, kept current'})")
        print(f"    best candidate found (adopted or not): {best_params}")

        final_params = best_params if improved else current_params
        applied_params[position] = final_params
        results[position] = {
            "current_loss": round(current_loss, 4),
            "current_mae": round(current_mae, 4),
            "best_loss": round(best_loss, 4),
            "best_mae": round(best_mae, 4),
            "improved": improved,
            "params": final_params,
            "best_candidate_params": best_params,
        }

    results["generated_at"] = datetime.now(timezone.utc).isoformat()
    results["search_seed"] = SEARCH_SEED
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    any_improved = any(r["improved"] for k, r in results.items() if isinstance(r, dict))
    if apply and any_improved:
        apply_params_to_source(applied_params)
    elif apply:
        print("\nNo position improved past the threshold -- nothing to apply.")
    else:
        print(f"\nSaved {OUTPUT_FILE}. Nothing in {MODEL_FILE} has been changed -- "
              f"re-run with --apply to write genuine improvements back, or review and update by hand.")


if __name__ == "__main__":
    ensure_utf8_stdout()
    main()
