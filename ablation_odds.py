"""Gate for adding bookmaker odds to the model: does the deployment replay rank better with them?

Same replay as backtest_deployment.py (train on 2024-25 + 2025-26 GW<=N, test GW N+1..N+5), run
with and without odds_features.ODDS_FEATURES attached. Closing-odds caveat applies (see
odds_features.py): the gain measured here is an upper bound on what deadline-time odds give.
"""
import json

import numpy as np
import pandas as pd
from understatapi import UnderstatClient

from backtest_deployment import CUTOFFS, run_config, summarise
import backtest_deployment
from backtest_model import build_archive_rows
from fpl_ml_model import ensure_utf8_stdout
from odds_features import ODDS_FEATURES, attach_odds

OUTPUT_FILE = "fpl_ml_odds_ablation.json"


def main():
    understat = UnderstatClient()
    prior = build_archive_rows("2024-25", "2024", understat=understat)
    current = build_archive_rows("2025-26", "2025", understat=understat)
    prior = attach_odds(prior, ["2024-25"])
    current = attach_odds(current, ["2025-26"])

    base_features = list(backtest_deployment.FEATURES)
    results = {}
    team_form = ("team_xg_for_form", "team_xg_against_form", "opp_xg_for_form", "opp_xg_against_form", "difficulty")
    # First run (2026-10-16): adding odds NEXT TO the team-form features made ranking slightly worse
    # (cross-position pool Spearman 0.147 -> 0.128). Trees on ~1-5k rows per position with five
    # near-duplicate team-strength inputs is a plausible reason, so the third variant swaps the
    # team-form features out for the odds instead of stacking them.
    for label, feats in (("without odds", base_features), ("with odds", base_features + ODDS_FEATURES),
                         ("odds REPLACING team form + FDR", [f for f in base_features if f not in team_form] + ODDS_FEATURES),
                         ("odds + team form only (no player xG)", [f for f in base_features if f not in ("expected_goals", "expected_assists", "npxg90", "xa90")] + ODDS_FEATURES)):
        backtest_deployment.FEATURES = feats
        runs = [run_config(prior, current, cutoff, 100000) for cutoff in CUTOFFS]
        results[label] = summarise(label, runs)
        early = [run_config(prior, current, cutoff, 100000) for cutoff in (3, 6)]
        results[label + " (early season)"] = summarise("  " + label + ", early season", early)
    backtest_deployment.FEATURES = base_features
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {OUTPUT_FILE}")


if __name__ == "__main__":
    ensure_utf8_stdout()
    main()
