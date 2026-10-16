"""Does training on MORE archive seasons improve ranking skill in the deployment setup?

Same replay as backtest_deployment.py (train on prior seasons + 2025-26 GW1..N, test GW N+1..N+5),
but varying how many prior seasons go in: just 2024-25, plus 2023-24, plus 2022-23. Recency
weighting off, so every season counts equally -- the question is whether more rows help at all.
FPL's xG fields only exist from 2022-23, so that is as far back as the feature set can go.
Defensive-contribution features are dropped throughout (absent before 2025-26), as in the
deployment backtest, so numbers are comparable to that script and not to backtest_model.py.
"""
import json

import numpy as np
import pandas as pd
from understatapi import UnderstatClient

from backtest_deployment import CUTOFFS, run_config, summarise
from backtest_model import build_archive_rows
from fpl_ml_model import ensure_utf8_stdout

OUTPUT_FILE = "fpl_ml_more_seasons_ablation.json"
CURRENT = ("2025-26", "2025")
PRIORS = [("2024-25", "2024"), ("2023-24", "2023"), ("2022-23", "2022")]


def main():
    understat = UnderstatClient()
    current_rows = build_archive_rows(*CURRENT, understat=understat)
    prior_frames = {}
    for season, understat_season in PRIORS:
        prior_frames[season] = build_archive_rows(season, understat_season, understat=understat)
        print(f"  {season}: {len(prior_frames[season])} rows")

    summaries = []
    for n_seasons in (1, 2, 3):
        seasons = [s for s, _ in PRIORS[:n_seasons]]
        prior_rows = pd.concat([prior_frames[s] for s in seasons], ignore_index=True)
        runs = [run_config(prior_rows, current_rows, cutoff, 100000) for cutoff in CUTOFFS]
        summaries.append(summarise(f"prior seasons: {', '.join(seasons)}", runs))
        early = [run_config(prior_rows, current_rows, cutoff, 100000) for cutoff in (3, 6)]
        summaries.append(summarise(f"  early season only, {n_seasons} prior", early))

    with open(OUTPUT_FILE, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"Saved {OUTPUT_FILE}")


if __name__ == "__main__":
    ensure_utf8_stdout()
    main()
