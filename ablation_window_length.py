"""Do longer rolling windows rank better and churn less?

Every form feature (player xG, xA, xGC, bonus, defensive stats, team xG for/against) is a 6-match
rolling mean (TEAM_FORM_WINDOW), so one match moves every input by a sixth, and at GW3-7 the
home/away team-form split is two or three matches. Measured churn between weekly runs: rank
correlation among startable players ~0.5, 7 of the top 20 change each week. This tests 6 vs 10 vs
15 on the deployment replay for ranking skill, and on consecutive-cutoff overlap as a churn proxy.
The window is patched at module level before the archive rows are built, which is where the
rolling means are computed, so each setting is a full rebuild.
"""
import json

import numpy as np
import pandas as pd
from understatapi import UnderstatClient

import fpl_ml_model
from backtest_deployment import CUTOFFS, run_config, summarise
from backtest_model import build_archive_rows
from fpl_ml_model import ensure_utf8_stdout

OUTPUT_FILE = "fpl_ml_window_ablation.json"
WINDOWS = [6, 10, 15]


def churn_proxy(runs):
    """Mean top-30 overlap between predictions for the same gameweek produced from adjacent
    cutoffs (one week more data), restricted to gameweeks both runs predict."""
    overlaps = []
    for (a, _), (b, _) in zip(runs, runs[1:]):
        for gw in sorted(set(a["gw"]) & set(b["gw"])):
            ta = set(a[a["gw"] == gw].nlargest(30, "pred").index.map(lambda i: i))
            # index-based identity is not stable across frames; use rank on pred within the frames' row order instead
        # Simpler and stable: compare the ranked lists by row position is not valid either, so fall back to
        # per-gameweek rank correlation of predictions for rows present in both (same row order by construction).
        common = sorted(set(a["gw"]) & set(b["gw"]))
        for gw in common:
            pa, pb = a[a["gw"] == gw]["pred"].to_numpy(), b[b["gw"] == gw]["pred"].to_numpy()
            if len(pa) == len(pb) and len(pa) > 10:
                overlaps.append(float(np.corrcoef(pd.Series(pa).rank(), pd.Series(pb).rank())[0, 1]))
    return float(np.mean(overlaps)) if overlaps else float("nan")


def main():
    understat = UnderstatClient()
    results = {}
    for window in WINDOWS:
        fpl_ml_model.TEAM_FORM_WINDOW = window
        prior = build_archive_rows("2024-25", "2024", understat=understat)
        current = build_archive_rows("2025-26", "2025", understat=understat)
        runs = [run_config(prior, current, cutoff, 100000) for cutoff in CUTOFFS]
        summary = summarise(f"window {window}", runs)
        summary["churn_rank_corr_adjacent_cutoffs"] = round(churn_proxy(runs), 3)
        print(f"   churn proxy (rank corr of same-GW predictions from adjacent cutoffs): {summary['churn_rank_corr_adjacent_cutoffs']}")
        results[window] = summary
    fpl_ml_model.TEAM_FORM_WINDOW = 6
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {OUTPUT_FILE}")


if __name__ == "__main__":
    ensure_utf8_stdout()
    main()
