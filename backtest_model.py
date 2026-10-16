"""Backtests the prediction methodology against real historical results.

fpl_ml_model.py never validates its own accuracy -- it just trains on everything available and
predicts forward, so there's no way to know from a normal run whether the predictions are any
good. This script holds out the last few gameweeks of last season's archive, trains each position
model on everything before that, and checks the predictions against what actually happened.

What it measures changed after the GW1-6 review (SEASON_REVIEW_GW1-6.md). MAE is still reported,
but it is no longer the number anything is judged on: on a target where 63% of outcomes are 2 or
less, MAE is minimised by predicting near the median for everyone, so it rewards exactly the
shrink-to-the-mean behaviour that destroys the top of the ranking -- and the top of the ranking is
the only thing captaincy and transfers use. The headline metrics are now:

- spearman_pool: rank correlation among the top quarter of predicted players each gameweek (the
  "startable" pool a manager actually chooses from). Live GW3-5 this was 0.05 / 0.12 / -0.02.
- pool_bias: mean predicted minus mean actual in that pool. Captaincy compares the four position
  models against each other, so a position that is under-predicted (FWD was, by 0.7-1.3 pts) can
  never win the armband however good its players are.
- top10_actual_mean: what the ten highest-predicted players per gameweek actually scored.
- haul classifier: AUC and calibration of P(points >= 8), the quantity the captain pick is a bet on.

selection_loss() combines spearman_pool and pool_bias into the single number tune_model.py searches
on. A live scorecard against prediction_history/ snapshots is produced too, so the archive numbers
can be checked against how the deployed model is actually doing this season.

Reuses fpl_ml_model.py's own feature-engineering functions rather than reimplementing them, so this
can't silently drift from what the real pipeline does.
"""
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import roc_auc_score
from understatapi import UnderstatClient

from fpl_ml_model import (
    BASE_URL,
    HAUL_THRESHOLD,
    POSITION_FEATURES,
    POSITION_MAP,
    PREDICTION_HISTORY_DIR,
    PRIOR_SEASON_ARCHIVE,
    PRIOR_SEASON_UNDERSTAT,
    QUANTILE_HIGH_ALPHA,
    QUANTILE_LOW_ALPHA,
    QUANTILE_MARGIN,
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
    make_haul_model,
    make_model,
    make_quantile_model,
    match_understat_players,
)

HOLDOUT_GAMEWEEKS = 5
OUTPUT_FILE = "fpl_ml_backtest.json"
HISTORY_FILE = "fpl_ml_accuracy_history.jsonl"
LIVE_SCORECARD_FILE = "fpl_ml_live_scorecard.json"
MODEL_FILE = "fpl_ml_model.py"

# The "startable pool": top quarter of predicted players per gameweek. A fraction rather than a
# points threshold so a model that simply predicts bigger numbers isn't rewarded with a bigger pool.
POOL_FRACTION = 0.25
TOP_K = 10
# Cross-position pool for the captaincy-relevant check: roughly the players a manager would consider.
CROSS_POSITION_POOL = 100
# How a 1-pt bias in the pool trades off against rank correlation in selection_loss. A quarter of
# a Spearman point per point of bias: enough that the old FWD under-prediction (0.7-1.3 pts) costs
# more than any plausible ranking gain could buy back, small enough that calibration doesn't
# dominate ranking skill, which is the thing being searched for. A design choice, not a measured
# constant -- revisit if tuning starts trading real ranking skill for tiny bias changes.
BIAS_WEIGHT = 0.25
# Live conformal margin only trusted once a position has this many scored live rows.
LIVE_MARGIN_MIN_ROWS = 50


def spearman(a, b):
    a = pd.Series(np.asarray(a, dtype=float)).rank()
    b = pd.Series(np.asarray(b, dtype=float)).rank()
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def top_fraction_per_gw(frame, fraction=POOL_FRACTION, minimum=5):
    """Rows in the top `fraction` of `pred` within each gameweek."""
    rank = frame.groupby("gw")["pred"].rank(ascending=False, method="first")
    size = frame.groupby("gw")["pred"].transform("size")
    keep = rank <= np.maximum(minimum, np.ceil(size * fraction))
    return frame[keep]


def top_k_per_gw(frame, k=TOP_K):
    rank = frame.groupby("gw")["pred"].rank(ascending=False, method="first")
    return frame[rank <= k]


def ranking_metrics(frame):
    """frame: columns gw, pred, actual. The metrics everything in this project is now judged on."""
    pool = top_fraction_per_gw(frame)
    top = top_k_per_gw(frame)
    return {
        "mae": float(np.abs(frame["pred"] - frame["actual"]).mean()),
        "spearman_all": spearman(frame["pred"], frame["actual"]),
        "spearman_pool": spearman(pool["pred"], pool["actual"]),
        "pool_rows": int(len(pool)),
        "pool_pred_mean": float(pool["pred"].mean()),
        "pool_actual_mean": float(pool["actual"].mean()),
        "pool_bias": float(pool["pred"].mean() - pool["actual"].mean()),
        "top10_pred_mean": float(top["pred"].mean()),
        "top10_actual_mean": float(top["actual"].mean()),
    }


def selection_loss(metrics):
    """Lower is better. What tune_model.py minimises instead of MAE -- see module docstring."""
    rho = metrics["spearman_pool"] if metrics["spearman_pool"] is not None else 0.0
    return (1.0 - rho) + BIAS_WEIGHT * abs(metrics["pool_bias"])


def apply_quantile_margin_to_source(margins):
    """Rewrites QUANTILE_MARGIN in fpl_ml_model.py in place -- unlike tune_model.py's hyperparameter
    search, this isn't gated behind an improvement threshold: it's a direct calibration correction,
    not a noisy search, so it's safe (and necessary) to keep it current on every run. Git tracks
    every change, so a bad update is always a `git revert` away."""
    text = open(MODEL_FILE, encoding="utf-8").read()
    pairs = ", ".join(f'"{k}": {v!r}' for k, v in margins.items())
    new_line = f"QUANTILE_MARGIN = {{{pairs}}}"

    start = text.index("QUANTILE_MARGIN = {")
    end = text.index("\n", start)
    text = text[:start] + new_line + text[end:]
    open(MODEL_FILE, "w", encoding="utf-8").write(text)
    print(f"Applied to {MODEL_FILE}: QUANTILE_MARGIN updated.")


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


def build_archive_rows(archive_season=PRIOR_SEASON_ARCHIVE, understat_season=PRIOR_SEASON_UNDERSTAT, understat=None, include_unplayed=False):
    """Feature rows for one archived season, built with the live pipeline's own functions. Shared by
    this script, tune_model.py, backtest_deployment.py, calibrate_captaincy.py and the ablations.
    include_unplayed keeps 0-minute rows (see build_prior_season_rows)."""
    understat = understat or UnderstatClient()
    team_matches = get_understat_team_matches(understat, understat_season)
    team_matches = add_prematch_team_form(team_matches)
    team_form_lookup = build_team_form_lookup(team_matches)

    merged_gw, difficulty_lookup, team_names, player_idlist, rest_days_lookup, prior_set_piece_lookup = load_prior_season_archive(archive_season)
    elements = build_backtest_elements(merged_gw, player_idlist)

    understat_players = get_understat_player_stats(understat, understat_season)
    understat_matches, _ = match_understat_players(elements, understat_players)
    prior_per90_by_understat_id = {p["understat_id"]: p for p in understat_players}
    id_map = {e: e for e in merged_gw["element"].unique()}

    rows = build_prior_season_rows(
        merged_gw, difficulty_lookup, team_names, id_map, team_form_lookup, understat_matches,
        prior_per90_by_understat_id, rest_days_lookup, prior_set_piece_lookup, include_unplayed=include_unplayed,
    )
    rows = add_features(rows)
    rows["season"] = archive_season
    # Ownership at that gameweek, for the template-captain comparisons in calibrate_captaincy.py.
    if "selected" in merged_gw:
        selected = merged_gw[["element", "GW", "selected"]].rename(columns={"element": "current_id", "GW": "gw"})
        rows = rows.merge(selected, on=["current_id", "gw"], how="left")
    return rows


def evaluate_position(train, test, position):
    model = make_model(position)
    weights = compute_sample_weights(
        train["date"], half_life_days=get_half_life_days(position), as_of=pd.Timestamp(train["date"].max(), tz="UTC")
    )
    model.fit(train[POSITION_FEATURES], train["total_points"], sample_weight=weights)
    predictions = model.predict(test[POSITION_FEATURES])
    multiplier = np.minimum(1.0, test["minutes"].to_numpy() / get_playing_time_denominator(position))
    predictions = np.clip(predictions * multiplier, 0, 15)
    actual = test["total_points"].to_numpy()

    baseline_pred = train["total_points"].mean()
    baseline_mae = float(np.mean(np.abs(baseline_pred - actual)))

    frame = pd.DataFrame({"gw": test["gw"].to_numpy(), "position": position, "pred": predictions, "actual": actual})
    metrics = ranking_metrics(frame)

    # Coverage check for the uncertainty bands: an 80% interval (10th-90th percentile) is only
    # meaningful if roughly 80% of real outcomes actually land inside it on data the model never
    # trained on -- trusting the theory without checking would be exactly the "we can just make
    # up a number" mistake this whole backtest exists to avoid.
    low_model = make_quantile_model(position, QUANTILE_LOW_ALPHA)
    low_model.fit(train[POSITION_FEATURES], train["total_points"], sample_weight=weights)
    high_model = make_quantile_model(position, QUANTILE_HIGH_ALPHA)
    high_model.fit(train[POSITION_FEATURES], train["total_points"], sample_weight=weights)
    low_raw = low_model.predict(test[POSITION_FEATURES])
    high_raw = high_model.predict(test[POSITION_FEATURES])
    low = np.clip(np.minimum(low_raw, high_raw) * multiplier, 0, 15)
    high = np.clip(np.maximum(low_raw, high_raw) * multiplier, 0, 15)
    low = np.minimum(low, predictions)
    high = np.maximum(high, predictions)
    coverage = float(np.mean((actual >= low) & (actual <= high)))
    mean_width = float(np.mean(high - low))

    # Conformal-prediction margin: widen both bounds by the (1 - miscoverage) empirical quantile
    # of how far outside the raw interval real outcomes actually landed, so the corrected interval
    # is calibrated to genuinely hit its target coverage on data like this, not just claim to.
    target_coverage = QUANTILE_HIGH_ALPHA - QUANTILE_LOW_ALPHA
    nonconformity = np.maximum(low - actual, actual - high)
    q_level = min(1.0, np.ceil((len(nonconformity) + 1) * target_coverage) / len(nonconformity))
    quantile_margin = max(0.0, float(np.quantile(nonconformity, q_level)))

    # Haul classifier: does P(points >= 8) rank hauls, and does its average match the real rate in
    # the startable pool (the players it will actually be read for)?
    haul_model = make_haul_model(position)
    haul_target = (train["total_points"] >= HAUL_THRESHOLD).astype(int)
    haul_auc = None
    haul_pool_prob = haul_pool_rate = None
    if haul_target.nunique() == 2:
        haul_model.fit(train[POSITION_FEATURES], haul_target, sample_weight=weights)
        haul_prob = np.clip(haul_model.predict_proba(test[POSITION_FEATURES])[:, 1] * multiplier, 0, 1)
        hauled = (actual >= HAUL_THRESHOLD).astype(int)
        if hauled.sum() > 0 and hauled.sum() < len(hauled):
            haul_auc = float(roc_auc_score(hauled, haul_prob))
        frame["haul_prob"] = haul_prob
        pool = top_fraction_per_gw(frame)
        haul_pool_prob = float(pool["haul_prob"].mean())
        haul_pool_rate = float((pool["actual"] >= HAUL_THRESHOLD).mean())

    result = {
        "position": position,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "model_mae": round(metrics["mae"], 3),
        "baseline_mae": round(baseline_mae, 3),
        "improvement_vs_baseline_pct": round(100 * (1 - metrics["mae"] / baseline_mae), 1) if baseline_mae > 0 else None,
        "spearman_all": round(metrics["spearman_all"], 3) if metrics["spearman_all"] is not None else None,
        "spearman_pool": round(metrics["spearman_pool"], 3) if metrics["spearman_pool"] is not None else None,
        "pool_bias": round(metrics["pool_bias"], 3),
        "pool_pred_mean": round(metrics["pool_pred_mean"], 2),
        "pool_actual_mean": round(metrics["pool_actual_mean"], 2),
        "top10_pred_mean": round(metrics["top10_pred_mean"], 2),
        "top10_actual_mean": round(metrics["top10_actual_mean"], 2),
        "selection_loss": round(selection_loss(metrics), 4),
        "max_prediction": round(float(predictions.max()), 2),
        "interval_coverage_pct": round(100 * coverage, 1),
        "interval_target_pct": round(100 * target_coverage, 1),
        "interval_mean_width": round(mean_width, 2),
        "quantile_margin": round(quantile_margin, 3),
        "haul_auc": round(haul_auc, 3) if haul_auc is not None else None,
        "haul_pool_mean_prob": round(haul_pool_prob, 3) if haul_pool_prob is not None else None,
        "haul_pool_actual_rate": round(haul_pool_rate, 3) if haul_pool_rate is not None else None,
    }
    return result, frame


def cross_position_metrics(frames):
    """Captaincy compares players ACROSS positions, so check the combined top-100 per gameweek:
    rank skill within it, and whether any position is systematically over/under-represented or
    mis-calibrated inside it."""
    combined = pd.concat(frames, ignore_index=True)
    rank = combined.groupby("gw")["pred"].rank(ascending=False, method="first")
    pool = combined[rank <= CROSS_POSITION_POOL]
    by_position = {}
    for position, group in pool.groupby("position"):
        by_position[position] = {
            "share_of_pool_pct": round(100 * len(group) / len(pool), 1),
            "pred_mean": round(float(group["pred"].mean()), 2),
            "actual_mean": round(float(group["actual"].mean()), 2),
            "bias": round(float(group["pred"].mean() - group["actual"].mean()), 2),
        }
    rho = spearman(pool["pred"], pool["actual"])
    top = top_k_per_gw(combined)
    return {
        "pool_size_per_gw": CROSS_POSITION_POOL,
        "spearman_pool": round(rho, 3) if rho is not None else None,
        "top10_actual_mean": round(float(top["actual"].mean()), 2),
        "top10_position_mix": top["position"].value_counts().to_dict(),
        "by_position": by_position,
    }


def get_json_with_retry(session, url, tries=4):
    """FPL's API drops connections now and then (seen as WinError 10053 mid-run). A dropped live
    scorecard shouldn't take the whole backtest down with it."""
    import time
    for attempt in range(tries):
        try:
            response = session.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError):
            if attempt == tries - 1:
                raise
            time.sleep(2 * (attempt + 1))


def score_live_snapshots(session):
    """How the DEPLOYED model has done this season: every prediction_history/gwNN snapshot whose
    gameweek has finished, scored against real points. Same metrics as the archive backtest, plus
    the captaincy comparison that matters in practice (model's top pick vs most-owned player vs
    hindsight best) and the live conformal margin for the intervals."""
    bootstrap = get_json_with_retry(session, f"{BASE_URL}/bootstrap-static/")
    finished = {e["id"]: e for e in bootstrap["events"] if e.get("finished")}
    per_gw = []
    played_rows = []
    for path in sorted(PREDICTION_HISTORY_DIR.glob("gw*.csv.gz")):
        gw = int(path.name[2:4])
        if gw not in finished:
            continue
        snap = pd.read_csv(path)
        snap = snap[(snap["weeks_ahead"] == 1) & (snap["is_blank"].astype(str).str.lower() != "true")]
        agg = snap.groupby("id", as_index=False).agg(
            web_name=("web_name", "first"), position=("position", "first"),
            pred=("predicted_points", "sum"), low=("predicted_points_low", "sum"),
            high=("predicted_points_high", "sum"), ownership_pct=("ownership_pct", "first"),
        )
        live = get_json_with_retry(session, f"{BASE_URL}/event/{gw}/live/")["elements"]
        points = {e["id"]: e["stats"]["total_points"] for e in live}
        minutes = {e["id"]: e["stats"]["minutes"] for e in live}
        agg["actual"] = agg["id"].map(points).fillna(0).astype(int)
        agg["minutes"] = agg["id"].map(minutes).fillna(0).astype(int)
        agg["gw"] = gw
        played = agg[agg["minutes"] > 0].copy()
        if played.empty:
            continue
        metrics = ranking_metrics(played)
        coverage = float(((played["actual"] >= played["low"]) & (played["actual"] <= played["high"])).mean())
        model_pick = agg.loc[agg["pred"].idxmax()]
        template_pick = agg.loc[agg["ownership_pct"].idxmax()]
        best = agg.loc[agg["actual"].idxmax()]
        per_gw.append({
            "event": gw,
            "players_scored": int(len(played)),
            "mae": round(metrics["mae"], 2),
            "spearman_all": round(metrics["spearman_all"], 3) if metrics["spearman_all"] is not None else None,
            "spearman_pool": round(metrics["spearman_pool"], 3) if metrics["spearman_pool"] is not None else None,
            "pool_bias": round(metrics["pool_bias"], 2),
            "top10_pred_mean": round(metrics["top10_pred_mean"], 2),
            "top10_actual_mean": round(metrics["top10_actual_mean"], 2),
            "interval_coverage_pct": round(100 * coverage, 1),
            "model_captain": {"web_name": model_pick["web_name"], "predicted": round(float(model_pick["pred"]), 2), "actual": int(model_pick["actual"])},
            "template_captain": {"web_name": template_pick["web_name"], "ownership_pct": float(template_pick["ownership_pct"]), "actual": int(template_pick["actual"])},
            "best_captain": {"web_name": best["web_name"], "actual": int(best["actual"])},
            "game_average": finished[gw].get("average_entry_score"),
        })
        played_rows.append(played)

    if not per_gw:
        return None

    all_played = pd.concat(played_rows, ignore_index=True)
    pooled = ranking_metrics(all_played)
    by_position = {}
    live_margin_extra = {}
    target_coverage = QUANTILE_HIGH_ALPHA - QUANTILE_LOW_ALPHA
    for position, group in all_played.groupby("position"):
        m = ranking_metrics(group)
        coverage = float(((group["actual"] >= group["low"]) & (group["actual"] <= group["high"])).mean())
        by_position[position] = {
            "rows": int(len(group)),
            "spearman_pool": round(m["spearman_pool"], 3) if m["spearman_pool"] is not None else None,
            "pool_bias": round(m["pool_bias"], 2),
            "mean_pred": round(float(group["pred"].mean()), 2),
            "mean_actual": round(float(group["actual"].mean()), 2),
            "interval_coverage_pct": round(100 * coverage, 1),
        }
        # The snapshot bounds already include whatever margin was live at the time, so the live
        # nonconformity quantile is the EXTRA widening needed on top of the archive margin.
        if len(group) >= LIVE_MARGIN_MIN_ROWS:
            nonconformity = np.maximum(group["low"] - group["actual"], group["actual"] - group["high"])
            q_level = min(1.0, np.ceil((len(nonconformity) + 1) * target_coverage) / len(nonconformity))
            live_margin_extra[position] = round(max(0.0, float(np.quantile(nonconformity, q_level))), 3)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gameweeks": per_gw,
        "pooled": {
            "players_scored": int(len(all_played)),
            "spearman_all": round(pooled["spearman_all"], 3) if pooled["spearman_all"] is not None else None,
            "spearman_pool": round(pooled["spearman_pool"], 3) if pooled["spearman_pool"] is not None else None,
            "pool_bias": round(pooled["pool_bias"], 2),
            "top10_actual_mean": round(pooled["top10_actual_mean"], 2),
            "model_captain_pts": int(sum(g["model_captain"]["actual"] for g in per_gw)),
            "template_captain_pts": int(sum(g["template_captain"]["actual"] for g in per_gw)),
            "best_captain_pts": int(sum(g["best_captain"]["actual"] for g in per_gw)),
        },
        "by_position": by_position,
        "live_margin_extra": live_margin_extra,
    }


def main():
    print(f"Backtesting against the {PRIOR_SEASON_ARCHIVE} archive: train on all but the last "
          f"{HOLDOUT_GAMEWEEKS} gameweeks, test on those held-out gameweeks.")

    rows = build_archive_rows()
    max_gw = rows["gw"].max()
    split_point = max_gw - HOLDOUT_GAMEWEEKS
    train_all = rows[rows["gw"] <= split_point]
    test_all = rows[rows["gw"] > split_point]
    print(f"Split at GW{split_point}: {len(train_all)} train rows (GW1-{split_point}), "
          f"{len(test_all)} test rows (GW{split_point + 1}-{max_gw})\n")

    results = []
    frames = []
    for position in POSITION_MAP.values():
        train = train_all[train_all["position"] == position]
        test = test_all[test_all["position"] == position]
        if len(train) < 30 or len(test) < 5:
            print(f"{position}: skipped, not enough rows (train={len(train)}, test={len(test)})")
            continue
        result, frame = evaluate_position(train, test, position)
        results.append(result)
        frames.append(frame)
        print(
            f"{position:>3}  pool Spearman {result['spearman_pool']}  pool bias {result['pool_bias']:+.2f} "
            f"(pred {result['pool_pred_mean']} vs actual {result['pool_actual_mean']})  "
            f"top-10 actual {result['top10_actual_mean']}  max pred {result['max_prediction']}  "
            f"selection loss {result['selection_loss']}"
        )
        print(
            f"     MAE {result['model_mae']:.2f} vs baseline {result['baseline_mae']:.2f}  all-rows Spearman {result['spearman_all']}  "
            f"[{result['train_rows']} train / {result['test_rows']} test rows]"
        )
        print(
            f"     {result['interval_target_pct']}% interval actual coverage: {result['interval_coverage_pct']}%  "
            f"(mean width {result['interval_mean_width']} pts)   haul AUC {result['haul_auc']}  "
            f"haul prob {result['haul_pool_mean_prob']} vs rate {result['haul_pool_actual_rate']} in pool"
        )

    cross = cross_position_metrics(frames) if frames else None
    if cross:
        print(f"\nCross-position top-{CROSS_POSITION_POOL} per GW: Spearman {cross['spearman_pool']}, "
              f"top-10 actual {cross['top10_actual_mean']}, top-10 mix {cross['top10_position_mix']}")
        for position, stats in cross["by_position"].items():
            print(f"   {position}: {stats['share_of_pool_pct']}% of pool, pred {stats['pred_mean']} vs actual {stats['actual_mean']} (bias {stats['bias']:+.2f})")

    overall_model_mae = float(np.mean([r["model_mae"] for r in results])) if results else None
    overall_baseline_mae = float(np.mean([r["baseline_mae"] for r in results])) if results else None
    overall_loss = float(np.mean([r["selection_loss"] for r in results])) if results else None
    print(f"\nOverall (mean across positions): selection loss {overall_loss:.4f}, MAE {overall_model_mae:.3f} vs baseline {overall_baseline_mae:.3f}")

    session = requests.Session()
    try:
        live = score_live_snapshots(session)
    except requests.RequestException as exc:
        # The archive results above are the backtest proper; the live scorecard is a bonus that
        # must not block the margin update or the refresh pipeline when FPL's API is flaky.
        print(f"\nLive scorecard skipped -- FPL API unreachable after retries ({exc}).")
        live = None
    if live:
        print(f"\nLIVE this season ({len(live['gameweeks'])} finished gameweek(s) with a snapshot):")
        for g in live["gameweeks"]:
            print(f"  GW{g['event']}: pool Spearman {g['spearman_pool']}, pool bias {g['pool_bias']:+.2f}, "
                  f"top-10 actual {g['top10_actual_mean']}, coverage {g['interval_coverage_pct']}% | "
                  f"captain: model {g['model_captain']['web_name']} {g['model_captain']['actual']}, "
                  f"most-owned {g['template_captain']['web_name']} {g['template_captain']['actual']}, "
                  f"best {g['best_captain']['web_name']} {g['best_captain']['actual']}")
        p = live["pooled"]
        print(f"  Pooled: pool Spearman {p['spearman_pool']}, bias {p['pool_bias']:+.2f}; captain pts model {p['model_captain_pts']} "
              f"vs most-owned {p['template_captain_pts']} vs best {p['best_captain_pts']}")
        for position, stats in live["by_position"].items():
            print(f"   {position}: n={stats['rows']} pool Spearman {stats['spearman_pool']} bias {stats['pool_bias']:+.2f} coverage {stats['interval_coverage_pct']}%")
        with open(LIVE_SCORECARD_FILE, "w") as f:
            json.dump(live, f, indent=2)
        print(f"  Saved {LIVE_SCORECARD_FILE}")

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "holdout_gameweeks": HOLDOUT_GAMEWEEKS,
        "split_gw": int(split_point),
        "max_gw": int(max_gw),
        "by_position": results,
        "cross_position": cross,
        "overall_selection_loss": round(overall_loss, 4) if overall_loss is not None else None,
        "overall_model_mae": round(overall_model_mae, 3) if overall_model_mae is not None else None,
        "overall_baseline_mae": round(overall_baseline_mae, 3) if overall_baseline_mae is not None else None,
        "live": live["pooled"] if live else None,
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    with open(HISTORY_FILE, "a") as f:
        f.write(json.dumps(output) + "\n")
    print(f"Saved {OUTPUT_FILE} (and appended to {HISTORY_FILE})")

    if results:
        # Archive margin, widened by the live extra where there is enough live data to measure it.
        # Live coverage ran 64-70% against an 80% target in GW3-5 with the archive-only margin: the
        # archive calibration did not transfer, so live evidence is allowed to widen but not narrow.
        margins = {}
        for r in results:
            extra = (live or {}).get("live_margin_extra", {}).get(r["position"], 0.0)
            margins[r["position"]] = round(r["quantile_margin"] + extra, 3)
            if extra:
                print(f"  {r['position']}: live coverage short, widening margin {r['quantile_margin']} -> {margins[r['position']]}")
        apply_quantile_margin_to_source(margins)


if __name__ == "__main__":
    ensure_utf8_stdout()
    main()
