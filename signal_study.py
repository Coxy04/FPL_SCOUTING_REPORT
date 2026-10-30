"""What actually predicts FPL points? A from-first-principles study on the archive.

Everything the project has built assumes the modelling choices made in August. This script
ignores the model and asks the data directly, on two full archived seasons (2024-25, 2025-26):

  1. Which pre-match signals rank players best for THIS week's points, for a HAUL (8+), and for
     the NEXT FIVE weeks' points? (Spearman per gameweek, averaged; among "startable" players.)
  2. How much weekly skill signal exists at all? Split-half reliability of points per game among
     regulars, and lag-1 autocorrelation. This is the ceiling any model is chasing.
  3. Captaincy: each week, who scores more -- the most expensive player, the most owned, the best
     season PPG, the best recent PPG, the best xGI/90, or a LightGBM fit on all signals?
  4. A LightGBM on ALL the signals, time-split, with importance: what a "kitchen sink" model gets
     and what it leans on -- including FPL's ICT/threat/creativity/BPS columns the live model has
     never used.

All signals are lagged: computed from gameweeks strictly before the one being predicted.
"""
import json
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ARCHIVE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{season}"
SEASONS = ["2024-25", "2025-26"]
OUT = "fpl_ml_signal_study.json"
POSITIONS = ["GK", "DEF", "MID", "FWD"]


def spearman(a, b):
    a, b = pd.Series(np.asarray(a, float)), pd.Series(np.asarray(b, float))
    m = a.notna() & b.notna()
    if m.sum() < 8 or a[m].std() == 0 or b[m].std() == 0:
        return np.nan
    return float(np.corrcoef(a[m].rank(), b[m].rank())[0, 1])


def per_gw_spearman(frame, signal, target):
    vals = [spearman(g[signal], g[target]) for _, g in frame.groupby("GW")]
    vals = [v for v in vals if not np.isnan(v)]
    return float(np.mean(vals)) if vals else np.nan


def load_season(season):
    gw = pd.read_csv(f"{ARCHIVE.format(season=season)}/gws/merged_gw.csv", encoding="utf-8", encoding_errors="ignore")
    fx = pd.read_csv(f"{ARCHIVE.format(season=season)}/fixtures.csv")
    gw = gw[gw["position"].isin(POSITIONS)].copy()
    for c in ["total_points", "minutes", "expected_goals", "expected_assists", "expected_goal_involvements", "expected_goals_conceded",
              "threat", "creativity", "influence", "ict_index", "bps", "bonus", "value", "selected", "transfers_balance",
              "transfers_in", "transfers_out", "starts", "goals_scored", "assists", "clean_sheets", "saves", "team_h_score", "team_a_score"]:
        if c in gw:
            gw[c] = pd.to_numeric(gw[c], errors="coerce").fillna(0)
        else:
            gw[c] = 0.0
    gw = gw.sort_values(["element", "GW"]).reset_index(drop=True)
    diff = fx.set_index("id")[["team_h_difficulty", "team_a_difficulty"]]
    gw = gw.join(diff, on="fixture")
    gw["fdr"] = np.where(gw["was_home"].astype(bool), gw["team_h_difficulty"], gw["team_a_difficulty"])
    gw["season"] = season
    # Team-level attack/defence strength prior to this GW: goals for/against per match so far.
    gw["gf"] = np.where(gw["was_home"].astype(bool), gw["team_h_score"], gw["team_a_score"])
    gw["ga"] = np.where(gw["was_home"].astype(bool), gw["team_a_score"], gw["team_h_score"])
    team_gw = gw.groupby(["team", "GW"]).agg(gf=("gf", "first"), ga=("ga", "first")).reset_index().sort_values(["team", "GW"])
    team_gw["team_gf_pg"] = team_gw.groupby("team")["gf"].transform(lambda s: s.shift(1).expanding().mean())
    team_gw["team_ga_pg"] = team_gw.groupby("team")["ga"].transform(lambda s: s.shift(1).expanding().mean())
    team_gw["team_gf6"] = team_gw.groupby("team")["gf"].transform(lambda s: s.shift(1).rolling(6, min_periods=1).mean())
    team_gw["team_ga6"] = team_gw.groupby("team")["ga"].transform(lambda s: s.shift(1).rolling(6, min_periods=1).mean())
    gw = gw.merge(team_gw[["team", "GW", "team_gf_pg", "team_ga_pg", "team_gf6", "team_ga6"]], on=["team", "GW"], how="left")
    # Opponent strength via opponent team name.
    teams = pd.read_csv(f"{ARCHIVE.format(season=season)}/teams.csv")[["id", "name"]].rename(columns={"id": "opponent_team", "name": "opp_name"})
    gw = gw.merge(teams, on="opponent_team", how="left")
    opp = team_gw.rename(columns={"team": "opp_name", "team_gf_pg": "opp_gf_pg", "team_ga_pg": "opp_ga_pg", "team_gf6": "opp_gf6", "team_ga6": "opp_ga6"})
    gw = gw.merge(opp[["opp_name", "GW", "opp_gf_pg", "opp_ga_pg", "opp_gf6", "opp_ga6"]], on=["opp_name", "GW"], how="left")
    return gw


def add_lagged_signals(gw):
    g = gw.groupby("element")
    lag = lambda col, fn: g[col].transform(lambda s: fn(s.shift(1)))
    roll = lambda w: (lambda s: s.rolling(w, min_periods=1).mean())
    exp_mean = lambda s: s.expanding().mean()
    exp_sum = lambda s: s.expanding().sum()
    gw["ppg_season"] = lag("total_points", exp_mean)
    gw["ppg_last3"] = lag("total_points", roll(3))
    gw["ppg_last6"] = lag("total_points", roll(6))
    gw["pts_last1"] = g["total_points"].shift(1)
    gw["min_season"] = lag("minutes", exp_sum)
    gw["min_last6"] = lag("minutes", roll(6))
    gw["min_last1"] = g["minutes"].shift(1)
    gw["starts_share6"] = lag("starts", roll(6))
    gw["games_prior"] = g.cumcount()
    xgi_cum = lag("expected_goal_involvements", exp_sum)
    gw["xgi90_season"] = np.where(gw["min_season"] > 0, 90 * xgi_cum / gw["min_season"].replace(0, np.nan), np.nan)
    xgc_cum = lag("expected_goals_conceded", exp_sum)
    gw["xgc90_season"] = np.where(gw["min_season"] > 0, 90 * xgc_cum / gw["min_season"].replace(0, np.nan), np.nan)
    gw["xgi_last6"] = lag("expected_goal_involvements", roll(6))
    gw["xg_last6"] = lag("expected_goals", roll(6))
    gw["xa_last6"] = lag("expected_assists", roll(6))
    gw["threat_last6"] = lag("threat", roll(6))
    gw["creativity_last6"] = lag("creativity", roll(6))
    gw["influence_last6"] = lag("influence", roll(6))
    gw["ict_last6"] = lag("ict_index", roll(6))
    gw["bps_last6"] = lag("bps", roll(6))
    gw["bonus_last6"] = lag("bonus", roll(6))
    gw["bps_season"] = lag("bps", exp_mean)
    gw["threat_season"] = lag("threat", exp_mean)
    gw["goals_season"] = lag("goals_scored", exp_sum)
    gw["cs_rate_season"] = lag("clean_sheets", exp_mean)
    gw["price"] = gw["value"]
    gw["ownership"] = gw["selected"]
    gw["flow"] = g["transfers_balance"].shift(1)
    gw["is_home"] = gw["was_home"].astype(int)
    # Targets
    gw["pts"] = gw["total_points"]
    gw["haul"] = (gw["total_points"] >= 8).astype(int)
    gw["played"] = (gw["minutes"] > 0).astype(int)
    fut = g["total_points"].transform(lambda s: s[::-1].rolling(5, min_periods=1).sum()[::-1])
    gw["pts_next5"] = fut  # includes this GW and the following 4
    gw["games_next5"] = g["total_points"].transform(lambda s: s[::-1].rolling(5, min_periods=1).count()[::-1])
    return gw


SIGNALS = ["ppg_season", "ppg_last3", "ppg_last6", "pts_last1", "price", "ownership", "flow", "xgi90_season", "xgi_last6", "xg_last6",
           "xa_last6", "threat_last6", "creativity_last6", "influence_last6", "ict_last6", "bps_last6", "bps_season", "bonus_last6",
           "threat_season", "min_last6", "starts_share6", "cs_rate_season", "xgc90_season", "fdr", "is_home", "team_gf_pg", "team_ga_pg",
           "team_gf6", "team_ga6", "opp_gf_pg", "opp_ga_pg", "opp_gf6", "opp_ga6"]
# Signals whose sign should be flipped so "higher = better" (so Spearman signs are comparable).
NEGATIVE = {"fdr", "opp_gf_pg", "opp_gf6", "team_ga_pg", "team_ga6", "xgc90_season"}


def main():
    frames = [add_lagged_signals(load_season(s)) for s in SEASONS]
    data = pd.concat(frames, ignore_index=True)
    data["season_gw"] = data["season"] + "-" + data["GW"].astype(str).str.zfill(2)
    for s in NEGATIVE:
        data[s] = -data[s]
    # Startable pool: regulars with history. Early-season rows (GW<4) excluded: signals too thin.
    pool = data[(data["min_last6"] >= 60) & (data["games_prior"] >= 3) & (data["GW"] >= 4)].copy()
    pool_gw = pool.rename(columns={"season_gw": "GW_key"}).copy()
    pool_gw["GW"] = pool_gw["GW_key"]
    full_next5 = pool_gw[pool_gw["games_next5"] == 5]
    print(f"Rows: {len(data)} total, {len(pool)} startable (min_last6>=60, 3+ games, GW>=4), "
          f"haul rate in pool {pool['haul'].mean():.3f}, mean pts {pool['pts'].mean():.2f}, played rate {pool['played'].mean():.3f}")

    print("\n=== 1. Which signals rank the startable pool? (mean per-GW Spearman, two seasons) ===")
    table = []
    for sig in SIGNALS:
        row = {"signal": sig,
               "this_week_pts": per_gw_spearman(pool_gw, sig, "pts"),
               "haul": per_gw_spearman(pool_gw, sig, "haul"),
               "next5_pts": per_gw_spearman(full_next5, sig, "pts_next5"),
               "played": per_gw_spearman(pool_gw, sig, "played")}
        table.append(row)
    tab = pd.DataFrame(table).sort_values("this_week_pts", ascending=False)
    print(tab.round(3).to_string(index=False))
    print("\n  by position, this-week pts (top 8 signals each):")
    by_pos = {}
    for pos in POSITIONS:
        sub = pool_gw[pool_gw["position"] == pos]
        rows = sorted(((sig, per_gw_spearman(sub, sig, "pts")) for sig in SIGNALS), key=lambda x: -(x[1] if not np.isnan(x[1]) else -9))[:8]
        by_pos[pos] = rows
        print(f"   {pos}: " + ", ".join(f"{s} {v:.3f}" for s, v in rows))

    print("\n=== 2. How much signal is there to find? ===")
    lag1 = spearman(pool["pts_last1"], pool["pts"])
    print(f"  lag-1 Spearman of weekly points (startable): {lag1:.3f}")
    # Split-half reliability of PPG among players with 20+ appearances in a season.
    rel = []
    for season, sd in data[data["minutes"] > 0].groupby("season"):
        counts = sd.groupby("element")["GW"].count()
        regs = sd[sd["element"].isin(counts[counts >= 20].index)]
        odd = regs[regs["GW"] % 2 == 1].groupby("element")["total_points"].mean()
        even = regs[regs["GW"] % 2 == 0].groupby("element")["total_points"].mean()
        both = pd.concat([odd, even], axis=1, keys=["odd", "even"]).dropna()
        r = float(both["odd"].corr(both["even"]))
        rel.append({"season": season, "players": len(both), "split_half_r": r, "spearman_brown": 2 * r / (1 + r)})
        print(f"  {season}: split-half reliability of PPG over a season (20+ apps, n={len(both)}): r={r:.3f} (full-season reliability {2*r/(1+r):.3f})")
    print("  -> a season of PPG is a reliable skill signal; a single week is ~90% noise. The ceiling for ranking ONE week is low;")
    print("     the ceiling for ranking the next 5-8 weeks is much higher. That is where a model should be judged and used.")

    print("\n=== 3. Captaincy: who scores most each week among startable players? ===")
    cap_rules = {"most expensive": "price", "most owned": "ownership", "best season PPG": "ppg_season", "best last-6 PPG": "ppg_last6",
                 "best xGI/90 (season)": "xgi90_season", "best threat (last 6)": "threat_last6", "best BPS (season)": "bps_season"}
    cap = {}
    for name, sig in cap_rules.items():
        pts = [g.loc[g[sig].idxmax(), "pts"] for _, g in pool_gw.groupby("GW") if g[sig].notna().any()]
        cap[name] = {"mean_pts": float(np.mean(pts)), "haul_rate": float(np.mean(np.array(pts) >= 8)), "weeks": len(pts)}
        print(f"  {name:<24} mean {np.mean(pts):.2f} pts/wk, hauls {100*np.mean(np.array(pts) >= 8):.0f}% of weeks")
    best = [g["pts"].max() for _, g in pool_gw.groupby("GW")]
    print(f"  {'hindsight best':<24} mean {np.mean(best):.2f}")
    # Haul rate by price band and position (what does a haul-capable player look like?)
    pool["price_band"] = pd.cut(pool["price"], [0, 50, 60, 75, 95, 200], labels=["<=5.0", "5.1-6.0", "6.1-7.5", "7.6-9.5", "9.6+"])
    hb = pool.groupby(["position", "price_band"], observed=True)["haul"].agg(["mean", "count"]).reset_index()
    hb = hb[hb["count"] >= 40]
    print("  haul rate by position and price band (n>=40):")
    for pos in POSITIONS:
        sub = hb[hb["position"] == pos]
        print(f"   {pos}: " + ", ".join(f"{r.price_band} {100*r['mean']:.0f}%" for _, r in sub.iterrows()))

    print("\n=== 4. Kitchen-sink LightGBM on all signals (time split within each season: train GW<=25, test GW 26-38) ===")
    feats = SIGNALS + [f"pos_{p}" for p in POSITIONS]
    for p in POSITIONS:
        pool[f"pos_{p}"] = (pool["position"] == p).astype(int)
    results = {}
    for target, objective in (("pts", "regression"), ("haul", "binary"), ("pts_next5", "regression")):
        preds, tests = [], []
        imps = pd.Series(0.0, index=feats)
        for season in SEASONS:
            sd = pool[pool["season"] == season]
            if target == "pts_next5":
                sd = sd[sd["games_next5"] == 5]
            tr, te = sd[sd["GW"] <= 25], sd[sd["GW"] >= 26]
            params = dict(n_estimators=400, learning_rate=0.03, num_leaves=15, max_depth=5, min_child_samples=40, reg_lambda=2.0, random_state=42, verbosity=-1)
            m = lgb.LGBMClassifier(**params) if objective == "binary" else lgb.LGBMRegressor(**params)
            m.fit(tr[feats], tr[target])
            p = m.predict_proba(te[feats])[:, 1] if objective == "binary" else m.predict(te[feats])
            te = te.copy(); te["pred"] = p
            preds.append(te); imps += pd.Series(m.booster_.feature_importance("gain"), index=feats)
        te = pd.concat(preds)
        te_gw = te.rename(columns={"season_gw": "GWk"}); te_gw["GW"] = te_gw["GWk"]
        rho = per_gw_spearman(te_gw, "pred", target)
        top = te.sort_values(["season_gw", "pred"], ascending=[True, False]).groupby("season_gw").head(10)
        best_single = max(((s, per_gw_spearman(te_gw, s, target)) for s in SIGNALS), key=lambda x: x[1] if not np.isnan(x[1]) else -9)
        line = {"spearman_model": rho, "top10_actual_mean": float(top[target].mean()), "best_single_signal": best_single[0], "best_single_spearman": best_single[1],
                "top_features": (imps / imps.sum()).sort_values(ascending=False).head(10).round(3).to_dict()}
        if objective == "binary":
            line["auc"] = float(roc_auc_score(te[target], te["pred"]))
            cap_pts = [g.loc[g["pred"].idxmax(), "pts"] for _, g in te.groupby("season_gw")]
            line["captain_by_haul_prob_mean_pts"] = float(np.mean(cap_pts))
        if target == "pts":
            cap_pts = [g.loc[g["pred"].idxmax(), "pts"] for _, g in te.groupby("season_gw")]
            line["captain_by_model_mean_pts"] = float(np.mean(cap_pts))
        results[target] = line
        print(f"  target {target:<9} model Spearman {rho:.3f} vs best single signal {best_single[0]} {best_single[1]:.3f} | top-10 actual {line['top10_actual_mean']:.2f}"
              + (f" | AUC {line['auc']:.3f} | captain-by-haul-prob {line['captain_by_haul_prob_mean_pts']:.2f}" if objective == "binary" else "")
              + (f" | captain-by-model {line['captain_by_model_mean_pts']:.2f}" if target == "pts" else ""))
        print("     importance: " + ", ".join(f"{k} {v:.2f}" for k, v in line["top_features"].items()))

    with open(OUT, "w") as f:
        json.dump({"signal_table": tab.round(4).to_dict("records"), "by_position": {k: [(s, float(v)) for s, v in v_] for k, v_ in by_pos.items()},
                   "reliability": rel, "lag1": lag1, "captaincy": cap, "models": results}, f, indent=2, default=float)
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    from fpl_ml_model import ensure_utf8_stdout
    ensure_utf8_stdout()
    main()
