"""signal_study.py's analysis on THIS season's data (every player's per-match FPL history, pulled
from element-summary), so the archive conclusions can be checked against 2026-27 specifically.
Eight gameweeks is a small sample; read the direction, not the decimals."""
import sys

import numpy as np
import pandas as pd

import signal_study as ss

SCRATCH = sys.argv[1] if len(sys.argv) > 1 else "."
POS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def load_current(path):
    h = pd.read_csv(f"{path}/season_2026_27_history.csv")
    fx = pd.read_csv(f"{path}/fixtures_2026_27.csv")
    teams = pd.read_csv(f"{path}/teams_2026_27.csv")[["id", "name"]]
    h = h.rename(columns={"round": "GW"})
    h["position"] = h["element_type"].map(POS)
    h["team"] = h["team"].map(teams.set_index("id")["name"])
    for c in ["total_points", "minutes", "expected_goals", "expected_assists", "expected_goal_involvements", "expected_goals_conceded",
              "threat", "creativity", "influence", "ict_index", "bps", "bonus", "value", "selected", "transfers_balance",
              "transfers_in", "transfers_out", "starts", "goals_scored", "assists", "clean_sheets", "saves", "team_h_score", "team_a_score"]:
        h[c] = pd.to_numeric(h.get(c, 0), errors="coerce").fillna(0)
    h = h.sort_values(["element", "GW"]).reset_index(drop=True)
    diff = fx.set_index("id")[["team_h_difficulty", "team_a_difficulty"]]
    h = h.join(diff, on="fixture")
    h["fdr"] = np.where(h["was_home"].astype(bool), h["team_h_difficulty"], h["team_a_difficulty"])
    h["season"] = "2026-27"
    h["gf"] = np.where(h["was_home"].astype(bool), h["team_h_score"], h["team_a_score"])
    h["ga"] = np.where(h["was_home"].astype(bool), h["team_a_score"], h["team_h_score"])
    tg = h.groupby(["team", "GW"]).agg(gf=("gf", "first"), ga=("ga", "first")).reset_index().sort_values(["team", "GW"])
    for col, src in (("team_gf_pg", "gf"), ("team_ga_pg", "ga")):
        tg[col] = tg.groupby("team")[src].transform(lambda s: s.shift(1).expanding().mean())
    for col, src in (("team_gf6", "gf"), ("team_ga6", "ga")):
        tg[col] = tg.groupby("team")[src].transform(lambda s: s.shift(1).rolling(6, min_periods=1).mean())
    h = h.merge(tg[["team", "GW", "team_gf_pg", "team_ga_pg", "team_gf6", "team_ga6"]], on=["team", "GW"], how="left")
    h["opp_name"] = h["opponent_team"].map(teams.set_index("id")["name"])
    opp = tg.rename(columns={"team": "opp_name", "team_gf_pg": "opp_gf_pg", "team_ga_pg": "opp_ga_pg", "team_gf6": "opp_gf6", "team_ga6": "opp_ga6"})
    h = h.merge(opp[["opp_name", "GW", "opp_gf_pg", "opp_ga_pg", "opp_gf6", "opp_ga6"]], on=["opp_name", "GW"], how="left")
    return h


def main():
    data = ss.add_lagged_signals(load_current(SCRATCH))
    data["season_gw"] = data["GW"].astype(str).str.zfill(2)
    for s in ss.NEGATIVE:
        data[s] = -data[s]
    pool = data[(data["min_last6"] >= 60) & (data["games_prior"] >= 2) & (data["GW"] >= 3)].copy()
    pool_gw = pool.copy(); pool_gw["GW"] = pool_gw["season_gw"]
    print(f"2026-27: {len(pool)} startable rows over GW3-8; haul rate {pool['haul'].mean():.3f}; mean pts {pool['pts'].mean():.2f}")
    rows = []
    for sig in ss.SIGNALS:
        rows.append({"signal": sig, "this_week_pts": ss.per_gw_spearman(pool_gw, sig, "pts"), "haul": ss.per_gw_spearman(pool_gw, sig, "haul"),
                     "next3_pts": ss.per_gw_spearman(pool_gw[pool_gw["games_next5"] >= 3], sig, "pts_next5")})
    tab = pd.DataFrame(rows).sort_values("this_week_pts", ascending=False)
    print(tab.round(3).to_string(index=False))
    print("\nCaptaincy this season (GW3-8), argmax among startable:")
    for name, sig in {"most expensive": "price", "most owned": "ownership", "best season PPG": "ppg_season", "best last-6 PPG": "ppg_last6",
                      "best xGI/90": "xgi90_season", "best threat last 6": "threat_last6", "best BPS season": "bps_season"}.items():
        picks = [(g.loc[g[sig].idxmax(), "web_name"], int(g.loc[g[sig].idxmax(), "pts"])) for _, g in pool_gw.groupby("GW") if g[sig].notna().any()]
        print(f"  {name:<20} mean {np.mean([p for _, p in picks]):.2f}  {picks}")
    # Who have the consistently good picks been? Top 15 by total points among regulars, with their pre-season price and xGI/90
    reg = data[data["GW"] == data["GW"].max()]
    tot = data.groupby("element").agg(name=("web_name", "first"), pos=("position", "first"), pts=("total_points", "sum"), mins=("minutes", "sum"),
                                      price0=("value", "first"), price_now=("value", "last"), xgi=("expected_goal_involvements", "sum"),
                                      bps=("bps", "sum"), own_now=("selected_by_percent_today", "first")).reset_index()
    tot["xgi90"] = 90 * tot["xgi"] / tot["mins"].replace(0, np.nan)
    print("\nTop 20 total points so far (what a good pick looked like, in hindsight):")
    print(tot.sort_values("pts", ascending=False).head(20)[["name", "pos", "pts", "mins", "price0", "price_now", "xgi90", "bps", "own_now"]].round(2).to_string(index=False))
    print("\nPoints per £m, regulars (mins>=450), top 15:")
    tot["ppm"] = tot["pts"] / (tot["price0"] / 10)
    print(tot[tot["mins"] >= 450].sort_values("ppm", ascending=False).head(15)[["name", "pos", "pts", "price0", "ppm", "xgi90", "own_now"]].round(2).to_string(index=False))


if __name__ == "__main__":
    from fpl_ml_model import ensure_utf8_stdout
    ensure_utf8_stdout()
    main()
