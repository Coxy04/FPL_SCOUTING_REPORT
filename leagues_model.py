"""match_model.py for the English and Scottish leagues, from football-data.co.uk.

Same ratings, Poisson grid and Dixon-Coles correction as match_model.py; the only difference is the
input. There is no free per-match xG below the Premier League, so each team-match gets an xG PROXY
built from football-data's shots on target (each league's goals-per-shot-on-target conversion
times shots on target), goals, or a blend -- which proxy is decided per league by validation, never
assumed. football-data also carries closing odds for every division, so every league is validated
the same way as the Premier League model: fit on 2024-25 (seeded by 2023-24), test on 2025-26
against the bookmakers, and the gap to the market is reported per league on the dashboard.

Upcoming fixtures come from football-data's fixtures.csv (the next round or so for every league,
refreshed on their side after each round). Outputs fpl_ml_leagues_predictions.json.
"""
import io
import itertools
import json
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

from fpl_ml_model import ensure_utf8_stdout
from match_model import Ratings, scoreline_grid, score, summarise, walk_forward
from odds_features import CACHE_DIR, market_probabilities

LEAGUES = {"E0": "Premier League", "E1": "Championship", "E2": "League One", "E3": "League Two",
           "SC0": "Scottish Premiership", "SC1": "Scottish Championship", "SC2": "Scottish League One", "SC3": "Scottish League Two"}
SEASON_CODES = {"2023-24": "2324", "2024-25": "2425", "2025-26": "2526", "2026-27": "2627"}
CURRENT_SEASON = "2026-27"
OUTPUT_FILE = "fpl_ml_leagues_predictions.json"
PARAMS_FILE = "fpl_ml_leagues_params.json"
GRID = {"half_life": [6, 10, 15, 25], "shrink": [1, 2, 4], "rho": [-0.14, -0.08, 0.0]}
PROXIES = ["sot", "goals", "blend"]
HEADERS = {"User-Agent": "Mozilla/5.0"}


def fetch_csv(url, cache_name, max_age_hours=None):
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / cache_name
    fresh = path.exists() and (max_age_hours is None or (datetime.now().timestamp() - path.stat().st_mtime) < 3600 * max_age_hours)
    if not fresh:
        response = requests.get(url, headers=HEADERS, timeout=60)
        response.raise_for_status()
        path.write_bytes(response.content)
    return pd.read_csv(io.StringIO(path.read_bytes().decode("utf-8-sig", "ignore")))


def load_results(div, season):
    code = SEASON_CODES[season]
    raw = fetch_csv(f"https://www.football-data.co.uk/mmz4281/{code}/{div}.csv", f"{div}_{code}.csv", max_age_hours=12 if season == CURRENT_SEASON else None)
    raw = raw[raw["HomeTeam"].notna() & raw["FTHG"].notna()].copy()
    probs = market_probabilities(raw)
    out = pd.DataFrame({
        "fixture": [f"{div}-{code}-{i}" for i in range(len(raw))],
        "date": pd.to_datetime(raw["Date"], dayfirst=True).dt.strftime("%Y-%m-%d"),
        "home": raw["HomeTeam"], "away": raw["AwayTeam"],
        "hg": pd.to_numeric(raw["FTHG"], errors="coerce"), "ag": pd.to_numeric(raw["FTAG"], errors="coerce"),
        "hst": pd.to_numeric(raw.get("HST"), errors="coerce"), "ast": pd.to_numeric(raw.get("AST"), errors="coerce"),
        "b_home": probs["h"].values, "b_draw": probs["d"].values, "b_away": probs["a"].values, "b_over": probs["over"].values,
    })
    return out.sort_values("date").reset_index(drop=True)


def with_proxy(frame, proxy, conversion):
    f = frame.copy()
    sot_h, sot_a = f["hst"].fillna(f["hg"] / conversion) * conversion, f["ast"].fillna(f["ag"] / conversion) * conversion
    if proxy == "sot":
        f["xg_h"], f["xg_a"] = sot_h, sot_a
    elif proxy == "goals":
        f["xg_h"], f["xg_a"] = f["hg"].astype(float), f["ag"].astype(float)
    else:
        f["xg_h"], f["xg_a"] = 0.5 * (sot_h + f["hg"]), 0.5 * (sot_a + f["ag"])
    return f


def conversion_rate(frames):
    all_ = pd.concat(frames)
    goals = all_["hg"].sum() + all_["ag"].sum()
    sot = all_["hst"].sum() + all_["ast"].sum()
    return float(goals / sot) if sot else 0.3


def validate_league(div):
    f23, f24, f25 = (load_results(div, s) for s in ("2023-24", "2024-25", "2025-26"))
    conv = conversion_rate([f23, f24])
    best, best_ll = None, 9e9
    for proxy in PROXIES:
        p23, p24 = with_proxy(f23, proxy, conv), with_proxy(f24, proxy, conv)
        for hl, sh, rho in itertools.product(GRID["half_life"], GRID["shrink"], GRID["rho"]):
            params = {"half_life": hl, "shrink": sh, "rho": rho, "proxy": proxy, "conversion": conv}
            pred, _ = walk_forward(p24, params, prior_fixtures=p23, start_index=40)
            ll = score(pred)["log_loss"]
            if ll < best_ll:
                best, best_ll = params, ll
    p24, p25 = with_proxy(f24, best["proxy"], conv), with_proxy(f25, best["proxy"], conv)
    pred, _ = walk_forward(p25, best, prior_fixtures=p24, start_index=40)
    merged = pred.merge(f25[["fixture", "b_home", "b_draw", "b_away"]], on="fixture")
    model, market = score(merged), score(merged, ("b_home", "b_draw", "b_away"))
    const = merged.assign(c_home=0.45, c_draw=0.26, c_away=0.29); naive = score(const, ("c_home", "c_draw", "c_away"))
    agree = float((merged[["p_home", "p_draw", "p_away"]].to_numpy().argmax(axis=1) == merged[["b_home", "b_draw", "b_away"]].to_numpy().argmax(axis=1)).mean())
    closed = (naive["log_loss"] - model["log_loss"]) / (naive["log_loss"] - market["log_loss"]) if naive["log_loss"] > market["log_loss"] else float("nan")
    summary = {"params": best, "fit_log_loss": best_ll, "test_matches": model["n"], "model_log_loss": model["log_loss"], "market_log_loss": market["log_loss"],
               "naive_log_loss": naive["log_loss"], "model_accuracy": model["accuracy"], "market_accuracy": market["accuracy"], "favourite_agreement": agree,
               "gap_closed_pct": 100 * closed}
    print(f"  {LEAGUES[div]:<24} proxy={best['proxy']:<5} hl={best['half_life']:<3} shrink={best['shrink']} rho={best['rho']:+.2f} | 2025-26: model {model['log_loss']:.4f} "
          f"market {market['log_loss']:.4f} naive {naive['log_loss']:.4f} | acc {100*model['accuracy']:.0f}% vs {100*market['accuracy']:.0f}% | agree {100*agree:.0f}% | closes {100*closed:.0f}% of the gap")
    return summary


def predict_league(div, params):
    f25, f26 = load_results(div, "2025-26"), load_results(div, CURRENT_SEASON)
    conv = params["conversion"]
    p25, p26 = with_proxy(f25, params["proxy"], conv), with_proxy(f26, params["proxy"], conv)
    _, ratings = walk_forward(p26, params, prior_fixtures=p25, start_index=10 ** 6)
    fixtures = fetch_csv("https://www.football-data.co.uk/fixtures.csv", "fixtures_upcoming.csv", max_age_hours=6)
    fixtures = fixtures[fixtures["Div"] == div]
    rows = []
    for f in fixtures.itertuples():
        lam_h, lam_a = ratings.expected(f.HomeTeam, f.AwayTeam)
        s = summarise(scoreline_grid(lam_h, lam_a, params["rho"]))
        rows.append({"date": pd.to_datetime(f.Date, dayfirst=True).strftime("%Y-%m-%d"), "time": str(getattr(f, "Time", "") or ""),
                     "home": f.HomeTeam, "away": f.AwayTeam, "xg_home": round(lam_h, 2), "xg_away": round(lam_a, 2),
                     **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()}})
    teams = sorted(set(p26["home"]) | set(p26["away"]))
    strength = sorted(((t, ratings._rating(ratings.att, t), ratings._rating(ratings.dfn, t)) for t in teams), key=lambda x: -(x[1] - x[2]))
    return {"name": LEAGUES[div], "fixtures": sorted(rows, key=lambda r: (r["date"], r["time"])),
            "team_ratings": [{"team": t, "attack": round(a, 2), "defence": round(d, 2)} for t, a, d in strength]}


def main():
    validate = "--validate" in sys.argv
    params = json.load(open(PARAMS_FILE)) if (not validate and __import__("os").path.exists(PARAMS_FILE)) else {}
    if validate or not params:
        print("Validating per league (fit 2024-25, test 2025-26 vs bookmakers):")
        params = {div: validate_league(div) for div in LEAGUES}
        with open(PARAMS_FILE, "w") as f:
            json.dump(params, f, indent=2)
        print(f"Saved {PARAMS_FILE}")
    out = {"generated_at": datetime.now(timezone.utc).isoformat(), "leagues": {}}
    for div in LEAGUES:
        league = predict_league(div, params[div]["params"])
        league["validation"] = {k: v for k, v in params[div].items() if k != "params"}
        out["leagues"][div] = league
        print(f"{LEAGUES[div]}: {len(league['fixtures'])} upcoming fixtures priced")
    with open(OUTPUT_FILE, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {OUTPUT_FILE}")


if __name__ == "__main__":
    ensure_utf8_stdout()
    main()
