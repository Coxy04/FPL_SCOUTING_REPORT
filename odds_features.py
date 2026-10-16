"""Bookmaker odds as model features, from football-data.co.uk's free historical files.

Why: closing odds already price in injuries, lineups, motivation and everything else the market
knows; implied team goals and clean-sheet probability are the backbone of every serious FPL
projection. The GW1-6 review found the model's ranking skill is capped by the information in its
features, not by data volume (adding three more seasons moved pool Spearman 0.147 -> 0.165), so a
genuinely new information source is the right next step. This module turns 1X2 and over/under
2.5 prices into per-team expected goals via an independent-Poisson fit, which gives:

  implied_team_goals  - the team's own expected goals this match
  implied_opp_goals   - the opponent's expected goals (what a defender/keeper faces)
  implied_cs_prob     - exp(-implied_opp_goals), the clean-sheet probability
  implied_win_prob    - the team's win probability
  implied_total_goals - the match total

Caveat for backtests: these are CLOSING odds, taken at kickoff after lineups are known. Odds
fetched at the FPL deadline are slightly less informed, so a replay using closing odds is a little
optimistic. Live use needs a pre-deadline source (The Odds API, free tier) -- see fetch_live_odds.

Files are cached in odds_cache/ so repeated runs don't hammer the site.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.optimize import minimize
from scipy.stats import poisson

FOOTBALL_DATA_URL = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"
SEASON_CODES = {"2022-23": "2223", "2023-24": "2324", "2024-25": "2425", "2025-26": "2526", "2026-27": "2627"}
CACHE_DIR = Path("odds_cache")
ODDS_FEATURES = ["implied_team_goals", "implied_opp_goals", "implied_cs_prob", "implied_win_prob", "implied_total_goals"]

# football-data.co.uk team names -> FPL team names (bootstrap-static `name`). Identical names omitted.
TEAM_NAME_MAP = {
    "Man City": "Man City", "Man United": "Man Utd", "Newcastle": "Newcastle", "Nott'm Forest": "Nott'm Forest",
    "Tottenham": "Spurs", "Wolves": "Wolves", "Brighton": "Brighton", "West Ham": "West Ham",
    "Sheffield United": "Sheffield Utd", "Luton": "Luton", "Leicester": "Leicester", "Southampton": "Southampton",
    "Ipswich": "Ipswich Town", "Leeds": "Leeds", "Burnley": "Burnley", "Sunderland": "Sunderland",
    "Coventry": "Coventry City", "Hull": "Hull City", "Arsenal": "Arsenal", "Aston Villa": "Aston Villa",
    "Bournemouth": "Bournemouth", "Brentford": "Brentford", "Chelsea": "Chelsea", "Crystal Palace": "Crystal Palace",
    "Everton": "Everton", "Fulham": "Fulham", "Liverpool": "Liverpool",
}


def download_season(season):
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / f"E0_{SEASON_CODES[season]}.csv"
    # The current season's file grows every week, so don't trust its cache for long.
    if path.exists() and (season != max(SEASON_CODES) or (pd.Timestamp.now() - pd.Timestamp(path.stat().st_mtime, unit="s")).days < 1):
        return pd.read_csv(path, encoding="utf-8-sig")
    response = requests.get(FOOTBALL_DATA_URL.format(code=SEASON_CODES[season]), headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    response.raise_for_status()
    path.write_bytes(response.content)
    return pd.read_csv(path, encoding="utf-8-sig")


def _pick(frame, *candidates):
    for c in candidates:
        if c in frame.columns:
            return pd.to_numeric(frame[c], errors="coerce")
    return pd.Series(np.nan, index=frame.index)


def market_probabilities(frame):
    """Normalised implied probabilities. Prefer closing market averages, then pre-close averages,
    then Bet365 alone."""
    h = _pick(frame, "AvgCH", "AvgH", "B365CH", "B365H")
    d = _pick(frame, "AvgCD", "AvgD", "B365CD", "B365D")
    a = _pick(frame, "AvgCA", "AvgA", "B365CA", "B365A")
    over = _pick(frame, "AvgC>2.5", "Avg>2.5", "B365C>2.5", "B365>2.5")
    under = _pick(frame, "AvgC<2.5", "Avg<2.5", "B365C<2.5", "B365<2.5")
    inv = pd.DataFrame({"h": 1 / h, "d": 1 / d, "a": 1 / a})
    book = inv.sum(axis=1)
    probs = inv.div(book, axis=0)
    ou = pd.DataFrame({"over": 1 / over, "under": 1 / under})
    probs["over"] = ou["over"] / ou.sum(axis=1)
    return probs


_GOALS = np.arange(0, 11)


def _poisson_market(lh, la):
    ph = poisson.pmf(_GOALS, lh)
    pa = poisson.pmf(_GOALS, la)
    grid = np.outer(ph, pa)
    home = np.tril(grid, -1).sum()
    away = np.triu(grid, 1).sum()
    draw = np.trace(grid)
    total = np.add.outer(_GOALS, _GOALS)
    over = grid[total > 2.5].sum()
    return home, draw, away, over


def fit_lambdas(p_home, p_draw, p_away, p_over):
    """Expected goals for each side that best reproduce the market's four probabilities under
    independent Poisson scoring. Two unknowns, four targets -- least squares."""
    target = np.array([p_home, p_draw, p_away, p_over])
    if np.isnan(target).any():
        return np.nan, np.nan

    def loss(x):
        lh, la = np.exp(x)
        return float(np.sum((np.array(_poisson_market(lh, la)) - target) ** 2))

    result = minimize(loss, x0=np.log([1.4, 1.1]), method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-8, "maxiter": 400})
    # A handful of markets (missing over/under, extreme prices) send the fit to a degenerate corner
    # (0 or 8+ goals). No Premier League side has a real expectation outside this range.
    lh, la = np.clip(np.exp(result.x), 0.25, 4.0)
    return float(lh), float(la)


def season_odds_features(season):
    """One row per (date, team, opponent, was_home) with the ODDS_FEATURES columns, FPL team names."""
    raw = download_season(season)
    raw = raw[raw["HomeTeam"].notna()].copy()
    probs = market_probabilities(raw)
    unmapped = set(raw["HomeTeam"]) - set(TEAM_NAME_MAP)
    if unmapped:
        raise RuntimeError(f"football-data team names without an FPL mapping: {sorted(unmapped)}")
    dates = pd.to_datetime(raw["Date"], dayfirst=True).dt.strftime("%Y-%m-%d")
    rows = []
    for idx, match in raw.iterrows():
        lh, la = fit_lambdas(probs.at[idx, "h"], probs.at[idx, "d"], probs.at[idx, "a"], probs.at[idx, "over"])
        if np.isnan(lh):
            continue
        home, away = TEAM_NAME_MAP[match["HomeTeam"]], TEAM_NAME_MAP[match["AwayTeam"]]
        p_home, p_draw, p_away = probs.at[idx, "h"], probs.at[idx, "d"], probs.at[idx, "a"]
        for team, opp, was_home, own, other, win in ((home, away, True, lh, la, p_home), (away, home, False, la, lh, p_away)):
            rows.append({
                "date": dates.at[idx], "team_name": team, "opponent_name": opp, "was_home": was_home,
                "implied_team_goals": own, "implied_opp_goals": other, "implied_cs_prob": float(np.exp(-other)),
                "implied_win_prob": float(win), "implied_total_goals": own + other,
            })
    return pd.DataFrame(rows)


def attach_odds(rows, seasons):
    """Left-join odds features onto feature rows that carry date, team_name, opponent_name, was_home.
    Rows with no matching market (postponed/rescheduled dates that differ, cup weeks) stay NaN,
    which LightGBM handles natively."""
    odds = pd.concat([season_odds_features(s) for s in seasons], ignore_index=True)
    # Feature rows carry was_home as int (add_features), the odds frame as bool; merging the two
    # produces an object column LightGBM rejects. Align on int.
    odds["was_home"] = odds["was_home"].astype(int)
    rows = rows.copy()
    rows["was_home"] = rows["was_home"].astype(int)
    merged = rows.merge(odds, on=["date", "team_name", "opponent_name", "was_home"], how="left")
    matched = merged["implied_team_goals"].notna().mean()
    print(f"  odds attached to {100 * matched:.1f}% of {len(rows)} rows")
    return merged


def fetch_live_odds(api_key, fixtures, teams):
    """Pre-deadline odds for upcoming fixtures via The Odds API (https://the-odds-api.com, free tier
    500 requests/month; one request covers every EPL match). Returns the same ODDS_FEATURES shape as
    season_odds_features. Needs ODDS_API_KEY in the environment; returns an empty frame without it so
    the live model can fall back to NaN odds features rather than crash."""
    if not api_key:
        return pd.DataFrame(columns=["date", "team_name", "opponent_name", "was_home"] + ODDS_FEATURES)
    url = "https://api.the-odds-api.com/v4/sports/soccer_epl/odds/"
    params = {"apiKey": api_key, "regions": "uk", "markets": "h2h,totals", "oddsFormat": "decimal"}
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    rows = []
    name_fix = {"Manchester City": "Man City", "Manchester United": "Man Utd", "Newcastle United": "Newcastle",
                "Nottingham Forest": "Nott'm Forest", "Tottenham Hotspur": "Spurs", "Wolverhampton Wanderers": "Wolves",
                "Brighton and Hove Albion": "Brighton", "West Ham United": "West Ham", "Ipswich Town": "Ipswich Town",
                "Leeds United": "Leeds", "Coventry City": "Coventry City", "Hull City": "Hull City",
                "Leicester City": "Leicester", "Sheffield United": "Sheffield Utd", "Luton Town": "Luton"}
    for event in response.json():
        home = name_fix.get(event["home_team"], event["home_team"])
        away = name_fix.get(event["away_team"], event["away_team"])
        h = d = a = over = under = []
        for book in event.get("bookmakers", []):
            for market in book.get("markets", []):
                if market["key"] == "h2h":
                    prices = {o["name"]: o["price"] for o in market["outcomes"]}
                    if event["home_team"] in prices and event["away_team"] in prices and "Draw" in prices:
                        h = h + [prices[event["home_team"]]]; d = d + [prices["Draw"]]; a = a + [prices[event["away_team"]]]
                if market["key"] == "totals":
                    for o in market["outcomes"]:
                        if o.get("point") == 2.5:
                            (over if o["name"] == "Over" else under).append(o["price"])
        if not (h and d and a and over and under):
            continue
        frame = pd.DataFrame({"AvgH": [np.mean(h)], "AvgD": [np.mean(d)], "AvgA": [np.mean(a)], "Avg>2.5": [np.mean(over)], "Avg<2.5": [np.mean(under)]})
        probs = market_probabilities(frame).iloc[0]
        lh, la = fit_lambdas(probs["h"], probs["d"], probs["a"], probs["over"])
        date = event["commence_time"][:10]
        for team, opp, was_home, own, other, win in ((home, away, True, lh, la, probs["h"]), (away, home, False, la, lh, probs["a"])):
            rows.append({"date": date, "team_name": team, "opponent_name": opp, "was_home": was_home,
                         "implied_team_goals": own, "implied_opp_goals": other, "implied_cs_prob": float(np.exp(-other)),
                         "implied_win_prob": float(win), "implied_total_goals": own + other})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    for season in ("2025-26",):
        feats = season_odds_features(season)
        print(season, len(feats), "team-match rows")
        print(feats.head(6).round(3).to_string(index=False))
        print(feats[ODDS_FEATURES].describe().round(2).to_string())
