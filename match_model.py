"""Match-result model: win/draw/loss, expected goals, scorelines and clean-sheet odds per fixture.

Team attack and defence ratings are exponentially-weighted means of xG for and against over past
matches, shrunk toward the league mean (so three games of data don't produce a wild rating), with
a fitted home-advantage factor and a Dixon-Coles low-score correction. Expected goals for a fixture
are home_attack x away_defence / league_mean x home_factor (and the mirror), then a Poisson grid
gives every scoreline.

Validation is walk-forward on two archived seasons against the bookmakers' closing 1X2 odds
(odds_features.py), scored on log loss and Brier. Parameters are fitted on 2024-25 and tested on
2025-26, never on the season they're scored on. Expect to approach the market, not beat it: closing
odds include lineups and news this model never sees. If the gap is small, the team-strength inputs
the player model relies on are sound; if it's large, they aren't -- either way it's a cleaner scorecard
than weekly player points.

Live: reads fpl_ml_team_matches.csv (written by fpl_ml_model.py from FPL's own per-match xG) and
the FPL fixture list, and writes fpl_ml_match_predictions.json for the next MAX_FUTURE_GAMEWEEKS.
"""
import itertools
import json
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
from scipy.stats import poisson

from fpl_ml_model import (
    BASE_URL, MAX_FUTURE_GAMEWEEKS, TEAM_MATCHES_FILE, ensure_utf8_stdout, load_prior_season_archive, team_matches_from_archive,
)

OUTPUT_FILE = "fpl_ml_match_predictions.json"
VALIDATION_FILE = "fpl_ml_match_model_validation.json"
PARAMS_FILE = "fpl_ml_match_model_params.json"
GOALS = np.arange(0, 10)
DEFAULT_PARAMS = {"half_life": 8.0, "shrink": 4.0, "rho": -0.05}
GRID = {"half_life": [6, 10, 15, 20, 30, 45], "shrink": [1, 2, 4, 6], "rho": [-0.18, -0.14, -0.10, -0.06, 0.0]}


def fixtures_from_team_matches(tm):
    """One row per fixture from the two team rows: date, home, away, xg_h, xg_a, goals (NaN if unplayed)."""
    home = tm[tm["was_home"]].rename(columns={"team": "home", "opponent": "away", "xg_for": "xg_h", "xg_against": "xg_a", "goals_for": "hg", "goals_against": "ag"})
    return home[["fixture", "date", "home", "away", "xg_h", "xg_a", "hg", "ag"]].sort_values("date").reset_index(drop=True)


class Ratings:
    """Walk-forward team ratings. Feed matches in date order; ask for expected goals before feeding
    the match itself."""

    def __init__(self, half_life, shrink, mean_xg, home_factor, away_factor):
        self.decay = 0.5 ** (1.0 / half_life)
        self.shrink = shrink
        self.mean = mean_xg
        self.home_factor, self.away_factor = home_factor, away_factor
        self.att = {}  # team -> (weighted sum xg_for, weight)
        self.dfn = {}

    def _rating(self, store, team):
        s, w = store.get(team, (0.0, 0.0))
        return (s + self.shrink * self.mean) / (w + self.shrink)

    def expected(self, home, away):
        lam_h = self._rating(self.att, home) * self._rating(self.dfn, away) / self.mean * self.home_factor
        lam_a = self._rating(self.att, away) * self._rating(self.dfn, home) / self.mean * self.away_factor
        return max(lam_h, 0.05), max(lam_a, 0.05)

    def update(self, team, xg_for, xg_against):
        for store, value in ((self.att, xg_for), (self.dfn, xg_against)):
            s, w = store.get(team, (0.0, 0.0))
            store[team] = (s * self.decay + value, w * self.decay + 1.0)


def scoreline_grid(lam_h, lam_a, rho):
    ph, pa = poisson.pmf(GOALS, lam_h), poisson.pmf(GOALS, lam_a)
    grid = np.outer(ph, pa)
    # Dixon-Coles adjustment to the four low-scoring cells.
    tau = {(0, 0): 1 - lam_h * lam_a * rho, (1, 0): 1 + lam_a * rho, (0, 1): 1 + lam_h * rho, (1, 1): 1 - rho}
    for (h, a), t in tau.items():
        grid[h, a] *= max(t, 0.0)
    grid /= grid.sum()
    return grid


def summarise(grid):
    home = float(np.tril(grid, -1).sum()); away = float(np.triu(grid, 1).sum()); draw = float(np.trace(grid))
    total = np.add.outer(GOALS, GOALS)
    h, a = np.unravel_index(int(grid.argmax()), grid.shape)
    return {"p_home": home, "p_draw": draw, "p_away": away, "p_over25": float(grid[total > 2.5].sum()),
            "p_cs_home": float(grid[:, 0].sum()), "p_cs_away": float(grid[0, :].sum()),
            "likely_score": f"{h}-{a}", "p_likely_score": float(grid[h, a])}


def walk_forward(fixtures, params, prior_fixtures=None, start_index=0):
    """Predict every fixture from matches strictly before it. prior_fixtures (previous season) seed
    the ratings; start_index skips the first few rounds of the scored season from evaluation."""
    played = fixtures.dropna(subset=["hg", "ag"])
    seed = prior_fixtures.dropna(subset=["hg", "ag"]) if prior_fixtures is not None else played.iloc[0:0]
    hist = pd.concat([seed, played], ignore_index=True)
    mean_xg = float(pd.concat([hist["xg_h"], hist["xg_a"]]).mean())
    home_factor = float(hist["hg"].mean() / ((hist["hg"].mean() + hist["ag"].mean()) / 2))
    away_factor = float(hist["ag"].mean() / ((hist["hg"].mean() + hist["ag"].mean()) / 2))
    r = Ratings(params["half_life"], params["shrink"], mean_xg, home_factor, away_factor)
    for m in seed.itertuples():
        r.update(m.home, m.xg_h, m.xg_a); r.update(m.away, m.xg_a, m.xg_h)
    out = []
    for i, m in enumerate(played.itertuples()):
        lam_h, lam_a = r.expected(m.home, m.away)
        if i >= start_index:
            s = summarise(scoreline_grid(lam_h, lam_a, params["rho"]))
            out.append({"fixture": m.fixture, "date": m.date, "home": m.home, "away": m.away, "lam_h": lam_h, "lam_a": lam_a, "hg": m.hg, "ag": m.ag, **s})
        r.update(m.home, m.xg_h, m.xg_a); r.update(m.away, m.xg_a, m.xg_h)
    return pd.DataFrame(out), r


def outcome(hg, ag):
    return 0 if hg > ag else (1 if hg == ag else 2)


def score(frame, cols=("p_home", "p_draw", "p_away")):
    p = frame[list(cols)].to_numpy(); y = np.array([outcome(h, a) for h, a in zip(frame["hg"], frame["ag"])])
    onehot = np.eye(3)[y]
    return {"log_loss": float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1)).mean()), "brier": float(((p - onehot) ** 2).sum(axis=1).mean()),
            "accuracy": float((p.argmax(axis=1) == y).mean()), "n": int(len(y))}


def bookmaker_frame(season):
    from odds_features import download_season, market_probabilities, TEAM_NAME_MAP
    raw = download_season(season); raw = raw[raw["HomeTeam"].notna()].copy()
    probs = market_probabilities(raw)
    return pd.DataFrame({"date": pd.to_datetime(raw["Date"], dayfirst=True).dt.strftime("%Y-%m-%d"),
                         "home": raw["HomeTeam"].map(TEAM_NAME_MAP), "away": raw["AwayTeam"].map(TEAM_NAME_MAP),
                         "b_home": probs["h"].values, "b_draw": probs["d"].values, "b_away": probs["a"].values, "b_over": probs["over"].values})


def archive_fixtures(season):
    mg, _, team_names, *_ = load_prior_season_archive(season)
    return fixtures_from_team_matches(team_matches_from_archive(mg, team_names))


def validate():
    print("Building fixtures from the 2023-24, 2024-25 and 2025-26 archives...")
    f23, f24, f25 = archive_fixtures("2023-24"), archive_fixtures("2024-25"), archive_fixtures("2025-26")
    # Fit on 2024-25 (seeded by 2023-24), scoring from match 40 onward (roughly GW5).
    best, best_ll = None, 9e9
    for hl, sh, rho in itertools.product(GRID["half_life"], GRID["shrink"], GRID["rho"]):
        params = {"half_life": hl, "shrink": sh, "rho": rho}
        pred, _ = walk_forward(f24, params, prior_fixtures=f23, start_index=40)
        ll = score(pred)["log_loss"]
        if ll < best_ll:
            best, best_ll = params, ll
    print(f"Fitted on 2024-25: {best} (log loss {best_ll:.4f})")
    # Test on 2025-26 with the 2024-25-fitted parameters, seeded by 2024-25.
    pred, _ = walk_forward(f25, best, prior_fixtures=f24, start_index=40)
    book = bookmaker_frame("2025-26")
    merged = pred.merge(book, on=["date", "home", "away"], how="inner")
    model = score(merged); market = score(merged, ("b_home", "b_draw", "b_away"))
    const = merged.copy(); const["c_home"], const["c_draw"], const["c_away"] = 0.45, 0.25, 0.30
    naive = score(const, ("c_home", "c_draw", "c_away"))
    blend = merged.copy()
    for c in ("home", "draw", "away"):
        blend[f"m_{c}"] = 0.5 * merged[f"p_{c}"] + 0.5 * merged[f"b_{c}"]
    blended = score(blend, ("m_home", "m_draw", "m_away"))
    y_over = (merged["hg"] + merged["ag"] > 2.5).astype(int)
    ou_model = float(-np.mean(y_over * np.log(np.clip(merged["p_over25"], 1e-9, 1)) + (1 - y_over) * np.log(np.clip(1 - merged["p_over25"], 1e-9, 1))))
    ou_market = float(-np.mean(y_over * np.log(np.clip(merged["b_over"], 1e-9, 1)) + (1 - y_over) * np.log(np.clip(1 - merged["b_over"], 1e-9, 1))))
    cs_h = merged["ag"] == 0
    cs_cal = {"p_cs_home_mean": float(merged["p_cs_home"].mean()), "cs_home_rate": float(cs_h.mean())}
    agree = float((merged[["p_home", "p_draw", "p_away"]].to_numpy().argmax(axis=1) == merged[["b_home", "b_draw", "b_away"]].to_numpy().argmax(axis=1)).mean())
    print(f"\n2025-26 test, {model['n']} matches with bookmaker odds (from ~GW5):")
    print(f"  {'':<22}{'log loss':>10}{'Brier':>9}{'accuracy':>10}")
    for name, s in (("xG model", model), ("bookmakers (closing)", market), ("50/50 blend", blended), ("constant 45/25/30", naive)):
        print(f"  {name:<22}{s['log_loss']:>10.4f}{s['brier']:>9.4f}{100*s['accuracy']:>9.1f}%")
    print(f"  over/under 2.5 log loss: model {ou_model:.4f} vs market {ou_market:.4f}")
    print(f"  home clean-sheet: predicted {100*cs_cal['p_cs_home_mean']:.1f}% vs actual {100*cs_cal['cs_home_rate']:.1f}%")
    print(f"  picks the same favourite as the market {100*agree:.0f}% of the time")
    out = {"fitted_params": best, "fit_log_loss_2024_25": best_ll, "test_2025_26": {"model": model, "market": market, "blend": blended, "naive": naive,
           "ou_log_loss_model": ou_model, "ou_log_loss_market": ou_market, "cs_calibration": cs_cal, "favourite_agreement": agree}}
    with open(VALIDATION_FILE, "w") as f:
        json.dump(out, f, indent=2)
    with open(PARAMS_FILE, "w") as f:
        json.dump(best, f, indent=2)
    print(f"Saved {VALIDATION_FILE}, {PARAMS_FILE}")
    return best


def predict_live(params):
    tm = pd.read_csv(TEAM_MATCHES_FILE)
    tm["was_home"] = tm["was_home"].astype(bool)
    current = fixtures_from_team_matches(tm)
    prior = archive_fixtures("2025-26")
    _, ratings = walk_forward(current, params, prior_fixtures=prior, start_index=10 ** 6)
    bootstrap = requests.get(f"{BASE_URL}/bootstrap-static/", timeout=60).json()
    teams = {t["id"]: t["name"] for t in bootstrap["teams"]}
    short = {t["id"]: t["short_name"] for t in bootstrap["teams"]}
    fixtures = requests.get(f"{BASE_URL}/fixtures/", timeout=60).json()
    upcoming_events = sorted({f["event"] for f in fixtures if f.get("event") and not f.get("finished_provisional")})[:MAX_FUTURE_GAMEWEEKS]
    rows = []
    for f in fixtures:
        if f.get("event") not in upcoming_events or f.get("finished_provisional"):
            continue
        home, away = teams[f["team_h"]], teams[f["team_a"]]
        lam_h, lam_a = ratings.expected(home, away)
        s = summarise(scoreline_grid(lam_h, lam_a, params["rho"]))
        rows.append({"event": f["event"], "kickoff": f.get("kickoff_time"), "home": home, "away": away, "home_short": short[f["team_h"]], "away_short": short[f["team_a"]],
                     "xg_home": round(lam_h, 2), "xg_away": round(lam_a, 2), **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()}})
    strength = sorted(((t, ratings._rating(ratings.att, t), ratings._rating(ratings.dfn, t)) for t in teams.values()), key=lambda x: -(x[1] - x[2]))
    out = {"generated_at": datetime.now(timezone.utc).isoformat(), "params": params, "fixtures": rows,
           "team_ratings": [{"team": t, "attack_xg": round(a, 2), "defence_xg_against": round(d, 2)} for t, a, d in strength]}
    with open(OUTPUT_FILE, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nNext gameweek ({upcoming_events[0] if upcoming_events else '?'}):")
    for r in sorted((r for r in rows if r["event"] == upcoming_events[0]), key=lambda r: r["kickoff"] or ""):
        print(f"  {r['home_short']} v {r['away_short']}: {100*r['p_home']:.0f}/{100*r['p_draw']:.0f}/{100*r['p_away']:.0f}  xG {r['xg_home']}-{r['xg_away']}  likely {r['likely_score']}  CS {100*r['p_cs_home']:.0f}%/{100*r['p_cs_away']:.0f}%")
    print(f"Saved {OUTPUT_FILE}")


def main():
    if "--validate" in sys.argv or not __import__("os").path.exists(PARAMS_FILE):
        params = validate()
    else:
        params = json.load(open(PARAMS_FILE))
    if "--validate-only" not in sys.argv:
        predict_live(params)


if __name__ == "__main__":
    ensure_utf8_stdout()
    main()
