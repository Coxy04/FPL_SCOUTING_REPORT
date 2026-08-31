"""Retrospectively builds a GW1 squad pick for the tracking history -- NOT a genuine blind
prediction, and clearly labeled as such (retrospective: true in the saved entry, and a "RETRO"
tag wherever the dashboard shows it).

Two ways this could have been done, both flawed, chosen deliberately over the worse one:
1. Use TODAY's current-season rolling form to predict GW1. Rejected: this is circular, not just
   "informed by hindsight" -- with only 1-2 gameweeks played, GW1 IS most of what "recent form"
   currently means, so this would use GW1's own result as an input to predicting GW1's result.
2. Use only what was genuinely knowable before a ball was kicked: GW1's real fixture list
   (opponent, venue, FPL's own pre-season difficulty rating) plus LAST season's Understat player
   quality (npxg90/xa90/xgchain90) as the closest available proxy for "known coming into the
   season". All in-season rolling stats (minutes, xG, xA, bonus, team form) are correctly zero,
   since no current-season match existed yet to compute them from. This is what's implemented.

This still isn't a fully blind test -- it uses TODAY's trained model (which has since trained on
GW1 results among its current-season rows), just fed pre-season-only inputs. That residual
hindsight is real and smaller than approach 1's; it's why this is labeled retrospective rather
than presented as a genuine advance pick.

Reuses fpl_ml_model.py's own training functions and pick_team.py's own squad/scoring functions so
this can't silently drift from what either actually does.
"""
import pandas as pd
import requests
from understatapi import UnderstatClient

from fpl_ml_model import (
    DEFAULT_REST_DAYS,
    POSITION_FEATURES,
    POSITION_MAP,
    PRIOR_SEASON_UNDERSTAT,
    UNDERSTAT_SEASON,
    UNDERSTAT_TEAM_MAP,
    add_features,
    build_prior_season_rows,
    build_training_rows,
    compute_sample_weights,
    ensure_utf8_stdout,
    get_half_life_days,
    get_json,
    get_understat_player_stats,
    get_understat_team_matches,
    add_prematch_team_form,
    build_team_form_lookup,
    load_prior_season_archive,
    make_fixture_lookup,
    make_model,
    match_prior_season_players,
    match_understat_players,
)
from pick_team import HISTORY_FILE, load_history, pick_squad, save_history, score_pick

GW1_EVENT = 1


def build_gw1_rows(elements, fixtures, teams, understat_matches, prior_per90_by_understat_id):
    difficulty_lookup = make_fixture_lookup(fixtures)
    gw1_fixtures = {f["id"]: f for f in fixtures if f.get("event") == GW1_EVENT}
    rows = []
    for player in elements:
        fixture = next(
            (f for f in gw1_fixtures.values() if player["team"] in (f["team_h"], f["team_a"])), None
        )
        if fixture is None:
            continue
        was_home = player["team"] == fixture["team_h"]
        pair = difficulty_lookup[fixture["id"]]
        entry = understat_matches.get(player["id"])
        prior_stats = prior_per90_by_understat_id.get(entry["understat_id"], {}) if entry else {}
        rows.append(
            {
                "id": player["id"],
                "web_name": player["web_name"],
                "team_name": player["team_name"],
                "position": POSITION_MAP.get(player["element_type"]),
                "now_cost": player.get("now_cost", 0),  # approximation -- today's price, not GW1's actual price
                "was_home": was_home,
                "difficulty": pair[0] if was_home else pair[1],
                # No in-season match existed yet -- these are genuinely unknown pre-season, not
                # just "unavailable", so zero is the honest value, not a missing-data placeholder.
                "minutes": 0, "minutes_sd": 0, "expected_goals": 0, "expected_assists": 0,
                "expected_goals_conceded": 0, "bonus": 0,
                "team_xg_for_form": 0, "team_xg_against_form": 0,
                "opp_xg_for_form": 0, "opp_xg_against_form": 0,
                "clearances_blocks_interceptions": 0, "recoveries": 0, "tackles": 0,
                "defensive_contribution": 0,
                # Rest days genuinely means something pre-season (the summer break, not fatigue),
                # so the neutral "typical week" default is the honest value here, not 0 -- 0 would
                # read as squeezed fixture congestion, which this isn't.
                "own_days_rest": DEFAULT_REST_DAYS, "opp_days_rest": DEFAULT_REST_DAYS,
                "npxg90": prior_stats.get("npxg90", 0),
                "xa90": prior_stats.get("xa90", 0),
                "xgchain90": prior_stats.get("xgchain90", 0),
            }
        )
    return pd.DataFrame(rows)


def train_current_models(session, elements, fixtures, teams):
    understat = UnderstatClient()
    team_matches = get_understat_team_matches(understat, UNDERSTAT_SEASON)
    team_matches = add_prematch_team_form(team_matches)
    team_form_lookup = build_team_form_lookup(team_matches)

    understat_players = get_understat_player_stats(understat, UNDERSTAT_SEASON)
    understat_matches, _ = match_understat_players(elements, understat_players)

    prior_team_matches = get_understat_team_matches(understat, PRIOR_SEASON_UNDERSTAT)
    prior_team_matches = add_prematch_team_form(prior_team_matches)
    prior_team_form_lookup = build_team_form_lookup(prior_team_matches)
    prior_per90_by_understat_id = {
        p["understat_id"]: p for p in get_understat_player_stats(understat, PRIOR_SEASON_UNDERSTAT)
    }

    merged_gw, prior_difficulty_lookup, prior_team_names, player_idlist, prior_rest_days_lookup = load_prior_season_archive()
    prior_id_map, _ = match_prior_season_players(elements, player_idlist)
    prior_rows = build_prior_season_rows(
        merged_gw, prior_difficulty_lookup, prior_team_names, prior_id_map,
        prior_team_form_lookup, understat_matches, prior_per90_by_understat_id, prior_rest_days_lookup,
    )

    print("Building training data (same as fpl_ml_model.py)...")
    training = build_training_rows(elements, fixtures, teams, session, team_form_lookup, understat_matches)
    training["source"] = "fpl_live_current_season"
    training = pd.concat([training, prior_rows], ignore_index=True)
    training = add_features(training)

    models = {}
    for position in POSITION_MAP.values():
        subset = training[training["position"] == position]
        weights = compute_sample_weights(subset["date"], half_life_days=get_half_life_days(position))
        model = make_model(position)
        model.fit(subset[POSITION_FEATURES], subset["total_points"], sample_weight=weights)
        models[position] = model
        print(f"  {position}: trained on {len(subset)} rows")

    return models, understat_matches, prior_per90_by_understat_id


def main():
    history = load_history()
    if any(e["event"] == GW1_EVENT for e in history):
        print(f"GW{GW1_EVENT} already in history, nothing to do.")
        return

    session = requests.Session()
    bootstrap = get_json(session, "bootstrap-static/")
    fixtures = get_json(session, "fixtures/")
    teams = {t["id"]: t["name"] for t in bootstrap["teams"]}
    elements = []
    for player in bootstrap["elements"]:
        player = dict(player)
        player["team_name"] = teams.get(player["team"], "Unknown")
        elements.append(player)

    models, understat_matches, prior_per90 = train_current_models(session, elements, fixtures, teams)

    print("\nBuilding GW1 pre-season-only feature rows...")
    gw1_rows = build_gw1_rows(elements, fixtures, teams, understat_matches, prior_per90)
    featured = add_features(gw1_rows)
    gw1_rows["predicted_points"] = 0.0
    for position, model in models.items():
        mask = (gw1_rows["position"] == position).to_numpy()
        if mask.any():
            gw1_rows.loc[mask, "predicted_points"] = model.predict(featured.loc[mask, POSITION_FEATURES])
    gw1_rows["predicted_points"] = gw1_rows["predicted_points"].clip(0, 15)

    players = gw1_rows[["id", "web_name", "team_name", "position", "now_cost", "predicted_points"]].to_dict("records")
    squad = pick_squad(players)
    predicted_total = sum(p["predicted_points"] * (2 if p["is_captain"] else 1) for p in squad if p["is_starter"])

    entry = {
        "event": GW1_EVENT,
        "picked_at": None,
        "retrospective": True,
        "squad": squad,
        "predicted_total": round(predicted_total, 2),
        "actual_total": None,
        "scored_at": None,
    }
    print(f"\nRetrospective GW1 squad picked. Predicted starting XI total: {entry['predicted_total']}")
    scored = score_pick(entry, session)
    print(f"Scored against real GW1 results: actual {scored['actual_total']}")

    history.append(entry)
    save_history(history)
    print(f"Saved {HISTORY_FILE}")


if __name__ == "__main__":
    ensure_utf8_stdout()
    main()
