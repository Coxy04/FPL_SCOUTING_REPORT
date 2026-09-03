"""Tests whether the model's minutes signal should know which DIRECTION a player's role is moving.

Prompted by a concrete miss. Cherki came off the bench in GW1 (27 mins) and started GW2 (81 mins),
scoring 8 then 14 -- a player winning his place. The pipeline averages those two matches flat, sees
"54 mins, +/-38", and applies a 0.72 playing-time multiplier, docking 28% off his score. Back that
discount out and he'd rate alongside the players actually being recommended. The flat rolling mean
cannot tell "bench -> starter" apart from "starter -> bench": it is blind to direction.

Three candidate fixes, tested against the current flat mean on the same holdout split and tuned
per-position hyperparameters as every other ablation here:

  ewm     -- recency-weighted minutes (exponential, halflife 2 matches) instead of a flat 6-match
             mean, feeding BOTH the model feature and the playing-time multiplier. The most recent
             match counts most, so a player breaking into a side climbs faster than a flat average
             lets him.
  starts  -- add the rolling share of matches STARTED as its own feature. Minutes conflate "plays
             a lot" with "starts"; a 60-minute starter and a 60-minute super-sub look identical by
             minutes alone but are different propositions.
  both    -- both changes together.

Deliberately tested rather than assumed: the last "this should obviously help" feature (set-piece
taker flags) came back at near-zero importance, so intuition doesn't get a free pass here.
"""
import numpy as np
import pandas as pd
from understatapi import UnderstatClient

from backtest_model import build_backtest_elements
from fpl_ml_model import (
    POSITION_FEATURES,
    POSITION_MAP,
    PRIOR_SEASON_UNDERSTAT,
    TEAM_FORM_WINDOW,
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
    make_model,
    match_understat_players,
)

HOLDOUT_GAMEWEEKS = 5
# Searched rather than picked: a halflife is exactly the kind of hand-chosen constant this project
# has been burned by before. Shorter reacts faster to a changing role but trusts a single match
# more; longer is steadier but slower to notice someone winning or losing their place.
EWM_HALFLIFE_CANDIDATES = [1.0, 1.5, 2.0, 3.0, 4.0]


def build_minutes_variants(merged_gw):
    """Alternative pre-match minutes signals, computed per player in gameweek order.

    Every one is shifted by a match before any window is applied -- same leakage rule the rest of
    the pipeline follows. Using a match's own minutes to predict that match's own points would be
    the bonus-feature bug all over again."""
    gw = merged_gw.copy()
    gw["minutes"] = pd.to_numeric(gw["minutes"], errors="coerce").fillna(0)
    gw["starts"] = pd.to_numeric(gw.get("starts", 0), errors="coerce").fillna(0)
    gw = gw.sort_values(["element", "GW"]).reset_index(drop=True)

    grouped = gw.groupby("element")
    columns = ["element", "GW", "starts_roll"]
    for halflife in EWM_HALFLIFE_CANDIDATES:
        column = f"minutes_ewm_{halflife}"
        gw[column] = grouped["minutes"].transform(
            lambda s, hl=halflife: s.shift(1).ewm(halflife=hl).mean()
        )
        columns.append(column)
    gw["starts_roll"] = grouped["starts"].transform(
        lambda s: s.shift(1).rolling(TEAM_FORM_WINDOW, min_periods=1).mean()
    )
    return gw[columns].copy().fillna(0)


def evaluate(train_all, test_all, features, minutes_column):
    """MAE per position for one variant. minutes_column also drives the playing-time multiplier,
    since that discount is the mechanism that actually penalised the motivating case."""
    maes = []
    for position in POSITION_MAP.values():
        train = train_all[train_all["position"] == position]
        test = test_all[test_all["position"] == position]
        if len(train) < 30 or len(test) < 5:
            continue
        model = make_model(position)
        weights = compute_sample_weights(
            train["date"], half_life_days=get_half_life_days(position),
            as_of=pd.Timestamp(train["date"].max(), tz="UTC"),
        )
        model.fit(train[features], train["total_points"], sample_weight=weights)
        predictions = model.predict(test[features])
        multiplier = np.minimum(1.0, test[minutes_column].to_numpy() / get_playing_time_denominator(position))
        predictions = np.clip(predictions * multiplier, 0, 15)
        maes.append((position, float(np.mean(np.abs(predictions - test["total_points"].to_numpy())))))
    return maes


def main():
    understat = UnderstatClient()
    team_matches = get_understat_team_matches(understat, PRIOR_SEASON_UNDERSTAT)
    team_matches = add_prematch_team_form(team_matches)
    team_form_lookup = build_team_form_lookup(team_matches)

    merged_gw, difficulty_lookup, team_names, player_idlist, rest_days_lookup, prior_set_piece_lookup = load_prior_season_archive()
    elements = build_backtest_elements(merged_gw, player_idlist)

    understat_players = get_understat_player_stats(understat, PRIOR_SEASON_UNDERSTAT)
    understat_matches, _ = match_understat_players(elements, understat_players)
    prior_per90 = {p["understat_id"]: p for p in understat_players}
    id_map = {e: e for e in merged_gw["element"].unique()}

    rows = build_prior_season_rows(
        merged_gw, difficulty_lookup, team_names, id_map, team_form_lookup,
        understat_matches, prior_per90, rest_days_lookup, prior_set_piece_lookup,
    )
    rows = add_features(rows)

    variants = build_minutes_variants(merged_gw)
    rows = rows.merge(
        variants, left_on=["current_id", "gw"], right_on=["element", "GW"], how="left"
    )
    variant_columns = [c for c in variants.columns if c not in ("element", "GW")]
    rows[variant_columns] = rows[variant_columns].fillna(0)

    max_gw = rows["gw"].max()
    split_point = max_gw - HOLDOUT_GAMEWEEKS
    train_all = rows[rows["gw"] <= split_point]
    test_all = rows[rows["gw"] > split_point]

    print(f"Minutes-signal ablation -- same GW{split_point}/{max_gw} split and tuned "
          f"hyperparameters for every variant; only the minutes signal changes.\n")

    features_plus_starts = POSITION_FEATURES + ["starts_roll"]
    setups = [("baseline (flat mean)", POSITION_FEATURES, "minutes")]
    for halflife in EWM_HALFLIFE_CANDIDATES:
        setups.append((f"ewm halflife {halflife}", POSITION_FEATURES, f"minutes_ewm_{halflife}"))
    setups.append(("+ starts feature", features_plus_starts, "minutes"))

    results = {}
    for label, features, minutes_column in setups:
        # The ewm variants swap the value of the `minutes` feature itself, not just the multiplier
        # input -- otherwise the model would still be reading the flat mean and only the post-hoc
        # discount would change, which isn't the comparison of interest.
        train, test = train_all.copy(), test_all.copy()
        if minutes_column != "minutes":
            train["minutes"] = train[minutes_column]
            test["minutes"] = test[minutes_column]
        results[label] = dict(evaluate(train, test, features, minutes_column))

    positions = [p for p in POSITION_MAP.values() if p in next(iter(results.values()))]
    print(f"{'Variant':<22}" + "".join(f"{p:<9}" for p in positions) + f"{'Overall':<10}{'vs base':<10}")
    base_overall = None
    for label, _, _ in setups:
        maes = results[label]
        overall = float(np.mean([maes[p] for p in positions]))
        if base_overall is None:
            base_overall = overall
            delta = ""
        else:
            change = 100 * (overall - base_overall) / base_overall
            delta = f"{change:+.1f}%"
        print(f"{label:<22}" + "".join(f"{maes[p]:<9.3f}" for p in positions) + f"{overall:<10.3f}{delta:<10}")

    print("\nNegative 'vs base' means better than the current flat-mean minutes signal.")


if __name__ == "__main__":
    ensure_utf8_stdout()
    main()
