"""Tests whether the market's price/transfer trajectory predicts points beyond what the model sees.

The model already has `now_cost` -- the price LEVEL. What it has no way to express is the
DERIVATIVE: whether a player is being backed into a rise or bled out into a fall. Two different
players at 7.5m are not the same proposition if one is climbing and the other is sinking.

The honest null hypothesis, stated up front so a null result isn't rationalised away afterwards:
FPL prices rise BECAUSE a player scored. Price momentum is therefore largely a lagging echo of past
points, and the model already has past points baked into its form features (xG, xA, bonus, minutes).
On that reading price momentum adds nothing and may just add noise.

The reason to test anyway is `transfers_balance`, which is a genuinely different animal. Price
follows net transfers, so net transfers LEAD price -- and they aggregate millions of managers acting
on information the model provably cannot see: press conferences, injury updates, predicted lineups,
rotation rumours, a manager saying "he'll be assessed". That is orthogonal information, not an echo.
If anything here works, the prediction is that it is the transfer flow rather than the price move.

Variants (each added to the current feature set, everything else held identical):

  price_momentum_{n}  -- change in `value` over the last n gameweeks. Literally "price rise
                         trajectory". Searched over n rather than picked, same as the minutes
                         halflife, because a hand-chosen window is exactly what this project has
                         been burned by before.
  net_transfer_rate   -- rolling mean of transfers_balance / selected. Normalising by ownership
                         matters: 50k net transfers into a 3m-owned player is noise, into a 60k-owned
                         player it is a stampede.
  ownership           -- log(selected). Not a trajectory at all, included as a control: if the level
                         does as well as the derivative, then "trajectory" was never the useful part.
  all three           -- do they add anything on top of each other, or measure the same thing.

Leakage rule, same as everywhere else in this pipeline: every variant is shift(1)'d to strictly
past gameweeks. A tempting extra variant -- the CURRENT gameweek's transfer flow, which is
technically pre-deadline and so not leaky in the archive -- is deliberately absent, because it isn't
computable at predict time: the current window's transfers are still accumulating when the refresh
runs, so training on a full week's flow and predicting on a partial one would mean the feature
quietly means two different things. Same train/predict-parity rule that governs the minutes EWM.
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
PRICE_MOMENTUM_WINDOWS = [2, 4, 6]
# transfers_balance/selected is a ratio with a small, noisy denominator: a 200-owned player with 400
# net transfers in is a 2.0 that means nothing. Clipped rather than dropped so those rows still
# contribute their other features.
TRANSFER_RATE_CLIP = 0.5
MIN_SELECTED = 1000


def build_price_variants(merged_gw):
    """Price/ownership trajectory signals, per player in gameweek order, all strictly lagged."""
    gw = merged_gw.copy()
    for column in ("value", "transfers_balance", "selected"):
        gw[column] = pd.to_numeric(gw[column], errors="coerce").fillna(0)
    gw = gw.sort_values(["element", "GW"]).reset_index(drop=True)
    grouped = gw.groupby("element")

    columns = ["element", "GW"]
    for window in PRICE_MOMENTUM_WINDOWS:
        column = f"price_momentum_{window}"
        # diff(window) on the shifted series: the price move over the n gameweeks ENDING with the
        # last completed one, so the current gameweek's own price never enters.
        gw[column] = grouped["value"].transform(
            lambda s, w=window: s.shift(1).diff(w)
        )
        columns.append(column)

    gw["_rate"] = gw["transfers_balance"] / gw["selected"].clip(lower=MIN_SELECTED)
    gw["net_transfer_rate"] = grouped["_rate"].transform(
        lambda s: s.shift(1).rolling(TEAM_FORM_WINDOW, min_periods=1).mean()
    ).clip(-TRANSFER_RATE_CLIP, TRANSFER_RATE_CLIP)
    gw["ownership"] = grouped["selected"].transform(
        lambda s: np.log1p(s.shift(1))
    )
    columns += ["net_transfer_rate", "ownership"]
    return gw[columns].copy().fillna(0)


def evaluate(train_all, test_all, features):
    """MAE per position for one feature set, on the same split and tuned hyperparameters."""
    maes = {}
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
        multiplier = np.minimum(1.0, test["minutes"].to_numpy() / get_playing_time_denominator(position))
        predictions = np.clip(predictions * multiplier, 0, 15)
        maes[position] = float(np.mean(np.abs(predictions - test["total_points"].to_numpy())))
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

    variants = build_price_variants(merged_gw)
    rows = rows.merge(
        variants, left_on=["current_id", "gw"], right_on=["element", "GW"], how="left"
    )
    variant_columns = [c for c in variants.columns if c not in ("element", "GW")]
    rows[variant_columns] = rows[variant_columns].fillna(0)

    max_gw = rows["gw"].max()
    split_point = max_gw - HOLDOUT_GAMEWEEKS
    train_all = rows[rows["gw"] <= split_point]
    test_all = rows[rows["gw"] > split_point]

    print(f"Price-trajectory ablation -- same GW{split_point}/{max_gw} split and tuned "
          f"hyperparameters for every variant; only the feature set changes.")
    print(f"{len(train_all)} train rows, {len(test_all)} holdout rows.\n")

    setups = [("baseline", POSITION_FEATURES)]
    for window in PRICE_MOMENTUM_WINDOWS:
        setups.append((f"+ price_momentum_{window}", POSITION_FEATURES + [f"price_momentum_{window}"]))
    setups.append(("+ net_transfer_rate", POSITION_FEATURES + ["net_transfer_rate"]))
    setups.append(("+ ownership (control)", POSITION_FEATURES + ["ownership"]))
    best_window = PRICE_MOMENTUM_WINDOWS[len(PRICE_MOMENTUM_WINDOWS) // 2]
    setups.append((
        "+ all three",
        POSITION_FEATURES + [f"price_momentum_{best_window}", "net_transfer_rate", "ownership"],
    ))

    results = {label: evaluate(train_all, test_all, features) for label, features in setups}

    positions = [p for p in POSITION_MAP.values() if p in results["baseline"]]
    print(f"{'Variant':<24}" + "".join(f"{p:<9}" for p in positions) + f"{'Overall':<10}{'vs base':<10}")
    base_overall = None
    for label, _ in setups:
        maes = results[label]
        overall = float(np.mean([maes[p] for p in positions]))
        if base_overall is None:
            base_overall = overall
            delta = ""
        else:
            delta = f"{100 * (overall - base_overall) / base_overall:+.1f}%"
        print(f"{label:<24}" + "".join(f"{maes[p]:<9.3f}" for p in positions) + f"{overall:<10.3f}{delta:<10}")

    print("\nNegative 'vs base' means better than the current feature set.")
    print("Expect small numbers either way: a 29-feature model is unlikely to be moved much by one "
          "more column, and anything under ~1% is inside the noise this split can resolve.")

    # A single holdout cannot distinguish a real 0.3% gain from a lucky one, and 0.3% is small
    # enough that the question is entirely whether it's consistent. Re-run the shortlist across
    # several split points: a genuine signal helps at most of them, noise changes sign.
    print("\nConsistency check across rolling split points -- overall MAE vs baseline at each:\n")
    shortlist = [
        ("price_momentum_4", POSITION_FEATURES + ["price_momentum_4"]),
        ("net_transfer_rate", POSITION_FEATURES + ["net_transfer_rate"]),
        ("ownership (control)", POSITION_FEATURES + ["ownership"]),
    ]
    split_points = [max_gw - n for n in (10, 8, 6, 5, 4)]
    print(f"{'Variant':<24}" + "".join(f"{'GW' + str(s):<9}" for s in split_points) + f"{'Mean':<9}{'Wins':<6}")
    for label, features in shortlist:
        deltas = []
        for split in split_points:
            train = rows[rows["gw"] <= split]
            test = rows[(rows["gw"] > split) & (rows["gw"] <= split + HOLDOUT_GAMEWEEKS)]
            base = evaluate(train, test, POSITION_FEATURES)
            variant = evaluate(train, test, features)
            shared = [p for p in base if p in variant]
            base_mae = float(np.mean([base[p] for p in shared]))
            variant_mae = float(np.mean([variant[p] for p in shared]))
            deltas.append(100 * (variant_mae - base_mae) / base_mae)
        wins = sum(1 for d in deltas if d < 0)
        print(f"{label:<24}" + "".join(f"{d:<+9.1f}" for d in deltas)
              + f"{np.mean(deltas):<+9.1f}{wins}/{len(deltas):<6}")

    print("\nA feature worth adding should be negative at most split points, not just on average -- "
          "one big win and four losses is noise wearing a good average.")


if __name__ == "__main__":
    ensure_utf8_stdout()
    main()
