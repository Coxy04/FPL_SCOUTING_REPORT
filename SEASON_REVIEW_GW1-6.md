# Season review, GW1-6: why the system isn't delivering, and what to change

Written 2026-10-15, after GW6. Everything below was measured against the real FPL API, the three archived
prediction snapshots (`prediction_history/gw03-05.csv.gz`), the model team history, and fresh holdout runs on
last season's archive. Scratch scripts used for the measurements are listed at the end.

---

## Headline

- The model is good at the easy half of the problem (who plays) and has **no measurable live skill at the hard
  half** (which starter scores big this week). Among players you'd actually start, its weekly ranking in GW3-5 was
  indistinguishable from a coin flip.
- Where it stuck its neck out, it was wrong for an identifiable reason: it captained **Fernandes three weeks running
  (2, 2, 2 pts)** over Haaland because the **FWD model is under-boosted and structurally cannot predict a 7**, and
  because price, minutes and set-piece flags from last season outweigh this season's xG.
- The tuning loop is **optimising the wrong number**: MAE on a heavy-tailed target rewards shrinking everyone toward
  the mean, which is exactly what destroys the top of the ranking. The FWD problem is a direct product of this.
- The league gap is mostly **captaincy** (27 pts from the armband vs 45 for "Haaland every week"), **bench points**
  (54 left on the bench), one **blank-everything week** (GW5, 34 pts), and a **missed refresh** (no model run at all
  for GW6). Six gameweeks is also a tiny, noisy sample.
- Three cheap "obvious" fixes were tested and **do not help**: Poisson objective, points-form features, a pooled
  model. One does: more boosting rounds for FWD. The real fix is changing what the tuner scores.

---

## 1. Where the points went

Nathan vs the Brackley Invitational, GW1-6 (captain pts = extra points earned by the armband, chips included):

| Rank | Team | Total | Captain pts | Bench pts | Chips | Captains GW1-6 |
|---|---|---|---|---|---|---|
| 1 | Bielsa Boys | 447 | 57 | 75 | TC@3 | Haaland, Haaland, Haaland(x3), Joao Pedro, Haaland, Haaland |
| 2 | Groove Kamada | 421 | 49 | 21 | - | Haaland, Mbeumo, Haaland, Haaland, Haaland, Joao Pedro |
| 3 | HaalandGlobeTrotters | 415 | 59 | 63 | FH@3 | Havertz, Fernandes, Haaland, Haaland, Haaland, Haaland |
| 4 | Bad to Wirtz | 414 | 44 | 62 | - | Haaland x3, Joao Pedro, Haaland, Barnes |
| 5 | The xG Files | 413 | 52 | 38 | BB@1, WC@3, FH@5 | Joao Pedro, Havertz, Haaland x3, De Cuyper |
| 10 | **Lord Farkesquad** | **355** | **36** | **54** | **TC@3, WC@4** | **Haaland, Mbeumo, Haaland(x3), Isak, Gibbs-White, Fernandes** |
| 11 | Lord of the Ings | 352 | 30 | 65 | - | Haaland, Szoboszlai, Szoboszlai, Haaland, Haaland, Haaland |

- Overall rank 4.77M (about 43rd percentile). Season total 355 vs the game average of 357. So "average overall,
  bottom of a strong league", not "bottom of the game".
- 92 points behind the leader. Roughly where it comes from:
  - **Captaincy: -18 to -23.** Nathan's armband earned 27 base pts (36 with the TC). Captaining Haaland every week
    would have earned 45 base. The top five in the league did almost exactly that.
  - **Bench: 54 pts left on the bench**, 15 of them in GW2 and 15 in GW4 (Davis 14 benched while Konsa, predicted
    1.3, started). The model's own lineup from his GW4 squad started Davis.
  - **GW5: 34 pts** vs a 48 average. Ten of eleven starters scored 4 or fewer. The model predicted 58 for that XI.
    That's variance, not a decision.
  - **GW1: 46** vs 50, with Haaland (C) on 2. Everyone in the league ate that one.
- Nathan's weekly totals: 46, 89, 63, 63, 34, 60. Game averages: 50, 81, 51, 69, 48, 58. Two good weeks, two bad,
  two average. He is not consistently below average; the league's top five are consistently above it.

## 2. The captain calls, one by one

| GW | Nathan's captain | Model's top overall | Model's top in his XI | Haaland | Best in his XI (hindsight) |
|---|---|---|---|---|---|
| 1 | Haaland 2 | no run | no run | 2 | Joao Pedro 11 |
| 2 | Mbeumo 11 | Haaland 13 | - | 13 | Fernandes 23 |
| 3 | Haaland 9 (TC = 27) | **Fernandes 2** | Fernandes 2 | 9 | Haaland 9 |
| 4 | Isak 2 | **Fernandes 2** | Fernandes 2 | 9 | Gvardiol 11 |
| 5 | Gibbs-White 2 | Haaland 6 | **Fernandes 2** | 6 | Isak 8 |
| 6 | Fernandes 1 | **no run** | no run | 6 | Joao Pedro 12 |

- Over GW2-5 the model's overall top pick earned 23 pts, Nathan's picks 24, Haaland 37.
- In GW3 Nathan overrode the model (Haaland over Fernandes) and was right. In GW4 and GW5 his pick and the model's
  pick scored the same 2. In GW6 there was no model: the last refresh was 17 Sept, the GW6 deadline was 10 Oct.
- Following the model's own XI and captain from his own 15 in GW3-5 would have scored **8 points fewer** than what
  he actually did. So "he didn't follow the model" is not the story. The model's captain calls were wrong whether he
  followed them or not.
- Fernandes this season: 2, 23, 2, 2, 2, 1. xG per match after GW2: 0.15, 0.03, 0.28, 0.02. The model still rated
  him top in GW5 (6.51) with these contributions: `now_cost +1.08`, `minutes +0.94`, `is_freekick_taker +0.36`,
  `expected_goals +0.35` (that last one still carried GW2's 2.02 xG in the 6-match window). Price and minutes, not
  current output.

## 3. What the model can and can't do, measured

### Live, GW3-5 snapshots vs real points

| | GW3 | GW4 | GW5 |
|---|---|---|---|
| Spearman, all 626-659 players | 0.78 | 0.70 | 0.76 |
| Spearman, players who played | 0.34 | 0.32 | 0.42 |
| **Spearman, startable (pred >= 3, n~120-130)** | **0.05** | **0.12** | **-0.02** |
| Spearman within the top-30 predicted (pooled GW3-5) | 0.08 | | |
| Price alone within top-30 (pooled GW3-5) | 0.25 | | |
| 80% interval coverage, players who played | 70% | 64% | 68% |
| Top-10 predicted: mean pred / mean actual | 5.9 / 4.7 | 5.3 / 4.2 | 5.6 / 4.0 |

- The high overall Spearman is the minutes signal: separating 0-minute players from starters. That is real and
  useful for transfers and bench order. It's worth nothing for captaincy.
- Within the pool that matters, ranking skill is zero. Price alone ranked the top-30 better.
- Decile calibration is fine in the body of the distribution, but the top decile and premiums (>= GBP 9.0m) are
  over-predicted (5.6 predicted vs 4.9 actual) and the "80%" intervals cover 64-70%. The conformal margin was
  calibrated on last season and doesn't transfer.
- Position bias among players who played (predicted / actual mean): DEF 2.58/3.36, MID 2.25/2.80, **FWD 1.96/2.76**,
  GK 2.88/4.03. Forwards and keepers are under-predicted most. GK live ranking skill: 0.04.

### Last season's archive, held-out GW29-38 (two folds)

| Signal | Spearman all | Spearman startable | Spearman top-30 | Captain argmax, mean pts over 10 GWs |
|---|---|---|---|---|
| Model | 0.340 | 0.146 | 0.139 | 5.7 |
| Last-6 points per game | 0.270 | 0.092 | 0.103 | 6.4 |
| Season points per game | 0.288 | 0.099 | | 5.5 |
| Price | 0.088 | 0.038 | 0.018 | 4.8 |

- On clean holdout the model does beat naive form among startable players, but by 0.05 of Spearman. Live, that edge
  has not shown up yet.
- Nobody's argmax reliably finds the top scorer. A 5-6 pt average from the captain pick is what a good signal looks
  like week to week; the gains are in expectation over a season, not in weekly "hits".

### How noisy is the target

- Lag-1 correlation of a player's points week to week: 0.13 (Spearman 0.20). 63% of scores by players who played
  were 2 or less; 10% were 8 or more. SD of actual points 3.0 vs SD of predictions 1.6.
- A prediction model that compresses toward the mean is doing the statistically right thing for MAE. It is doing the
  wrong thing for "who do I captain".

## 4. Root causes in the code

### A. The FWD model is under-boosted, so no forward can ever be the captain pick

- `POSITION_MODEL_PARAMS["FWD"]` is 100 trees at learning rate 0.02 ([fpl_ml_model.py:141](fpl_ml_model.py#L141)).
  With L2 boosting that leaves roughly 0.98^100 = 13% of the signal un-fitted; the model is shrunk toward the mean by
  construction. MID gets 150 x 0.02 (5% left), DEF and GK are fully fitted.
- Measured on the archive holdout: **the maximum FWD prediction in 10 held-out gameweeks is 4.92**. Startable
  forwards are under-predicted in both folds (3.69 vs 4.50 actual; 3.70 vs 5.09 in the later fold) and live (1.96 vs
  2.76). Only 7 of 150 top-30 slots went to forwards.
- Giving FWD 400 trees at lr 0.04: bias gone (4.14 vs 4.33), max FWD prediction 6.5, forwards take 32 top-30 slots,
  captain argmax over the 10 holdout GWs goes from 5.4 to 8.5 pts. MAE gets worse by 0.04 (1.99 -> 2.03), which is
  exactly why `tune_model.py` rejected it.
- This is why Haaland (xG 0.74-1.09 every week, 45 pts, most-captained in the game every week) sat below Fernandes
  for three straight weeks.

### B. The tuner and backtest score MAE, which selects for shrink-to-mean

- `tune_model.score_params` and `backtest_model.evaluate_position` both judge on MAE
  ([tune_model.py:96-116](tune_model.py#L96-L116), [backtest_model.py:88-100](backtest_model.py#L88-L100)). On a
  target where 63% of outcomes are <= 2, MAE is minimised by predicting near the median for everyone. Any setting
  that lets the model commit to a high prediction (more trees, higher learning rate, less regularisation) is
  penalised when the big score doesn't land.
- Every ablation in the repo (per-player models, recency weighting, pooled vs per-position, minutes half-life,
  now_cost) was judged on MAE too. Conclusions drawn on MAE are conclusions about the body of the distribution, not
  the top, and the top is the only thing captaincy and transfers care about.
- The 1% improvement gate means a correct-but-MAE-neutral change can never be adopted by `--apply`.

### C. Static last-season priors outweigh current form

- 9,318 of 10,554 training rows (88%) are last season. `half_life_days` is 100,000 (recency weighting off) for GK,
  MID and FWD. `now_cost` is the 2nd most important feature for MID and FWD (11-14% gain) and is itself FPL's own
  encoding of last season's points.
- `tune_model.py` concluded recency weighting doesn't help, but it can only test *within* one season (train GW1-28,
  test GW29-33). It has never tested the deployment situation (last season + a few current gameweeks, predict the
  next), so it cannot say anything about cross-season drift, which is precisely the Fernandes failure.
- The current-season Understat per-90s (`npxg90` etc.) are season aggregates applied to every row, including
  training rows from before those matches happened. Checked: removing all six per-90 features changes holdout
  Spearman from 0.340 to 0.343, so they are inert rather than leaking materially. They just aren't doing anything.

### D. Captaincy compares four separately-calibrated models

- Captain = argmax of predicted mean across the XI (`pick_squad`, `pick_best_lineup`, `pick_with_transfers`). That's
  only valid if the GK, DEF, MID and FWD models are calibrated against each other. Section A shows they aren't, and
  a per-position linear recalibration fitted on one fold did not transfer to the next (DEF bias flipped sign), so
  the fix has to be in the models, not a post-hoc constant.

### E. The squad optimiser ignores the captain multiplier

- `pick_squad` ([pick_team.py:60-63](pick_team.py#L60-L63)) and `pick_with_transfers`
  ([fetch_my_team.py:360-366](fetch_my_team.py#L360-L366)) maximise the plain sum of starters' predicted points. The
  captain is chosen afterwards. A GBP 15.5m player predicted 6.5 is therefore valued at 6.5, not 13, and the MILP
  prefers three GBP 5-8m players at 5 each. This structurally undervalues the one asset the whole league is winning
  with. The weekly-captaincy correction added on 11 Sept only re-ranks wildcard candidates after the search.
- Nathan's GW4 wildcard sold Haaland. The model's own squads (GW4 pick, GW5 `wildcard_now_squad`) kept him, so
  that specific call was Nathan's. But with E in place the tool was never going to argue hard for him either.

### F. The "captain ceiling" display carries no information

- The 80% bands for the top captain options in GW5 were Haaland 0.1-13.7, Fernandes 0.0-12.7, Gibbs-White 0.0-12.8.
  Every band spans the whole plausible range, so "highest ceiling" is noise. The FWD low quantile is literally 0.0.

### G. GK model

- Zero live ranking skill (0.04), under-predicts keepers by 1.1 pts, and the retrospective GW1 pick captained a
  goalkeeper (Steele, 0 minutes). Lowest priority because keepers rarely decide a week, but it's broken.

### H. Process

- **No GW6 refresh.** Last run 17 Sept; GW6 deadline 10 Oct. The GW6 captain call (Fernandes, 1 pt) had no model
  behind it. The deadline lock works; nothing reminds you to run before the next one.
- **Wildcard built on a stale model.** Wildcard made 12 Sept off a 7 Sept run; no `my_fpl_team.json` existed for the
  new squad until 17 Sept.
- **Lineups didn't use the tool.** 54 bench points. The model's XI from his own 15 would have added Davis/Ajayi for
  Konsa/Gvardiol in GW4-5 (net +1 base point over GW3-5, but +12 in GW4 alone from Davis).

## 5. Honest sizing

- **Model:** wrong captain for a structural reason three weeks running; otherwise at the noise floor among starters.
  Cost vs "Haaland every week": about 14 captain pts over GW2-5.
- **Usage:** 54 bench pts, a missed refresh, a wildcard without Haaland. Cost: order of 20-30 pts.
- **Variance:** GW5 alone is -14 vs average on a squad predicted 58. Six gameweeks is not enough to separate a
  2-pt/week edge from noise (weekly SD of a team score is ~15).
- The top of the league is not using a better model. They are captaining the most-captained player in the game
  and benching less.

## 6. What to do, ranked, each with the test that decides it

1. **Change what the tuner scores.** Replace MAE in `tune_model.score_params` and `backtest_model.evaluate_position`
   with a composite that reflects use: Spearman among startable (pred >= 3), mean actual points of the top-10
   predicted per gameweek, and a per-position bias term (mean pred - mean actual among startable, penalised).
   Report MAE alongside, don't optimise it. Test: re-run tuning; the FWD result in A should fall out of it.
2. **Fix FWD now** (`n_estimators` 400, `learning_rate` 0.04) as a stopgap until 1 lands. Measured: bias removed,
   captain argmax 5.4 -> 8.5 over 10 holdout GWs, MAE +0.04. Then check that Haaland ranks above Fernandes live.
3. **Put the captain inside the MILP.** Add a binary `captain_i <= starter_i`, `sum(captain) == 1`, objective
   `+ sum(captain_i * pred_i)`. Linear, cheap, fixes E for squad, transfer and lineup searches at once. Test: re-run
   the GW5 wildcard options and transfer plans; expect premiums to appear more and the MILP to stop spreading money.
4. **Add a captaincy rule that respects the model's measured skill.** Captain the model's pick only when it beats the
   most-captained player in the game (from `bootstrap-static` `most_captained`, or effective ownership) by a margin;
   otherwise take the template captain. Calibrate the margin on the archive: simulate "template captain" vs "model
   argmax" vs the rule across all gameweeks and pick the margin that maximises points. Expect the margin to be large
   (>= 1.5 pts) while the model's startable Spearman is 0.1.
5. **Build the backtest that matches deployment.** Pull the 2024-25 archive too. Train on 2024-25 + current-season
   GW1..N, test on GW N+1..N+5 for N = 3..30. This is the only setup that can answer the recency-weighting and
   "how much does last season matter" questions. The current within-season folds cannot.
6. **Replace the ceiling band with a haul probability.** Train a classifier for P(points >= 8) per position (or take
   it from the quantile models properly). That is the captaincy quantity. Drop "highest ceiling" until it exists.
7. **Recalibrate intervals on live data.** Live coverage is 64-70%. `prediction_history` now has 3 scored gameweeks;
   compute the conformal margin on those as they accrue and report live coverage next to the archive number.
8. **Process.** Refresh on deadline day, every week (a reminder, not autorun, per the standing rule). Read the
   tool's XI before setting the lineup. Rebuild the My Team view on deadline day after any wildcard.
9. **Reset the success metric.** Track cumulative points vs the game average and vs a "template" team (highest
   effective ownership XI, template captain) in `fpl_ml_team_history.jsonl`. A good model shows up as +1-2 pts a week
   over a season, not as correct captain calls. Weekly captain hit-rate will never look good and isn't the goal.

## 7. Tested and rejected (so nobody re-tries them)

All on the archive holdout, two folds, same metrics as section 3.

| Variant | Spearman startable | Spearman top-30 | Captain argmax | Note |
|---|---|---|---|---|
| Baseline per-position L2 | 0.146 | 0.139 | 5.7 | |
| Poisson objective | 0.110 | 0.122 | 5.0 | worse, FWD bias worse |
| + last-6 ppg, season ppg, points SD | 0.123 | 0.141 | 4.9 | no gain; model already has the information |
| Pooled single model + position one-hots | 0.138 | 0.160 | 4.6 | no gain |
| Per-position linear recalibration (fit fold 1, apply fold 2) | 0.106 | 0.136 | 5.4 | didn't transfer; DEF bias flipped |
| Remove all Understat per-90s | 0.140 | 0.150 | 5.1 | inert features, not a leak |
| Remove now_cost | 0.153 | 0.130 | 5.4 | neutral |
| FWD 400 trees lr 0.04 | 0.116 | 0.151 | **8.5** | FWD bias removed, max FWD pred 4.9 -> 6.5, MAE +0.04 |

The captain-argmax column is a mean over only 10 gameweeks and moves by a point or two with noise. The FWD bias
and the 4.92 ceiling are structural and reproduce in every fold and live.

## 8. What was done about it (2026-10-16)

Items 1-4, 6, 7 and 9 from section 6 are implemented; item 5 was built and run; item 8 is a reminder only.

| Item | Change | Where |
|---|---|---|
| 1 | Tuner and backtest score `selection_loss` (1 - pool Spearman + 0.25·bias) instead of MAE. MAE still reported. | `backtest_model.py` (`ranking_metrics`, `selection_loss`), `tune_model.py` |
| 2 | FWD model 400 trees / lr 0.04, with the reason in the comment | `fpl_ml_model.py` `POSITION_MODEL_PARAMS` |
| 3 | Captain binary in the objective of all three optimisers | `pick_team.pick_squad`, `fetch_my_team.pick_with_transfers`, `pick_best_lineup` |
| 4 | Captaincy rule: model overrides the most-owned starter only when ahead by `CAPTAIN_TEMPLATE_MARGIN` | `fetch_my_team.recommend_captain`, `calibrate_captaincy.py` |
| 5 | Deployment-matching backtest on 2024-25 + 2025-26 | `backtest_deployment.py` |
| 6 | P(8+ haul) classifier replaces the ceiling band in the captain UI | `fpl_ml_model.make_haul_model`, `haul_prob` column, `dashboard_template.html` |
| 7 | Live scorecard from `prediction_history/` snapshots; live conformal extra widens the interval margin | `backtest_model.score_live_snapshots` |
| 8 | Deadline countdown printed at the top of every model run | `fpl_ml_model.main` |
| 9 | Template (ownership-optimised) squad picked and scored next to the model's every week, plus game average | `pick_team.py` |

Calibration results that set the constants:

- **Captaincy margin** (`calibrate_captaincy.py`, 35 weekly decisions replayed on 2025-26): always follow the
  model 196 captain pts, always take the most-owned player 204, the rule at margin 0.75 → 221, 1.0 → 240,
  **1.5 → 242**, 2.0 → 232, 3.0 → 218. A clear hump above both pure strategies; 1.5 adopted. The rule follows
  the model in 11 of 35 weeks.
- **Recency weighting** (`backtest_deployment.py`, 9 cutoffs × 5-GW windows, 2024-25 + 2025-26): cross-position
  pool Spearman 0.12-0.14 at every half-life from 60 days to off, and prior-season-only ≈ current-season-only ≈
  both. Recency weighting is neutral for ranking skill, so the half-lives were left alone. Section 4C's concern
  was reasonable but the data does not support acting on it. One real finding from it: GK is over-predicted by
  about a point in every configuration, which is unresolved.
- **FWD bias** is reduced, not gone: the single-fold archive backtest with the new params still shows FWD pool
  bias −1.1 on GW34-38 (a hot finish for forwards last season). Watch the live scorecard.

Tested along the way and not adopted: the per-position recalibration and the other rejected variants in
section 7 stand.

## 9. Stability work and the data-source experiments (2026-10-16, evening)

Nathan's complaint: the recommendations swing every week. Measured on the archived snapshots: the model's
view of the SAME gameweek, from two runs a week apart, has rank correlation 0.92-0.96 over all players
but only 0.47-0.55 among startable players; 7 of the top 20 change each week; mean movement 0.65 pts on
predictions that sit ~0.5 pts apart at the top.

Shipped:

| Change | Where |
|---|---|
| Tuning runs at most every 28 days (or `-Tune`), seeded by calendar month, 60 trials | `refresh_dashboard.ps1`, `tune_model.py` |
| Tier column on the player table: bands of one noise unit (1 pt per GW, 3 pts for the 5-GW total) below the best prediction | `dashboard_template.html` |
| Recent-buy protection: fit, unflagged players bought in the last 3 GWs are not offered for sale; wildcard/free-hit weeks excluded | `fetch_my_team.py` (`RECENT_BUY_WEEKS`, `LOCKED_IDS`) |
| Stickiness: a new recommended move must beat last run's (same GW) by `CHURN_MARGIN = 1.5` net pts, else last run's stands; runs logged to `my_fpl_team_history.jsonl` | `fetch_my_team.py` |
| Pending transfers already made for the upcoming GW are applied to the squad, bank and free-transfer count | `fetch_my_team.py` |
| Discount notes: fit players discounted by a one-off absence (João Pedro: raw 6.6 shown as 4.8) are flagged for a manual overrule | `fetch_my_team.py`, dashboard captain panel |
| Transfer options exclude on (incoming, outgoing) so equal-value sales for the same buy are all listed | `fetch_my_team.pick_with_transfers` |

Tested on the deployment replay (2024-25 + 2025-26 GW≤N → N+1..N+5) and NOT adopted:

| Experiment | Result | Verdict |
|---|---|---|
| Bookmaker odds (football-data.co.uk closing 1X2 + O/U 2.5 → Poisson team goals; `odds_features.py`, `ablation_odds.py`) | cross-position pool Spearman 0.147 without → 0.128 added, 0.125 replacing team form, 0.113 replacing player xG; early-season worse still | Fails the gate in every form. Team xG form already carries the strength signal; odds add correlated noise on small per-position samples. Infrastructure kept; not in the live model. |
| Start-probability model (`start_probability.py`, trained on all squad rows incl. 0-minute, AUC 0.95) as the playing-time multiplier | current rule Spearman 0.152 / top-10 4.65 / 10.1% non-players in top 30; P(≥60) 0.111 / 4.62 / 5.9%; blended 0.123 / 4.72 / 6.0%; max(rule, P) 0.138 / 4.54; normalised P 0.119 / 4.71 | Halves non-players at the top but loses ranking skill. "Rescue the discounted regular" hurts on average: a player who just missed a match does score less next time. Rule kept; the exception is flagged (discount notes) rather than modelled. |
| Longer rolling windows 6 / 10 / 15 (`ablation_window_length.py`) | Spearman 0.147 / 0.129 / 0.149; top-10 4.89 / 4.86 / 4.97 | Indistinguishable. Window kept at 6. |
| More seasons 2022-23..2024-25 (`ablation_more_seasons.py`) | 0.147 → 0.165, top-10 4.89 → 5.12 | Small, real. Worth adopting in the live training set when the defensive-stat columns can be handled (they don't exist before 2025-26). |

Net: the churn is mostly irreducible input noise at the top of a flat ranking. The fixes that help are
about not *acting* on it (tiers, stickiness, protection, monthly tuning), not about a better signal.
None of the three new signals cleared the bar, and that is recorded here so nobody re-tries them without
a new idea.

## Scripts used

Scratch, in the session scratchpad (not in the repo): `skill_test.py` (section 3 archive numbers and the Understat /
now_cost ablations), `variant_test.py` (section 7), `recal_test.py` (recalibration), `fwd_test.py` (section 4A),
plus inline snippets that scored `prediction_history/gw03-05.csv.gz` against `/event/{gw}/live/` and pulled every
league entry's picks. Worth promoting `skill_test.py`'s metrics into `backtest_model.py` as part of item 1.
