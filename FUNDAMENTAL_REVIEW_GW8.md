# Fundamental review after GW8: what the data says a good pick, transfer and captain look like, and what the system got wrong

Written 2026-10-30. Position: 414 points, 11th of 11 in the league (leader 586), overall rank 7.45M. GW7 37 points
(average 59), GW8 22 (average 45). This document sets the model aside and asks the data directly, on two full
archived seasons (2024-25, 2025-26: 57,040 player-gameweeks) and on this season's eight gameweeks (every player's
per-match FPL history). Every signal is computed strictly from gameweeks before the one being predicted. Scripts:
`signal_study.py`, `signal_study_live.py`, `ablation_fixed_pool.py`, `ablation_train_serve_skew.py`.

---

## 1. The verdict in five lines

1. A single week of FPL points is about 90% noise. Season points-per-game is highly reliable (split-half 0.73-0.81),
   a single week barely correlates with the next (0.20). Nothing ranks one week well; several things rank the next
   five weeks well. The system has been judged, used and tuned on the wrong horizon.
2. The model is not broken as a ranker. On a fixed pool of regulars across last season's replay it is the best
   ranker tested (Spearman 0.215 vs 0.153 for season PPG, 0.140 price, 0.129 ownership) and the best captain picker
   (6.74 pts/week vs 6.51 for PPG). Live this season it has shown none of that (0.04, captain 5.0 vs 8.4 for the
   most-owned player). No bug found explains the gap; the train/serve skew hypothesis was tested and rejected. The
   honest reading is six noisy weeks in an unusually chaotic season, on an edge that is small even when real.
3. The damage came from decisions, not predictions. Your transfers net −45 points (players sold went on to score
   287, players bought 242). The wildcard alone: Haaland → Isak −14, Verbruggen → Sels −20, Kinsky → Dubravka −23,
   João Pedro → Furo −26. Captaincy earned 27 base points in six weeks where Haaland every week earned 45, and he
   has scored 42 in the five weeks since you sold him. 54 points sat on the bench. The model's small edge cannot
   outrun that.
4. What a good pick looks like is not a mystery, and the top of your league already knows: one premium attacker who
   is captained every week (Haaland: 66 points, 30% haul rate for £9.6m+ attackers vs 6% for cheap ones), then
   nine nailed players at £4.5-8m with high season PPG and BPS. All three leaders have exactly that shape, 32-33%
   mean ownership, one premium, 6-7 transfers, no hits (one 4-point hit).
5. The fix is a different decision framework, not another feature. Three features, three data sources and a
   start-probability model were tested this fortnight and none moved the needle. The framework below does.

---

## 2. What predicts points: the evidence

### 2a. Two seasons, 11,913 "startable" player-gameweeks (60+ recent minutes, 3+ appearances, GW4+)

Mean per-gameweek Spearman rank correlation between each pre-match signal and the outcome:

| Signal | This week's points | Haul (8+) | Next 5 weeks' points |
|---|---|---|---|
| Season points per game | **0.222** | **0.139** | **0.358** |
| Season BPS per game | 0.217 | 0.125 | 0.332 |
| Last week's points | 0.200 | 0.078 | 0.221 |
| Price | 0.194 | 0.138 | 0.294 |
| Last-6 PPG | 0.187 | 0.102 | 0.274 |
| ICT index, last 6 | 0.170 | 0.119 | 0.257 |
| Ownership | 0.166 | 0.130 | **0.305** |
| Clean-sheet rate (season) | 0.159 | 0.085 | 0.250 |
| xGI per 90 (season) | 0.136 | 0.120 | 0.214 |
| Threat (season) | 0.135 | 0.129 | 0.230 |
| Fixture difficulty (easier = better) | 0.137 | 0.085 | 0.057 |
| Opponent goals conceded | 0.083 | 0.046 | 0.027 |
| Home | 0.052 | 0.021 | 0.009 |

- Sustained output (season PPG, BPS, price, ownership) beats underlying process metrics (xG, xA, threat) for every
  target, and the gap widens over five weeks. xGI/90 is a real signal but a weaker one than "has been scoring".
- Fixture difficulty matters this week (0.137) and almost not at all over five weeks (0.057): fixtures wash out,
  player quality doesn't. Home advantage barely registers.
- Ownership is the third-best five-week signal. The crowd is a decent model. That is not a reason to copy it blindly,
  but it is a reason never to be far from it without a specific, measured edge.

By position, this week's points: GK is ranked best by minutes and starts (0.24); DEF by season PPG, minutes and
fixture difficulty (0.18-0.21); MID by season PPG, last week's points and BPS (0.21); FWD by BPS and season PPG (0.17).

### 2b. This season (GW3-8, 1,078 startable rows)

| Signal | This week | Haul | Next 3 weeks |
|---|---|---|---|
| Minutes, last 6 | **0.158** | 0.112 | **0.250** |
| Ownership | 0.154 | **0.150** | **0.254** |
| Last week's points | 0.153 | 0.057 | 0.149 |
| Last-3 PPG | 0.129 | 0.111 | 0.194 |
| Fixture difficulty | 0.119 | 0.113 | −0.021 |
| Season PPG | 0.109 | 0.120 | 0.192 |
| xG last 6 | 0.105 | 0.121 | 0.156 |
| xGI/90 (season) | 0.052 | 0.070 | 0.112 |

- Same shape with fewer games: who plays, who the crowd owns, who has been scoring. Early in a season the process
  metrics are thin and noisy; the model leans on them.

### 2c. How much signal exists at all

- Lag-1 Spearman of weekly points among startable players: 0.20.
- Split-half reliability of season PPG (odd vs even gameweeks, 20+ appearances): 0.68 and 0.58 → full-season
  reliability 0.81 and 0.73. A season of PPG is skill; a week is noise.
- The best kitchen-sink LightGBM on all 33 signals reaches 0.244 for this week and 0.369 for the next five. Season
  PPG alone reaches 0.227 and 0.379. There is no hidden weekly signal waiting to be found in this data.

---

## 3. What a good captain looks like

Two seasons, captain = argmax of each rule among startable players, scored on real points:

| Rule | Points/week | Hauls |
|---|---|---|
| Best season PPG | **6.99** | **41%** |
| Best season BPS | 6.66 | 39% |
| Best xGI/90 | 6.39 | 36% |
| Most owned | 5.96 | 30% |
| Best last-6 PPG | 5.91 | 30% |
| Most expensive | 5.04 | 26% |
| Kitchen-sink model's top prediction | 4.81 | |
| Haul-probability classifier's top | 3.96 | |
| Hindsight best | 16.56 | |

Haul rate (8+ points) by position and price band, two seasons: MID £9.6m+ **31%**, FWD £9.6m+ **30%**, MID £7.6-9.5m
21%, FWD 6.1-7.5m 18%, DEF 6.1-7.5m 16%, anything £5m or under 6-9%.

Restricting the captain to attackers priced £9m+ (ablation_fixed_pool.py, 2025-26 replay): most-owned premium 7.18,
best-PPG premium 7.18, BPS 7.11, the production model 6.61.

This season: most-owned 9.0/week, most expensive 8.5 (Haaland every week), best season PPG 3.5, model's pick 5.0.

**Conclusion:** the captain is the premium attacker with the best sustained output, which in practice is the
most-owned premium. A mean-regression model picks worse captains than that rule, and a haul classifier picks worst
of all. The armband is a tail bet and the tail belongs to £9m+ attackers. The production model's cross-position
flattening (defenders at the top of its table) is wrong for this decision specifically.

---

## 4. What a good long-term pick and a good transfer look like

- Five-week points are ranked by season PPG (0.36), BPS (0.33), ownership (0.31), price (0.29). Fixture run is
  nearly irrelevant over five weeks (0.06). A transfer target should be a nailed player with high sustained PPG and
  BPS, and the fixture swing should be a tie-break, not the reason.
- Minutes are the first filter: minutes-last-6 is the top signal this season for the next three weeks (0.25), and
  starts share dominates for keepers and defenders. Cherki (54 recency-weighted minutes when bought) and Furo (0)
  failed this filter; both cost you.
- Your own transfers (section 1, point 3) confirm it from the other side: every bad move was a sale of a nailed
  high-PPG player (Haaland, João Pedro, Verbruggen, Kinsky) or a buy of a non-starter (Furo, Dubravka, Cherki). Every
  good move was a nailed, in-form, mid-priced regular (Konsa +17, Calafiori +18, Rogers +18, Saka +10).
- What the league leaders own (GW8 XI): Haaland and ten of Rogers, Mbeumo, Ødegaard, Groß, Calvert-Lewin,
  Calafiori, Hall, Bogle, Konsa, Verbruggen, Barnes, De Cuyper, Raya, Guéhi. Nailed, £4.6-8m, mostly top-20 by
  points per £m. Not one speculative buy between them.

---

## 5. Where the model stands, honestly

| Test | Production model | Best alternative |
|---|---|---|
| Fixed pool of regulars, 2025-26 replay, Spearman per GW | **0.215** | Season PPG 0.153; kitchen-sink LightGBM 0.196 |
| Same, top-10 predicted → actual points | 5.07 | 4.86-5.03 |
| Same, captain = model's top pick | **6.74** (40% hauls) | PPG 6.51; most-owned premium 7.18 |
| Adding PPG, BPS, ownership, ICT as features | 0.211-0.224 | no gain |
| Live GW3-8, Spearman in model's top 120 | 0.037 | ownership 0.094 |
| Live GW3-8, captain pick, pts/week | 5.0 | most expensive 9.0, most owned 8.4 |
| Model's own squad, scored GWs | 78 | template 106, game average 104 |

- On replay the architecture earns its keep: a small but real edge in ranking and captaincy over every naive rule.
- Live it has delivered nothing in six weeks. Tested explanations: train/serve skew in the per-90 rate features
  (trained on full-season rates, served on 90-300-minute rates) costs 0.01 of Spearman, not 0.17; early-season
  cutoffs on replay sit at 0.14-0.15, not 0.04. What remains is sample size (six weeks, SE ≈ 0.05) and a season in
  which GW8's average was 45 and the top scorers were Gabriel, Hill, Bobb and Tel. The model can be right on average
  and still lose six weeks in a row; so can any 0.2-Spearman signal.
- The edge, where it exists, is worth roughly +0.5 to +1 point a week in captaincy and a few points a month in
  transfers. The decisions above cost 10-20 points a week. The order of magnitude is the lesson.

---

## 6. Data audit: what comes in, what is missing, what is unused

**In the model today:** FPL per-match xG, xA, xGC, bonus, minutes (recency-weighted), minutes volatility, saves,
CBIT/recoveries/tackles/defensive contribution (lagged), price, home/away, FDR, Understat team xG for/against form
(home/away split, 6 matches), Understat season per-90s (npxG, xA, xGChain, xGBuildup, shots, key passes), rest days,
set-piece flags. Trained on last season plus this one.

**Available and unused:** FPL's own `total_points` history (season PPG and recent PPG, the single best signal),
`bps` (second best), `ict_index`/`threat`/`creativity`/`influence`, `starts`, `selected` (ownership) and
`transfers_balance`. All in both the live API and the archive. Adding PPG/BPS/ownership/ICT to the model on replay
did not improve its ranking (0.211-0.224 vs 0.215), so the information is largely already captured through xG,
minutes and price. But these are exactly the signals the *decision rules* should use directly (section 7).

**Tested and rejected this fortnight** (deployment replay, 2024-25 + 2025-26): bookmaker odds as features (0.147 →
0.125-0.128 in four variants), a start-probability model as the minutes multiplier (AUC 0.95, but ranking 0.152 →
0.11-0.14), rolling windows of 10/15 matches (flat), three more seasons of training data (+0.02), Poisson objective,
pooled model, per-position recalibration, point-in-time shrunk rate features (no change). The feature ceiling is
near. Further model work is low-yield until the decision layer stops throwing away ten times the edge.

**Genuinely missing and not cheaply available:** predicted lineups / press-conference team news before the deadline
(the only input that would have stopped Cherki, Furo and the Gvardiol benching); set-piece and penalty *share* over
time rather than a flag; injury return dates. These are human-readable inputs, not model inputs.

---

## 7. The redesign: decide on the horizon where the signal lives

1. **Captain by rule, not by model.** Captain = the most-owned attacker priced £9m+ in your XI; if you don't own one,
   buy one before you worry about anything else. Model override only when its pick leads by 1.5+ points *and* is
   itself a £9m+ attacker. The current rule already defers to the game's template captain; this closes the gap it
   left when you don't own him, which has been every week since GW4. Measured: 7.18/week vs 5.0 live.
2. **Squad shape before squad members.** One premium attacker captained every week, two mid-priced attackers at
   £7-8m with high season PPG, keepers and defenders chosen on minutes and clean-sheet rate, bench at £4.0-4.5m that
   actually plays. This is the shape of all three league leaders and of the best 5-week signals.
3. **Transfer filters before transfer gains.** A target must have 70+ recency-weighted minutes (or 80% starts share),
   no flag, and a season PPG in the top half of its price band. The optimiser's 5-GW gain is the tie-break among
   players who pass, not the reason to buy a player who doesn't. Cherki, Furo and Dubravka all fail this filter.
4. **Never sell a nailed top-quartile-PPG player for a punt.** Haaland → Isak, Verbruggen → Sels and Kinsky →
   Dubravka were all sales of exactly that. Recent-buy protection exists; add "protect top-quartile PPG regulars
   unless flagged".
5. **Judge the model on the 5-week horizon and on cumulative points vs the template squad**, which the tracker now
   records weekly (model 78, template 106 so far). If the model is not beating the ownership template over ten
   scored gameweeks, the weekly point estimates should stop driving anything but tie-breaks.
6. **One refresh, on deadline morning, and no mid-week punts.** The model's view of the same gameweek moves 0.65
   points per player between runs; acting on Thursday's numbers for Saturday's game is acting on noise.

---

## 8. Scripts and outputs

`signal_study.py` → `fpl_ml_signal_study.json`; `signal_study_live.py` (this season's history, pulled to the session
scratchpad); `ablation_fixed_pool.py` → `fpl_ml_fixed_pool_ablation.json`; `ablation_train_serve_skew.py` →
`fpl_ml_train_serve_skew.json`. Earlier experiments: `SEASON_REVIEW_GW1-6.md` sections 7 and 9.
