# Species repulsion inside the batch (pre-registration, 2026-10-15)

**Question.** The proposer collapses to a few species line-ups (176 → ~30 distinct compositions per 512
over six generations), and every elite-side diversity filter tried so far (DPP, MMR, entropy, taboo) costs
win rate or is a hard rule. Does coupling the 96 teams of one sampling batch *during* decoding, so that
later teams are pushed away from the species the earlier teams already chose, give more distinct line-ups
and more winning line-ups at the same strength?

**Source.** Lamont et al., *Free Lunch for Pass@k? Low Cost Diverse Sampling for Diffusion Language Models*
(arXiv 2603.04893, preprint, no venue): at every denoising step, sample i's feature vector is projected onto
the orthonormal basis of samples 1..i−1 and the redundant component is penalised with one gradient step on
the logits. Found in the 2026-10-14 diversity sweep as the one batch-coupling mechanism written for masked
diffusion.

**Port (`src/sampler_repel.py`).** Feature = species-presence vector of a team (sum over the six slots of the
one-hot of each decoded species, plus the current species column's probabilities). Repulsion acts on the six
species columns only, in decode order, with strength α·(1 − s/6) on species slot s. The gradient flows
through the softmax of the (tempered, legality-masked) species logits; the basis is held fixed. Everything
else is today's sampler (`lossdown_battle.sample`, F1 + G2 clean mixture, guidance 2).

**Win rate** = wins / battles against the top-50 meta teams, one battle per (team, top-50 column) = 49
battles per team, behaviour-cloning policy both sides (policy 1), all in the matchup matrix with origin
`repel:<cell>`. A real top-50 team against the other 49 scores 0.531.

## Strength chosen without battles (`check`, `results/sampler_repel_check.json`)
asked 0.5, guidance 2, 3 batches × 96 per α; rule fixed before the run: the α with the most distinct
line-ups per Showdown-valid team among α with validity ≥ 90% (ties → smaller α).

| α | valid | line-ups / valid | species used | exp(species entropy) |
|---|---|---|---|---|
| 0 (today) | 100% | 96 / 287 | 52 | 13.5 |
| 64 | 99% | 122 / 285 | 57 | 16.8 |
| 128 | 99% | 135 / 286 | 55 | 19.6 |
| **256** | 99% | **137 / 285** | 63 | 22.6 |
| 384 | 99% | 118 / 286 | 52 | 21.7 |
| 512 | 98% | 96 / 283 | 40 | 17.9 |
| 1024 | 98% | 66 / 281 | 30 | 16.4 |

Above 384 the repulsion overshoots: every team is pushed onto the same few unpopular species, so line-ups
fall below today's. **α = 256.** (A first two-batch sweep over 0/4/16/64/256/1024 had a pick rule that fell
back to the largest valid α and chose 1024; the rule was corrected before any battle, and the sweep rerun.)

## Arms (`sampler_qd_run.py` protocol; seed 6161)
| arm | sampler |
|---|---|
| `today` | one field per pass in `diffusion.ORDER`, temperature 1 |
| `settemp05` | "free species, focused sets": species at temperature 1, every other field at 0.5 |
| `repel` | today + species repulsion α = 256 |
| `repel_settemp05` | both |

asked y* ∈ {0.1, 0.3, 0.5, 0.7, 0.9}, guidance 2, the first 64 Showdown-valid teams per cell, corpus Stat
Point spreads. 4 arms × 5 asked × 64 teams × 49 battles = 62,720 battles, plus the confirm stage.

## Decision rule (fixed before any battle)
1. **Diversity.** Distinct species line-ups among the 64 teams: `repel` above `today` at every asked value.
2. **Quality-diversity** (`sampler_qd.py`): winning line-ups (distinct line-ups whose best team scores
   ≥ 0.531) and QD-score (sum over line-ups of the best team's win rate), `repel` minus `today` with
   leave-one-team-out jackknife standard errors; beyond 2 standard errors is a difference.
3. **Strength guard.** Mean win rate over the five asked values, `repel` minus `today`; the lower bound of
   a 95% interval (standard error over teams) must be above −0.03.
4. **Confirm.** The best team of every winning line-up gets 49 fresh battles (one more per top-50 column);
   "confirmed" = the fresh battles alone score ≥ 0.531. `repel`'s confirmed count must not be below `today`'s.

**PASS** = (1) and (3) and (4). A pass sends species repulsion into the cross-entropy loop as the next
experiment (the loop is where the collapse happens; this run only shows the per-batch effect). A **FAIL**
closes batch repulsion as a lever; the only open variant would be a different feature (full composition,
or species + item).

`repel_settemp05` is read against `settemp05` with the same rule, as a secondary result.

Caveats written in advance: one model mixture, one seed per cell, 64 teams per cell, 49 battles per team
(standard error ≈ 0.07 per team), so cell-level win rates are noisy and only the five-asked means are
compared. Distinct line-ups among 64 is a count of species sets; it says nothing about strategic variety.

## Result (2026-10-15, 65,268 battles, `results/sampler_repel.json`, report in `/tmp/vgc-pilot/supervisor_report.out`)

**Both repulsion arms FAIL the pre-registered rule, on the diversity clause.** Strength is unharmed.

| arm | mean win rate over asked | vs today (95%) | line-ups /320 | winning line-ups | QD-score | confirmed (of winning) |
|---|---|---|---|---|---|---|
| today | 0.254 | | 235 | 1 | 51.3 | 0 of 1 |
| settemp05 | 0.333 | +0.079 ± 0.008 | 237 | 12 | 71.5 | 2 of 12 |
| repel α 256 | 0.249 | −0.005 ± 0.009, lower −0.021 | 250 | 3 | 55.8 | 1 of 3 |
| repel + settemp05 | 0.327 | +0.073 ± 0.008 | 260 | 14 | 78.8 | 8 of 14 |

Line-ups per 64 by asked value, today → repel: 0.1 64 → 64, 0.3 57 → 63, 0.5 29 → 42, 0.7 42 → 31, 0.9 57 → 63.
The lift the battle-free check measured at asked 0.5 (+13 line-ups per 64) is real there, but at asked 0.7
repulsion *lost* 11 line-ups, so clause (1) fails; over all five asked values the gain is +15 ± 11.6 line-ups
per 320, inside noise. Winning line-ups +2 ± 1.7 and QD-score +4.6 ± 3.5 versus today: no difference.

Secondary read, `repel_settemp05` against `settemp05`: line-ups 260 vs 237, winning line-ups 14 vs 12, QD-score
+7.3 ± 4.4, and **8 of 14 winners confirmed on 49 fresh battles (fresh mean 0.563) against 2 of 12** (+6 ± 3.2,
Poisson). That confirm count is the one number worth a second look, but it comes from one seed per cell and
fails the same every-asked diversity clause (asked 0.7: 36 vs 43), so it is a lead, not a result.

**Verdict.** Species repulsion at the batch level is closed as a diversity lever for one-shot sampling: it
moves line-ups by about the per-cell noise and does not change which species the model prefers once the
batch is larger than the handful of alternatives the model holds. The 2026-10-14 Haiku/Opus reading was
right that the mechanism ports in a day; it was wrong that a 96-team batch has enough slack to spread.
Open variant per the pre-registration: a different feature (full composition, or species + item), or the
loop setting where the collapse actually happens.
