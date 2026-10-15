# Species repulsion + focused sets: three-seed replication (pre-registration, 2026-10-15)

**Why.** The one-shot run in `species-repulsion.md` failed its rule, but one secondary number stood out:
`repel_settemp05` confirmed **8 of 14** winning line-ups on 49 fresh battles (fresh mean 0.563) against
**2 of 12** for `settemp05` (+6 ± 3.2 on Poisson counts), from one sampling seed per cell. The owner asked
for a replication at the two asked values where the line-up lift appeared (0.5 and 0.7).

**Design (`src/sampler_repel_rep.py`).** Arms `settemp05` ("free species, focused sets": species at
temperature 1, every other field at 0.5) and `repel_settemp05` (the same plus species repulsion at α = 256,
the strength chosen battle-free on 2026-10-15). Asked y* ∈ {0.5, 0.7}, sampling seeds 1–3, so 12 cells of
64 Showdown-valid teams, guidance 2, F1 + G2 clean mixture, one battle per top-50 column (49 per team),
behaviour-cloning policy both sides (policy 1), origin `repelrep:<cell>`. 12 × 64 × 49 = 37,632 battles
plus the confirm stage. The model is fixed; "seed" varies only the sampling draws.

**Win rate** = wins / battles against the top-50 meta teams. A real top-50 team against the other 49 = 0.531.

## Decision rule (fixed before any battle)
- **Primary: confirmed winners.** Pool each arm's six cells. For every distinct species line-up whose best
  team scored ≥ 0.531 in the search battles, that team gets 49 fresh battles (one more per top-50 column);
  "confirmed" = the fresh battles alone score ≥ 0.531. **REPLICATED** if `repel_settemp05` − `settemp05`
  exceeds 2 Poisson standard errors (√(a + b)) AND the per-seed difference is ≥ 0 in at least 2 of 3 seeds.
- **Guards.** Mean win rate over the six cells, `repel_settemp05` − `settemp05`, 95% lower bound > −0.03
  (standard error over teams). Pooled winning line-ups not fewer than `settemp05`'s.
- **Secondary.** Distinct line-ups per 64 by cell; QD-score with leave-one-team-out jackknife SE.

Anything short of REPLICATED closes the lead; a replication sends `repel_settemp05` into the cross-entropy
loop as the next experiment. Caveats in advance: the first run's 8-vs-2 came from a pick-the-best
procedure and carries a winner's curse of its own; a confirmed count at exactly the bar passes half the time,
so confirmed counts are compared between arms, never read as absolute.

## Result (2026-10-15, 40,768 battles incl. prior cells on duplicate teams, `results/sampler_repel_rep.json`)

**NOT REPLICATED.** Confirmed winners 11 (repel_settemp05) vs 10 (settemp05), +1 ± 4.6; per seed 4/9, 4/14, 3/9
against 4/11, 2/9, 4/6. The first run's 8-vs-2 was a winner's-curse draw from one seed.

| arm | mean win rate (6 cells) | line-ups /384 | winning line-ups | QD-score | confirmed |
|---|---|---|---|---|---|
| settemp05 | 0.430 | 153 | 19 ± 3.7 | 63.3 ± 3.8 | 10 of 19 (fresh mean 0.565) |
| repel_settemp05 | 0.424 | 170 | 25 ± 4.1 | 71.5 ± 3.8 | 11 of 25 (fresh mean 0.559) |

Guards held (win rate −0.007, lower bound −0.021; winning line-ups not fewer). Secondary: line-ups +17 ± 13,
QD-score +8.2 ± 5.4, winning line-ups +6 ± 5.5, all inside 2 SE. By asked value the pattern is consistent
across seeds but small: at asked 0.5 repulsion widens line-ups per 64 from 21–28 to 42–48 and costs 0.03–0.05
win rate; at asked 0.7 line-ups are equal (33–41 vs 34–44) and win rate is 0.01–0.05 higher. Species repulsion
in the batch is closed as a one-shot lever; the loop setting remains the only untested place for it.
