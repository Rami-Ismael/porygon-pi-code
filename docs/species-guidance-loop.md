# Species guidance 1.5 inside the cross-entropy loop (pre-registration, 2026-10-16)

Written before any battle. Results and verdict are appended after the run; nothing above the
"Result" heading is edited afterwards. Earlier notes (`docs/cem-v3-loop.md`,
`docs/guidance-timing-cads.md`) are not modified.

**Question.** On the generation-0 proposer, classifier-free guidance 1.5 on the six species columns
(2 on the other 42 fields; `foundation.Gen.sample(g_species=1.5)`) passed its pre-registered rule
(`docs/guidance-timing-cads.md`, second section): +54 / +52 / +48 distinct species line-ups per 512
proposals in three seeds at a win-rate difference of +0.001 [−0.026, +0.027]. That rule says a pass
earns the arm a place inside the cross-entropy loop. Does the extra width survive the re-steer — the
loop fine-tunes both members on the elites every generation and the clean run collapsed from 156 to
76 line-ups in three generations — and does the loop's output hold its strength?

**Win rate** (every number here) = wins / battles against the top-50 meta teams, Showdown battles
with the behaviour-cloning policy on both sides (policy 1), unless a line says "vs ranks 51–100".
Monte Carlo evaluation, not ground truth.

## Protocol: the clean `cem_v3` loop, unchanged except the proposer's species weight

Both arms run `src/cem_v3.py` exactly as the clean re-run of 2026-10-14
(`CEM_TAG=cem_v3_noheldout`, `results/cem_v3_noheldout.json`): the `_noheldout_regmb_v3` members
(never trained on ranks 51–100), the foundation ridge `cem_v3_ridge_base_noheldout.npz`, cumulative
elites (every labelled team and the 49 top-50 anchors at or above 0.531, topped up to 64), both
members fine-tuned from the original v3 checkpoints on the elites for 40 epochs at learning rate
1e-4, the ridge refit with loop labels at weight 10, 512 validator-accepted proposals per generation
at ask 0.5, guidance 2, temperature 1, ridge-gradient guidance at the ridge's own λ, nature repair,
held-out ranks 51–100 refused as proposals, the 128 highest by ridge score battled once per top-50
column (49 battles), and the stop rule as written there (copies > 10 %, species sets below half of
the run's own generation 0, saturation of the running best, at most 10 generations). End of loop:
the 16 best distinct battled teams by loop label, re-battled fresh with 192 battles each against the
top-50 schedule (`pool.score`), then once against ranks 51–100 (`heldout.evaluate`).

| arm | proposer, every generation | `CEM_G_SPECIES` |
|---|---|---|
| `plain` | `Gen.sample(...)`, species weight = guidance 2 | unset |
| `species15` | `Gen.sample(..., g_species=1.5)`: weight 1.5 on the six species columns, 2 elsewhere | `1.5` |

The weight applies at every generation's proposer, generation 0 included, on the re-steered members
as on the original ones. Fine-tuning, the ridge, acquisition, elites and the stop rule are untouched.

**Seeds.** 4041, 4042, 4043 (`CEM_SEED`), new for both arms. The settled clean run (seed 4040) is not
rerun; it is reported beside the three new `plain` runs as a fourth reading of the plain loop and
is not part of the rule. Within a seed both arms start every generation's proposal from the same
torch and numpy state (`SEED + 100·g`), fine-tune from the same state (`SEED + g`) and battle with
the same matrix seed, so the arms differ only by the species weight. Three seeds: intervals are
exploratory, as in the generation-0 experiment.

**Files and origins.** Tags `sp15loop_<arm>_s<seed>`: results `results/sp15loop_<arm>_s<seed>.json`,
checkpoints `sp15loop_<arm>_s<seed>_g<k>_<member>.pt`, ridges `sp15loop_<arm>_s<seed>_ridge_g<k>.npz`,
matchup-matrix origin `sp15loop_<arm>_s<seed>:g<k>`. The summary and verdict go to
`results/cem_v3_species15_loop.json` (`src/cem_species15_loop.py report`). No earlier results file,
checkpoint or origin is touched. A plumbing smoke test (`CEM_SMOKE=1`, tag suffix `_smoke`, own
origin) runs first and is not a result.

**Order.** Six runs in one queue, sequential, paired by seed: `plain` s4041, `species15` s4041,
`plain` s4042, … so that an interrupted queue still leaves matched pairs.

## Measures

Per generation and run (`results/sp15loop_*.json`, written by the loop):
1. distinct species line-ups among the 512 proposals (`sets`), among the 128 battled
   (`battled_sets`), and among the elites the generation was fine-tuned on (`elites.sets`, new);
2. **Vendi score** of the 512 proposals' line-ups, kernel K_ij = shared species / 6, as in
   `guidance_timing_cads.py` (`vendi`, new);
3. battled mean win rate with a bootstrap 95 % interval (teams resampled, 4,000 draws, computed by
   the report), p90, max, count ≥ 0.531;
4. copies, valid fraction, new-species teams, elites used, the stop generation and reason.

End of loop, per run: the 16 finalists' fresh win rate (mean, best, count ≥ 0.531), their loop
labels (winner's curse), their vs-ranks-51–100 win rate, and the distinct species line-ups among
the 16.

Per arm, mean over seeds, with the plain loop at the same seed as the reference: line-ups per 512
proposals at generation 0 and over generations ≥ 1, Vendi, battled win rate per generation,
finalists' fresh win rate; the difference arm − plain with a bootstrap 95 % interval (finalists
resampled within seed, 4,000 draws, numpy seed 0).

## Decision rule (fixed before battles)

`species15` **passes** if
(a) in every seed, its mean distinct species line-ups per 512 proposals over the generations ≥ 1
it ran exceeds the `plain` loop's at the same seed (generation 0 is left out: it repeats the settled
generation-0 result; the question is what survives the re-steer), AND
(b) the bootstrap 95 % lower bound of the mean over seeds of (`species15` finalists' fresh win rate
− `plain` finalists' fresh win rate), 16 teams × 192 battles each against the top-50, is above −0.03.

Vendi score, elite line-ups, battled-per-generation win rates, the vs-ranks-51–100 numbers and the
number of generations each run survives are reported, not part of the rule. A pass makes
`g_species=1.5` the proposer setting for later loop experiments (the owner's call to adopt); a fail
retires the guidance-timing lever inside the loop. Not rerun either way.

**Prediction, written before the run.** Generation 0 repeats the earlier result (~200 vs ~155
line-ups). After the first re-steer both arms narrow — the fine-tuned members concentrate on the
elite core — and the arm's lead shrinks to +20–40 line-ups per 512 at generations ≥ 1 but stays
positive in every seed. Finalists' fresh win rate within ±0.03 of the plain loop's (the plain loop's
one reading: 0.504 fresh, best 0.599). Because the species-set guardrail is relative to each run's own
generation 0 and the arm starts higher, the arm may run one generation longer; this is reported, not
scored. If (a) fails, the lever acts only on the original members and the re-steer erases it.

**Budget.** Per run at most 10 × 128 × 49 = 62,720 search battles, expected ~4 generations
(25,088) as in the clean run, plus 16 × 192 × 2 = 6,144 final battles; six runs ≈ 190,000 battles
at the expected length. Fast evaluator (six workers per process) for the loop, the live Showdown
pool for the finals.

## Amendments before the run

(none yet)

## Result (2026-10-16, six runs 10:47–12:11, `results/sp15loop_<arm>_s<seed>.json`, summary `results/cem_v3_species15_loop.json`)

**`species15` FAILS the pre-registered rule: clause (a) fails in seed 4041, clause (b) holds.**

Distinct species line-ups per 512 proposals, by generation (plain → species15):

| seed | g0 | g1 | g2 | g3 | g4 | mean g ≥ 1 | stop |
|---|---|---|---|---|---|---|---|
| 4041 | 155 → 205 | 153 → 128 | 66 → 69 | | | 109.5 → 98.5 (**−11**) | both g2 |
| 4042 | 156 → 205 | 84 → 185 | 45 → 108 | – → 57 | | 64.5 → 116.7 (+52) | plain g2, arm g3 |
| 4043 | 132 → 189 | 187 → 145 | 71 → 71 | 69 → – | 40 → – | 91.8 → 108.0 (+16) | plain g4, arm g2 |

Generation 0 repeats the settled result (+50 / +49 / +57 line-ups, Vendi 4.0–4.4 → 5.8–6.1). After
the first re-steer the lead is gone in two seeds of three: at g1 the arm proposes fewer line-ups than
the plain loop in seeds 4041 and 4043, and by g2 both arms sit at 66–71 (seed 4041, 4043) with Vendi
1.7–1.9. Elite line-ups are 29–41 in every run, arm or not. The one seed where the arm keeps a lead
(4042) is the seed where the plain loop collapsed fastest (84 at g1). Every run stopped on the
species-set guardrail; because the guardrail is relative to the run's own generation 0, the arm's
threshold is ~100 instead of ~70, which ended `species15_s4041` at 69 sets where the plain loop with
the same count would have continued.

Finalists, 16 teams × 192 fresh battles vs the top-50 (behaviour-cloning policy both sides), then vs
ranks 51–100:

| run | fresh mean | best | ≥ 0.531 | line-ups /16 | vs 51–100 |
|---|---|---|---|---|---|
| plain 4041 / 4042 / 4043 | 0.550 / 0.577 / 0.551 | 0.615 / 0.688 / 0.677 | 11 / 14 / 12 | 5 / 2 / 4 | 0.561 / 0.568 / 0.544 |
| species15 4041 / 4042 / 4043 | 0.552 / 0.579 / 0.554 | 0.625 / **0.734** / 0.630 | 10 / 11 / 12 | 4 / 3 / 2 | 0.523 / 0.561 / 0.531 |
| reference, plain 4040 (2026-10-14) | 0.504 | 0.599 | 5 | 4 | 0.509 |

species15 − plain, finalists' fresh win rate: **+0.002 [−0.019, +0.024]** (clause (b) holds: the
lower bound clears −0.03). Vs ranks 51–100 the arm's finalists are 0.004–0.038 below the plain
loop's in every seed (reported, not scored). The best single team of the experiment is
`species15_s4042`'s 0.734 fresh; the plain loop's best is 0.688.

Reading. The lever acts on the original members only: 1.5 on the species columns widens what the
untuned model proposes, and the first fine-tune on the elites erases that width — the fine-tuned
members propose from the elite core whatever the species weight, and the elites hold 29–41 line-ups
in every run. The prediction (+20–40 line-ups at g ≥ 1 in every seed) was wrong in two seeds. Strength
is unaffected either way. Per the rule, the guidance-timing lever is retired inside the loop; not rerun.

Side reading, not part of the rule: three fresh plain-loop seeds give finalists at 0.550 / 0.577 /
0.551 fresh against 0.504 for the settled seed-4040 run — the clean loop's run-to-run spread is
about 0.03 and the 4040 reading is its low end. A new method compared against 0.504 alone would look
better than it is; compare against these four readings.

Battles: 22 search generations × 6,272 plus 6 × 6,144 finals ≈ 175,000, origins `sp15loop_*`.
