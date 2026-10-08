# Experiment A — asked vs got (pre-registration, 2026-10-04)

**Question.** The owner's loop asks the diffusion model for a win rate just above the
best in the dataset (best 0.10 → ask 0.11) and retrains. That only works if the win
rate the model is *asked* for moves the win rate its teams *get*. Does it?

**Win rate** = wins / 24 battles against opponents drawn uniformly from the top-50
meta pool (`activesearch.opponents()`), VGC-Bench behaviour-cloning policy on both
sides. Real meta team ≈ 0.47–0.49.

## Model
`ContinuousWR` = `hpsdiffusion.TeamDiffusionHPS` with the 4-bin token replaced by a
continuous win-rate embedding (scalar + 8 Fourier features → MLP) and a learned null
vector. Warm-started from `results/temperature_p0.pt` (the shared unconditional p0),
fine-tuned on the real corpus (null condition) + every labelled team that survives
in `results/{activesearch,dsame_experiment,entropyloop,gradguide,gradloop}.json`
and the matchup DB (deduplicated by canonical hash, battle-weighted mean label).
Condition dropped with p = 0.15 so classifier-free guidance is defined.

## Arms
- asked y* ∈ {0.0, 0.1, …, 0.9} (0.9 sits above almost every training label — extrapolation)
- guidance w ∈ {1, 2, 4}
- plus one unconditional cell
- 64 Showdown-valid teams per cell; temperature 1.0, no top-p; corpus Stat Point
  spreads (same handicap in every cell). 31 cells × 64 × 24 = 47,616 battles.

## Measures
Per cell: mean win rate ± team-clustered SE, Showdown validity, distinct species
sets, exact copies of a training team. Per w: OLS slope of got on asked (teams as
units, 2,000 team bootstraps within cell for the 95% CI).

## Decision rule (fixed before battles)
1. Slope CI includes 0 at every w → **asking does not move got**; stepping y* by
   +0.01 is moot. Stop the y*-stepping idea.
2. Slope > 0 but got flattens → report the ceiling (highest cell mean) and the
   asked value where it flattens; stepping is only meaningful below it.
3. Slope > 0 and no flattening → the idea is alive; the next question is whether
   the step size is detectable: report n per arm needed to see slope × 0.01.
Copies of training teams above 10% in a cell invalidate that cell's "got".

## Amendment (2026-10-04, before any battle result was read)
Owner: every battle goes into the matchup matrix. Battling switched from
`pool.score` (24 battles, opponents drawn at random, totals only) to
`matchup_db.ensure` with per_cell = 1: each generated team plays each of the 49
distinct top-50 columns once (49 battles), recorded as cells under policy 1, team
origin `askedgot:<cell>`. Score = the DB row mean over top-50 columns (set weights).
A first `pool.score` chunk was killed mid-run; none of its output was read or kept.

## Result (2026-10-04) — decision rule 2: works, then flattens
1,984 teams × 49 battles (97,216 + 3,577 fill), all in the matchup matrix; figure `results/asked_vs_got.png`.
- Slope of got on asked: w=1 +0.062 [0.035, 0.087], w=2 +0.125 [0.100, 0.152], w=4 +0.208 [0.188, 0.228].
- Ceiling 0.322 ± 0.013 (asked 0.5, w=4); flat from asked 0.4 to 0.7; falls back to 0.21 / 0.13 at 0.8 / 0.9
  (above the best training label, 0.792). Unconditional 0.185; top-50 meta team vs the other 49: 0.531.
- w=4 tracks asked from 0.2 to 0.4 (0.12 → 0.19 → 0.30), local slope ≈ 0.9; seeing a +0.01 step there needs ~1,700 teams per arm.
- Diversity collapses exactly where it climbs: 26–30 distinct species sets of 64 at w=4, asked 0.4–0.7 (64 at low asks).
- Zero copies of training teams in any cell. Validity ≥ 89% except w=4 at asked ≥ 0.7 (68–78%).
Implication: "best in dataset + 0.01" asks at ≈ 0.80, where got drops to 0.21. Step from what the model delivers, in 0.1s.
