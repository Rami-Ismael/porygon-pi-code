# Cross-entropy loop with Diffusion-BBO's learning rule (pre-registration, 2026-10-16)

**Question.** Diffusion-BBO (arXiv 2407.00610) retrains its conditional diffusion model on all data
gathered so far and asks for a target tied to what the last round delivered. Our cross-entropy loop
re-steers on elites only, and inside it the ask stopped mattering (ask-range loop, 2026-10-12: got vs
ask correlation −0.03). Does Diffusion-BBO's rule make the ask a lever again, or lift the finalists?

**Win rate** = wins / battles against the top-50 meta teams, behaviour-cloned policy (policy 1) on both
sides. Battled teams: one battle per top-50 column (49). Finalists: 192 fresh battles.

## The arm (`src/cem_dbbo.py`)
Shared generation 0 = `cem_v3_noheldout`'s generation 0. Generations 1–3:
1. **Model.** Both v3 members fine-tuned from their original checkpoints, 300 steps, batch 64,
   learning rate 1e-4. Each batch: 32 rows from the member's own original training set, 32 from the
   labelled pool (49 anchors + every battled loop team). No elite cut.
2. **Shrinkage.** Each battled batch's win rates shrink toward the batch mean under a beta-binomial
   prior fitted by the method of moments; anchors are their own batch. The ridge keeps raw labels.
3. **Delivered.** The 10 best distinct battled teams of a generation re-battled fresh (98 battles
   each, `pool.score`); delivered_k = their mean.
4. **Ask.** y = min(delivered_k + 0.1, 90th percentile of the shrunk labels, training max).
   384 of 512 proposals ask y, 128 ask y − 0.2; the ridge picks the top 96 and top 32 per stratum.
Kept from the base: taboo list (k = 2), sets-only temperature 0.5 (species at 1), ridge refit,
guidance and ranking, guidance 2, nature repair. No uncertainty or disagreement term.

Finalists as `cem_v3.final`: 16 best distinct battled teams by loop label, 192 fresh battles vs
top-50, then once vs ranks 51–100.

## Decision rule (fixed before battles)
Keep the arm only if (a) finalists beat `cem_v3_noheldout` (0.504 vs top-50, 0.509 vs ranks 51–100)
with the 95% interval of the difference above zero, or (b) got-vs-ask correlation inside a generation
is clearly positive (95% interval above zero) at equal battled win rate.

Extra cost over the baseline: 4 × 10 × 98 delivered battles.

## Results (2026-10-16)

| gen | ask | battled win rate (mean) | median | top 10% | max | max-label team, fresh | asked y | asked y − 0.2 | got vs ask r [95%] | compositions / 512 | battled compositions | delivered | baseline win rate (compositions) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 (shared) | 0.500 | 0.458 ± 0.009 | 0.460 | 0.580 | 0.680 | 0.500 | – | – | – | 156 | 60 | 0.508 | 0.458 (156) |
| 1 | 0.557 | 0.484 ± 0.010 | 0.480 | 0.620 | 0.720 | 0.520 | 0.478 | 0.501 | −0.09 [−0.24, +0.07] | 448 | 119 | 0.543 | 0.506 (149) |
| 2 | 0.569 | 0.539 ± 0.008 | 0.540 | 0.640 | 0.720 | 0.582 | 0.541 | 0.533 | +0.04 [−0.14, +0.22] | 452 | 108 | 0.541 | 0.506 (109) |
| 3 | 0.579 | 0.525 ± 0.008 | 0.530 | 0.640 | 0.700 | 0.490 | 0.526 | 0.523 | +0.01 [−0.14, +0.17] | 446 | 111 | 0.537 | 0.528 (76) |

Median tracks the mean within 0.014. The max label (49 battles) is mostly luck: the max-label team re-battled fresh at 0.49–0.58.
The 90th-percentile cap set the ask in every generation (delivered + 0.1 was always higher).

**Finalists** (16 × 192 fresh battles): **0.521** vs top-50 (best 0.646, 7 ≥ 0.531), **0.511** vs ranks 51–100
(best 0.578). Baseline (`cem_v3_noheldout`): 0.504 / 0.509. Difference +0.017 [−0.026, +0.059] vs top-50,
+0.002 [−0.036, +0.038] vs ranks 51–100 (two-sample bootstrap over finalist teams).

**Decision: not kept.** Neither interval clears zero, and got vs ask stays flat (r ≈ 0 in every generation).
Training on all data with shrunk labels does not make the ask a lever again. The arm keeps about 3–4× the
compositions of the baseline, but that comes from the taboo list, which the baseline did not use.

## Ablation: what moving the ask does on its own (2026-10-16)

Two more arms with the same sampling, taboo list, contrast stratum, shared generation 0 and finalist protocol,
re-steering on the elites only (cem_v3's rule): **fixed** asks 0.5 (contrast 0.3), **moving** uses the ask rule above.

| arm | battled win rate g1 / g2 / g3 | got vs ask r g1 / g2 / g3 | finalists vs top-50 | vs ranks 51–100 |
|---|---|---|---|---|
| fixed ask, elite re-steer (`cem_fixed_elite`) | 0.507 / 0.521 / 0.516 | −0.10 / +0.15 / −0.11 | 0.540 | 0.527 |
| moving ask, elite re-steer (`cem_moving_elite`) | 0.513 / 0.530 / 0.518 | +0.09 / +0.08 / −0.14 | 0.544 | 0.507 |
| moving ask, all-data + shrinkage (`cem_dbbo`) | 0.484 / 0.539 / 0.525 | −0.09 / +0.04 / +0.01 | 0.521 | 0.511 |
| baseline, old sampling (`cem_v3_noheldout`) | 0.506 / 0.506 / 0.528 | – | 0.504 | 0.509 |

All got-vs-ask intervals span zero. Differences in finalists (two-sample bootstrap, 95%):
- moving − fixed ask: +0.004 [−0.028, +0.036] vs top-50, −0.020 [−0.051, +0.009] vs ranks 51–100.
- all-data − elite training: −0.023 [−0.061, +0.016] vs top-50, +0.004 [−0.024, +0.031] vs ranks 51–100.
- fixed ask (new sampling) − baseline: +0.035 [−0.002, +0.072] vs top-50, +0.018 [−0.021, +0.058] vs ranks 51–100.

**Reading.** Moving the ask changes nothing detectable, and neither does training on all the data. The only
near-signal is the sampling change (taboo list + sets-only temperature 0.5), and its interval just touches zero.

## Per-generation top-20% elites (`cem_v3_topfrac`, 2026-10-16)

The owner's intended elite rule, on the old loop otherwise unchanged (old sampling, no taboo, ask 0.5). Elites for
generation g+1 are the top 26 of generation g's 128 battled teams, with no anchors and nothing carried over. The
model is fine-tuned from the original members for 10 epochs. Shared generation 0, capped at g0–g3.

| gen | battled win rate (mean / median / top 10% / max) | species sets / 512 | baseline (`cem_v3_noheldout`) |
|---|---|---|---|
| 1 | 0.527 / 0.540 / 0.646 / 0.720 | 106 | 0.506 (149) |
| 2 | 0.500 / 0.500 / 0.640 / 0.720 | 106 | 0.506 (109) |
| 3 | 0.526 / 0.520 / 0.620 / 0.760 | 97 | 0.528 (76) |

Finalists: **0.509** vs top-50 (best 0.641, 5 ≥ 0.531), **0.506** vs ranks 51–100. Against the baseline:
+0.005 [−0.032, +0.042] and −0.003 [−0.045, +0.039]. Against the all-labels arm: −0.012 [−0.052, +0.029] and
−0.005 [−0.037, +0.030]. **No detectable difference.** The species-set count drops less by generation 3
(97 vs 76), but starts lower at generation 1 (106 vs 149).

## Fixed ask 0.5 / 0.6 / 0.7 / 0.8 (2026-10-16)

Same as the fixed-ask-0.5 arm (elite re-steer, new sampling, contrast at ask − 0.2); only the fixed ask changes.

| ask | battled win rate g1 / g2 / g3 | got vs ask r g1 / g2 / g3 | finalists vs top-50 | vs ranks 51–100 | minus ask 0.5 (top-50) |
|---|---|---|---|---|---|
| 0.5 | 0.507 / 0.521 / 0.516 | −0.10 / +0.15 / −0.11 | 0.540 | 0.527 | – |
| 0.6 | 0.509 / 0.505 / 0.517 | +0.08 / −0.01 / −0.01 | 0.551 | 0.538 | +0.011 [−0.025, +0.046] |
| 0.7 | 0.491 / 0.522 / 0.503 | +0.01 / −0.07 / +0.12 | 0.547 | 0.519 | +0.007 [−0.026, +0.039] |
| 0.8 | 0.496 / 0.510 / 0.529 | −0.09 / −0.09 / +0.11 | 0.547 | 0.520 | +0.007 [−0.029, +0.043] |

**Reading.** The fixed ask does not matter inside the loop, not even 0.8, which sits above the highest training
label (0.792). Every got-vs-ask interval spans zero.

## Paper-faithful Diffusion-BBO training (`cem_scratch`, pre-registration 2026-10-16)

The arm above fine-tuned v3 for 300 steps. The paper instead trains its conditional model on the labelled
dataset each round ("Train the conditional diffusion model with 𝒟"; ensemble members from different random
seeds). This arm does that:
- Every generation, including generation 0, trains one fresh model from random weights: the F1 member's
  architecture with a scalar condition, 20 epochs, batch 256, learning rate 5e-4 with warmup and cosine decay, EMA 0.999.
  The label is hidden 15% of the time (classifier-free guidance, as in the paper).
- Data: labelled teams only. That is every team in the matchup matrix with full top-50 coverage (48,924 at
  smoke time, growing as runs add teams), with no corpus and no synthetic rows. Labels are shrunk toward the
  global mean (beta-binomial, method of moments, n = battles).
- No ensemble. The paper's ensemble only feeds its uncertainty score, which this project does not use.
- Ask: generation 0 asks 0.5. Afterwards, min(delivered + 0.1, 90th percentile, max) of the training labels.
- Everything else as `cem_dbbo`: sets-only temperature 0.5, taboo, ridge refit, tilt and pick, contrast
  stratum, top-10 delivered re-battle, finalists 16 × 192 vs top-50 and ranks 51–100.

Compare with `cem_dbbo` (fine-tuned v3) and `cem_fixed_elite`, using the same decision rule as the other arms.

### Result: `cem_scratch` (2026-10-16)

| gen | ask | battled win rate | asked y → got | asked y − 0.2 → got | got vs ask r [95%] | compositions / 512 |
|---|---|---|---|---|---|---|
| 0 | 0.500 | 0.510 ± 0.010 | 0.535 | 0.436 | +0.39 [+0.22, +0.53] | 425 |
| 1 | 0.589 | 0.535 ± 0.009 | 0.556 | 0.472 | +0.35 [+0.19, +0.50] | 383 |
| 2 | 0.589 | 0.528 ± 0.007 | 0.538 | 0.500 | +0.21 [+0.04, +0.37] | 312 |
| 3 | 0.589 | 0.531 ± 0.008 | 0.539 | 0.506 | +0.15 [−0.04, +0.33] | 244 |

Finalists (16 × 192 fresh battles): **0.580** vs top-50 (best 0.688, 14 ≥ 0.531), **0.561** vs ranks 51–100 (best 0.656).
- Against `cem_v3_noheldout` (0.504 / 0.509): +0.076 [+0.036, +0.114] / +0.051 [+0.010, +0.093]. Rule (a) passes, and rule (b)
  passes too: got-vs-ask is positive with its interval above zero in generations 0–2.
- Fairer references, because seed 4040's 0.504 looks unlucky (the plain loop at seeds 4041–4043 gave 0.550–0.577):
  - The same-machinery cross-entropy arm `cem_fixed_elite` (0.540 / 0.527): +0.040 [+0.004, +0.078] / +0.034 [−0.002, +0.068].
  - The pooled plain loop at three fresh seeds (0.559 / 0.558): +0.021 [−0.011, +0.054] / +0.003 [−0.027, +0.035].
- One seed. The ask works in this loop. The win-rate gain is clear against the cross-entropy arm with the same machinery,
  not against the fresh-seed plain loops. Compositions shrink each generation (425 → 244).


## Ask = w × best label so far: w = 0.6 added (2026-10-17)

Owner: add w = 0.6 (`cem_dbbo_bestw06`, `cem_scratch_bestw06`) after the w = 1.0 / 0.8 arms, same settings, so the ask stays inside the labelled data.


**Withdrawn 2026-10-17 before they started (owner):** `cem_dbbo_bestw08`, `cem_scratch_bestw08`, `cem_dbbo_bestw06` and
`cem_scratch_bestw06`. Only the w = 1.0 arms ran (`cem_dbbo_bestw10` done, `cem_scratch_bestw10` finishing). The queue goes on
to `dbbo_allw` and then the four fix arms.
