# Foundation swap (pre-registration, 2026-10-08)

**Question.** The loss-down retrain (docs/loss-down.md) gave a generator whose teams already win more
than the old model's with no tricks (unconditioned 0.257 vs 0.185). Every decode-time and filtering
method this project tried on the old generator was measured on top of that old model. Re-run each one
on both models under one protocol: is the new model a better foundation, and do the methods still
add on top of it?

**Win rate** = wins / battles against the top-50 meta teams, one battle per (team, top-50 column),
VGC-Bench behaviour-cloning policy on both sides (policy 1), every battle in the matchup matrix.
64 Showdown-valid teams per cell, corpus Stat Point spreads, SE over teams. A top-50 team scores
0.531 against the other 49.

## Models (each with the decoder it is deployed with)
- **old** — `results/asked_vs_got.pt`, decoded in `diffusion.ORDER` with `diffusion.Constraints.mask_for`
  (exactly `asked_vs_got.sample`).
- **new** — F1 + G2 mixture (`lossdown`), decoded in the same order with `lossdown.Tables` masks
  (exactly `lossdown_battle.sample`).

## Arms (one-shot methods; the loops come after, as stage 2)
| arm | method | from |
|---|---|---|
| base | temperature 1, unconditioned | reused: asked_vs_got `uncond`, lossdown_battle `uncond` |
| cfg | ask y* = 0.5, guidance 2 | reused: `y0.5_g2` cells of the same two runs |
| lowtemp | temperature 0.7 + top-p 0.9 | decoding schemes, 2026-08-27 (+8.7 points on the old line) |
| protect3 | keep only teams with ≥ 3 Protect-family Pokémon | streamliner, 2026-09-01 (+0.081 as a filter) |
| surrogate | ridge picks the top 64 of 512 valid proposals | value acquisition, active search 2026-09-01 |
| guide | ridge-gradient tilt on the logits at decode time ("naive") | gradient guidance, 2026-09-01 (0.140 → 0.230) |
| repair | nature matched to the move split after decoding | stat-alignment repair, 2026-08-27 (+1.3 points) |
| stack | lowtemp + cfg + guide + repair, then protect3, then surrogate top 64 of 512 | all of the above |

**One surrogate for both models**: ridge on field indicators + species pairs (`activesearch.Feats`),
fit once on every labelled team in `results/lossdata.npz` (~45k, labels pooled from the matrix and the
logs). Guidance strength: the old run used naive λ = 36 with a ridge fit on ~2–4k labels; λ here is
rescaled so the species tilt has the same spread (std over the species vocabulary) as that run's.

## Measures and decision rules (fixed before battles)
1. Per arm, **new − old** with its SE. The new model is the better foundation for that method if z > 2.
2. **Method gain** on each model = arm − base. **Interaction** = gain(new) − gain(old):
   > 0 → the method helps the new model more; < 0 → diminishing returns on a better model.
3. Report validity, distinct species sets and exact copies of any known team per cell, as before.
4. Headline: the best arm on each model, against 0.531.

## Results (2026-10-08, 12 new cells × 64 teams × 49 battles + 4 reused cells)

| arm | old | new | new − old | gain old | gain new | interaction |
|---|---|---|---|---|---|---|
| base | 0.185 ± 0.014 | 0.257 ± 0.015 | +0.072 (3.5σ) | — | — | — |
| cfg (ask 0.5, g2) | 0.282 ± 0.017 | 0.402 ± 0.013 | +0.121 (5.6σ) | +0.097 | +0.145 | +0.049 (1.6σ) |
| lowtemp | 0.296 ± 0.012 | 0.400 ± 0.012 | +0.103 (6.1σ) | +0.111 | +0.142 | +0.031 (1.2σ) |
| protect3 | 0.195 ± 0.013 | 0.253 ± 0.016 | +0.059 (2.8σ) | +0.010 | −0.004 | −0.013 (−0.5σ) |
| surrogate | 0.288 ± 0.012 | 0.374 ± 0.014 | +0.086 (4.6σ) | +0.103 | +0.117 | +0.014 (0.5σ) |
| guide | 0.303 ± 0.016 | 0.353 ± 0.013 | +0.050 (2.4σ) | +0.118 | +0.096 | −0.022 (−0.8σ) |
| repair | 0.189 ± 0.014 | 0.264 ± 0.018 | +0.076 (3.3σ) | +0.004 | +0.007 | +0.003 (0.1σ) |
| **stack** | 0.503 ± 0.009 | **0.527 ± 0.013** | +0.025 (1.6σ) | +0.318 | +0.270 | −0.048 (−1.9σ) |

Rule 1: the new model is the better foundation in 7 of 8 arms (z > 2), and ahead in all 8.
Rule 2: no interaction clears 2σ. CFG, low temperature, the surrogate pick and guidance each add
+0.10–0.15 on either model; protect3 and nature repair add nothing on either (both models already put
Protect on 3+ Pokémon, and repair's old +1.3 points does not reproduce). The stack shows diminishing
returns near the top (−0.048, −1.9σ).

Rule 3 / headline: **new stack 0.527 ± 0.013, level with a real top-50 team (0.531)**: 31 of 64 teams
≥ 0.531, best single 0.740 (49 battles, winner's-cursed), 17 distinct line-ups, 100% valid, 0 copies.
The old stack's 0.503 is ONE line-up repeated 64 times (Basculegion, Charizard, Floette-Eternal,
Garchomp, Kingambit, Whimsicott — the gradient-loop core of 2026-09-01). The new stack's most common
line-up (16/64) is different: Ceruledge, Floette-Mega, Garchomp, Incineroar, Milotic, Sinistcha.
Inside the stack the surrogate no longer ranks (predicted vs battled r = 0.02; predicted mean 0.635 vs
0.527 got). "copies 64" on the reused old cells is bookkeeping: those teams are in lossdata itself.

**Limit (2026-10-08, Codex regmb v3 audit):** both models here use the corpus vocabulary (189 of 347
legal species), so every arm searches only that subset of the regulation. See docs/loss-down.md,
"Known limits found later".

## Version 3 rerun (2026-10-08, Codex regmb v3: full Regulation M-B vocabulary)

`FOUNDATION_SET=v3 python foundation.py gen|battle|report` → results/foundation_v3.json. Same seeds per
method name, same ridge, 64 teams × 49 battles; v3 runs all 8 arms (no reused cells). New species
without a corpus spread get a random legal Stat Point spread (as Codex's v3 decoder).

| arm | old | new | v3 | v3 − new | new-species teams |
|---|---|---|---|---|---|
| base | 0.185 | 0.257 | 0.099 ± 0.016 | −0.158 (−7.2σ) | 33/64 |
| cfg | 0.282 | 0.402 | 0.369 ± 0.015 | −0.033 (−1.7σ) | 0/64 |
| lowtemp | 0.296 | 0.400 | 0.358 ± 0.018 | −0.042 (−1.9σ) | 4/64 |
| protect3 | 0.195 | 0.253 | 0.205 ± 0.017 | −0.049 (−2.1σ) | 2/64 |
| surrogate | 0.288 | 0.374 | 0.331 ± 0.012 | −0.043 (−2.3σ) | 0/64 |
| guide | 0.303 | 0.353 | 0.185 ± 0.023 | −0.168 (−6.5σ) | 26/64 |
| repair | 0.189 | 0.264 | 0.159 ± 0.020 | −0.106 (−3.9σ) | 26/64 |
| **stack** | 0.503 | 0.527 | **0.556 ± 0.012** | +0.029 (+1.7σ) | 0/64 |

v3's plain teams that hold a newly reachable species win 0.010 (n = 33) against 0.194 without (n = 31):
the new species came from 6,000 unlabelled synthetic teams. Steering methods drop them (0/64). Fully
stacked, v3 is the best set so far (38/64 teams ≥ 0.531, best 0.760) but concentrated: 38/64 teams are
one line-up (Basculegion, Chandelure, Floette-Eternal, Garchomp, Kingambit, Whimsicott; 0.585), 12
line-ups in all. Reading: the fixes make the vocabulary complete; they do not by themselves make a
better team writer. Page: https://claude.ai/artifact/ErVVWphQwLJ5ERY4zHkHnz
