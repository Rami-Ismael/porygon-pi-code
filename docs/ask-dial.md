# Making asked = got (pre-registration, 2026-10-16)

**Question.** Teams generated alone (no loop) track the asked win rate only up to about 0.5, and inside the
loop the ask has no effect (`ask-vs-got-loop.png`). Which of four changes makes the ask a working dial?

Win rate = wins / battles against the top-50 meta teams, behaviour-cloned policy (policy 1) on both sides.
Every sweep: asks 0.1–0.9, 64 valid teams per ask, 49 battles each, the loop's sampling (species at
temperature 1, other fields 0.5), no ridge guidance.

1. **Shrunk labels on all data** (`ask_sweep_dbbo.json`): original v3 members vs the Diffusion-BBO
   generation-3 members, guidance 2.
2. **Precise labels** (`ask_dial.py relabel/train`, `ask_sweep_precise.json`): ~1,000 matrix teams,
   stratified by label, re-battled to 196 battles each. Both members are fine-tuned from the originals for 600 steps:
   each batch is half original rows with the label hidden, half precise teams.
3. **Stronger guidance** (`ask_sweep_g4.json`): all three models at guidance 4.
4. **Ridge guidance off inside the loop** (`cem_fixed_elite_notilt`): the fixed-ask-0.5 loop (contrast 0.3)
   with the ridge tilt removed from the decode; the ridge still picks the 128.
5. **Calibration** (`ask_dial.py calibrate`): the sweep curve with the widest rising range, inverted
   (got → ask); 4 target win rates inside that range, 64 teams × 49 battles each.

**What counts as working.** A change helps if its rising range (lowest-ask got to peak got) is wider than
the original model's at guidance 2, or if its peak sits at a higher ask. Inside the loop, the dial works if
the got-vs-ask interval clears zero. Calibration works if the mean absolute miss is below 0.03, about
one team-level standard error of a 64-team mean.

## Result 1: shrunk labels on all data, no loop (2026-10-16)

Got (mean win rate of 64 teams, 49 battles each) by ask, guidance 2, the loop's sampling, no ridge:

| ask | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|---|---|---|---|---|---|---|---|---|---|
| v3 as trained | 0.218 | 0.325 | 0.414 | 0.446 | 0.496 | 0.499 | 0.497 | 0.397 | 0.142 |
| v3 after Diffusion-BBO generation 3 | 0.189 | 0.326 | 0.410 | 0.437 | **0.533** | 0.511 | 0.507 | 0.487 | 0.436 |

**Reading.** The v3 model alone has a working dial from 0.1 to 0.5: got rises 0.22 → 0.50, close to the
diagonal. It stops at about 0.50, which is where every loop sits. The all-data training lifts the peak a little
(0.533 at ask 0.5) and removes the collapse above the training max (0.44 vs 0.14 at ask 0.9), but does not move
the ceiling far. Inside the loop, teams are already at that ceiling, and low asks get lifted to it.

## Results 2–5 (2026-10-16)

**Precise labels.** 1,105 matrix teams, stratified by label, re-battled to about 196 battles each (mean label SE 0.026;
new labels correlate 0.97 with the old ones). The v3 members, retrained on them (label hidden on the original rows):

| ask | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|---|---|---|---|---|---|---|---|---|---|
| precise, guidance 2 | 0.143 | 0.360 | 0.421 | 0.503 | **0.593** | 0.590 | 0.588 | 0.565 | 0.449 |
| precise, guidance 4 | 0.065 | 0.363 | 0.434 | 0.542 | 0.574 | 0.584 | 0.571 | **0.591** | 0.543 |
| v3 as trained, guidance 4 | 0.132 | 0.323 | 0.448 | 0.481 | 0.518 | 0.443 | 0.467 | 0.347 | 0.175 |
| Diffusion-BBO g3, guidance 4 | 0.100 | 0.279 | 0.442 | 0.455 | 0.508 | 0.511 | 0.523 | 0.500 | 0.451 |

The ceiling rises from about 0.50 to 0.59. No generated team is an exact copy of a precise-label training team (0–1 of 64
per ask). About 46 of 64 reuse a line-up that appears among the 218 precise teams labelled ≥ 0.6, with new sets; the
original v3 model also reuses known line-ups (38 of 64). Guidance 4 alone does not raise the ceiling (0.52).

**Ridge guidance off inside the loop** (`cem_fixed_elite_notilt`). Got vs ask r = +0.19 [+0.01, +0.35], +0.20 [+0.04, +0.36],
+0.16 [−0.02, +0.33]. Pooled, asked 0.3 → 0.477 and asked 0.5 → 0.515. The ask becomes a lever again (+0.04 per +0.2)
once the ridge stops pushing proposals. Finalists 0.534 / 0.510, against 0.540 / 0.527 with the push on
(−0.006 [−0.036, +0.024]): no detectable loss.

**Calibration** on the precise model, guidance 4: targets 0.085 / 0.247 / 0.409 / 0.571 gave 0.111 / 0.130 / 0.442 / 0.558.
The mean absolute miss is 0.047, above the 0.03 bar. The miss comes from the steep stretch between asks 0.1 and 0.2, where the
interpolated ask (0.161) lands badly. Above 0.4 it is within 0.035.

**Reading.** The ceiling is a label-noise problem: precise labels lift it by about 0.09. Inside the loop the ridge push is what
erases the ask. Next: precise labels inside the loop with the ridge push off.

**Random pick inside the loop** (`cem_fixed_elite_random`: ridge push off and the 128 battled chosen at random within each
ask group). Battled win rate 0.441 / 0.466 / 0.439, about 0.057 below the ridge-picked arm. Got vs ask r = +0.04 / +0.19 / −0.05;
pooled, asked 0.3 → 0.437 and asked 0.5 → 0.453 (+0.016). Finalists 0.513 / 0.517, against the ridge-picked arm's 0.534 / 0.510
(−0.021 [−0.056, +0.011]).

**Verdict on the three suspects.** The ridge push flattened the ask: removing it gives +0.04 per +0.2 asked. The ridge pick did
not; removing it too leaves the ask weaker (+0.016), and the battled teams drop 0.06. Elite retraining also compresses the
dial: the loop's model gives +0.016–0.04 per +0.2 asked, against about +0.08 for v3 alone over the same asks.

## Next arm: precise labels inside the loop, ridge push off (`cem_precise_notilt`, pre-registration 2026-10-16)

Generation 0 proposes from the precise-label members (no fine-tune). Each later generation chooses elites by cem_v3's rule
(loop label ≥ 0.531, at least 64), re-battles them to 4 battles per top-50 column (about 196) and fine-tunes the
precise-label members, not the original v3, for 40 epochs on those labels. Ridge push off, the ridge still picks; fixed ask
0.5, contrast 0.3; sets-only temperature 0.5, taboo; finalists 16 × 192 vs top-50 and ranks 51–100. Compared with
`cem_fixed_elite_notilt` (same loop from the original v3 with noisy elite labels): it is kept if the finalists' interval
clears zero, or if got-vs-ask stays positive with the battled win rate at least equal.

## Is the flat ask inside the loop a bug? (diagnosis, 2026-10-16)

No. Checks on the loop's own decode path (original v3 and the fixed-ask loop's generation-1 and generation-3 models):
- Every row of a mixed 0.3/0.5 batch is conditioned on its own ask. Its output equals the row computed alone
  (max difference ≤ 0.001), and the ask moves the output (max log-probability change 2.4–2.6).
- `propose` labels each kept team with the ask it was decoded at, and the battled order matches the labels.
- Inside the decode (generation-1 model and ridge, λ = 34.8), the ridge push and the ask push are about the same size
  overall (spread 0.33 vs 0.38). On species, though, the ridge flips the top choice 46% of the time and the ask only 8%.
  With the ridge push on, the low- and high-ask groups share all of the 8 most-used species by generation 3, and 8 of 31
  low-ask line-ups also appear in the high-ask group; with it off, 1–2 of 32.

So the ridge chooses the Pokémon and the ask only adjusts the sets; species decide most of the win rate. Guard:
`scripts/test_ask_rows.py` (fails if rows of a mixed batch leak or the ask stops changing the output).

## Does the loop's model become a forward model? (owner's hypothesis, 2026-10-16)

**Hypothesis (owner).** The fixed-ask loop retrains the model on its elites while the ask stays fixed. The model stops
conditioning on the ask and just generates teams like its training set. So the win rate it implies for a team disagrees
with the battles, and the ask-vs-got chart looks stuck.

**What the retraining feeds the model** (generations 1–3 of `cem_fixed_elite`, rebuilt exactly as `run()` does): the battle
win rate, never the ask (0 labels equal 0.5). But every elite label lies between 0.52 and 0.72 (64 → 110 → 168 elites).
The model is retrained only on that band and then asked 0.5 (and 0.3), below anything it was retrained on.

**What each model implies** (`src/implied_label.py`). For the 1,105 precise-label teams (about 196 battles each), the ask on a
0.1–0.8 grid that best explains each team under the model (maximum-likelihood, same masks for every ask):

| model | implied vs true win rate (r) | implied mean, true < 0.3 / 0.3–0.5 / ≥ 0.5 | teams judged 0.3–0.6 |
|---|---|---|---|
| v3 as trained | +0.58 | 0.29 / 0.52 / 0.59 | 492 |
| fixed-ask loop, generation 3 (elite re-steer) | **+0.46** | 0.16 / 0.37 / 0.46 | **188** |
| Diffusion-BBO fine-tune, generation 3 | +0.64 | 0.21 / 0.45 / 0.57 | 480 |
| precise labels (in-sample for these teams) | +0.83 | 0.20 / 0.46 / 0.65 | 519 |
| from scratch, generation 2 | +0.84 | 0.19 / 0.39 / 0.59 | 643 |

**Verdict: supported.** The elite-retrained model reads the ask worst (r 0.46). It has lost the middle of the scale: it judges
most teams either 0.1–0.2 (658) or 0.7–0.8 (259), and only 188 of 1,105 anywhere in 0.3–0.6. It behaves like an
"elite or not" switch, not a graded win rate. The fixed asks 0.3 and 0.5 both fall in that dead middle, so they mean the same
thing to it, which is why the loop's chart is flat. Models trained on the full range of labels (from scratch, precise) read the
ask best (r 0.83–0.84), and they are the ones where the ask works: the from-scratch loop's got-vs-ask is r +0.39 and +0.35.

## Fixed ask or elites-only training? (2026-10-16)

The owner's mechanism: the cross-entropy loop holds the ask fixed, so the model stops conditioning on it. A 2×2 table
separates the fixed ask from the elites-only retraining:

| | fixed ask | moving ask |
|---|---|---|
| elites-only retraining | flat (`cem_fixed_elite`, asks 0.5–0.8) | flat (`cem_moving_elite`) |
| full-range training (from scratch) | `cem_scratch_fixed`, queued | the ask works (`cem_scratch`) |

Implied-label check (no battles; 1,105 precise-label teams): every elites-only model degrades the same way, whatever its ask.
- Fixed 0.5: generation 1 r +0.66 (421 of 1,105 placed in 0.3–0.6), generation 3 r +0.46 (188).
- Fixed 0.8, generation 3: r +0.49 (247).
- Moving ask, generation 3: r +0.47 (208).

The damage grows with generations of elites-only retraining, not with the ask rule. **Prediction, written before the battles:**
if the elites-only label band is the cause, `cem_scratch_fixed` keeps a working ask (got-vs-ask interval above zero in
most generations). If the fixed ask is the cause, its lines go flat.

### Result: `cem_precise_notilt` (2026-10-16)

| gen | battled win rate | asked 0.5 → got | asked 0.3 → got | got vs ask r [95%] | compositions / 512 |
|---|---|---|---|---|---|
| 0 | 0.528 ± 0.010 | 0.552 | 0.455 | +0.39 [+0.24, +0.53] | 432 |
| 1 | 0.540 ± 0.008 | 0.550 | 0.511 | +0.18 [−0.01, +0.35] | 469 |
| 2 | 0.543 ± 0.008 | 0.554 | 0.512 | +0.20 [+0.02, +0.38] | 483 |
| 3 | **0.560 ± 0.008** | 0.563 | 0.550 | +0.07 [−0.10, +0.23] | 482 |

Finalists 0.559 vs top-50 (best 0.646) and 0.546 vs ranks 51–100. Against `cem_v3_noheldout`: +0.055 [+0.015, +0.092] /
+0.036 [−0.004, +0.075]. Against `cem_fixed_elite_notilt`, `cem_fixed_elite` and the pooled fresh-seed plain loops (0.559):
no detectable difference (−0.000 [−0.032, +0.030]). The battled win rate climbs every generation to 0.560, the highest of
any loop, and the line-ups stay wide (482 of 512). The ask works in generations 0 and 2 and fades by generation 3.

### Result: fixed ask with full-range training (`cem_scratch_fixed`, 2026-10-16)

| gen | battled win rate | asked 0.5 → got | asked 0.3 → got | got vs ask r [95%] |
|---|---|---|---|---|
| 0 (shared with `cem_scratch`) | 0.510 | 0.535 | 0.436 | +0.39 [+0.22, +0.53] |
| 1 | 0.506 | 0.533 | 0.427 | +0.39 [+0.22, +0.54] |
| 2 | 0.514 | 0.551 | 0.402 | +0.52 [+0.36, +0.66] |
| 3 | 0.505 | 0.541 | 0.398 | +0.48 [+0.30, +0.64] |

**Verdict on the 2×2:** the ask stays a working dial in every generation even though it is held fixed at 0.5. It is
stronger than in the moving-ask from-scratch arm (r +0.48 vs +0.15 at generation 3).

| | fixed ask | moving ask |
|---|---|---|
| elites-only retraining | flat | flat |
| full-range training | **works** (r +0.39 to +0.52) | works (r +0.39 → +0.15) |

The owner's forward-model reading is right: the cross-entropy model stops conditioning on the ask. The mechanism is the
elites-only retraining (labels 0.52–0.72 only), not the fixed ask. Holding the ask fixed does no harm when the model trains on
the full range of labels.

Finalists of `cem_scratch_fixed` (16 × 192): 0.549 vs top-50 (best 0.672) and 0.538 vs ranks 51–100. Against `cem_v3_noheldout`:
+0.045 [−0.000, +0.089] / +0.029 [−0.017, +0.071]. The moving-ask `cem_scratch` scored 0.580 / 0.561. Holding the ask fixed keeps
the dial sharper; letting it move may help the best teams. One seed each, so that difference is not established.
