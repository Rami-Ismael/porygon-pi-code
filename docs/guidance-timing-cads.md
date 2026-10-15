# Guidance timing and condition noise: two cheap diversity arms (pre-registration, 2026-10-15)

**Question.** The 2026-10-14 diversity-in-diffusion sweep left two untested levers in the sampler itself:
the classifier-free guidance weight is applied at every one of the 48 decode steps, and the win-rate
condition enters clean. Applying Guidance in a Limited Interval (Kynkäänniemi et al., NeurIPS 2024) and
Improving Classifier-Free Guidance in Masked Diffusion (ICLR 2026) both find that guidance while the
sample is still mostly masked is what drops modes; CADS (Sadat et al., ICLR 2024) recovers diversity by
corrupting the condition early and annealing the corruption away. Does either lever widen the species
line-ups of the generation-0 proposer without costing win rate?
Already failed, not retried: taboo list (diversity yes, −0.055 win rate), DPP / MMR elite filters,
entropy bonus in elite selection, island, peeling, negative prompting, sampler swaps other than
free-species / focused-sets temperature (`docs/diversity-methods.md`, `docs/redundancy-penalty.md`).

**Win rate** = wins / battles against the top-50 meta teams, one battle per (team, top-50 column),
behaviour-cloning policy both sides (policy 1), all in the matchup matrix (origin `gtc:<arm>_s<seed>`).

## Where the change goes
The loop's proposer is `foundation.Gen.sample` (species → ability/item → moves → nature in
`diffusion.ORDER`, D-CFG on the win-rate condition, ridge tilt, legality mask), not
`diffusion.sample_constrained`, which belongs to the older style-conditioned model. The condition is a
scalar win rate embedded by `lossdown.Net.wvec` (Fourier features → two-layer MLP → one 192-d vector added
to every token; a learned null vector when the condition is dropped). Each of the two v3 members has its
own `wvec`.

## Shared design (`src/guidance_timing_cads.py`)
- Model: the generation-0 v3 members without held-out leakage (`F1_final_medium_noheldout_regmb_v3`,
  `G2_family_matrix_noprtrain_noheldout_regmb_v3`), the foundation ridge `foundation_ridge_noheldout.npz`,
  no fine-tuning, no loop labels. Identical to `cem_v3_noheldout` generation 0.
- Control settings, as in that generation: ask 0.5, guidance 2, temperature 1, ridge guidance at the
  ridge's own λ, nature repair, Showdown validator, held-out ranks 51–100 refused as proposals.
- Per (arm, seed): the first **512 validator-accepted proposals in sampling order** (cap 8 attempts per
  accepted proposal); the **64 highest by ridge score** are battled, 49 battles each — the loop's own
  acquisition rule at generation 0 with half its battle budget.
- Seeds 1, 2, 3: every arm starts each seed from the same torch and numpy state, so arms differ only by
  their mechanism. Three seeds: intervals are exploratory.

## Arms
| arm | change to the control sampler |
|---|---|
| `control` | none |
| `interval` | classifier-free guidance weight 1 (conditional only, no unconditional mix) while the six species columns decode — the first six steps of `diffusion.ORDER` — and the usual weight 2 on the remaining 42 fields. The ridge tilt is unchanged on every column, so the arm isolates guidance timing. (`Gen.sample(g_species=1.0)`) |
| `cads` | Gaussian noise on the win-rate condition embedding, per member, fresh at every step and row: at step k of K = 48, γ = k / (K − 1), ŷ = √γ · y + s · σ_y · √(1 − γ) · ε, then rescaled to y's own mean and standard deviation across its 192 coordinates (CADS with ψ = 1, τ₁ = 0, τ₂ = 1). s = 0.1, stated relative to σ_y because the embedding's scale is a free parameter of the network. Noise touches only the conditional pass; the unconditional pass keeps the null vector. Guidance stays at 2 everywhere. (`lossdown.COND_NOISE`) |

Per-step noise-to-signal ratio in `cads`: s · √((1 − γ) / γ) — 0.68 at the first species column, 0.26 at
the sixth, 0.10 at the midpoint, 0 at the last field. The first step (γ = 0) is a random direction at the
embedding's scale, as in CADS.

## Verification
On the 512 accepted proposals of every (arm, seed):
1. distinct species line-ups /512, exp(H) of the line-up frequencies, largest line-up share;
2. **Vendi score** (Friedman & Dieng, TMLR 2023) with kernel K_ij = shared species / 6 (the Gram matrix of
   the species indicators / √6; K_ii = 1): exp of the von Neumann entropy of K / 512;
3. 48-field unique fraction (exact repeats).
On the 64 battled: mean win rate with a bootstrap 95% interval (teams resampled), best, count ≥ 0.531,
distinct line-ups among the battled; per arm, the mean over seeds of (arm − control) with a bootstrap 95%
interval (teams resampled within seed, 4,000 draws).
Reference points: `cem_v3_noheldout` generation 0 has 156 line-ups /512 and battled mean 0.458 (128 teams).

## Decision rule (fixed before battles)
An arm **widens line-ups without losing win rate** if
(a) its distinct line-ups /512 exceed `control`'s in all 3 seeds, AND
(b) the bootstrap 95% lower bound of its battled win rate minus `control`'s is above −0.03.
Vendi score and largest share are reported, not part of the rule. A pass earns the arm a place inside
the cross-entropy loop (the next experiment); a fail retires it. Do not rerun an arm that passes or fails.

## Result (2026-10-15, `results/guidance_timing_cads.json`)
Line-ups /512 by seed: control 149 / 135 / 160; interval 282 / 247 / 292; cads 380 / 399 / 386. Vendi: 4.1,
10.3, 28.5 (mean of seeds). Battled win rate (64 teams × 49 battles, mean of seeds): control 0.427,
interval 0.414, cads 0.393. Arm − control: interval −0.013 [−0.040, +0.015]; cads −0.034 [−0.059, −0.008].
**Both arms fail** rule (b): every seed widens line-ups, but neither win-rate lower bound clears −0.03.
Interval is the near miss (point −0.013); CADS pays for its 2.6× line-ups with a real win-rate loss.
Count ≥ 0.531 among the 64: control 21 / 21 / 11, interval 15 / 9 / 14, cads 10 / 6 / 12.
