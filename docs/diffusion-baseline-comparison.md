# Diffusion versus Ling: frozen masked-completion pilot

Models: F1_final_medium_regmb_v2, G2_family_matrix_noprtrain_regmb_v2. This comparison reuses all 63 tasks, three random legal Champions M-B starting teams, masks and opponent panels from the Ling pilot.

**Coverage: diffusion supports 63/63 tasks and produces 62/63 legal completions. Ling produced 27/63 legal completions.** Unsupported tasks count as completion failures; they are not assigned invented battle losses.

One candidate per task, no retries or repairs. Diffusion uses target win rate 0.5, CFG 2, temperature 1, fixed-order inpainting. Known categorical fields are clamped. Missing vocabulary entries are rejected rather than replaced with masks. Partial move ranks cannot be inferred from hidden values: known moves are sorted, masked moves appended, alphabetical constraints disabled, and distinctness enforced.

Both battle sides use the same frozen behaviour-cloning policy, including learned team preview. Each candidate plays the frozen 50-battle schedule over 49 opponent teams. Runtime fingerprints and the exact-panel cache are shared with the original baseline. Policy RNG seeds match; Showdown RNG is not controlled.

Input limitations: diffusion cannot read opponent pastes or Stat Points; its win-rate condition is a scalar. Stat Point fills use same-species corpus spreads, with a random legal spread for species absent from the corpus. These are system comparisons, not equal-input neural architecture comparisons. There is no search or surrogate reranking in this arm.

| Task | Diffusion coverage | Legal diffusion | Legal Ling |
|---|---:|---:|---:|
| stats | 9/9 | 9/9 | 4/9 |
| item | 9/9 | 9/9 | 4/9 |
| move | 9/9 | 9/9 | 3/9 |
| pokemon | 9/9 | 8/9 | 2/9 |
| ability | 9/9 | 9/9 | 3/9 |
| alignment | 9/9 | 9/9 | 9/9 |
| mixed | 9/9 | 9/9 | 2/9 |

Battle win rates below are conditional on legal outputs. Only rows with both model scores are paired comparisons.

| Task | Original | Random fill | Ling | Diffusion | Δ diffusion − Ling (pp) |
|---|---:|---:|---:|---:|---:|
| s0-stats-1 | 0.0% | 2.0% | 0.0% | 6.0% | +6.0 |
| s0-stats-3 | 0.0% | 0.0% | 2.0% | 2.0% | +0.0 |
| s0-stats-6 | 0.0% | 0.0% | 2.0% | 6.0% | +4.0 |
| s0-item-1 | 0.0% | 2.0% | 2.0% | 0.0% | -2.0 |
| s0-item-3 | 0.0% | 0.0% | 0.0% | 2.0% | +2.0 |
| s0-item-6 | 0.0% | 2.0% | — | 0.0% | — |
| s0-move-1 | 0.0% | 0.0% | 2.0% | 0.0% | -2.0 |
| s0-move-12 | 0.0% | 2.0% | — | 0.0% | — |
| s0-move-24 | 0.0% | 2.0% | — | 10.0% | — |
| s0-pokemon-1 | 0.0% | 2.0% | — | 0.0% | — |
| s0-pokemon-3 | 0.0% | 2.0% | — | 8.0% | — |
| s0-pokemon-6 | 0.0% | 2.0% | — | 20.0% | — |
| s0-ability-1 | 0.0% | 2.0% | 0.0% | 0.0% | +0.0 |
| s0-ability-3 | 0.0% | 0.0% | — | 0.0% | — |
| s0-ability-6 | 0.0% | 4.0% | — | 4.0% | — |
| s0-alignment-1 | 0.0% | 2.0% | 0.0% | 2.0% | +2.0 |
| s0-alignment-3 | 0.0% | 0.0% | 4.0% | 0.0% | -4.0 |
| s0-alignment-6 | 0.0% | 6.0% | 0.0% | 0.0% | +0.0 |
| s0-mixed-1 | 0.0% | 2.0% | — | 4.0% | — |
| s0-mixed-20 | 0.0% | 2.0% | — | 2.0% | — |
| s0-mixed-40 | 0.0% | 2.0% | — | 4.0% | — |
| s1-stats-1 | 2.0% | 4.0% | — | 4.0% | — |
| s1-stats-3 | 2.0% | 2.0% | — | 2.0% | — |
| s1-stats-6 | 2.0% | 2.0% | — | 4.0% | — |
| s1-item-1 | 2.0% | 6.0% | 2.0% | 4.0% | +2.0 |
| s1-item-3 | 2.0% | 4.0% | — | 0.0% | — |
| s1-item-6 | 2.0% | 8.0% | 12.0% | 4.0% | -8.0 |
| s1-move-1 | 2.0% | 2.0% | 2.0% | 0.0% | -2.0 |
| s1-move-12 | 2.0% | 0.0% | — | 0.0% | — |
| s1-move-24 | 2.0% | 0.0% | — | 12.0% | — |
| s1-pokemon-1 | 2.0% | 6.0% | 0.0% | 2.0% | +2.0 |
| s1-pokemon-3 | 2.0% | 0.0% | — | — | — |
| s1-pokemon-6 | 2.0% | 0.0% | — | 32.0% | — |
| s1-ability-1 | 2.0% | 0.0% | 2.0% | 0.0% | -2.0 |
| s1-ability-3 | 2.0% | 4.0% | — | 0.0% | — |
| s1-ability-6 | 2.0% | 0.0% | — | 10.0% | — |
| s1-alignment-1 | 2.0% | 6.0% | 4.0% | 4.0% | +0.0 |
| s1-alignment-3 | 2.0% | 4.0% | 4.0% | 8.0% | +4.0 |
| s1-alignment-6 | 2.0% | 8.0% | 4.0% | 2.0% | -2.0 |
| s1-mixed-1 | 2.0% | 2.0% | 2.0% | 2.0% | +0.0 |
| s1-mixed-20 | 2.0% | 4.0% | — | 2.0% | — |
| s1-mixed-40 | 2.0% | 0.0% | — | 6.0% | — |
| s2-stats-1 | 4.0% | 2.0% | — | 0.0% | — |
| s2-stats-3 | 4.0% | 2.0% | 2.0% | 0.0% | -2.0 |
| s2-stats-6 | 4.0% | 4.0% | — | 2.0% | — |
| s2-item-1 | 4.0% | 0.0% | — | 2.0% | — |
| s2-item-3 | 4.0% | 2.0% | — | 2.0% | — |
| s2-item-6 | 4.0% | 2.0% | — | 4.0% | — |
| s2-move-1 | 4.0% | 0.0% | 4.0% | 2.0% | -2.0 |
| s2-move-12 | 4.0% | 2.0% | — | 4.0% | — |
| s2-move-24 | 4.0% | 4.0% | — | 4.0% | — |
| s2-pokemon-1 | 4.0% | 4.0% | — | 20.0% | — |
| s2-pokemon-3 | 4.0% | 0.0% | 20.0% | 6.0% | -14.0 |
| s2-pokemon-6 | 4.0% | 0.0% | — | 24.0% | — |
| s2-ability-1 | 4.0% | 4.0% | 8.0% | 8.0% | +0.0 |
| s2-ability-3 | 4.0% | 4.0% | — | 0.0% | — |
| s2-ability-6 | 4.0% | 0.0% | — | 2.0% | — |
| s2-alignment-1 | 4.0% | 2.0% | 0.0% | 4.0% | +4.0 |
| s2-alignment-3 | 4.0% | 4.0% | 2.0% | 0.0% | -2.0 |
| s2-alignment-6 | 4.0% | 6.0% | 0.0% | 2.0% | +2.0 |
| s2-mixed-1 | 4.0% | 0.0% | 2.0% | 4.0% | +2.0 |
| s2-mixed-20 | 4.0% | 2.0% | — | 2.0% | — |
| s2-mixed-40 | 4.0% | 2.0% | — | 4.0% | — |

27 jointly legal, scored tasks. With only three starting teams, this pilot cannot establish which model is better across the requested task distribution.

Next experiment: expand the frozen cohort and keep development selection separate from final battle evaluation. Report coverage and legality separately from conditional battle gains. Stat Points still need a learned representation before calling this a fully neural completion system.

Unsupported values:

Equal-start-weighted diffusion minus Ling on jointly scored tasks: -0.52 percentage points. This is descriptive, not a significance claim.

- s1-pokemon-3: no legal token under clamped context
