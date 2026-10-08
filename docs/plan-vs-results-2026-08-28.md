---
created_at: 2026-08-28
updated_at: 2026-08-28
type: audit
---

# The 2026-08-28 plan, item by item, against what has been measured

Every number below is a mean win rate against the 50 top-placement Reg M-B meta teams, both
sides piloted by the VGC-Bench behaviour-cloning policy, errors clustered by team. An
untouched real corpus team scores **0.470** (median 0.458). Companion: the primary-source
review in `discrete-diffusion-plan-review.md`.

| Plan item | Status | What was measured | Where |
|---|---|---|---|
| 1. Features: 6 species × (4 moves, ability, item, nature, 6 Stat Points) | done | 8 categorical fields per slot (`diffusion.py`, 48 columns); the 14-field variant also generates Stat Points (`legaldiffusion.py`) — generated spreads win 2.1 points less than copied real spreads (−2.3 σ, 150 paired twins); steering the spread alone costs 7.7 points (−6.6 σ). Spreads are copied from real sets. | `results/legal_eval.json`, `results/paired_eval.json` |
| 2. Model-based search with a diffusion generator | done, closed on the objective | From scratch 0.093 → 0.193 (win-rate condition, g=4) → **0.260** (temp 0.7 / top-p 0.9). The copy-paste cross-entropy loop reaches 0.494 (ρ=0.20) and **0.540 (ρ=0.01)** with the same battle budget. Inside the loop the diffusion proposer does not climb (0.165 → 0.172 → 0.153). | `results/loop_results.json`, `results/filterrate_results.json` |
| 3.2 CFG on playstyle | works mechanically, cost unresolved | rain 48/48, sand 27× base rate; steered 0.371 vs unsteered 0.406 is 0.73 σ at 16 teams — needs ~700 teams per arm. | `results/steer_eval.json` |
| 3.2 CFG on legality | done, negative ×3 | Global token, heavy-negative token, per-slot token: 0/2,880 whole legal teams unconstrained; +0.8 / −1.4 / +1.4 σ on top of constraints. Learned legality classifier at majority-class rate three times. | `results/legal_eval*.json`, `results/slotlegal_eval.json` |
| 3.2 CFG on win rate, "set it as high as possible" | done | 6-bin condition, top bin [0.5, 1]: 0.128 → 0.193 ± 0.012 at g=4 (4.8 σ); bottom-bin control 0.006. Legality 88 → 77 → 29 % as g goes 1 → 2 → 4; species entropy 5.79 → 3.42 bits from g=1 to g=8. | `results/wr_eval.json`, `results/guidance.json` |
| 4.1.1 Masking schedule (mask a % per feature, unmask n % per step) | done as MDLM; per-feature variant not published | `loss()` draws t ~ U[0,1] and masks each column with prob t, weight 1/t (the ELBO weight for α_t = 1 − t). MD4's per-value schedule is learned and overfits; MDLM proves the ELBO is schedule-invariant — see review §A. | `src/wrdiffusion.py:72-92` |
| 4.1.2 Legality via data: random legal teams ≫ VGCPastes, battled, multiprocess, resumable, balanced brackets | done except balance | 4,850 labelled teams / 116,400 battles (`ladder.py`, `label_strata.py`); random legal teams score median **0.000** — no signal; the pool runs 63 battles/s on 7 Showdown servers (`pool.py`). Bin counts under the shipped edges: [1397, 481, 785, 614, 608, 965]; only 147 teams ≥ 0.667, none ≥ 0.833. **Balance / growth: run 2026-08-28, see below.** | `results/ladder_labels.json`, `results/strata_labels.json` |
| 4.1.2 alt: mutate real teams (slotcopy, shuffled spreads, model regenerates a slot) | done | slotcopy 0.436 (1 paste), 0.431 (2), 0.410 (3); shuffled 0.449; model edit 0.381; random candidate −8.7 points each, real candidate −1.9 each. | `results/strata_labels.json`, `gui/static/hist.html` |
| 4.2.1 Inference schemes + visual | done | 15 schemes at one checkpoint: order is null (dependency 0.174 = confidence 0.176; random 0.134, entropy-first 0.105, parallel 0.127, remask 0.154); **temperature 0.7 / top-p 0.9 → 0.260 at 97 % valid**; no-mask arms 0 valid in 20,000 × 4. Visual: `gui/static/schemes.html`. | `results/schemes_results.json`, `src/build_schemes.py` |
| 4.2.2 Whole team / fill in the blank | done | whole team 0.260 (best scheme); regenerate one move on a real team 0.436; one candidate 0.368; build around a pinned Garchomp 0.269. | `results/schemes_results.json` |
| 5.1 Objective: VGC-Bench BC policy from Hugging Face | done | `/tmp/bc_100.zip` (checkpoint 100 of `results/saves_bc/seed1/`, the only checkpoints on the Hub) via `shard.py`. | `src/shard.py:40` |
| 5.2 Top-50 meta from the VGCPastes sheet | done | 50 top-placement teams (`top50_evs.json`, `loop.opponents()`); the CSV export exposes Tournament / Event and Rank columns — review §F. | `results/top50_evs.json` |
| 6. Slot Copy baseline | done | 0.420 (16 teams) / 0.436 (300 teams) for one paste; as a proposer between elite teams, 0.540 at ρ=0.01. | `results/strata_labels.json`, `results/filterrate_results.json` |
| 7.1.1 Bigger network | done | 0.6M / 2.1M / 7.9M params → 0.162 / 0.172 / 0.195 (+3 points per 10×, 3.0 σ). | `results/scale_results.json` |
| 7.1.2 Bigger / balanced dataset | data-size unresolved; balance run 2026-08-28 | 25 / 50 / 100 % → 0.150 / 0.183 / 0.172 (single runs, ±3-4 points seed noise); keeping only winners costs 5–11 points (losers are load-bearing for CFG contrast). Balance and top-end growth: below. | `results/sampling_results.json`, `results/brackets_results.json` |
| 7.2 Ablation | partly | k-sweep of the local move (`ablation.py`); Stat Alignment repair +1.3 points (2.6 σ pooled); decode-order ablation above. No ablation of the conditioning tokens or of the model depth yet. | `results/ablation.json`, `results/alignment_results*.json` |
| 7.3 Grid | not run | No joint (guidance × temperature × bin) grid exists; the one-at-a-time sweeps are guidance 1/2/4/8 and temperature 1.0/0.7. | — |
| 7.4 CEM elite filter rate vs the meta | done | ρ 1.0 → 0.005: 0.244 → 0.548, monotone; 0.01 is the tutorial's own recommendation (review §G). Diffusion density under the same filter caps at 0.187. | `results/filterrate_results.json` |

## Bracket balance and top-end growth (run 2026-08-28, `src/brackets.py`)

Every arm fine-tunes the shipped `wrdiffusion.pt` for 4,560 steps (filterrate.py's recipe,
lr 1e-4), decodes 150 Showdown-valid teams with the best scheme (dependency order, temp 0.7,
top-p 0.9, guidance 2, top bin) and battles each 24 times. The grown pool adds the 1,600
copy-paste proposals from the ρ sweep, re-labelled by `src/label_cp.py` (`results/cp_labels.json`;
the re-battle reproduces the sweep's means, ρ=0.01 0.557 vs 0.540). The 8-bin edges add cuts at
0.6 and 0.7, so the top bin is [0.7, 1] instead of [0.5, 1]. "Balanced" draws every bin equally
often (DDOM-style reweighting); uniform draws every team equally often.

| arm | pool | bins | top bin | balanced | win rate | Δ vs base | valid | p90 | ≥ real median |
|---|---|---|---|---|---|---|---|---|---|
| base | 4,850 | 6 | [0.5, 1] · 880 teams | no | **0.276 ± 0.012** | — | 98 % | 0.458 | 12 % |
| balanced6 | 4,850 | 6 | [0.5, 1] | yes | 0.262 ± 0.011 | −1.4 (−0.9 σ) | 94 % | 0.417 | 8 % |
| base_fine | 4,850 | 8 | [0.7, 1] · 72 teams | no | 0.254 ± 0.010 | −2.2 (−1.4 σ) | 88 % | 0.417 | 9 % |
| grown | 6,450 | 6 | [0.5, 1] · 1,635 teams | no | 0.298 ± 0.011 | +2.1 (+1.3 σ) | 94 % | 0.500 | 16 % |
| grown_fine | 6,450 | 8 | [0.7, 1] · 161 teams | no | 0.334 ± 0.011 | +5.8 (+3.5 σ) | 94 % | 0.542 | 23 % |
| grown_fine_bal | 6,450 | 8 | [0.7, 1] | yes | **0.356 ± 0.012** | +8.0 (+4.9 σ) | 79 % | 0.542 | 28 % |

- **Balancing the brackets on the existing pool does nothing** (balanced6 −0.9 σ); neither does a
  finer top bracket on the existing pool (base_fine −1.4 σ, with 72 teams in the bracket the
  condition is too thin). The owner's "balance the number per bracket" is not the lever on its own.
- **Growing the top end is the lever.** Adding 1,600 labelled teams whose mean is 0.47 lifts the
  same model +2 points with the old bins and +6 points once the top bracket is cut at 0.7 — the
  bracket only pays when it is populated (161 vs 72 teams). Balancing on top of that adds ~2 more
  points (+1.4 σ over grown_fine, not resolved) at a visible legality cost (94 → 79 % valid).
- 0.356 is the best number the generator has produced (previous best 0.260), 28 % of proposals at
  or above the median real team, still 11 points under a real team (0.470) and 18 under the
  copy-paste proposer at ρ = 0.01 (0.540) — whose labelled output is exactly what closed the gap.
  The generator is being taught by the dictionary proposer, not the other way round.
- Caveat: one seed per arm; run-to-run noise on this model is ~3–4 points, so only grown_fine and
  grown_fine_bal vs base are outside it. Novelty is unchanged across arms (PMI surprise 2.29–2.36
  vs 1.84 for a real team).
