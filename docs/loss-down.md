# Loss-down retrain (2026-10-07)

**Goal (owner).** Retrain the team diffusion model on everything we own — the matchup
matrix and every team the loops wrote — get the loss below every earlier run, and use
current papers to find techniques that push it further.

Code: `src/lossdata.py` (the data table), `src/lossdown.py` (model, benchmark, runs).
Results: `results/lossdown.json`; checkpoints in `~/.local/share/vgc-pilot-runtime/lossdown/`
(outside iCloud on purpose).

## The benchmark (one, fixed, never trained on)

The 500 held-out teams of `asked_vs_got.train()` — the split `loss_diagnosis.py` used.
Every model, old checkpoints included, is scored on **the same masks** (same slot
shuffles, same t, same mask draws — common random numbers), conditioned on the teams'
win-rate labels as the old "held-out cond" number was. Differences between models are
therefore paired per team; the ± on a difference is the paired standard error over the
500 teams.

Three numbers:

| name | what it is | old model |
|---|---|---|
| **legacy** | what every earlier training log printed ("held-out cond ~4.9"): Σ CE/t over masked fields ÷ number of masked fields, pooled, with a draw that masks nothing forced to mask one field | 4.960 |
| **ELBO** | the proper MDLM/MD4 bound on −log p(team), nats per field: E_t[Σ_masked (w_c/t)·CE_c]/48; a draw with nothing masked contributes 0 | 2.368 |
| ce | plain cross-entropy per masked field at the same masks | 2.574 |

legacy ≈ 2 × ELBO (pooling divides by ≈ 24 masked fields, not 48). The forced-mask
convention inflates both legacy and an ELBO computed that way: ∫(1−t)⁴⁸/(48t)dt
diverges logarithmically as t→0, about +0.06–0.17 nats/field depending on the t floor.
The first numbers of this run had that bias; every number below is from the corrected
evaluator (`lossdown.py rescore`).

## Data

`lossdata.py` walks every `results/*.json`, the matchup matrix (`~/vgc-data/matchup_regmb.sqlite`,
policy 1, top-50 columns), the 692-team corpus and the 100k HPS file, and deduplicates on
the 48-field encoding. **146,994 distinct teams** (the old model saw 14,736), 44,887
labelled, 10,272 with a matrix row. Any team whose encoding equals a test team's is
removed (every occurrence, all sources). Each team carries a source-family bitmask (real, old pool, Aug–Sep
loops, temperature campaign, cem/reverse-score, redundancy/asked-vs-got, matrix, random
fill, HPS). 95,193 of the HPS teams use a move or item outside the corpus vocabulary and
are not encodable; 4,807 are kept.

## Techniques tried (papers)

| technique | source | here |
|---|---|---|
| structured output: logits restricted to values the visible fields allow (species↔ability, species↔learnset, species↔Mega stone, Species/Item Clause, canonical move order) | domain knowledge; the joint decoder that zeroes inconsistent values is also Prime's output layer (Chao et al., NeurIPS 2025) | verified exact on all 146,994 teams and the test 500 (leave-one-out, the most restrictive case: 0 exclusions) |
| more data, 10× | owner | |
| source tags as metadata conditioning | Gao et al., *Metadata Conditioning Accelerates Language Model Pre-training*, ICML 2025 | |
| pretrain on all, fine-tune on the target family | standard transfer | |
| width/depth | Nie et al. scaling; Sahoo et al., *Scaling Beyond Masked Diffusion Language Models*, ICML 2026 | |
| per-field-type polynomial schedule αc = 1 − t^wc (non-uniform unmasking order) | Shi et al., *Simplified and Generalized Masked Diffusion for Discrete Data* (MD4/GenMD4), NeurIPS 2024; Garg et al., *Masked Diffusion Models are Secretly Learned-Order Autoregressive Models*, EurIPS 2025 workshop | |
| time-independent network | Ou et al., *Your Absorbing Discrete Diffusion Secretly Models the Conditional Distributions of Clean Data* (RADD), ICLR 2025 | |
| stratified t across the batch | MD4 Algorithm 1 | on in every new run |
| partial masking (each field = 2 sub-tokens) | Chao et al., *Beyond Masked and Unmasked: Discrete Diffusion Models via Partial Masking* (Prime), NeurIPS 2025 | |
| EMA weights, ensembles | standard | |
| matchup-row conditioning | the matchup matrix | |

Read but not run: DUEL (exact likelihood under a deterministic unmasking policy, ICML 2026)
— a different metric, not a training technique; P-POTS / MIRROR variance reduction
(Jia et al., ICLR 2026) — reports stability and downstream accuracy, not lower loss.

## Results (test 500, conditioned, corrected evaluator; paired diff vs old best)

Four numbers. **legacy** and **ELBO** as defined above. **own** = ELBO under the run's own
unmasking schedule (its training objective; a valid bound for the model sampled in that
order). **exact** = −log p(team) of the sampler that reveals columns in `diffusion.ORDER`
(species → ability/item → moves → nature, the order every loop already decodes in) —
computed exactly, not bounded (DUEL, ICML 2026), 4 slot shuffles per team.

| run | legacy | ELBO | own | exact | ELBO vs old (paired) |
|---|---|---|---|---|---|
| old `asked_vs_got.pt` | 4.960 | 2.368 | – | 2.362 | 0 |
| old + structured output at eval only | 3.785 | 1.830 | – | 1.593 | −0.538±0.006 |
| A1 all 145k teams + structured output | 3.773 | 1.823 | – | | −0.545±0.008 |
| B1 + source tags | 3.805 | 1.838 | – | | −0.530 |
| D1 pretrain all → fine-tune family | 3.689 | 1.784 | – | | −0.584 |
| B2 test's family only (15k), forced-mask training | 3.683 | 1.782 | – | | −0.586±0.006 |
| B3 = B2 without forced masks in training | 3.604 | 1.743 | – | 1.489 | −0.625±0.007 |
| N1 = B3 without time input (RADD) | 3.606 | 1.743 | – | 1.493 | −0.625±0.007 |
| C2 family, d384×6 (forced-mask training) | 3.614 | 1.751 | – | | −0.617±0.007 |
| P1 Prime ℓ=2 | – | 2.026 | – | | −0.342 |
| S1 species-first schedule (2,1,1,1,1) | 3.633 | 1.757 | 1.642 | | −0.726±0.008 (own) |
| S5 block order (8,3,3,1.5,1) | 3.707 | 1.792 | 1.541 | 1.452 | −0.827±0.014 (own) |
| **S7 sharper block order (16,4,4,1.5,1)** | 3.775 | 1.829 | **1.508** | **1.451** | **−0.860±0.016 (own)** |
| MO1 = S5 with moves in popularity order | 3.703 | 1.796 | 1.558 | 1.478 | −0.810±0.014 (own) |
| F2 / F3 = S7 order, val folded in, seeds 1 / 2 | 3.69 | 1.82 | 1.506 / 1.504 | 1.449 / 1.445 | |
| ensemble F1+F2+F3+S7 (mixture of predictions) | 3.694 | 1.793 | 1.484 | | worse than F1 alone |
| F1: S7 order, d256×6, val folded in, 80 epochs — loop teams only | 3.678 | 1.786 | 1.481 | 1.426 | −0.887±0.017 (own) · exact −0.935±0.010 |
| G0: same arch, **all 145,994 teams + matrix rows**, 6 epochs | | 1.865 | 1.556 | 1.501 | |
| G1: G0 → fine-tune on loop + matrix teams (25,914), 40 epochs | | 1.801 | 1.502 | 1.448 | |
| G2: loop + **all 9,272 matrix teams, row-conditioned** (25,914), 60 epochs | | 1.787 | 1.490 | 1.434 | |
| **FINAL = F1 + G2 (mixture of predictions)** | | | **1.473** | **1.418** | **exact −0.944±0.010 (−40%); −0.0085±0.0012 vs F1** |
| F1 + G2 + G1 | | | 1.476 | 1.421 | |

Re-scored with 64 mask draws and two fresh seeds: S5 own 1.543 / 1.552, S1 own 1.632 / 1.639 —
the 16-draw numbers hold to ±0.01.

Read: (1) structured output is the biggest single lever (−0.54); (2) in-distribution data
beats 10× off-distribution data; (3) dropping the forced-mask training convention −0.04;
(4) unmasking order is the biggest paper technique: species-first −0.10, full block order
−0.20 vs B3 — the order lets the legality masks bind before the fields they constrain are
predicted; (5) width −0.03; (6) time input, source tags, all-data pretraining: no effect;
Prime: worse (its half-revealed fields can't use the legality masks).

A sharper order (S7) tightens the bound toward the exact loss of the block-order sampler
(own 1.508 vs exact 1.451) without changing the model: S5 and S7 have the same exact loss
(1.4517 vs 1.4514). Moves in popularity order (Protect first) are worse than alphabetical
(moves 0.610 vs 0.583): alphabetical order plus the order mask prunes the later moves.
Two block-order models mixed (S5+S7, same seed) gain only −0.002±0.0004 exact.

Exact-order breakdown, S5 (nats per field, summed over the type's columns ÷ 48): species
0.379 · ability 0.024 · item 0.252 · moves 0.583 · nature 0.214. Moves are 40% of what is left.

## Final model

**F1 + G2**, predictions mixed field by field (`lossdown.ensemble_fn`); checkpoints
`~/.local/share/vgc-pilot-runtime/lossdown/{F1_final_medium,G2_family_matrix_noprtrain}.pt`.
Both: legality-masked logits, block unmasking order 16,4,4,1.5,1, d256 × 6 layers, validation
teams folded back in. What each trained on:

- **F1** — the ~16k Aug–Sep loop teams (the old pool, activesearch, entropyloop, gradguide,
  gradloop, dsame, mdts) + the 692-team corpus.
- **G2** — the same plus **every team in the matchup matrix that has a row against the top-50**
  (9,272; 1,000 more held out to measure the row), each conditioned on its per-opponent row.

On the test 500: exact loss of the deployed decoder **2.362 → 1.418 nats/field (−0.944±0.010
paired, −40%)**; ELBO under its own schedule **2.368 → 1.473**. Every one of the 500 test teams
is better predicted by F1 alone; the mixture adds −0.0085±0.0012 on top.

The other ~120k teams (temperature campaign, cem / reverse-score, redundancy, random fill,
HPS) were tried three ways and none lowered the test loss: A1 (all data, one stage) 1.823 vs
B2 1.782 uniform ELBO; G0 → G1 (pretrain on all 146k, then fine-tune) exact 1.448 vs G2 1.434
without the pretraining; adding G1 to the final mixture 1.421 vs 1.418. They are a different
distribution from the test teams (other generators, other temperatures).

Reproduce: `python lossdata.py`; then
`python lossdown.py run F1_final_medium data=family final=1 sched=16,4,4,1.5,1 d=256 nlayer=6 nhead=8 lr=4e-4 epochs=80` and
`python lossdown.py run G2_family_matrix_noprtrain data=family+matrix final=1 mrow=1 sched=16,4,4,1.5,1 d=256 nlayer=6 nhead=8 lr=4e-4 epochs=60`;
`python lossdown.py exact F1_final_medium G2_family_matrix_noprtrain`.

Sampler: `lossdown_battle.sample` (fixed-order constrained decode of the mixture, CFG on win rate).
Battled — see the next section.

## Battles: does the better fit write better teams? (2026-10-08)

`src/lossdown_battle.py`, same protocol as experiment A (asked_vs_got): 64 Showdown-valid teams per
cell, one battle per (team, top-50 column), BC policy both sides, all 31,360 battles in the matrix
(origin `lossdown:<cell>`). Win rate vs the top 50; a top-50 team scores 0.531 against the other 49.

| cell | new (F1+G2) | old (asked_vs_got.pt) | diff | valid new/old | species sets new/old |
|---|---|---|---|---|---|
| unconditioned | 0.257 ± 0.015 | 0.185 ± 0.014 | +0.072 (3.5σ) | 100% / 97% | 57 / 59 |
| ask 0.3 g1 | 0.251 ± 0.014 | 0.203 ± 0.015 | +0.048 (2.3σ) | 98% / 97% | 57 / 60 |
| ask 0.5 g1 | 0.316 ± 0.015 | 0.211 ± 0.013 | +0.105 (5.3σ) | 100% / 98% | 39 / 52 |
| ask 0.7 g1 | 0.326 ± 0.013 | 0.208 ± 0.014 | +0.118 (6.0σ) | 100% / 89% | 40 / 58 |
| ask 0.3 g2 | 0.308 ± 0.016 | 0.185 ± 0.012 | +0.123 (6.2σ) | 98% / 98% | 47 / 62 |
| **ask 0.5 g2** | **0.402 ± 0.013** | 0.282 ± 0.017 | +0.121 (5.6σ) | 97% / 97% | 24 / 43 |
| ask 0.7 g2 | 0.335 ± 0.016 | 0.240 ± 0.015 | +0.095 (4.3σ) | 97% / 93% | 43 / 54 |
| ask 0.3 g4 | 0.300 ± 0.013 | 0.191 ± 0.011 | +0.108 (6.4σ) | 100% / 97% | 50 / 60 |
| ask 0.5 g4 | 0.359 ± 0.014 | 0.322 ± 0.013 | +0.038 (1.9σ) | 100% / 94% | 34 / 26 |
| ask 0.7 g4 | 0.330 ± 0.019 | 0.319 ± 0.015 | +0.011 (0.4σ) | 97% / 74% | 48 / 30 |

The better density model does write better teams: ahead in all 10 cells, 8 beyond 2σ; best cell 0.402
vs the old model's best 0.322. Still below a real team (0.531). The gap closes at guidance 4, and the
best cell is the least varied (24 line-ups in 64). 0 copies of any known team.

## The matchup matrix as a condition (M1)

One model (all data, structured output, uniform schedule, 5 epochs) conditioned on the scalar
win rate **and** the team's matrix row (win rate vs each top-50 column, NaN = untested; the
row is dropped whole 30% of the time and column by column 20%). Scored on 1,000 matrix teams
with ≥ 40 tested columns that it never trained on, three ways on the same masks:

| condition | ELBO (nats/field) |
|---|---|
| nothing | 2.2276 |
| scalar win rate | 2.2055 |
| scalar + per-opponent row | 2.1908 |

Row beyond scalar: **−0.704 ± 0.066 nats per team** (paired, 1,000 teams); scalar beyond
nothing: −1.063 ± 0.057. Who a team beats carries about two-thirds as much information about
the team as how often it wins — measured from one battle per cell. The test 500 barely
appear in the matrix (15 of 500), so this does not move the headline number; it is the
evidence that counter-conditioning ("write a team that beats these opponents") has signal.

Re-measured on the same 1,000 holdout teams inside the bigger row-conditioned models: G0 −0.0139±0.0013,
G1 −0.0061±0.0008, G2 −0.0036±0.0007 nats/field (own schedule). Still clearly non-zero, but it shrinks as the
model is trained harder on the loop teams; the M1 number (−0.0147) is the one measured on a model trained on all data.

Ops: three runs at once filled 24 GB and swapped; evaluation batch is now 500 with
`torch.mps.empty_cache()`. Run at most two at a time.

## Known limits found later (2026-10-08, Codex regmb v2/v3 audit)

A masked-completion benchmark against an LLM (Codex session, `regmb-vocabulary-retraining.md`,
`empty-moves-and-mega-stone-diagnosis.md` in the GitHub mirror) found that everything above runs on
the **corpus vocabulary**: 189 of 347 legal species, 127/200 abilities, 111/148 items, 316/496 moves,
19/25 natures. Consequences for this page:
- F1, G2 and every model here can never write the other 158 species; completing a real team that
  holds an out-of-vocabulary value fails (7/63 inputs representable in that benchmark).
- The 500-team test set and all training rows are the teams that encode fully under that vocabulary
  (95,193 HPS teams and ~2.3k others were dropped as "unencodable") — a selection, not the whole
  regulation.
- `Vocab.encode` maps unknown values to [MASK] silently; checkpoints here store no token identities.
- The "species↔Mega Stone" rule in the structured output is a **stone-ownership heuristic**, not
  legality: Showdown accepts any species holding any stone. It held on all our data, which is why the
  check above passed.
The fixes (regulation vocabulary from the pinned simulator, token-identity checkpoints, exact-forme
stone policy, variable move counts, joint ability×item table) are Codex's `F1/G2_*_regmb_v3`; their
losses are on a larger output vocabulary and are not interchangeable with the numbers on this page.
