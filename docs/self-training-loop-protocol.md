# Self-training loop — stopping rules and per-generation evaluation

Goal (owner, 2026-08-29): use the diffusion model as a search tool — sample teams; any Showdown-valid team not already in the dataset is added; retrain; repeat. Open questions: when does the loop stop, and what evaluation shows the model keeps working.

This is the *self-consuming generative model* setting. Six papers, each verified 2026-08-29 by a second adversarial agent reading the full text (all links open free; grade = does it change this loop's design):

1. 🟢 [Alemohammad et al., ICLR 2024](https://arxiv.org/abs/2307.01850) — **HIGH.** Three loop types: fully-synthetic degrades; *synthetic augmentation* (fixed real set + growing synthetic — structurally this loop, their §4.2/Fig 8) "only delays the inevitable degradation"; fresh-real-data loops don't degrade given enough fresh data. Biased sampling (temperature, top-p, guidance) preserves precision but makes recall/diversity collapse **faster** (§3.4, Fig 6; empirical, not a theorem).
2. 🟢 [Bertrand et al., ICLR 2024 spotlight](https://arxiv.org/abs/2310.00429) — **HIGH.** Theorem 1 (sufficient, local): iterative retraining stays near the optimum when λ(1+Lε/α) < 1/2, where λ is the synthetic-to-**real** ratio — synthetic under ~half the real count, i.e. ~a third of the training mix. The constants are unmeasurable in practice, so treat it as a heuristic budget, not a guarantee (their EDM run was stable even at λ=1). Pure self-consuming loops provably collapse (Prop 1, Gaussian: √covariance shrinks geometrically).
3. 🟢 [Shumailov et al., Nature 631, 2024](https://www.nature.com/articles/s41586-024-07566-y) — **MED.** Recursive training loses the tails first, then converges to low diversity; keeping even 10% original data limited damage to minor degradation (this loop retains 100%). Borrowable diagnostic: score each generation's samples under the generation-0 model, watch the per-sample loss histogram.
4. 🟢 [Feng et al., ICLR 2025](https://arxiv.org/abs/2406.07515) — **MED.** A filter helps only if it keeps good examples at a higher rate than bad ones (Theorem 4.2: p* = 1/(1+ψ/φ); φ ≈ ψ ⇒ p* ≈ 1/2, stated as equivalent to no pruning). The legality validator keeps constrained-decoded samples regardless of quality, so it contributes zero verification. (Caveat: the theorem makes collapse conditional on the corruption level exceeding p*; the collapse-at-p*=1/2 reading is from their experiments.)
5. 🟢 [Gillman et al., ICML 2024](https://arxiv.org/abs/2402.07087) — **MED.** A *corrector* (maps samples toward the true distribution) stabilizes the loop exponentially (Thm 4.3). A discard-only validator is no such map under their Def 4.1/C.2 — an inference; the paper never discusses rejection filters — so γ=0, which exactly recovers Bertrand's bound (their Remark 4.4).
6. 🟢 [Gerstgrasser et al., COLM 2024](https://arxiv.org/abs/2404.01413) — **MED.** Replace ⇒ test error grows linearly in generations; accumulate (never replace) ⇒ bounded by a constant: **plateau, not collapse — but also no improvement**. Linear-regression theory + unfiltered experiments; sets expectations rather than design.

Bottom line: legality-filtered self-training sits in the literature's no-verification, accumulate regime. Predicted outcome with the fixes below: bounded plateau. Predicted failure without them: silent diversity collapse in which **every sample stays legal** — validity is structurally unable to detect the failure mode.

## Loop specification (fixes that must land before generation 1)

**A. Admission key — spread-free, persistent.** `hps_generate.team_key` (src/hps_generate.py:123) hashes Stat Points, but the model does not generate spreads — they are copied from a random corpus slot at render time (src/hps_eval.py:38–49) — so under that key novelty is a coin flip and the same 48-field team is admissible repeatedly with different random spreads. Use `schemes.key_of` (src/schemes.py:496 — slot-order-invariant, no spreads). Re-key all 100k jsonl rows once into a persistent index; admission = Showdown-valid AND key not in index. Today no code checks a sample against the stored dataset at all (in-run `seen` set only).

**B. Admission sampling — unbiased, pinned.** temp 1.0, top-p 1.0, guidance 1.0 (`sample_constrained` defaults; the feasibility masks are support enforcement, not quality bias, and are fine). The production settings (temp 0.7 / top-p 0.9 / guidance 2, gui/server.py:127) are exactly the biased sampling Alemohammad shows accelerates diversity collapse — end-use only, never feeds the dataset. Evaluation sampling identical to admission sampling.

**C. Synthetic-fraction cap — mandatory.** Tag every jsonl row with provenance (`hps_gen0` / `hps_fresh` / `diffusion_gen_g`) — rows currently carry none, so the fraction can't even be computed. Each generation, either inject fresh HPS teams (free at ~1,400 teams/s) or subsample the synthetic pool at train time so diffusion-sourced rows stay under ~half the **real**-row count (≈ a third of the mix — Bertrand's λ < 1/2 is a synthetic-to-real ratio, and sufficient-only). This is the load-bearing stability condition in both Bertrand and Gillman, and fresh HPS injection moves the loop toward Alemohammad's non-degrading fresh-data regime.

**D. Versioning.** Per-generation checkpoints (`results/hpsdiffusion_gen{g}.pt`) and per-generation team files. Today `torch.save` overwrites one path (src/hpsdiffusion.py:157) — without versioning, rollback on a tripped stop rule is impossible.

**E. Frozen references, built once before generation 1:**
- *Frozen holdout*: 2,000 teams by line index from the original 100k, excluded from every retrain. Today `train` re-draws the holdout from the current dataset (src/hpsdiffusion.py:120–122), so losses are not comparable across generations.
- *Frozen generation-0 checkpoint* (scorer for E5).
- *Frozen fresh-HPS reference*: 100k newly HPS-sampled teams, never trained on (yardstick for E7).

## Per-generation evaluation

N = 10,000 unbiased samples per checkpoint, fixed protocol. Noise bands: sample twice from the generation-0 checkpoint, set each metric's band at ~3× the repeat-draw spread. Pre-register E0, E3, E4 as the decision metrics; the rest are diagnostics.

| # | Metric | What it catches |
|---|---|---|
| E0 | Masked-prediction loss on the frozen holdout | The primary health number (every paper's main detector); flat = plateau, rising = drift off the real distribution |
| E1 | Validity rate (diagnostic only, two-sided) | A sharp **rise** toward the 0.95 HPS ceiling is a memorization signature (a model drifting toward copying its all-legal training data validates more often); a fall is a training pathology. Never a health certificate |
| E2 | Novelty vs frozen original 100k (health) and vs current dataset (admission stat) — reported separately | The current-dataset number mechanically declines as the dataset grows; never plot it as model health |
| E3 | Coarse discovery yield: new distinct species sets per 1,000 accepted; `matches_real` / `distinct_species_sets` (src/filterrate.py:175) | Exact-key novelty stays ~100% forever in a ~10¹⁹ space (one move swap mints a "new" team); species-set yield is where saturation is actually visible |
| E4 | Coverage: species-marginal entropy; TV distance of species-**pair** frequencies vs the frozen 100k; mean min-Hamming from holdout teams to the sample set | Pair TV catches joint collapse that per-field marginals can't see; rising mean min-Hamming = tails receding (Shumailov early collapse) |
| E5 | Per-sample loss histogram of generation-g samples scored under the frozen generation-0 checkpoint | Shumailov's second diagnostic — drift and a growing tail signal the model leaving the original distribution |
| E7 | Beyond-HPS reach: fraction of admitted teams with min Hamming (48-column grid, src/pilot.py:33) beyond a calibrated radius from the fresh-HPS reference; count of species sets HPS never produced | The loop's entire value proposition — HPS makes fresh legal teams at ~1,400/s for free, so a loop whose admissions are all HPS-reachable is a slow HPS |

Do **not** use win rate vs the top-50 meta as the per-generation quality track: the pool sits at 0.008–0.015 mean win rate, where 24-battle labels are almost pure binomial noise (73% of label variance — results/hps_surrogate.json). If a battle check is ever wanted, battle against fixed HPS-sampled opponents (win rates near 0.5, maximum power per battle).

## Stopping rules

- **S0 — plateau (the expected exit).** Stop when E0 (frozen-holdout loss) AND E3 (species-set yield) are both flat within their noise bands for 2 consecutive generations. Gerstgrasser's theorem says the accumulate regime plateaus — nothing degrades and nothing improves — so a design whose only exits are collapse detectors runs to the budget cap while nothing changes.
- **S1 — value exit.** E7 ≈ 0 for 2 consecutive generations: the loop reaches nothing HPS can't, stop and just run HPS.
- **S2 — collapse tripwire.** Any pre-registered E4 metric more than its band below generation 0 for 2 consecutive generations: stop, roll back to the previous generation's checkpoint.
- **S3 — anomaly alarm, not a stop.** Two-sided validity alarm (see E1).
- **S4 — budget cap** on generations, set in advance.

## Run 1 — 2026-08-29: stopped at generation 2 (S0 + S1)

Implementation: `src/selftrain.py` (init / calibrate / run / status). 20,000 admission
attempts and 10,000 eval samples per generation, 30-epoch from-scratch retrains
(~26 min each), E7 radius calibrated at Hamming 41 (p99 nearest-neighbour distance
of 5k fresh HPS teams to the 100k reference). Full numbers: `results/selftrain_results.json`.

| metric | gen 0 | gen 1 | gen 2 | band |
|---|---|---|---|---|
| E0 frozen-holdout loss | 9.784 | 10.250 | 10.157 | ±0.201 |
| E1 validity | 0.477 | 0.508 | 0.535 | ±0.003 |
| E4 species entropy | 5.2281 | 5.2294 | 5.2282 | ±0.0009 |
| E4 pair TV vs original 100k | 0.1516 | 0.1535 | 0.1574 | ±0.0024 |
| E4 mean NN Hamming, holdout→samples | 40.77 | 40.74 | 40.75 | ±0.15 |
| E5 sample loss under frozen gen 0 | 4.119 | 4.042 | 4.010 | ±0.0024 |
| E7 beyond-HPS fraction (admitted) | — | 0.000 | 0.000 | radius 41 |
| admitted / λ | — | 9,568 / 0.098 | 10,095 / 0.201 | cap 0.5 |

**Stop rules fired:** S1 (both generations' admissions sit entirely within HPS's own
spacing — the loop is a slow HPS) and S0 (holdout loss flat between generations 1 and 2).
The S3 anomaly alarm also fired: validity rose monotonically 0.477 → 0.508 → 0.535.

**Reading.** Novelty is saturated at 1.0 at every granularity — all 10k eval samples
each generation are new teams with species sets absent from the original 100k, so
exact-novelty admission admits ~every valid sample and discovery-yield can never
flatten (the space is ~10¹⁹; even 188-choose-6 species sets never collide). The
value metric E7 is therefore the binding one, and it reads zero twice: nothing the
diffusion model finds is outside what HPS reaches for free at ~1,400 teams/s.
Meanwhile three needles moved together in the concentration direction — validity up
0.058, samples 0.11 more probable under the frozen gen-0 model, pair TV drifting up —
with coverage (entropy, holdout distance) not yet damaged: exactly Alemohammad's
synthetic-augmentation regime, degradation delayed rather than absent. Two
generations deep, the loop neither collapses nor pays rent.

## Existing code map

Closest template for the per-generation table: src/scale.py (per-checkpoint held-out loss + generation + battles). The only closed loop in the repo, src/loop.py, is label-filtered (admission requires a measured win rate), has no novelty test, fixed 3 rounds, and never touched the 100k model — it is not this loop. Gaps to build: persistent key index (A), provenance field (C), per-generation checkpointing (D), frozen holdout split (E), coverage metrics E3/E4/E5/E7 (none exist today; the README's coverage facts were ad hoc).
