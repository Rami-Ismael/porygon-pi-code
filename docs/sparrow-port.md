# SPARROW port — rank-based archive refinement with a frozen v3 generator

Written 2026-10-16, BEFORE any battle run. Pre-registration only: nothing below has been run.
Source: "Generative Refinement for Low-Budget Black-Box Optimization" (Dufour, Heinimann & Fua,
CVLab, EPFL — arXiv 2607.00691v2, 30 Sep 2026; no venue listed). SPARROW stands for Sequential
Proposal via Archival Rank-based Refinement for Optimization under Weak feedback. The paper is graded
"rejected at the owner's request, 2026-09-06" in the vault's Reading List; this port is the owner's
override. Results and verdict are appended after the run; everything above them is frozen first.

**Win rate** (every number in this doc) = wins / battles against the top-50 meta teams, Showdown
battles with the behaviour-cloning policy on both sides (policy 1), unless a line says "vs ranks
51–100". This is Monte Carlo evaluation, not ground truth.

## 1. Algorithm 1, as the paper writes it

Inputs: initial archive A_0 of n_0 evaluated candidates, a fixed corruption–refinement operator T_t,
budget B (objective calls), selection pressure β, corruption exponent γ, step size λ.

For k = 0 … B − n_0 − 1:
1. Normalized rank r_k(x_i) = rank(x_i) / n_k in the current archive, lower is better; the worst
   member has r = 1 (Appendix B), so ranks are 1-indexed and the best sits at 1/n_k.
2. One parent, p(x_i) ∝ exp(−β · r_k(x_i)) — the maximum-entropy distribution under an expected-rank
   constraint (Appendix A). β = 0 is uniform; every member keeps probability ≥ e^−β / n_k.
3. Noise level t_k = 1 − r_k(parent)^γ (t = 1 is data, t = 0 pure noise): good parents get local
   refinement, the worst parent gets a fresh unconditional sample.
4. Differential-evolution step: draw x_a, x_b uniformly from the archive;
   x_guided = parent + λ(1 − t_k) · sign(r(x_b) − r(x_a)) · (x_a − x_b).
5. x' ~ T_{t_k}(x_guided): partially noise to level t_k with the sampler's own forward process, then
   integrate back to data (a generalized SDEdit). The generator is never trained or guided by f.
6. Evaluate x' once, add it to the archive. One candidate per step, no batching (§4.2).

Return the best member of the final archive. Failed evaluations count against the budget and do not
enter the archive (§5.6). The convergence proof (Appendix B) assumes noiseless evaluations; the paper
never re-evaluates a point and never mentions duplicates.

Paper settings worth carrying: defaults β = 50, γ = 1, λ = 0.1; per-task picks (β, γ, λ) = (50, 1,
0.1) tube, (5, 0.1, 0.1) HopperController, (10, 0.5, 2.0) airfoil. Budgets B = 100 (and 1,000 on the
tube). n_0 is not stated. γ is "the most sensitive parameter" (Table 7: γ = 0.5 → 0.19, γ = 1 →
0.30, γ = 1.25 → 0.32 on the 64-dimensional tube); high β causes "excessive elitism". The baseline
the paper calls RDS (Random Diffusion Sampling) draws from the prior with no archive or selection;
γ = 0 reduces SPARROW to it. Every prior in the paper is a continuous flow-matching model — masked or
discrete generators are not discussed anywhere.

## 2. The mapping

### 2a. Generator — frozen

The v3 members used by the clean cross-entropy re-run: `F1_final_medium_noheldout_regmb_v3` +
`G2_family_matrix_noprtrain_noheldout_regmb_v3` (never trained on ranks 51–100), ensembled exactly
as in `scripts/diffusion_baseline_v3.py`: ask 0.5, guidance 2 (logits = 2·conditional −
unconditional), temperature 1, legality tables with rules `clause, compat`, distinct-move restriction,
masked columns decoded in `diffusion.ORDER`, Stat Points drawn from the same species' corpus spreads
(random legal spread if the species has none). No ridge guidance, no surrogate, no nature repair, no
fine-tuning at any step. The weights' sha256 are recorded in the run config and checked at every
resume.

### 2b. Corruption = re-masking whole Pokémon

T_t becomes: re-mask k whole slots of the parent (species, ability, item, four moves, nature — all 8
columns of the slot, plus its Stat Points), keep the other 6 − k slots as clamped context, and let v3
fill the masked columns through the masked-context decode in `diffusion_baseline_v3.generate`.

    k = ceil(6 · r^γ),   the k slots chosen uniformly without replacement (seeded)

r^γ is the paper's corruption amount 1 − t_k, measured in slots. ceil keeps k ≥ 1 (k = 0 would
re-propose the parent), and the worst parent (r = 1) gets k = 6, a fresh unconditional sample — the
paper's T_0. Why whole slots and not single fields:
- Ruggedness per edit type (`docs/ruggedness2.md`, 2026-09-28, 384 battles per edit): species edits
  are the only edit type rugged beyond battle noise (26% / 33% of species edits beyond noise against 8%
  of repeats); ability, alignment and Stat Points edits stay within noise, and move and item edits do
  not clear a 0.05 change. A single-field child is, to 24 battles, a copy of its parent.
- Mask Lab (2026-10-08, v3 at the same settings): net gain per mask +14.7 points on whole-Pokémon masks
  against +1.08 averaged over all mask families.

### 2c. Dropped: the differential-evolution step (λ = 0)

The step x_a − x_b is a vector difference in a continuous latent. A team is categorical; there is no
difference to add. The paper's own ablation (Table 7, tube, B = 100, β = 50, γ = 1) gives λ = 0 →
0.28 [0.24, 0.34] against the default λ = 0.1 → 0.30 [0.28, 0.32] — overlapping intervals; the paper
calls it "useful but not essential". λ = 0 is the port.

### 2d. Rank, ties, and parents

r = rank / n_k over the archive's current labels, best = 1. Labels at 24 battles take values in
multiples of 1/24, so ties are common; ties are broken by canonical team hash (deterministic, unrelated
to quality). One parent per step, drawn with a seeded generator.

What β and γ mean at this archive size (1,045 members at the end, exact sums over ranks):

| β | γ | expected parent rank | effective parents (1/Σp²) | k = 1 | 2 | 3 | 4 | 5 | 6 | mean k |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 0.5 | 0.19 | 412 | 0.13 | 0.30 | 0.29 | 0.18 | 0.08 | 0.02 | 2.85 |
| 5 | 1 | 0.19 | 412 | 0.57 | 0.25 | 0.11 | 0.05 | 0.02 | 0.01 | 1.73 |
| 50 | 0.5 | 0.02 | 42 | 0.75 | 0.25 | 0 | 0 | 0 | 0 | 1.25 |
| 50 | 1 | 0.02 | 42 | 1.00 | 0 | 0 | 0 | 0 | 0 | 1.00 |

At β = 50 and n = 128 (the start) the effective parent count is 5.2 and the top-ranked team is the
parent 32% of the time. β = 50, γ = 1 is therefore "swap one Pokémon of one of the ~40 best teams",
every step, for the whole run.

### 2e. Evaluation of one candidate

Showdown validator first (`gen9championsvgc2026regmb`). A candidate that fails it, or equals a held-out
team (ranks 51–100), gets no battles and does not enter the archive, but its step IS charged to the
budget (reject-and-resample counts). Otherwise: 24 battles, one each against 24 distinct top-50 columns
drawn uniformly without replacement for that candidate (seeded by run seed and step), written to the
matchup matrix through `matchup_db.ensure(rows=[team], cols=those 24, per_cell=1)`; the label is
`matchup_db.score` over the team's covered columns. Fresh columns per candidate keep the label an
unbiased estimate of the top-50 win rate; a single fixed 24-column schedule (common random numbers, as
the Trust-Region Noise Search port used) would make the search optimize a 24-team sub-meta.
Standard error of one label ≈ 0.10 at p = 0.5.

Duplicates (the paper is silent): a refill identical to its own parent never reaches the objective —
it is re-decoded with the next seed, up to 8 times, uncharged (decoding is free; the budget counts
objective calls). A candidate equal to another archive member is charged one step and battles 24 of
that member's not-yet-covered top-50 columns (until all 49 are covered; after that, the step is
charged with no battles); the member's label becomes its pooled score. This is the noise-aware reading
of "add the evaluated point to the archive" — the archive holds one row per distinct team.

### 2f. Initial archive

n_0 = 128 unconditional v3 samples (k = 6 decodes) per seed, each evaluated as in §2e — 128 = the
cross-entropy loop's generation-0 battled count. The same 128 samples (same seed) are shared by all
four SPARROW arms and form the first 128 steps of the unconditional arm, so arms differ only after
step 128. The initial archive is charged to the budget.

## 3. Arms

| Arm | Parents | Corruption | Steps (incl. n_0) |
|---|---|---|---|
| S(5, 0.5) | β = 5 | γ = 0.5 | 1,045 |
| S(5, 1) | β = 5 | γ = 1 | 1,045 |
| **S(50, 1) — primary** | β = 50 | γ = 1 | 1,045 |
| S(50, 0.5) | β = 50 | γ = 0.5 | 1,045 |
| Unconditional (the paper's RDS) | none | k = 6 every step | 1,045 |
| Clean cross-entropy loop | standing numbers, no new battles | — | 25,088 battles spent |

S(50, 1) is pre-registered as the primary arm because it is the paper's default and its tube setting;
the other three are secondary and cannot rescue a failed primary. Three seeds per arm (6060, 6061,
6062) → 15 runs. The unconditional arm is best-of-1,045 from the prior: it isolates what the archive
and parent selection add on top of v3 itself. Uniform random legal teams are not run (median win rate 0.000 over
4,850 labelled teams, `docs/plan-vs-results-2026-08-28.md`).

## 4. Budget arithmetic

Matched to the clean cross-entropy re-run (`results/cem_v3_noheldout.json`, seed 4040): it stopped
after generation 3 on the diversity guardrail, battling 4 generations × 128 teams × 49 distinct top-50
columns = **25,088 search battles** (the matrix's batch log shows 6,125 + 6,272 + 6,272 + 6,272 =
24,941 run; 147 cells were already cached). It then spent 16 × 192 = 3,072 fresh battles vs the top-50
and 3,072 vs ranks 51–100: 31,232 nominal in all. Its 512 proposals per generation were filtered by the
validator and the ridge for free, and it fine-tuned two members three times; SPARROW pays for its
validator rejects and fine-tunes nothing.

SPARROW: 25,088 / 24 = **1,045 steps** per run = 25,080 battle-equivalents (each step charged 24
whether battled, rejected, or a fully covered duplicate). Battles actually run ≤ 25,080 and are
reported.

| | per run | 15 runs |
|---|---|---|
| search (≤) | 25,080 | 376,200 |
| finals: 16 × 192 vs top-50 + 16 × 192 vs ranks 51–100 | 6,144 | 92,160 |
| **total (≤)** | **31,224** | **468,360** |

Plus a decode-only gate (0 battles, §6) and a smoke test (2 arms × 1 seed × 20 steps ≤ 960 battles).

## 5. Cost estimate

Throughput: the matrix path measured 50.9 battles/s on 7 servers (`docs/matchup-matrix.md`); the
clean cross-entropy generations ran 6,272 battles in 342–751 s (8–18 battles/s with fine-tuning
contention). At 50 battles/s the 468k battles are **~2.6 hours of server time**. But SPARROW is
sequential — step k + 1's parent depends on step k's label — so one run cannot fill the servers: each
step is one 24-battle job on one worker. The 15 runs therefore go concurrently (≤ 360 battles in
flight). The smoke test measures per-step latency L (decode + validator + 24 battles + matrix write);
wall clock ≈ max(1,045 · L, 468,360 / measured aggregate rate). If L = 30 s, that is ~8.7 hours; if
L = 60 s, ~17 hours. Concurrency changes wall time only, never the algorithm. The runs use the `/tmp`
staging copy (iCloud evicts the repo's virtual environment), resume from the persisted archive at any
step, and run under `battle_supervisor`.

## 6. Decode-only gate (0 battles, before anything is battled)

On 64 valid unconditional v3 teams, for each k = 1…6, 16 refills each (6,144 decodes): fraction
identical to the parent, fraction whose re-masked slots came back with the same species, fraction
valid, mean fields changed. Decision rule, fixed now: if at k = 1 more than 50% of refills reproduce
the parent even after the 8 uncharged re-decodes, the port stops before battles — the primary arm
(k = 1 almost always) would be spending its budget on duplicate top-ups, not search. If validity at any
k is below 80%, the charged rejects are reported as a budget handicap and the port proceeds.

## 7. Hypotheses and pass rule (fixed before any battle)

**Endpoint.** Per run: the 16 best distinct archive teams by label (ties: more battles first, then
hash), each re-battled fresh with 192 battles via `pool.score` against the top-50 (and, once,
afterwards, 192 vs ranks 51–100 via `heldout.evaluate`). Arm value = mean over its 3 × 16 = 48 fresh
numbers. Confidence intervals: bootstrap over runs, then teams within run, 10,000 resamples. The
cross-entropy loop is one seed; its 16 finalists give 0.504 (standard error 0.015) vs top-50 and
0.509 (0.017) vs ranks 51–100, and its run-to-run variance is unknown — disclosed, not modelled.

**H1 — the archive and parent selection help.** S(50, 1) − Unconditional > 0, 95% interval excluding 0.
If H1 fails, SPARROW is FAIL and H2 is not read.

**H2 — frozen SPARROW against the re-steered loop** (primary arm only):
- **PASS (beats the loop):** S(50, 1) − 0.504 has a 95% interval above 0, and its vs-ranks-51–100 mean
  is above 0.509 (point estimate).
- **MATCH (equal without fine-tuning):** the interval for S(50, 1) − 0.504 lies inside [−0.03, +∞).
  This is still a useful outcome: same strength, no training at any step.
- **FAIL:** anything else.

**Predictions, written down to be checked.** (i) The label-to-fresh gap (winner's curse) will be
larger than the loop's −0.175 (0.679 → 0.504), because labels rest on 24 battles not 49 and the archive
never re-battles a lucky team except through a duplicate; β = 50 arms will show it most. (ii) β = 50
arms will converge onto few species sets; the loop's clean finalists hold 4 distinct species sets in
16. (iii) γ = 0.5 arms will look closer to the unconditional arm, as in the paper's Table 7.

**Measured for every run** (reported, not part of the pass rule): label vs fresh per finalist; fresh
vs ranks 51–100 and its paired difference; distinct species sets among the archive, the finalists, and
per 100 steps; copy rate vs the training corpus and the top-50 (exact six-species match and max field
overlap); validator reject rate and identical-refill rate; duplicate top-ups; distribution of k and of
parent rank per 100 steps; best-so-far label curve against steps.

**Not tested here, would need its own pre-registration:** re-battling incumbents before they can be
selected as parents (the noise fix the Trust-Region Noise Search port used); per-field masks;
a SPARROW loop on a re-steered checkpoint.

## Amendments before the run

(none yet)

## Results

(appended after the run)
