# TRS port — Trust-Region Noise Search on the HPS diffusion generator

Written 2026-09-02, BEFORE any battle run. Source: "Trust-Region Noise Search for
Black-Box Alignment of Diffusion and Flow Models" (Schweiger, Cremers, Ram —
arXiv 2603.14504v2; short version ReALM-GEN workshop at ICLR 2026, full version
accepted to ECCV 2026). Released code: github.com/niklasschweiger/trust-region-noise-search
(single commit 8823c76, read in full). The paper is graded rejected/LOW in the
vault's Reading List ("noise-space search against a costly noisy reward"); the owner
overrides the grade for this port. Sections 6–7 (results, verdict) are appended
after the run; everything above them is frozen first.

## 1. Algorithm 1, exactly (Step 1)

Inputs: total budget N_total, warm-up budget N_warm, batch B, regions k,
generator F, reward R.

1. Warm-up: draw N_warm noise vectors from the prior p0, evaluate r = R(F(x))
   in batches of B. Paper default N_warm = 20% of N_total in all three domains.
2. Init: centers = top-k warm-up points; every region's side length l_j = l_init.
3. Loop while budget >= B:
   a. Each region j proposes B/k candidates (integer split: base = B//k, the
      remainder goes to the LOWEST-indexed regions, which after re-centering are
      the highest-valued centers — undocumented greedy bias, confirmed in code
      `tr_utils.py:1438-1444`).
   b. Per candidate: p ~ Uniform(p_min, p_max) (one scalar per region per
      iteration in code), mask m ~ Bernoulli(p) per coordinate (rows that come
      out all-zero get one random coordinate forced on), perturbation inside the
      hypercube [c − l/2, c + l/2]: Sobol fill when dim <= 21,201 (torch
      SobolEngine MAXDIM), else Gaussian with sigma = l/sqrt(12), clamped to the
      box. Candidate = center + (perturbation ⊙ mask); masked-off coordinates
      stay pinned at the center.
   c. Evaluate all B candidates once. The reward oracle is queried EXACTLY ONCE
      per candidate — no repeats, no averaging, no variance model anywhere.
   d. Adapt lengths per region: strict improvement of the region's batch max
      over the region's best value (+1e-12) increments c_succ and resets c_fail,
      else the mirror. c_succ >= 3 → l = min(l·1.5, l_max); c_fail >= 3 →
      l = max(l/1.5, l_min). Contraction already at l_min triggers a restart:
      l reset to l_init (the restart's re-centering half is dead code — global
      re-centering overwrites it immediately).
   e. Global re-centering (the distinction from TuRBO): ALL k centers are
      replaced by the global top-k points over the entire evaluation history,
      descending; then region i's "best value" = the i-th best global value.
      Regions are not independent and may collapse onto one neighbourhood; the
      paper's ablation (D.1) says this GlobalTopk rule is what makes TRS work.
4. Return the argmax over every point ever evaluated (warm-up included).

Defaults (paper Table 4): l_init 0.8, l_min 0.05, l_max 2.4, alpha_l 1.5,
c_succ = c_fail = 3, p_min/p_max 0.1/0.9, warm-up 20%. Per-domain: T2I B=20 k=15,
molecules B=100 k=20, proteins B=8 k=5. Reported budgets in reward calls: 400
(T2I main table), 2,000 (molecules), 160 (proteins) — the 5k–100k figures are
NFE (generator integration steps), not reward calls. Noise dimension M: 16,384 /
65,536 / 291 / 150–300; only SDXL (65,536) used the Gaussian regime.

Code-vs-paper deviations worth knowing: the image reproduction script actually
runs alpha_l = 1.75 and l_init = 0.6 (not 1.5/0.8); the README's "scores
candidates, keeps the best" is false — default scoring is random, all proposals
are evaluated; the CLI seed never reaches TRS on the image/molecule paths (runs
are seed-42 always); the QM9/protein reproduction scripts are syntactically
broken (unterminated quote). This port follows the PAPER defaults and Algorithm 1
as written, with the code resolving ambiguities (allocation remainder, counter
semantics, restart rule).

The paper assumes noise → sample → reward is DETERMINISTIC (DDIM eta = 0, ODE
integration; Appendix E shows TRS degrading toward random search when the map is
made stochastic). It never handles a noisy reward. That is the central tension of
this port and is addressed in §3.

## 2. The mapping — what is the source noise here? (Step 2)

### 2a. Source noise definition

The generator is `hpsdiffusion.sample_constrained`: MDLM-style masked discrete
diffusion over 6 slots × 8 categorical fields = 48 columns, decoded in a FIXED
dependency order (species → ability/item → moves → natures, one column per step,
`diffusion.ORDER`), with a deterministic per-step legality projection
(`Constraints.mask_for`: infeasible logits → −1e9). The ONLY randomness in a
decode is one `torch.multinomial` categorical draw per (team, column) — 48 draws.
There is no continuous source latent.

Chosen noise: **x ∈ [0,1]^M, the unit hypercube, via the Gumbel-max
reparametrization.** Per column c with vocabulary size V_c, the noise slice
u_c ∈ [0,1]^{V_c} maps to Gumbel noise G = −log(−log(u)) (u clamped to
[1e-6, 1−1e-6]), and the decode picks argmax over the feasible set of
(logits/temp + G). By the Gumbel-max trick this reproduces the multinomial
draw's distribution exactly — decoding x ~ Uniform[0,1]^M IS sampling from the
generator prior, so TRS's warm-up from p0 and the random-search baseline are
both exact prior sampling, and the arms are distribution-matched by
construction.

Stat Points are not one of the 48 columns (the encoding carries them alongside;
the baseline pipeline `hps_eval.row_to_paste` draws a spread from the per-species
corpus pool with an np RNG). To close the map: 6 extra noise coordinates, one
per slot; spread index = floor(x_s · pool_size(species_s)), x clamped to
[1e-6, 1−1e-6] like every other coordinate (pool order is arbitrary, so index
locality is meaningless anyway). Same distribution as the baseline draw up to
the negligible edge mass, now deterministic in x.

Rejected candidates for the noise: (b) per-step schedule uniforms — vacuous;
the constrained decode has no stochastic schedule (fixed ORDER, one column per
step). (c) the unmasking-order permutation — the constrained sampler has no
order randomness, and an arbitrary order breaks the species-first dependency
the constraints assume. (d) the draws inside `hps_generate.py` (hierarchical
product sampling) — rejection loops (dead slot draws, duplicate-team skips,
validator rejections) consume a data-dependent number of draws, so no
fixed-dimension noise → team map exists there.

### 2b. Is noise → team deterministic?

Yes, by construction, on CPU. The unmasking order is fixed, the time input is a
deterministic function of the step, `mask_for` is a deterministic function of
the partial row (including its duplicate-move relaxation fallback), the model is
in eval mode (no dropout), classifier-free guidance is deterministic given
weights, and argmax replaces the multinomial. The legality projection is an
in-step feasibility mask, NOT accept/reject — it consumes no randomness. The
94.9%-acceptance rejection loop belongs to `hps_generate.py` (dataset
construction), not to this decode path; the one stochastic piece downstream of
the model, the spread draw in `hps_eval.row_to_paste`, is replaced by the
noise-indexed rule above. Decode runs on CPU: MPS kernels are not certified
deterministic and argmax near-ties could flip across batch sizes. Verified
empirically by the decode-only determinism test (§2c) before any battle.

Residual nondeterminism: none in the map. The ESTIMATE of f(team) is noisy
(battles), which is §3's problem, not a moving-target decode.

### 2c. The piecewise-constant geometry — where the port most plausibly dies

f(x) = f(decode(x)) is piecewise constant: a perturbation changes nothing until
an argmax flips at some column, and an early flip (species) redraws the whole
downstream conditional while a late flip (a move, a nature) is a one-field edit.
Contracting l below the typical cell width proposes the center's own team over
and over; with a noisy reward the success/failure counters then run on pure
binomial noise.

What l_min means here: the paper's l_min = 0.05 is 5% of the coordinate range —
in this geometry that is meaningful only if a 5%-of-range perturbation on
~10–90% of coordinates still crosses cell boundaries. This is measured, not
assumed, by a decode-only test that costs zero battles (`trs.py geom`):

- Determinism check: same x decoded twice, and at batch sizes 1 vs 20 —
  identical teams required, else the port stops here.
- Plateau curve: for centers drawn from the prior and
  l ∈ {0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 2.4}, decode 64 masked uniform box
  perturbations each; record the fraction of proposals identical to the
  center's team, mean number of the 48 fields changed, distinct teams per
  batch, and validity rate.

Decision rule (pre-registered): let l* be the smallest tested l at which >= 50%
of proposals differ from the center in at least one field. The port's l_min is
set to l* (floored at the paper's 0.05 if l* is smaller). If even l = 2.4 (the
paper's l_max, 2.4× the full coordinate range — the whole box) leaves > 50% of
proposals identical to the center, the geometry has no usable resolution band
and the port is dead — kill criterion 1, report and stop. Plateau is also
tracked at runtime: identical-to-center proposals are counted per region per
iteration and reported; identical teams are label-cached and never re-battled.

### 2d. Dimension and proposal regime

M = 6·(189 + 169 + 112 + 26) + 24·315 + 6 = 10,542
(species/ability/item/nature vocabularies per slot + 24 move columns sharing the
315-entry move vocabulary + 6 spread coordinates; sizes include the inert
[MASK] index, which never wins the feasible argmax). M = 10,542 < 21,201 →
**Sobol regime**, same as three of the paper's four settings. Perturbations are
scrambled-Sobol fills of [c − l/2, c + l/2] ∩ [0,1]^M (clamped — the paper's
domains were unbounded, ours is the unit cube; disclosed adaptation).

## 3. The reward is not the paper's reward (Step 3)

f = expected win rate against the meta — the collected Reg Set M-B meta-team
list (the top-50 pool, `results/top50_evs.json`), estimated by Showdown battles
through the poke-env fork, behaviour-cloning policy both sides (`pool.score` →
`shard.py`). This is Monte Carlo evaluation, not ground truth — the only ground
truth is ladder validation.

### Budget arithmetic, done before anything runs

- Paper budgets in reward calls: 400 / 2,000 / 160. At the owner's 192-battle
  precision (±0.036) the T2I budget costs 400 × 192 = 76,800 battles; measured
  harness throughput is ~85,000–100,000 battles/hour end-to-end (uncontended,
  7 servers) — so even the paper-faithful budget is ~1 hour, and the day-scale
  kill criterion does not trigger. The port keeps the T2I shape: N_total = 400
  calls, B = 20, k = 15, warm-up 80 calls (20%), 16 search iterations.
- Battles per call: 96 (SE at p≈0.5 is 0.051; at the prior's p≈0.14, 0.035).
  400 × 96 = 38,400 battles ≈ 25–40 min. 24-battle labels (SE ≈ 0.10) would
  make every counter decision noise; 192 everywhere would double cost for
  precision the search does not need at proposal time. Precision is spent where
  decisions are made: promotion and the final report (below).
- Wait — the paper's 5k–20k / 25k–100k / 16k–64k figures are NFE (generator
  steps), not reward calls; the reward-call budgets above are the honest unit
  here, since one team decode is cheap (~0.1 s) and one reward call is 96
  battles (~2 s of 7-server time).

### The three required noise mitigations, all implemented

(i) **Common random numbers.** A fixed opponent schedule — 96 opponents drawn
once from the top-50 pool, seeded — is shared by every candidate in every arm
(new opt-in `opp_schedule`/`seed` parameters through `pool.score` → shard spec;
a deterministic cycling Teambuilder replaces the unseeded `random.choice`, and
the shard seeds torch/np/random so policy action sampling is aligned as far as
battle divergence allows). Comparisons between candidates are paired on the
opponent multiset. Battle-engine seeds are NOT reachable through poke-env
(would need a server patch); opponent pairing is the first-order win, engine
seeds a second-order refinement — accepted and disclosed.
(ii) **Incumbent re-evaluation before promotion.** A candidate whose block-A
(96-battle) estimate would enter the global top-k is immediately re-battled on
an independent fixed schedule B; its archive score becomes the pooled 192-battle
estimate, and top-k membership is decided on the pooled score. Winner's-curse
promotions get caught by the fresh block. Success/failure counters compare
block-A estimates only (like against like).
(iii) **Report re-evaluated win rate, never selection-time score.** Every
arm's finalists are re-battled at 192 FRESH battles on schedule F (drawn
independently of A and B, shared across arms). Only those numbers are reported.

### Surrogate: not used, and why

The ridge surrogate's Spearman is 0.83 in the 500-label active-search setting
but 0.16 on the 100k HPS pool. TRS warm-up and early proposals ARE
HPS-prior-like teams — exactly the 0.16 regime (compressed win rates near the
prior mean, labels mostly binomial noise). A 0.16-Spearman filter inside the
loop would corrupt region statistics more than it saves battles. Battles are
affordable at this budget; the two-stage reward is declined.

## 4. Legality is a constraint, not a reward term (Step 4)

Every decoded team passes through Showdown's own validator
(`scripts/validate-teams-batch.js`, format gen9championsvgc2026regmb) before it
is battled. Invalid candidates get no battles, cannot enter the archive or
become centers, and are simply infeasible points (for the counters, a region
whose whole allocation is invalid counts as no-improvement). No legality
penalty appears in the reward. Validity rate is reported per iteration next to
win rate. (The in-step projection should keep validity high — the baseline
constrained-decode pipeline measured ~high-80s–90s% — the geom test measures it
exactly.) Reg Set M-B: no Tera type, Stat Points (max 32 per stat, 66 total),
no IVs — all inherited from the existing encode/decode path.

## 5. Pre-registered comparison (Step 5)

Frozen before the run. All new arms use the same opponent schedules A/B/F, the
same battles-per-call, the same promotion rule, and the same reporting rule.

| Arm | Source of candidates | Battle budget (cap) |
|---|---|---|
| TRS | Algorithm 1 on x ∈ [0,1]^10542, paper defaults, l_min from §2c | 50,000 |
| Random search over the same noise space | x ~ U[0,1]^M i.i.d. (the paper's own baseline; identical in distribution to prior sampling) | 50,000 |
| Best-of-N from the generator prior | the existing multinomial decode path, seeded | 50,000 |
| Decode-time gradient guidance + combined loop | standing numbers, no new battles | (spent: 73,728) |
| Elite filter rho = 0.01 | standing number 0.5398 ± 0.0083 | (spent: ~121k pool amortized) |
| graft / invent (GUI) | standing numbers 0.420 / 0.093 | — |

Notes. Random-search-over-noise and best-of-N-prior are the same distribution
by the Gumbel-max construction — running both is a deliberate A/A test of the
harness and the CRN plumbing; a significant difference between them flags a bug,
not a finding. Budget-matching: each new arm stops when its battle cap is hit
(dedup cache hits consume no battles); TRS's cap equals the baselines' caps
exactly, and all three are SMALLER than the combined loop's historical spend
(73,728), so no new arm is budget-advantaged over the standing numbers.

**Primary endpoint:** mean re-evaluated win rate (192 fresh battles each,
schedule F) of each arm's top-8 teams by pooled selection score. TRS "beats"
random search iff its top-8 mean exceeds RS's outside the 95% CI of the
difference (team-level bootstrap). **Secondary:** each arm's single best team,
re-evaluated (vs the standing best 0.589 ± 0.036 and the real-team mean 0.491 ±
0.024). **Kill criterion 3:** TRS inside the CI of budget-matched random search
at this budget → clean negative, written up as such.

## 6. Diversity, measured (Step 6)

Per iteration and per arm at close: distinct teams proposed / battled; distinct
species sets; plateau fraction (proposals identical to their region's center);
mean nearest-neighbour Hamming distance (48 fields) among battled teams; copy
rate vs teams/reg_mb (exact six-species-set match, and max per-team field
overlap). TRS is an exploitation method whose global top-k re-centering is
expected to collapse coverage — if the top-8 are near-copies of one tournament
team or two-field edits of it, that is reported as a failure mode regardless of
win rate.

## Amendments before the run (2026-09-02, after adversarial code review, before
any battle)

An independent three-lens review of src/trs.py against this document and the
released code produced fixes, folded in before anything ran: (1) the 50,000
battle cap is enforced inside the battle wrapper itself (a call never starts
battles it has no budget for), so no arm can overshoot; (2) the final top-16
ranking is rebuilt from the persisted label store, gets one last promotion pass
(with the report's 8×192 reserve protected), and prefers pooled (A+B) teams —
no block-A-only winner's-curse entrant can lead it; (3) all rankings break
pooled-score ties by team hash (set iteration order must never decide);
(4) the §2c "identical to center" event is defined on the 48 decoded columns
(spread-coordinate churn is tracked separately via the team hash); (5) plateau
is recorded per region per iteration; (6) promotion retries per team are
bounded at 2 so a persistently failing shard cannot loop; (7) a finalist whose
schedule-F battles are lost is retried once, then EXCLUDED and named — never
reported as 0.0; (8) label and results writes are atomic; (9) the shard
reseeds numpy alongside random/torch; (10) the number of regions is frozen at
the post-warm-up center count, as in the reference solver.

## 7. Results

### Geometry gate (decode-only, before any battle — 2026-09-02)

Determinism: PASS (same x → same team on repeat and across batch sizes 1 vs 20,
CPU). Prior validity through the Gumbel decode: 105/200 = 52.5%, statistically
consistent with the multinomial baseline's recorded 45.1% — the decode is
distribution-faithful. The plateau curve (8 centers × 64 masked perturbations
per l, identity on the 48 decoded columns):

| l | identical to center | mean fields changed (of 48) | validity of changed |
|---|---|---|---|
| 0.05 | 0% | 16.85 | 50% |
| 0.10 | 0% | 23.44 | 47% |
| 0.20 | 0% | 29.29 | 54% |
| 0.40 | 0% | 35.64 | 59% |
| 0.80 | 0% | 40.40 | 61% |
| 1.60 | 0% | 43.43 | 57% |
| 2.40 | 0% | 44.62 | 44% |

The feared failure mode (§2c: plateaus, contraction stalls) does NOT occur —
the geometry fails in the OPPOSITE direction: there is no small move. The
inverse-CDF map is tail-sensitive (a coordinate near u ≈ 1 explodes under a
±l/2 nudge), and any early-column flip cascades through the conditional decode,
so even the paper's l_min changes a third of the team. The trust region is a
coarse locality dial, not a fine one: contraction spans "redraw a third" to
"redraw nearly all", and TRS here behaves as re-centered stochastic local
search with tunable mutation strength, not as a converging trust region. By
the pre-registered rule, l* = 0.05 (every tested l has 100% of proposals
differing) → l_min = 0.05, and the port PROCEEDS to battles; kill criterion 1
does not trigger. Interpretation of any positive TRS result must credit the
coarse dial + global top-k re-centering, not fine-grained convergence.

### The run (2026-09-02, 15:20–17:26, ~145k battles total, 7 servers)

All three new arms ran to their battle caps under the frozen protocol
(schedules A/B/F, seed 7000, promotion re-evaluation, hard cap in the battle
wrapper). TRS: warm-up 36/80 valid, then 22 search iterations, 15 regions.
Runtime plateau count was ZERO across all 22 iterations — no proposal ever
reproduced its region's center, even at l = 0.07, confirming the geometry
gate's no-small-move finding in vivo. TRS search validity rose from ~50%
(prior level) to 100% as regions contracted onto valid centers — 80.0% overall
vs 47.7% (random search) and 44.3% (best-of-N): perturbing a valid team is
much likelier to stay legal than a fresh prior draw, an unplanned but real
efficiency of the trust region in this domain.

### Pre-registered endpoints (all win rates re-evaluated on 192 FRESH
schedule-F battles; selection-time scores never reported)

| Arm | battles spent | top-8 mean f (se) | best team f | copy rate | distinct sets top-8 | NN-Hamming battled |
|---|---|---|---|---|---|---|
| TRS | 48,384 | **0.0814 (0.0089)** | 0.1094 | 0.0 | 6/8 | 27.6 |
| Random search (noise space) | 48,768 | 0.0247 (0.0042) | 0.0469 | 0.0 | 8/8 | 40.7 |
| Best-of-N (prior decode) | 48,000 | 0.0293 (0.0031) | 0.0417 | 0.0 | 8/8 | 40.8 |

Bootstrap difference CIs (team-level, 10,000 resamples):
**TRS − random search = +0.0567, 95% CI [+0.0365, +0.0743]** — excludes zero.
TRS − best-of-N = +0.0521, 95% CI [+0.0326, +0.0684] — excludes zero.
A/A check: random search vs best-of-N differ by 0.0046 with overlapping
intervals — the two prior-sampling arms are statistically indistinguishable,
validating the harness, the common-random-numbers plumbing, and the Gumbel-max
construction (they are the same distribution by design).

Standing context, unchanged budgets, quoted not re-measured: real-team mean
0.491 ± 0.024; standing best single team 0.589 ± 0.036; elite filter rho=0.01
0.5398 ± 0.0083; combined gradient-guidance loop 0.4108 at g10 (73,728
battles); graft 0.420; invent 0.093.

### Verdict

**Does TRS beat budget-matched random search on f? YES** — by +0.057 win rate
on the pre-registered top-8 endpoint, CI excluding zero, at matched ~48k-battle
budgets, with the A/A control clean. Kill criterion 3 does not trigger. The
paper's mechanism survives the port: with common random numbers, pooled
promotion, and legality as a hard constraint, trust-region noise search
extracts a ~3× improvement over prior sampling from a FROZEN generator, and
its diversity cost is visible and bounded (6/8 distinct species sets in the
top-8; battled-set nearest-neighbour Hamming 27.6 vs 40.7 for random search —
concentration, not collapse; no team within 8 fields of any meta team).

**Does TRS matter for this project? On this evidence, no.** The frozen HPS
prior is so weak (mean f ≈ 0.01) that TRS's endpoint — best team 0.1094 —
sits ~5× below the real-team mean (0.491) and the standing arms (elite filter
0.5398, combined loop 0.4108, at comparable or 1.5× budgets). Noise-space
search cannot move the generator's weights, and that is exactly the lever the
winning arms pull (re-steer fine-tuning, elite filtering of a labelled pool).
The Reading List's rejection reason ("noise-space search against a costly
noisy reward") is half-refuted — the reward-noise problem was tamed by CRN +
re-evaluation at ~48k battles, a budget this harness produces in half an hour
— but the grade's practical conclusion stands: as a team-building arm, TRS
optimizes the wrong thing (the noise, not the model), and every
weight-touching arm in this repo beats it by 0.3+ win rate. Possible salvage,
NOT run (out of scope, would need pre-registration): TRS on top of the
re-steered activesearch_p0.pt checkpoint, where the same +0.06-vs-prior
mechanism would start from 0.14 instead of 0.01 — worth considering only if a
frozen-generator constraint ever matters (e.g. no fine-tune budget).
