# CEM score sign and diversity across iterations

Question: does ranking on negative measured score preserve more Pokemon-ID
diversity through cumulative CEM updates than ordinary CEM?

## Fixed first experiment

- Arms: normal CEM ranks by `score`; reversed CEM ranks by `-score`; the random
  control selects a uniform subset. Stored scores remain wins divided by battles.
- Shared starting checkpoint: `results/temperature_p0.pt`, trained on the same
  692-team corpus. Three new trajectory seeds: 2101, 2202, 2303.
- Five cumulative updates. Each update starts from that arm's preceding
  checkpoint, with a fresh optimizer. No return to p0, replay pool, or anchors.
- Each selection batch contains 192 raw attempts. Select 96 legal attempts
  uniformly without replacement for evaluation, keeping duplicate teams as
  distinct draws. If fewer than 96 are legal, stop that trajectory as infeasible;
  do not refill or discard the failure.
- Each evaluated candidate receives 24 fresh battles using the same fixed BC
  policy on both sides and a shared 24-opponent schedule from the frozen top-50
  pool. Schedules and policy seeds match across arms within each seed/iteration.
  Showdown's internal battle RNG is not seeded by the policy RNG.
- Retain 20 candidates: the existing 20% rule with a minimum of 20 at pool size
  96. Shuffle before stable ranking to randomize ties. Random control retains
  20 using the same shuffle. Only the ranking score sign changes for reversal.
- Every update uses 64 optimizer steps, batch size 64 sampled with replacement,
  AdamW with initial learning rate 1e-4, cosine schedule, weight decay .01,
  gradient clipping 1.0, and random slot permutations. Use the same masked
  denoising loss, null conditioning, and zero additional entropy bonus in all arms.
- Generation always has decoding temperature 1.0 and no score guidance. Species,
  legality constraints, and stat-spread sampling remain fixed.
- After every update, draw 256 fresh raw evaluation attempts with an independent
  phase seed. These never enter selection or training. Preserve repeats and
  invalid outputs. Each seed also has a shared 256-attempt p0 baseline.
- The first candidate batch and its fresh battle results are shared across arms
  within a seed, since the starting model is identical. Later batches come from
  the respective updated generators. Rotate arm execution order over iterations.
- At iteration five, uniformly select 32 legal evaluation outputs per arm/seed
  for 48 fresh battles each. These secondary quality results never affect training.
  If fewer than 32 are legal, record quality evaluation as unavailable.

There are at most 19,776 physical raw generation attempts, 103,680 fresh battles,
and 2,880 optimizer steps. Shared first-round battles are performed once and
exposed equally to all arms. Infrastructure retries are recorded separately;
completed candidate labels are reused on resume.

## Measurements and interpretation

The primary endpoint is Pokemon-ID diversity among legal evaluation outputs at
iteration five: `100 * mean_pair(6 - shared_ID_count) / 6`. Slot order and all
non-ID attributes are ignored. Distinct form IDs remain distinct. Repeated
proposals remain in the calculation. Report the same metric on all raw outputs,
legality, distinct rosters, effective IDs, and ID coverage alongside it.

Show every iteration and the initial baseline. Report paired final differences
against normal CEM and the random control, in percentage points. Exploratory
three-seed t intervals use Bonferroni adjustment for those two comparisons and
assume normally distributed seed effects. Trajectory measurements and secondary
quality results are descriptive. Do not interpret individual generated teams as
independent training replicates.

An infeasible trajectory retains all completed results and prevents a full
three-seed final comparison for the affected arm; do not silently drop its seed.
Conclusions apply to this checkpoint, representation, policy, budgets, and five
updates. A ranking-pool result alone does not establish generated diversity.

## Reproduction

The runner is `scripts/cem_score_direction.py`; `--smoke` uses a separate output,
one seed, two iterations, and reduced budgets to test cumulative updates and
fresh battles. Run the full experiment with the stable runtime Python and eight
isolated local Showdown servers on ports 8134 through 8141. Input fingerprints
and per-phase checkpoints protect resume. The worker does not own those servers;
their launcher records ownership for cleanup.
