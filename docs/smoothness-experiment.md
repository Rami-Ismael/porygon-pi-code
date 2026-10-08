# Legal-move sensitivity experiment

The implementation is `scripts/smoothness_experiment.py`; battle workers are in
`scripts/smoothness_worker.py`. This experiment measures local sensitivity, not
mathematical non-smoothness. Earlier `ruggedness*.py` runs and outputs are preserved.

## Verified on 2026-10-01

All **14 statistical/protocol/integration tests pass**. A real simulator pilot
completed **4,800 battles**, using four random originals, the first four entries
of the frozen placement snapshot, two edits per original, an unchanged-team control,
and three panels of 50 battles per candidate. All 96 panels finished with exactly
50 battles. The report and PNG/PDF figure were generated and the figure inspected.

- [Pilot report](../results/smoothness_pilot/report.md)
- [Pilot figure](../results/smoothness_pilot/smoothness.png)
- [Pilot raw analysis](../results/smoothness_pilot/analysis.json)
- [Confirmed full-study manifest](../results/smoothness_top50_50x50/manifest.json)
- [Full-study completion audit](../results/smoothness_top50_50x50/verification.json)
- [Full measured report](../results/smoothness_top50_50x50/report.md)
- [Full measured figure](../results/smoothness_top50_50x50/smoothness.png)

**The full 50+50 study completed all 400,000 battles on 2026-10-01 at
3:54 p.m. America/Chicago**, after launching at 12:39 p.m. The completed
[run status](../results/smoothness_top50_50x50/run-status.md) records 4,000 complete
panels and successful analysis. A separate completion audit checked every saved
job, all frozen input/code/runtime hashes, and independently recomputed all 900
edit/control effects from the raw win counts. Each group received 200,000 battles.
The PNG figure was visually inspected; the report, PDF, CSV and JSON outputs exist.
The supervisor and its battle/server processes have stopped.
The small pilot is an implementation check, not the
requested full study. Its first four placement entries are not a random sample of
the full placement cohort. Dedicated pilot servers on ports 8170 and 8171 were
stopped after completion. Existing servers were not stopped.

The pilot was evaluated before the display label changed from "tournament" to
"Frozen top placements" after the user's cohort decision. Its original manifest
is retained; analysis uses the corrected display labels with unchanged numerical
estimators. The confirmed full-study manifest fingerprints the final code. Use a
fresh directory for another pilot rather than trying to resume its older code hash.
The inherited evaluator logged pending inference-task cleanup warnings; all saved
panels nevertheless contained the full required battle count. Logs are retained.

## Full measured results — 2026-10-01

The primary outcome is the mean noise-corrected squared expected win-rate change.
Values below are in squared percentage points (pp²), not percentage points. The
intervals are 95% percentile bootstrap intervals over original-team identities.

| Group | Original win rate | Corrected squared change (pp²) | 95% interval (pp²) |
|---|---:|---:|---|
| Random legal, frozen HPS sample | 1.845% | 0.292 | [0.001, 0.690] |
| Frozen top placements | 49.950% | 18.870 | [12.924, 25.511] |
| Top placements minus random | — | 18.579 | [12.698, 25.262] |

The positive group contrast supports greater local move sensitivity around this
placement cohort under the frozen evaluator, after the specified battle-noise
correction. It does not establish mathematical non-smoothness. Random teams are
near the win-probability floor: their low measured sensitivity may partly reflect
that one move seldom rescues a poorly performing team. The comparison does not
isolate team quality from this bounded-outcome effect or generalize to optimal play.

Unchanged-team controls were consistent with zero in both groups: random
**0.403 pp² [−0.173, 1.113]**, placements **−1.673 pp² [−5.480, 2.694]**.
These controls do not indicate a resolved nonzero null effect at this precision;
they do not prove that every noise assumption is exact. Negative corrected
estimates remain unmodified. In particular, the random-edit estimate is small
relative to its control uncertainty and should not be described as strong evidence
of sensitivity on its own.

The observed signed edit changes had median **+0.25 pp** and interquartile range
**[−0.25, +0.75] pp** for random teams, versus **−2.25 pp** and
**[−5.3125, +0.25] pp** for placements. These distribution summaries include battle
noise; only the primary squared-effect statistic has the stated noise correction.

Under the declared objective-bound normalization, the one-stage Pest Control
benchmark gave **4.4482e−5 [3.6107e−5, 5.3387e−5]**, while a Hartmann-6 coordinate
step of 0.01 gave **2.3822e−6 [8.9747e−7, 4.3902e−6]**. Pokemon's normalized means
were **2.9167e−5** for random teams and **1.8870e−3** for placements. Hartmann-6's
mean fell to **2.3843e−8** at step 0.001, approximately a hundredfold decrease for
a tenfold smaller step. These are descriptive comparisons using different input
spaces and step definitions, not a universal smoothness ranking or evidence that
Bayesian optimisation cannot work. All benchmark conditions and intervals are
saved in [benchmarks.json](../results/smoothness_top50_50x50/benchmarks.json).

## Frozen design

- Format: `gen9championsvgc2026regmb`, using the existing simulator and BC checkpoint.
- Originals: 50 rows sampled without replacement from `teams/hps_reg_mb_100k.jsonl`
  and all 50 entries in the frozen `results/top50_evs.json` snapshot. On 2026-10-01
  the user explicitly chose to retain this snapshot, including six ranked-season
  and four Showdown-ladder entries. The displayed group is **Frozen top placements**,
  not a strictly tournament-only cohort. The internal JSON key `tournament` names
  this placement group for compatibility with the first pilot manifest.
- The HPS corpus is a particular hierarchical product sampler, **not uniform over
  all legal teams**. Its known exclusions, including sets with fewer than four
  moves, remain part of this study's scope. We sample existing legal teams rather
  than substitute ordinary tournament teams for the random group.
- Tournament anchors retain the existing placement selection and duplicate entries.
  MB493 and MB494 are identical; these belong to one bootstrap cluster. Identity
  ignores capitalization/blank lines but preserves roster and move order because
  those may affect the fixed policy. The 50 entries are not 50 independent teams.
- Opponents: the same fifty entries for everyone, including anchors that also appear
  in the pool. Duplicate entries retain their original weight. Nothing is removed
  to avoid self matchups because that would change the objective across originals.
- Edits: eight distinct **single move replacements** per original. Select uniformly
  among move positions with alternatives, then uniformly among the species' legal
  move table excluding its current moves. Reject invalid/no-op/duplicate proposals.
  This is not uniform over all legal neighbors; the procedure is identical in both
  groups. Each final team passes Showdown's validator. All other text is preserved,
  including IVs, spreads, nature, item, species and ordering.
- Each original, edit, and unchanged-team control gets four independently seeded
  panels of 100 battles. Every panel cycles the same 50 opponents twice. The full
  default run costs **400,000 battles**, with equal budgets for both groups.
- Both sides use the same frozen behavior-cloning policy. A tie is a non-win:
  the objective is probability of winning, not expected tournament points.

Seeds fix sampling, proposals, and Python/NumPy/Torch action RNG initialization.
Showdown's battle RNG is not controlled by the existing network evaluator, and
concurrent action scheduling can change random-number consumption. We therefore
claim reproducible **design and saved analysis**, not bit-identical battle replay
or paired common battle RNG. Each candidate/replicate gets its own seed.

## Noise and uncertainty

For each edit, let `d[r]` be its win-rate estimate minus its original's estimate
in replicate panel `r`. The primary statistic is

```
mean(d)**2 - sample_variance(d) / number_of_replicates
```

This equals the average `d[r] * d[s]` over distinct replicate pairs. Under independent
replicate panels it estimates the **squared expected win-rate change**, removing
the sampling-noise contribution without assuming identical matchup win probabilities.
Take the mean across edits to give each original equal weight. Keep negative
estimates: clipping or taking their square root would bias the result upwards.
The unchanged-team control estimates the same quantity with true change zero.

The two-group contrast is tournament minus random mean corrected squared change.
Its 95% percentile interval resamples whole original-identity clusters, carrying all
their neighbors and duplicate placement entries together. Thus shared-baseline
dependence does not create hundreds of fake independent observations. Bootstrap
replicates for the groups use independent streams. These are descriptive intervals
for resampling these empirical groups, not evidence that tournament teams were
randomly sampled from all competitive play. With very few pilot clusters, intervals
are exploratory. The bootstrap conditions on the sampled edit protocol and pool.

Signed changes and observed absolute changes are also saved. The observed empirical
distribution **still contains noise**; it is not a recovered distribution of latent
absolute effects. Edit-level error bars use pointwise Newcombe/Wilson score intervals
for two proportions, pooling wins across replicates. These are approximate for the
fixed opponent strata and do not collapse to zero width after zero wins. They are
not multiplicity-adjusted; do not count interval exclusions as confirmed discoveries.
No practical-effect threshold or power claim is invented by the software.

## Benchmarks and figure

Reuses the existing `ruggedness.py` implementations:

- [COMBO Pest Control](https://github.com/QUVA-Lab/COMBO/blob/master/COMBO/experiments/test_functions/multiple_categorical.py):
  25 categorical stages, five choices, 100 simulated trajectories per evaluation.
  Replace one stage for the primary comparison; also evaluate two and three distinct
  stage replacements. Replicate evaluations use independent RNG streams.
- [Hartmann-6](https://www.sfu.ca/~ssurjano/hart6.html): smooth deterministic function
  on the unit cube. Replace one coordinate by a signed step of 0.01 for the primary
  comparison, plus 0.001 and 0.05. Original points are sampled from `[0.05,0.95]^6`
  so all steps have their stated size; there is no boundary clipping.

The common display divides squared effects by squared **fixed objective bounds**:
1 for win probability, 50 for Pest Control, and 8.4 for Hartmann-6. Those are
declared analytic bounds, not ranges fitted to sampled outcomes. They do not equate
the neighborhood sizes or make a universal smoothness ranking possible. Benchmark
points are random reference regions, not analogues of tournament placements.
The distance plot uses different labeled units for the two benchmarks. The Pokemon
implementation deliberately tests one move at distance one; it does not claim a
multi-distance Pokemon result.

## Commands

The completed full study was managed by `scripts/run_smoothness_study.py`, with
dedicated servers on ports 8170–8173 and four workers at reduced operating-system
priority. It saved completed panels, ran analysis automatically, and stopped its
own servers; unrelated campaigns were left alone. `caffeinate` prevented idle sleep
during the run. The completion heartbeat was disabled after final review.
For any future run, do not launch a second copy while `run-status.json` identifies
a live supervisor.

Run from the project root with the installed battle runtime:

```sh
cd /Users/ramiismael/Documents/code/vgc-team-generator-pilot
PY=/tmp/vgc-pilot/.venv/bin/python

# Statistical and protocol tests; no server required.
$PY -m unittest discover -s scripts -p 'test_smoothness_experiment.py' -v

# Freeze the entire 50+50 design and validate every input. No battles yet.
$PY scripts/smoothness_experiment.py prepare --output results/smoothness_top50_50x50

# Compute the inexpensive reference functions.
$PY scripts/smoothness_experiment.py benchmarks --output results/smoothness_top50_50x50

# Use verified local Showdown servers. Resume with exactly the same command.
$PY scripts/smoothness_experiment.py battle --output results/smoothness_top50_50x50 \
  --ports 8123 8124 8125 8126 8127 8128 8129

# Requires every expected battle result; writes data, report and PNG/PDF figure.
$PY scripts/smoothness_experiment.py analyse --output results/smoothness_top50_50x50

# Separate small, real-battle pilot. Never confuse this with the full study.
$PY scripts/smoothness_experiment.py all --output results/smoothness_pilot \
  --anchors 4 --neighbors 2 --replicates 3 --battles 50 --ports 8123 8124
```

Preparation options only set a new manifest. On resume, the stored manifest is
authoritative. Use a new output directory for a new design. Each analysis requires
all jobs with exactly the specified number of battles; missing, corrupted,
conflicting, or partial records cause an error. Each finished panel is committed
atomically, and an exclusive run lock prevents concurrent writers. Worker failures
stop the run; already completed panels remain resumable.

The manifest contains input copies/hashes, code hashes, checkpoint hash, runtime
source/build hashes, exact edits, sources, counts, seeds, and opponent IDs. Battle
resume refuses changed frozen code/runtime or team inputs. Analysis can run from
the saved counts without the original runtime. Benchmarks require the source
implementation. Archive the whole result directory and the implementation for
publication. Local server ports must be running the same built simulator as the
validator; the port connection check alone cannot prove that, so use servers started
from the manifest's runtime. No online ladder games or remote paid services are used.

Outputs: `manifest.json`, `teams/`, `opponents/`, `labels/`, worker logs,
`benchmarks.json`, `analysis.json`, `edits.csv`, `smoothness.png`, `smoothness.pdf`,
and `report.md`. Raw labels are the audit trail. The figure distinguishes observed
changes from noise-corrected squared changes and labels small runs as pilots in the
report. A significant local sensitivity result is not evidence that BO cannot work.
