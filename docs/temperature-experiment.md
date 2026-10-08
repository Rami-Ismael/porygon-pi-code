# Selection-temperature experiment

Completed 2026-09-08. All five arms across three seeds passed the full protocol
audit: 165 generations and 921,600 battles. Annealing averaged 33.25% fresh
generator win rate, versus 36.05% at fixed 0.10, while retaining more species-set
diversity. It was the only arm without a diversity guardrail failure, but the
three-seed comparisons did not establish a win-rate improvement. See the
[completed findings](/Users/ramiismael/Documents/code/vgc-team-generator-pilot/docs/temperature-results.md)
for all arms, uncertainty, and verification artifacts.

This study tests finite-pool resampling with `w_i ∝ exp(measured_win_rate_i / T)`.
It does not implement DiffUCO's joint-KL diffusion objective. DiffUCO lowers its
training temperature linearly to zero ([v1 §4](https://arxiv.org/html/2406.01661v1#S4));
our positive selection-temperature floor reflects noisy battle scores.

## Locked comparison

Five arms: linear **0.30 → 0.10**, fixed **0.10**, **0.20**, **0.30**, and adaptive.
The linear schedule includes both endpoints over 11 generations and holds 0.10
thereafter. The adaptive arm finds `ESS / pool_size = 0.50` by bisection within
`[0.10, 0.30]`; an unreachable target returns the nearest boundary, recorded in
the results. ESS is `1 / sum(w_i²)`. Equal scores remain uniformly weighted.
This target and these bounds are experimental choices, not claims from DiffUCO.

All arms share a checkpoint, anchor/gen0 labels, corpus, fixed BC battle policy,
50-opponent pool, acquisition method, gloss guidance strength 108, decoding
temperature **1.0**, and loss entropy bonus **0**. Each generation restarts from
the shared checkpoint, as in the existing loop. Selection uses replacement and
`max(100, floor(.25 * pool_size))` draws. Every arm receives the same 40 epochs
and batch size 64. Completed battle counts and optimizer steps are checked.
Pool rows, including repeated teams, retain the historical empirical measure;
duplicates can therefore accumulate extra mass. Proposal diversity counts repeats.

Default seeds: **101, 202, 303**. The selection, refit, proposal and battle-policy
seeds are paired across arms and differ across repetitions/phases. Arm execution
order is shuffled per seed. Opponent schedules are balanced shuffled cycles and
identical across arms at a given seed/generation. Policy RNG pairing does not
control Showdown's internal battle RNG, so exact replay is not claimed.

Each arm/seed runs **11 × (512 valid proposals, 128 labels × 24 battles)**.
Diversity deterioration is recorded without stopping early, because arm-specific
stopping would confound the matched budget. Failed/incomplete generation or battle
jobs raise an error; their partial scores never enter the pool.

## Fresh evaluation and decision

The primary endpoint is mean win rate of **128 new, unranked samples** from the
final generator, each playing **192 fresh battles**. These samples are generated
after search with an independent seed and cannot enter training. The final model
is the model used to propose generation 11; no extra post-search refit is added.

A separate endpoint is the mean fresh win rate of **16 unique finalists**, ranked
using search scores alone, at 192 battles each. The report retains search ranking
and gives the fresh score of the original rank-1 team. The maximum fresh score
must not be treated as an unbiased estimate of the best team.

Default total: **921,600 battles**, plus the already shared anchor/gen0 budget
and any separately recorded smoke/restoration tests. Three seeds give a pilot,
not a high-power confirmatory test. Compare paired seed-level differences; neither
battles nor successive generations are independent training replicates. Multiple
comparisons against four alternatives require care if making significance claims.

Measure diversity on equal-sized valid proposal streams before acquisition and
without deduplication, plus the independent final sample: distinct complete teams,
distinct species sets, effective species sets (`exp(H)`), mean pairwise species
Jaccard distance, and corpus nearest-neighbor/copy metrics. Pool ESS is a selection
diagnostic, not generated-team diversity. Assess win-rate gains jointly with all
diversity trajectories; a higher finalist score alone is insufficient.

The final report also records every generation that crosses the original
`entropyloop.py` diversity guardrails: corpus copy rate above 0.10 or fewer than
200 distinct species sets in 512 proposals. These are diagnostics, not early-stop
rules for this study. Passing those absolute thresholds does not establish
noninferiority to another arm; no relative diversity margin was specified.
The copy and nearest-neighbor metrics use the historical 48-field representation,
which excludes Stat Point spreads. Complete-team uniqueness includes those spreads.

## Commands

From the repository root, using a Python environment with NumPy:

```sh
python src/entropyloop.py compare preflight
python scripts/test_temperature_experiment.py
python src/entropyloop.py compare run
python src/entropyloop.py compare report
```

The runtime requires Torch, the VGC-Bench/poke-env fork, its BC policy, a Showdown
validator/server, the exact corpus, legality table, and original starting model.
Preflight lists missing inputs without importing the heavy runtime. The runner
hashes checkpoint, policy, labels, corpus, opponents, and implementation files;
resuming with changed inputs or settings is rejected. New experiments need new
`--output` paths. Completed proposal and battle phases are saved atomically;
partial failed simulator calls may spend extra attempts, which are not counted as
completed experiment observations. Never merge them with another run's results.

The default dedicated server ports are 8130–8133 (`--ports` overrides them). This
avoids sharing the historical campaign's 8123–8129 servers. A file lock prevents
two processes from writing the same result file.

The active campaign was continued with independent seeds running concurrently to
use the available 12-core host. Seed 101 finishes in the original process on
8130–8133; seed 202 uses 8134–8137 and seed 303 uses 8138–8141. Each seed's five
temperature arms share its four-server pool. The additional instances have
separate mutable runtime directories and identical simulator code (472 compiled
JavaScript files checked for each instance). No treatment, training, or battle
budget changed. The original controller is stopped after all five seed-101 arms
finish, before it can duplicate the independently running seeds.

`scripts/parallel_temperature_campaign.py` retains the three raw result files and
creates `results/temperature_comparison_parallel.json` with explicit seed/source
and server-pool provenance. It rejects any mismatch in checkpoint/data/code
fingerprints, opponents, arms, or experimental configuration; seed lists and
port numbers are the only permitted configuration differences. The merged artifact
is schema 2 and is resumed through the collector, not the sequential runner.
The same budget auditor and seed-level analysis are run after all 15 runs finish.

The collector binds each source version to the file descriptor used for its JSON
read. Taking the path's timestamp after a read could otherwise tag an older
snapshot with a newer atomic replacement's timestamp and miss that update. A
deterministic regression test replaces the source during a real collector read and
checks that the next iteration collects it. Final source hashes also wait for the
independent workers' terminal status write. On 2026-09-08, only the collector was
restarted to apply these fixes; all three experiment PIDs continued unchanged. The
restart is recorded in the runtime's `temperature-campaign-processes.json`.

Seed 202's fixed-0.20 generation 10 encountered a simulator team-validator crash
while accepting team 0073's thirteenth scheduled opponent. The challenge remained
pending while both clients waited for the battle to start. The existing acceptance
was reissued through the normal simulator handler after checking both packed teams
against the worker specification. The same worker then completed its original
shard. Verification found all 128 teams had 24 battles, all 114 previously completed
team scores were unchanged, and all earlier generations and saved training/proposal
data were unchanged. The before/after evidence and the exact recovery helper are
preserved in `results/recoveries/seed202-fixed020-g10-20260908T115607Z/`.

Before accepting the final result, also run the independent protocol audit below.
It requires all 15 runs, 165 generations, and 921,600 battles; reconstructs the
training pools to check temperatures and weight concentration; recomputes diversity
from the full proposal streams; verifies surrogate acquisition and search-only
finalist selection; and checks raw-source provenance, input hashes, and runtime
versions. It does not import the frozen training implementation. The copy-rate and
nearest-neighbor values still rely on that fingerprinted implementation; the audit
checks their bounds and independently checks their species counts.

```sh
/Users/ramiismael/.local/share/vgc-pilot-runtime/venv/bin/python \
  scripts/audit_temperature_protocol.py results/temperature_comparison_parallel.json
/Users/ramiismael/.local/share/vgc-pilot-runtime/venv/bin/python \
  scripts/test_temperature_protocol_audit.py results/temperature_comparison_parallel.json
```

The audit writes `results/temperature_comparison_parallel.protocol-verification.json`.
During execution, `--allow-incomplete` checks only the observed records and reports
`partial_checks_passed`; that is not the completion gate. Fault-injection tests use
temporary copies of one completed real run and reject wrong temperatures, altered
budgets, nonzero entropy loss, acquisition changes, unsupported diversity values,
reused search seeds in fresh evaluation, and finalist reranking.

### Restoration for this run (2026-09-07)

The original checkpoint and most `/tmp` runtime files had been cleared. A new
shared checkpoint is trained for 600 epochs, seed 0, on the current **692 valid
corpus teams**. This is a new controlled experiment conditional on that checkpoint;
it is not a reproduction of the older 543-team baseline. Its corpus, source, and
checkpoint hashes are recorded in `results/temperature_p0.training.json`.

The restored BC policy is the immutable [VGC-Bench checkpoint](https://huggingface.co/cameronangliss/vgc-bench-models/resolve/204c76741829ca0681629e41382043c385850d5c/results/saves_bc/seed1/100.zip),
SHA-256 `57f5edcab415cf6ccc1b6231923c8b66d3b1b7249b6b2562b97531471e4ca60b`.
Runtime source revisions: [VGC-Bench d79f953](https://github.com/cameronangliss/vgc-bench/tree/d79f9532947ac114dce1dda2456a590afcd375b2),
[Showdown 913da36](https://github.com/cameronangliss/pokemon-showdown/tree/913da3602a3aa1db79f9fdc5d5222eaf8d39569d),
and [poke-env 379a046](https://github.com/cameronangliss/poke-env/tree/379a04628e42b2a2c94790c0788204b652f7ae1b).
The new environment uses Torch 2.12.0, NumPy 2.4.4, SB3 2.8.0, and Gymnasium 1.2.3.
The policy successfully completed a four-battle restoration smoke test, saved
separately in `results/temperature_runtime_smoke.json`.

```sh
/Users/ramiismael/.local/share/vgc-pilot-runtime/venv/bin/python \
  scripts/prepare_temperature_checkpoint.py
/Users/ramiismael/.local/share/vgc-pilot-runtime/venv/bin/python \
  src/entropyloop.py compare run --checkpoint results/temperature_p0.pt
```

`scripts/audit_temperature_selection.py` produces separate battle-free diagnostics
on identical historical pools. At generation 1, T=0.10 has ESS/pool 0.181, versus
0.684 at T=0.30. The adaptive target chooses approximately 0.204. These demonstrate
different selection pressure; they provide no new generator win-rate evidence.

For a plumbing run, use a distinct output and explicitly reduced budgets:

```sh
python src/entropyloop.py compare run --seeds 999 --generations 2 \
  --propose 8 --battle 4 --battles 2 --final-sample 4 --final-battles 2 \
  --finalists 2 --epochs 1 --output results/temperature_smoke.json
```

Historical `entropyloop.json` compares boltz/niche/refit arms against cached runs.
It is not a temperature-controlled comparison and must not supply the new study's
fixed-temperature or fresh-evaluation results.
