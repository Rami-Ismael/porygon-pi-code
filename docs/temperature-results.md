# Boltzmann selection temperature: completed results

Completed 2026-09-08: five temperature arms, three training seeds, 165 search
generations, and **921,600 measured battles**.

Annealing selection temperature from 0.30 to 0.10 produced a tradeoff: it retained
more species-set diversity than fixed 0.10 or 0.20, but had lower observed fresh
generator win rates than both in every seed. Its average win rate exceeded fixed
0.30 and ESS-adaptive selection, with mixed differences across seeds. All four
adjusted confidence intervals include zero, so this study does not establish a
win-rate improvement.

## Results

Means across training seeds 101, 202, and 303:

| Selection temperature | Fresh generator win rate | Fresh finalist mean | Distinct species sets | Effective species sets | Seeds crossing a diversity guardrail |
|---|---:|---:|---:|---:|---:|
| Annealed 0.30 → 0.10 | 33.25% | 46.77% | 255.0 | 97.5 | 0/3 |
| Fixed 0.10 | 36.05% | 47.57% | 172.0 | 38.4 | 2/3 |
| Fixed 0.20 | 34.58% | 46.88% | 204.3 | 45.0 | 1/3 |
| Fixed 0.30 | 32.15% | 46.57% | 276.3 | 112.8 | 1/3 |
| Adaptive ESS/pool = 0.50 | 32.87% | 46.78% | 265.7 | 134.5 | 1/3 |

Generator evaluation uses 128 new, unranked legal samples per run, each playing
192 fresh battles. Finalist evaluation separately tests 16 unique teams selected
using search scores, also with 192 fresh battles each. Fresh scores never feed
training or determine which finalist is reported as search rank 1.

Diversity uses the full 512-team final proposal stream before surrogate ranking.
Effective species sets are the exponential of Shannon entropy, so repeated species
combinations reduce this measure. The existing guardrails flag fewer than 200
distinct species sets or a 48-field corpus copy rate above 10%. Annealing was the
only arm that avoided both guardrails throughout all three seeds. Every arm's
final corpus copy rate was zero. The adaptive target was attained in all 33
adaptive generations, but one seed still fell below the species-set threshold.

Annealing's average diversity was below fixed 0.30 and adaptive on distinct and
effective species sets. Its diversity advantage therefore depends on the
comparator. No relative noninferiority margin was specified; these comparisons
remain descriptive.

## Paired win-rate comparisons

Annealed minus comparator, in percentage points:

| Comparator | Mean difference | Adjusted 95% paired t interval | Seed differences (101, 202, 303) |
|---|---:|---|---|
| Fixed 0.10 | −2.81 | [−19.22, +13.61] | −1.59, −0.38, −6.45 |
| Fixed 0.20 | −1.33 | [−7.01, +4.35] | −1.45, −2.37, −0.16 |
| Fixed 0.30 | +1.10 | [−15.96, +18.16] | +3.89, +2.01, −2.59 |
| Adaptive | +0.38 | [−24.98, +25.74] | +0.05, +5.49, −4.41 |

The inference unit is the paired training seed. These exploratory t intervals use
two degrees of freedom and a Bonferroni adjustment for four comparisons within
the endpoint. Three seeds cannot establish the normal-effects assumption; the
paired sign-flip p-values under a symmetric zero-effect null are 0.25, 0.25,
0.75, and 0.75. The large number
of battles does not increase the number of independent retraining replicates.

![Search, diversity, temperature, and fresh evaluation trajectories](/Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/temperature_comparison_parallel.png)

## Controls and scope

All arms shared one starting checkpoint, cached initial labels, corpus, BC battle
policy, 50 opponents, training recipe, and battle budgets. Decoding temperature
was 1.0 and the separate loss entropy bonus was zero. Each generation refitted
from the shared checkpoint for 40 epochs using the expanded evaluated pool;
each run used exactly 1,880 training steps and 61,440 battles. The final model
was the one used to propose generation 11, with no additional post-search refit.
Battle schedules and policy seeds were paired across arms within each seed.
Identical initial-temperature controls reproduced all 512 proposals and the
128-team surrogate selection in all three seeds.

The original temporary checkpoint was unavailable. The shared replacement was
trained for 600 epochs on the current 692 valid corpus teams. Results are
conditional on that checkpoint, the shared cached initial pool, the fixed policy,
and this opponent population. Showdown's internal battle RNG was not controlled.
Search labels used 24 battles per team and can be noisy.

This is an evaluated-pool Boltzmann selection experiment inspired by
[DiffUCO](https://arxiv.org/html/2406.01661v1). The full DiffUCO variational training
objective is outside this experiment. Selection temperature is scheduled or
adapted from weight concentration; it is not learned as a network parameter.

## Verification and execution record

The independent full protocol audit passed all 15 runs, 165 generations, and
921,600 battles. It checked 1,295 input-file hashes, four runtime package versions,
temperature and concentration calculations, training steps, fresh-evaluation
separation, paired controls, recomputed species diversity, and equality between
combined results and their raw sources. Copy/nearest-neighbor values rely on the
fingerprinted producer; their bounds and species counts were checked separately.

Two collector races were fixed with deterministic regression checks; the
training processes and frozen experiment code continued unchanged. One crashed
team-validator subprocess left an existing battle challenge pending. Resuming
that same acceptance allowed the original worker to finish: all 114 previously
completed team scores were unchanged, and all 128 teams retained exactly 24
battles. The [recovery receipt](/Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/recoveries/seed202-fixed020-g10-20260908T115607Z/verification.json)
records this check. The eight additional simulators matched the baseline across
472 compiled files each. All experiment-owned simulator pools were stopped after
their seed completed.

The eight experiment tests, three collector regression tests, and ten independent
audit tests passed during implementation. The four-battle runtime check and
140-battle plumbing run are separate from the full-study budget and findings.

## Artifacts

- [Protocol and reproduction commands](/Users/ramiismael/Documents/code/vgc-team-generator-pilot/docs/temperature-experiment.md)
- [Raw combined results and source provenance](/Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/temperature_comparison_parallel.json)
- [Independent full protocol audit](/Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/temperature_comparison_parallel.protocol-verification.json)
- [Final 21-test receipt](/Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/temperature_comparison_parallel.tests.json)
- [All individual results and guardrail crossings](/Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/temperature_comparison_parallel.analysis.md)
- [Paired estimates and diversity metrics as JSON](/Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/temperature_comparison_parallel.analysis.json)
- [Final source archive manifest](/Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/temperature_comparison_final_source/manifest.json)
