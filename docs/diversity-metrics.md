# Team diversity metrics

`src/diversity_metrics.py` measures accepted proposal draws before ranking. It keeps duplicates and uses normalized records, so an unknown vocabulary token does not become a shared mask value. It exposes three representations:

| Representation | Equality |
| --- | --- |
| `categorical_48` | Species/form, ability, item, moves and nature for each of six Pokémon |
| `with_spreads` | The same fields plus six Stat Points per Pokémon |
| `composition` | The unordered six species/form identifiers, retaining multiplicity |

Moves and complete Pokémon records are sorted. Nicknames and text formatting do not contribute. An explicitly omitted nature becomes Serious, matching the Reg M-B Showdown validator. Missing required parsed keys are errors. The corpus parser supplies zero for omitted Stat Points. These keys cover the current pilot representation; fields such as Tera type, gender, IVs or level need a new key version if they vary in a future format.

For N accepted draws and U distinct keys, unique fraction is U/N and repeat fraction is (N−U)/N. Novel-draw fraction counts every draw absent from a fixed reference. Novel-unique fraction counts distinct absent keys divided by U. Each representation records numerators and denominators. Composition output also contains the complete frequency table and largest composition share. Empty samples produce null fractions and an `empty` status; a sample-size mismatch is explicit.

## Logging and reproducibility

The temperature comparison and `entropyloop.py` log these metrics alongside their existing results. They freeze normalized reference records and their hash; a changed reference prevents resuming those metric records. Live logging labels the reference `corpus`, since the presence of a corpus alone does not prove checkpoint training membership.

The guided sampler records exact attempted, accepted and rejected counts and uses a hard limit of eight attempts per requested accepted draw: 4,096 for the 512-proposal stream. Every accepted repeat is retained. Temperature-comparison failures save `sampling_failure.json` before rejecting an incomplete sample. This is conditional-on-acceptance diversity. Conversion into paste text occurs before validation; no post-validation repair is introduced.

The separate historical report verifies the training manifest against the campaign checkpoint, hashes every training file, and checks the parser and its data. Only then does it label novelty `initial_training`. It stores the frozen normalized reference and source hashes. Historical attempt counts remain null because the saved campaign only contains acceptance fractions.

```sh
.venv/bin/python scripts/report_team_diversity.py results/temperature_comparison_parallel.json \
  --training-manifest results/temperature_p0.training.json \
  --output results/temperature_diversity_metrics.json
```

The report covers 165 generation streams and 15 final unranked samples: 86,400 accepted draws. It checks sample sizes and battle budgets. Generation win rates describe score-selected candidates, while generation diversity describes all proposals. Final-sample win rate and diversity describe the same unranked draws. Do not compare unique fractions at different sample sizes as if they shared a denominator.

In these saved streams, categorical and with-spreads uniqueness and novelty are all 100%. Composition counts still decline: the fixed-0.10 arm averages 477.7 compositions in its first generation and 172.0 in its last, out of 512 proposals. Exact novelty alone therefore misses this observed concentration.

## Completed baseline and comparison

The original checkpoint baseline is complete: three seeds, 1,920 accepted draws across six samples, and 73,728 battles. Its 512-proposal samples average 498.3 distinct compositions, with a largest-roster share of 1.4%. The fixed-0.10 arm ends at 172.0 compositions and a largest-roster share of 40.5%. Both retain 100% exact uniqueness in the pilot representation. The baseline's independently sampled generator win rate is 18.00%; final arms range from 32.15% to 36.05%. These are descriptive three-seed means for a proposal procedure that updates both generator and surrogate.

- [Before/after report and chart](../results/diversity_before_after.md)
- [Baseline samples, reference and all battle counts](../results/diversity_baseline.json)
- [Full source and measurement audit](../results/diversity_verification.json)

```sh
/Users/ramiismael/.local/share/vgc-pilot-runtime/venv/bin/python scripts/report_diversity_baseline.py \
  --baseline results/diversity_baseline.json --history results/temperature_diversity_metrics.json \
  --output results/diversity_before_after.md
/Users/ramiismael/.local/share/vgc-pilot-runtime/venv/bin/python scripts/audit_diversity_results.py
```

The baseline was captured retrospectively from the original checkpoint, with zero generator refitting. `scripts/capture_diversity_baseline.py` records its configuration and resumes only matching inputs. It uses private simulator instances and shuts those down when complete.

## Validation

```sh
.venv/bin/python scripts/test_diversity_metrics.py
.venv/bin/python scripts/test_diversity_logging.py
/Users/ramiismael/.local/share/vgc-pilot-runtime/venv/bin/python scripts/test_diversity_sampler.py
.venv/bin/python scripts/test_temperature_experiment.py
/Users/ramiismael/.local/share/vgc-pilot-runtime/venv/bin/python scripts/test_diversity_baseline.py
/Users/ramiismael/.local/share/vgc-pilot-runtime/venv/bin/python scripts/test_diversity_baseline_report.py
```

Metric tests cover ordering, formatting, defaults, spreads, forms, unseen identifiers, repeated novel teams, missing data and frozen-reference membership. Logging tests check unranked duplicates, recorded failures, reference stability on resume, and historical sample/battle completeness. Sampler tests use a stub decoder and validator to check real-loop attempt accounting and the hard rejection cap.

The sampler test uses the local battle runtime because the project virtual environment stalled reading an iCloud-backed Torch cache file during verification. The local-runtime test completed successfully.

The original campaign started after its first fine-tuning step; the separate baseline now supplies the original checkpoint comparison. Structural generalization (for example a held-out hard Trick Room strategy), team distance, and compression-based model scores remain follow-up experiments. Neither exact novelty nor a new composition establishes a new playstyle.
