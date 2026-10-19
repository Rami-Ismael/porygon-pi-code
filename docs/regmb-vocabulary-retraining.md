# Regulation M-B vocabulary and retraining

**Both F1 and G2 were retrained with a simulator-derived vocabulary. All 63 frozen baseline contexts are representable.**

The original checkpoints remain unchanged. New checkpoints add `_regmb_v2` to their names. `regmb_model.model()` loads the new pair; older experiments retain explicit historical checkpoints.

## What changed

The corpus no longer defines the vocabulary for this pipeline. The exporter queries the pinned Champions M-B simulator, including legal unevolved species and forme representations, native and item-dependent abilities, usable items, learnable moves and all 25 named natures. Empty items, move padding and omitted/default natures have explicit tokens; they are distinct from the mask token. Accepted species/form entries are not a count of distinct evolution families.

| Field | Named/usable values | New tokens versus old vocabulary |
|---|---:|---:|
| species | 347 | 158 |
| ability | 200 | 73 |
| item | 148 | 38 |
| move | 496 | 181 |
| nature | 25 | 7 |

Ability/item compatibility is checked jointly: for example, Metagross can name Tough Claws when holding Metagrossite, but that combination is not accepted without the stone. The complete held-out set was retained after verifying these conditional forms and preserving move-padding order.

Token migration copies shared embedding rows, output weights and biases by token identity, and preserves the transformer backbone. New rows are initialized near existing rows and trained. Checkpoints store token identities, the full rules snapshot and parent/data/source fingerprints. Loaders reject a mismatched vocabulary or rules snapshot rather than silently loading the wrong indices.

## Training actually run

Generated 6,000 distinct simulator-validated teams. Every nonempty legal token appears at least 14 times. Synthetic examples carry no invented win-rate or opponent-row labels. Original test teams and frozen pilot inputs/legal Ling outputs are excluded from synthetic data.

The remapped historical pool retains 146,994/146,994 teams under compatibility checks. The original test set retains 500/500 teams. F1 uses its historical family cohort; G2 also uses matrix-labelled teams, preserving its matrix holdout.

Each model receives 150 embedding/output-head warm-up steps with the backbone frozen, then two full-network epochs. Batch size 128; warm-up learning rate 0.0003, full-network rate 0.0001. Training seed 20261009 is a fixed integer, not the execution date. This is continued training of the existing models, not training from scratch.

| Model | Training rows incl. synthetic | Migration diagnostic CE | After training diagnostic CE | Held-out 500-team CE |
|---|---:|---:|---:|---:|
| F1_final_medium_regmb_v2 | 21,179 | 2.1652 | 2.1987 | 2.1976 |
| G2_family_matrix_noprtrain_regmb_v2 | 29,920 | 2.1690 | 2.1737 | 2.1850 |

All 457 newly added embedding rows changed during training in each model, verified against deterministic migration initialization. Parent checkpoint hashes are unchanged.

Migration/after diagnostics use the same 256 historical validation rows and mask seeds; those rows were seen by the parent final checkpoints, so they are not an unseen generalization test. The held-out test remains separate. Expanded-vocabulary losses are not directly interchangeable with published losses from the smaller output vocabulary. Battle performance is evaluated separately.

## Reproduce

From the repository root:

```sh
node code/scripts/export_regmb_vocab.js "$HOME/.local/share/vgc-pilot-runtime/validator-913da36" code/data/regmb-vocabulary.json
/tmp/vgc-pilot/.venv/bin/python code/scripts/retrain_regmb.py prepare
/tmp/vgc-pilot/.venv/bin/python code/scripts/retrain_regmb.py train
/tmp/vgc-pilot/.venv/bin/python code/scripts/report_regmb.py
```

Existing derived checkpoints are reused only when the recorded data, trainer and settings match. `--replace` explicitly reruns and replaces only the derived v2 checkpoint names. Parent checkpoints are never overwritten. Data and weights are under `~/.local/share/vgc-pilot-runtime/`; the rules snapshot and training summary are in `code/data/`.

## Remaining limits

Vocabulary coverage is not a guarantee of legal or strong completions. Final Showdown validation still gates every output. Stat Points remain outside the neural grid: fills use corpus spreads, with a random legal spread when the new species has no corpus spread. Explicit opponent pastes are still not model inputs. There were no new LLM API calls during this retraining.

The updated frozen-mask battle comparison is documented in `diffusion-baseline-comparison.md`.

Completed pilot: 63/63 contexts supported; 62/63 legal completions; 62 scored 50-battle panels (exact cached panels reused where applicable). On 27 jointly legal tasks, the equal-start-weighted difference versus Ling is -0.52 percentage points. Only three starting teams: no statistically established winner. One decoder dead end was recorded without repair or retry.
