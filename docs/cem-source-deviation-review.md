# CEM source-freeze deviation

Recorded during the run, before seed two and seed three were complete, on 2026-09-09 UTC.

The CEM controller began at 00:25:22 UTC. A later independent smoke audit detected three shared source files whose hashes no longer matched the frozen manifest. Their current modification times were 00:41:21 (`gradguide.py`) and 00:45:26 (`entropyloop.py`, `temperature_experiment.py`). The experiment controller remained uninterrupted.

The original 104-file source/protocol snapshot is preserved in `results/cem_score_direction_source`. Copies of the three observed edits and their hashes/timestamps are preserved separately in `results/cem_score_direction_source_deviation`. Neither snapshot nor the raw result manifest has been rewritten.

Source inspection found:

- `entropyloop.py` and `temperature_experiment.py` are outside the conservative local import closure of the CEM controller/runtime and battle worker.
- The only syntax-tree change in `gradguide.py` is the `propose_guided` helper. This experiment calls `sample_guided` directly and never calls `propose_guided`. The sampler itself and every other syntax-tree node in the module are unchanged.
- All remaining frozen inputs still match, including the CEM ranking/training code, battle code, model, corpus, metric, policy, and simulator assets.

The edits therefore do not alter the computation used by this experiment. This is a post-hoc source review, not a successful strict source-freeze check. The original controller is expected to fail its final whole-source hash check after completing the experimental phases; its original status and error will remain preserved.

The separate reporting/audit path permits only these exact observed file hashes. It requires an idle controller, every planned update and final battle evaluation to be complete, and the specific final source-change error. It rejects any additional changed input. All result-level checks, independent Pokémon-ID calculations, legality revalidation, label accounting, checkpoint chains, and statistical checks remain required. Reports and verification receipts explicitly disclose this deviation.

This review does not authorize a change to seeds, selection rules, training, battle budgets, metrics, or inference. It does not establish a treatment effect; that is assessed only after the full result is available.
