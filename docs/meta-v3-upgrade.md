# V3 handoff during the meta-starter experiment

The user requested that the current diffusion generation finish before retraining,
while allowing Ling's response generation to continue in parallel.

Confirmed on October 8: all 4,998 v2 attempts finished, both v3 checkpoints were
trained and saved, and v3 regeneration started while Ling was still running.
Checkpoint policy checks reject Floettite for Kingambit in both models. Historical
test cross-entropy was 2.174873 for F1 and 2.169275 for G2; these are reconstruction
metrics, not battle-win improvements.

The upgrade coordinator performs these stages in order:

1. Wait for the existing v2 generation lock to release and verify all 4,998 outputs.
2. Train the two v3 checkpoints using the prepared 6,000-team dataset, with variable
   move counts and exact-form Mega Stone filtering. Existing v2 checkpoints remain.
3. Generate 4,998 new diffusion completions in
   `code/results/diffusion-baseline-meta-starters-v3`, using the same frozen masks,
   starters, seed schedule, ask 0.5, guidance 2, and temperature 1.
4. Wait for Ling's existing workers. Saved Ling responses are reused. Missing
   responses are resumed only after those workers release their locks; uncertain
   sent requests stop for reconciliation.
5. Score v3 against Ling with the same per-starter opponent schedules and BC policy,
   then publish the meta-starter results.

The original coordinator was paused before battle scoring. Its generation workers
continue independently. It is retired only after their saved outputs are complete.
V2 diffusion outputs are retained for audit; they are not overwritten with v3.

## Recovery

From the repository directory:

```sh
~/.local/share/vgc-pilot-runtime/venv/bin/python code/scripts/upgrade_meta_v3.py
```

The coordinator skips saved generations and completed model checkpoints. A partial
training run resumes at the unfinished model, not at an individual gradient step.
An API key is requested only if Ling calls remain after its existing workers finish.

The live page at [meta-starter progress](http://127.0.0.1:8767/meta.html) shows the
retained v2 count, v3 training progress, new v3 generation count, Ling progress,
and actual API cost. Training and diffusion generation are local: this handoff
does not rerun paid Ling responses.

Whether v3 training and regeneration finish before Ling depends on measured
runtime; the coordinator waits for both before starting battle scoring.
