# DiBO replacement verification — October 12, 2026

The current search is `src/search.py` → `src/dibo_v3.py`. The former executable
`cem_v3.py` interface also delegates to DiBO. The new path never calls the old
elite selector, ridge acquisition, or elite-only generator retraining. Historical
comparison helpers and results remain preserved.

## Evidence

- `scripts/test_dibo.py`: **7 passing tests**. Exact categorical density and
  sampling, RTB convergence to prior-times-reward, zero-reward identity, positive
  low-score training weights, independent ensemble disagreement, full-density
  ranking/deduplication, and malformed-input checks.
- `scripts/test_heldout_separation.py`: **15 passing tests** on the existing
  project data, training filters, semantic identities, battle guards and final
  evaluation boundary.
- `results/dibo_verification.json`: the final implementation completed **two
  real-model rounds**, with 16 initial observations, five predictors per round,
  weighted prior training, two RTB updates per round, eight legal proposals per
  round, and two selected/evaluated distinct teams per round.
- Every stored proposal satisfies `log_target = log_prior + beta * UCB`.
- The four selected grids are distinct from the frozen initial observations
  and from previously selected grids. All passed the simulator legality gate.
- Final selection was saved before measurement. Two selected teams each
  completed eight fresh top50 confirmation battles and eight held-out battles.
- The manifest's implementation hashes match the current adapter and core.
- A completed search was resumed without retraining or starting further search
  battles. `cem_v3.py report --smoke --output results/dibo_verification.json`
  correctly reports the DiBO result.

Search objectives use the matchup cache, so verification can reuse previously
measured cells instead of spending the same battle budget again. The separate
final confirmation and held-out measurements are fresh. An earlier integration
attempt encountered no running battle server; its saved candidate batch resumed
after a loopback server was started. Pre-final-guard development manifests and
checkpoints were archived separately rather than reused under changed source.

## Limits

This verifies implementation and integration. Two RTB updates and eight final
battles are deliberately small plumbing checks, not a converged posterior or an
experiment establishing improved competitive performance. A full configured run
and a matched-budget CEM comparison remain necessary before making that claim.
The adaptation searches the existing 48 categorical fields; Stat Points are
deterministically rendered, not separately learned or optimized.

## Reproduce

```sh
/tmp/vgc-pilot/.venv/bin/python scripts/test_dibo.py
/tmp/vgc-pilot/.venv/bin/python scripts/test_heldout_separation.py
/tmp/vgc-pilot/.venv/bin/python src/search.py report --smoke --output results/dibo_verification.json
```

For a fresh integration run, use a new manifest path with the same small setup:

```sh
/tmp/vgc-pilot/.venv/bin/python src/search.py run --smoke --output results/dibo_my_check.json
/tmp/vgc-pilot/.venv/bin/python src/search.py final --smoke --output results/dibo_my_check.json
```

Full-run configuration and the paper-to-code mapping are in [dibo-v3.md](dibo-v3.md).
