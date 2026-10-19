# Saved outputs and restarting the full grid

Run this from the repository directory after a restart or lost connection:

```sh
~/.local/share/vgc-pilot-runtime/venv/bin/python code/scripts/resume_full_grid.py
```

This resumes missing work and rebuilds the dashboard. It skips all saved Ling
requests, retains invalid answers, reuses diffusion completions, and reuses exact
completed battle panels. If API calls remain, it requests the key without echoing
or saving it. No key is needed once all responses are present.

The experiment data live on disk in the repository's `code/results` directory:

- `llm-baseline-full-grid/manifest.json`: starting teams, opponent pool, every
  mask and random-fill control, seeds and generation settings.
- `llm-baseline-full-grid/completions/<task>.json`: full raw provider response,
  generated text, usage/cost receipt, request fingerprint, validation outcome,
  and accepted team when legal. Rejected outputs are preserved too.
- `diffusion-baseline-full-grid/completions/<task>.json`: generated diffusion
  team or failure details. `generation-config.json` identifies the checkpoints,
  vocabulary, source code, and baseline settings.
- Both runs' `labels/`: scored 50-battle panels. During parallel scoring,
  checkpoints also live in `llm-baseline-full-grid/workers/*/labels/`.
- `llm-baseline-full-grid/pipeline-progress.json` and `pipeline.log`: the resume
  command's stage status and detailed log.

The shared exact-panel cache is in `~/vgc-data/matchup_regmb.sqlite`. Writes use
SQLite transactions. JSON checkpoints are flushed to disk and atomically renamed;
the parent directory is synced. If a crash occurs after a raw response is saved
but before validation finishes, restart validates that saved response locally.

A `.pending` file without its matching response means a request may have reached
OpenRouter but its response was not saved. The runner stops for reconciliation
rather than automatically paying again. There is an unavoidable window between
the provider processing a request and the local machine receiving it.

An interrupted battle panel may need its unfinished 50-battle panel rerun; finished
panels are reused. Resume does not promise recovery at an individual battle turn.
The resume command restores missing temporary runtime aliases and the verified BC
checkpoint from `~/.local/share/vgc-pilot-runtime` if the operating system clears
`/tmp`. The interpreter, checkpoint backup, and results are stored outside `/tmp`.

These result directories are gitignored local experiment storage, not an off-device
backup. Keep the worktree and database. A restart is covered; deletion of the
worktree or loss of the drive is not.
