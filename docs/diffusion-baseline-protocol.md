# Reproduce the strict diffusion comparison

**Selected baseline: ask (target win rate) 0.5, classifier-free guidance 2.**
The user selected these settings on October 8. Temperature remains 1, with the
F1 + G2 full-regulation v2 mixture. The existing v2 results already used these
settings; this selection does not require a rerun. The explicit profile is
`code/data/diffusion-baseline-profile.json`; the dashboard checks recorded settings
against it. This is the selected configuration, not a new claim that the small
masked-completion pilot proves it optimal.

## Full-regulation update

### Dense mask extension

The full integer grid retains the original 63 trials at their original manifest
indices and adds 243 trials, giving 306 tasks over the same three starting teams.
Mixed masks extend through all 48 eligible fields. Diffusion's index-derived
seeds and the original prompts remain unchanged for reused trials.

```sh
/tmp/vgc-pilot/.venv/bin/python code/scripts/dense_mask_grid.py prepare
# Supply OPENROUTER_API_KEY through the process environment; never commit it.
/tmp/vgc-pilot/.venv/bin/python code/scripts/llm_baseline.py complete --output code/results/llm-baseline-full-grid
/tmp/vgc-pilot/.venv/bin/python code/scripts/diffusion_baseline.py generate --baseline code/results/llm-baseline-full-grid --output code/results/diffusion-baseline-full-grid --members F1_final_medium_regmb_v2 G2_family_matrix_noprtrain_regmb_v2
/tmp/vgc-pilot/.venv/bin/python code/scripts/dense_grid_battles.py --workers 6
/tmp/vgc-pilot/.venv/bin/python code/scripts/dense_mask_grid.py report
python3 code/dashboard/build.py
```

Independent battle panels run in isolated processes. Each panel still uses one
concurrent battle, the original BC policy and exact schedule; its cache key is
unchanged. Identical teams are assigned to the same worker. Scoring can overlap
completion generation, but rerun scoring after generation ends to include late
completions. Finished panels reuse the exact-panel cache. The final report refuses
incomplete scoring, and the dashboard selects the dense run only after its summary
exists. Earlier pilot directories remain available.

The latest trained pair uses the simulator-derived M-B vocabulary. Its training
report is `regmb-vocabulary-retraining.md`. Run its comparison in a separate
directory so the earlier seven-supported-task experiment stays intact:

```sh
/tmp/vgc-pilot/.venv/bin/python code/scripts/diffusion_baseline.py generate --output code/results/diffusion-baseline-f1g2-regmb-v2 --members F1_final_medium_regmb_v2 G2_family_matrix_noprtrain_regmb_v2
/tmp/vgc-pilot/.venv/bin/python code/scripts/diffusion_baseline.py battle --output code/results/diffusion-baseline-f1g2-regmb-v2
/tmp/vgc-pilot/.venv/bin/python code/scripts/diffusion_baseline.py report --output code/results/diffusion-baseline-f1g2-regmb-v2
python3 code/dashboard/build.py
```

The checkpoint supplies the exact vocabulary and legality snapshot to inference.
New species without a corpus Stat Point spread use a random legal spread, explicitly
recorded in the generation configuration. This remains a hybrid completion system.

## Original corpus-vocabulary run

The report is `diffusion-baseline-comparison.md`. The comparator uses the existing F1/G2 checkpoints; it does not train another model or issue any LLM calls.

```sh
/tmp/vgc-pilot/.venv/bin/python code/scripts/diffusion_baseline.py generate
/tmp/vgc-pilot/.venv/bin/python code/scripts/diffusion_baseline.py battle
/tmp/vgc-pilot/.venv/bin/python code/scripts/diffusion_baseline.py report
python3 code/dashboard/build.py
```

Defaults read `code/results/llm-baseline-instant-pilot` and write `code/results/diffusion-baseline-f1g2-strict`. Both directories are gitignored. Generation uses one attempt per task with independent deterministic seeds; unsupported fixed context is recorded before inference. The configuration saves checkpoint, vocabulary, corpus and source hashes. It rejects a changed generation configuration on resume. The exact original manifest is copied and checked, and the dashboard build compares both complete battle configurations before embedding results.

All known categorical fields are supplied. Stat Point values are outside this model's representation and are not conditioning inputs; masked spreads are sampled from the corpus. Whole-slot masks include spreads. No hidden value is used to determine move rank. Alphabetical move ordering is disabled for this completion adapter so a fixed move does not falsely constrain the rank of a missing move. Clause and compatibility masks remain active, with distinct moves enforced explicitly and final authoritative Showdown validation. This decoding adaptation differs from the earlier full-team generation experiment.

The model has no interface for the explicit opponent pastes Ling receives. Its scalar win-rate conditioning is trained on historical project data, which may include this meta pool. These are in-pool system results and cannot establish held-out generalization.

Seven supported tasks generated seven legal teams and completed 350 attributed battles. One has a legal Ling counterpart. Unsupported inputs remain separate from invalid outputs and are included in the 63-task completion denominator. Do not compare the unconditional means of the two different supported subsets. A fairer shared-domain study needs a newly frozen cohort and new Ling calls; this run deliberately did not substitute starting teams after inspecting results.

Validation: nine baseline tests pass, including hidden-value independence, fixed OOV rejection and original move-path mapping. All seven output masks were checked, all battle panels have 50 outcomes, and battle configurations match exactly. The existing battle worker emits pending inference-loop cleanup warnings between panels; all seven panels nevertheless passed attribution and count checks. Showdown random draws are not fixed, so repeated fresh battles can differ.
