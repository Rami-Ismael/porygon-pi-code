# Held-out evaluation set: placement ranks 51–100 (2026-10-12)

Fifty teams that are never trained on, searched against, or used to pick a candidate. The one permitted
use is to score the final selected counter once, after selection.

Code: `src/heldout.py`. Check: `scripts/test_heldout_separation.py` (18 tests, no battles).
Set file: `results/heldout_next50_evs.json`, same record format as `results/top50_evs.json`.

    /tmp/vgc-pilot/.venv/bin/python src/heldout.py verify
    /tmp/vgc-pilot/.venv/bin/python scripts/test_heldout_separation.py

## The set
Same source, same order, same format as the top-50: the VGCPastes Reg M-B sheet as scraped by VGC-Bench,
EVs = Yes, ordered by placement ordinal ascending and then MB id descending (the order `top50_evs.json`
follows). The top-50 followed by this file is one strictly ordered list of 100 (0 violations), so these
are ranks 51–100.

| Fact | Value |
|---|---|
| Teams | 50, all distinct on 48 fields, all on disk, ordinals 4–28, MB50 … MB754 |
| Overlap with the top-50 | 0 ids, 0 teams with the same 48 fields |
| Ids pinned in `heldout.PINNED` | a silent edit of the file raises instead of moving the boundary |
| Top-50 | untouched: 50 entries, 49 distinct teams (MB493 = MB494), as before |

`heldout_worlds50_evs.json` (Worlds 2026, posted after the snapshot) is a different set and is not touched.

## What "identical" means: the same 48 fields
A team is its six slots' species, ability, item, four moves and alignment. Stat Points, slot order, move
order, nicknames and the spelling of a neutral alignment are not part of it. One rule serves every filter
(`heldout.grid_key_*`, `row_key`): files, pastes, parsed teams, encoded rows and matrix teams.

Pool teams that are a held-out team under this rule: MB575 (of MB574), MB524, MB540, MB545 (of MB522),
MB185, MB80 (of MB355). Training corpus: 692 teams, 56 dropped, 636 kept; all 50 top-50 files stay. A
candidate that matches a held-out team on 48 fields is not a candidate, whatever its spread.

## Where it is enforced
| Place | What happens |
|---|---|
| `corpus.load_corpus()`, `activesearch.load_corpus_files()` | the pool without the set (default); `include_heldout=True` for the whole pool |
| `encode.Vocab`, `encode.Legality` | read the excluded teams back in: **token ids and rule tables do not move** |
| `asked_vs_got.setup / gather / train`, `lossdown.load_data`, `lossdata.py`, `gradguide.all_labels` | rows, spread donors and labels lose the set |
| `cem_v3.py` proposals | a proposal that is a held-out team is not a candidate; ridge cache renamed `…_noheldout` |
| `dsame_experiment.py` | 15 of 328 start teams, mutation children and the baseline arm's battle pick skip the set |
| `matchup_db.score / labels` | held-out columns dropped from any set; held-out rows never labelled; asking for the set or a held-out team raises |
| `matchup_db.ensure` | a held-out row or column raises `HeldoutLeak`, 0 battles run |
| `matchup_db.score_pastes` | a held-out paste comes back `{"heldout": True, "score": None}` and is never battled |
| `pool.score` | a held-out team as candidate or opponent raises before any server is touched |
| `team.grid_hash` (new column) | filled on connect for databases written before it existed |

Why the vocabulary is not filtered: removing the 56 teams shrinks it (species 190 → 187, moves 317 → 311),
which would move every token id and silently corrupt every saved checkpoint. A test pins this.

## The permitted use
    matchup_db.evaluate_final(con, policy_id, [team_id, ...])   # matrix route; builds set 'heldout51_100'
    heldout.evaluate([team_file, ...], battles=192)             # battle-pool route (cem_v3.final calls this)
    python src/matchup_db.py evaluate-final --teams 123,456

Both refuse a held-out team as the candidate. Nothing they write is readable by `score` / `labels`
(tested: fitness before and after an evaluation is identical).

## What was deleted from the live matrices (2026-10-12)
`python src/matchup_db.py purge-heldout` (`--dry-run` first). Raw backups, integrity-checked, are in
`~/vgc-data/backups/*.before-heldout-purge-20261012T100133.sqlite`.

| Database | Teams matching | Cells deleted | Battles | Contributions | Teams / top-50 set |
|---|---|---|---|---|---|
| `matchup_regmb.sqlite` | 100 (55 `vgcpastes`, 45 `randomfill:*`) | 2,597 | 2,940 | 2,597 | 43,645 kept / 49 |
| `mask_lab_matrix.sqlite` | 100 | 2,597 | 2,940 | 2,597 | 16,745 kept / 49 |

The 45 `randomfill:*` teams are held-out teams with random Stat Points: invisible under a full-paste
identity, caught by the 48-field rule. Team rows and pastes stay (they are how candidates are recognised
and the columns `evaluate_final` uses); only entries were deleted.

`plan_cell` / `plan_contribution` (8,100 / 24,300 rows in each database) held the MB522-vs-MB763 plan matrix
and were deleted the same way (`purge_heldout` covers them). The published analysis can only be re-run by
restoring them from the backup above.

## Refit clean
| Artifact | State |
|---|---|
| `results/foundation_ridge_noheldout.npz` | refit: 44,887 → 44,879 labelled teams; the old file is kept, unused (`FD.RIDGE` points at the new one) |
| `results/temperature_p0_noheldout.pt` | trained on the 636-team corpus (600 epochs); the old file is kept |
| `results/cem_v3_ridge_base_noheldout.npz` | built on first `cem_v3.py run` from the filtered `lossdata` |
| `F1_final_medium_noheldout`, `G2_family_matrix_noprtrain_noheldout` and their `…_regmb_v3` members | retrained 2026-10-14 with the original configs and partitions (F1 22 min, G2 33 min, regmb 3 min); `foundation.V3_MEMBERS` now points at them, `FOUNDATION_MEMBERS=legacy` restores the old ones |

On the shared 500-team test the clean parents match the originals (ELBO 1.783 vs 1.786 for F1, 1.787 vs
1.787 for G2); the regmb_v3 members differ by 0.0003–0.005 in ELBO. They trained on 50 and 49 fewer rows.
The originals stay on disk, unchanged.

| `results/asked_vs_got_noheldout.pt`, `asked_vs_got_data_noheldout.json` | retrained from the clean p0 on the filtered labels; `foundation`'s "old" arm loads it |
| `dsame_experiment`, `mdts_experiment`, `redundancy_loop`, `reverse_score_diversity`, `cem_score_direction`, `asked_vs_got` | now start from `temperature_p0_noheldout.pt` |
| `lossdown_battle`, `diffusion_baseline*`, `upgrade_meta_v3`, `resume_full_grid`, `retrain_regmb` | member names point at the `_noheldout` models |

## Contaminated checkpoints cannot be loaded by accident
Everything trained or fitted with the set in its data was **moved, not deleted**, on 2026-10-14:

| Moved to | What | Size |
|---|---|---|
| `lossdown/_contaminated_heldout/` | the original F1/G2 and their v2/v3 members, every ablation run A1…S7, per-generation loop checkpoints, the `dibo_*` rounds | 475 `.pt`, 13.6 GB |
| `results/_contaminated_heldout/` | `temperature_p0.pt`, `asked_vs_got.pt`, `hpsdiffusion`, `legaldiffusion*`, `selftrain_gen*`, `slotlegal`, `wrdiffusion`, `foundation_ridge.npz`, `cem_v3_ridge_base.npz`, 51 per-iteration `model.pt`, the 2026-09-30 matrix snapshot | 63 files + 1 `.sqlite` |
| `~/vgc-data/backups/_contains_heldout/` | the two 2026-10-01 smoothness snapshots, the pre-purge copies of both live matrices, and the pre-purge copy of `vgcpastes_fast_labels_20261013.sqlite` | 3.0 GB |

`lossdown.load_net` raises `HeldoutLeak` for a name in that folder or any checkpoint without `noheldout`
in its name. Re-scoring a historical benchmark means moving a file back deliberately **and** setting
`ALLOW_CONTAMINATED_CHECKPOINTS=1` (only `1`: `0`, `false` or any other value keeps the guard closed). The pre-purge backups are the restore point for the deleted matrix
entries and the plan matrix; they hold the leak by design, so they live in the quarantine folder.

`regmb v2` was retrained clean too (2026-10-14). Its `training-regulation.json` is the manifest itself and
rebuilds the original v2 vocabulary exactly; `retrain_regmb.py` accepts the old v2 audit (no version or
regulation hash) only for `--version v2`.

Also purged: `vgc-data/vgcpastes_fast_labels_20261013.sqlite`, created after the first purge. It carried the
plan matrix (8,100 + 24,300 rows, deleted) and no cell touched the set; its `llm_panel_outcome` rows
reference no held-out team.

## Results already produced
They cannot be un-produced. `results/heldout_provenance.json` lists the 116 result files written before the
separation (2026-10-12 10:00): re-run them with the `_noheldout` models before citing one as out-of-sample.
The filters protect every run from now on.

## Decided by the owner (2026-10-14), measured
- **Near-copies are not excluded.** Identical is the same 48 fields, as defined. Measured by shared slots with
  the nearest held-out team: 12 of the 636 training teams share 5 of 6 slots (23 share 4, 45 share 3), and 103
  of 47,762 legal matrix teams share 5 (233 share 4, 414 share 3). Excluding them is a one-line change in
  `heldout.training_pairs` / `row_mask` if it is ever wanted.
- **`cem_v3` was re-run clean (2026-10-14)** with the `_noheldout` members and scored once on ranks 51-100: 16 finalists, fresh 0.504 on the top-50 and 0.509 on the held-out teams (paired +0.005, se 0.013), best 0.599 / 0.609. The earlier run's 0.581 came from models that had seen the set. Details in `docs/cem-v3-loop.md`. The other finished results are annotated, not re-run; three that held a held-out team were repaired and a re-scan finds none.

## What this does not undo
- **Second-order copies.** A team a contaminated model generated is a normal training row if it is not the
  same 48 fields as a held-out team.
- **The held-out teams' pastes and team rows** stay in the pool and the matrices (as evaluation columns and
  so candidates can be recognised); only entries, labels, rows and models were removed.
- `dsame_experiment.py validate` tests the surrogate on real anchors and edits from `ruggedness2.py`;
  that is validation, not search.
- The sheet itself was not re-fetched (no network in this environment): the set is the existing file,
  verified for count, order, format and separation, not re-derived from the live sheet.
