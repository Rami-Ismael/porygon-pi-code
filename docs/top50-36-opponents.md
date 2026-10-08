# Results scored against 36 opponents, not the top-50 (found 2026-09-30)

14 of the 50 teams in `results/top50_evs.json` live in `teams/reg_mb/featured/` (MB759 MB686 MB684
MB604 MB582 MB494 MB691 MB683 MB581 MB580 MB682 MB583 MB728 MB681). Five scripts built the
opponent list as `f"{root}/{id}.txt"` and kept only paths that exist, so they battled the other 36.
Nothing below was re-run; the recorded numbers stand as measured against 36.
Future runs resolve all 50 through `src/top50.py`, which raises if any one is missing.

The top-50 holds **49 distinct teams**: MB493 and MB494 are the same paste byte for byte. Both stay
in the list, so that team has weight 2 (as in the matchup database's `top50` set). MB494 is in
`featured/`, so the 36 kept MB493 at weight 1.

## Produced against 36
| Result | Write-up | What used the 36 |
|---|---|---|
| `results/hps_eval.json` | README, "Status (2026-08-29)" | every arm's win rate |
| `results/al_experiment.json` | README, "Status (2026-08-29, later)" | held-out labels and both arms' labels |
| `results/hps_surrogate.json`, key `transfer_ladder_to_hps` only | README, same section | trained on `al_experiment.json` random-arm labels |
| `results/ruggedness.json`, `results/ruggedness_smoke.json` (both record `"opponents": 36`) | `docs/ruggedness.md` | every VGC anchor, edit and replicate |
| `results/ruggedness2.json`, `results/ruggedness2_parts/vgc_battles.json` | `docs/ruggedness2.md` | new anchors, edits and replicates |
| `results/interaction_order.json`, `results/interaction_order_parts/` | `docs/interaction-order.md` | every cube battle; the 16 primary anchors were also drawn from the 36, not the 50 |
| `results/bench_features.json`, VGC column | `docs/bench_features.md` | reuses `ruggedness2_parts` |

## Not affected
Scripts that already resolve `featured/`: `counter_matrix.py`, `matchup_db.py`, `winrate*.py`,
`eval_*.py`, `loop.py`, `scale.py`, `activesearch.py`, `alignment.py`, `ridgeguide.py`,
`label_strata.py`, `matchup.py`. Not checked: `results/hps_labels.json`; the script that wrote it is
not in the repo.
