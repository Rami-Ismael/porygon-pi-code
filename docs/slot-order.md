# Slot order: the battle policy picks by list position

Found 2026-10-15, measured and fixed 2026-10-16. Win rate below is the share of battles won against the top-50 meta teams, behaviour-cloning policy (`/tmp/bc_100.zip`) on both sides.

## The bug

At team preview, `vgc_bench` `policy_player.teampreview` returns slot numbers (`/team 2346`), not Pokémon. Nothing makes the policy treat a team as the same team in any order, so the order a paste is written in changes which four it brings.

Team 47355 (Sylveon, Chandelure, Garchomp, Basculegion, Kingambit, Floette-Eternal), 1,176 battles per listing:

| Listing | Sylveon brought | Basculegion brought | Win rate |
|---|---|---|---|
| Sylveon first | 21% | 95% | 0.651 [0.623, 0.677] |
| Alphabetical (database order) | 78% | 7% | 0.710 [0.684, 0.734] |

## Every slot

12 teams (47355 plus 11 random top-50 teams), each written in 6 cyclic rotations so every Pokémon sits in every slot once; 192 battles per listing, 13,848 battles. 95% intervals bootstrap over teams.

| Slot | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| Brought (chance 0.67) | 0.43 | 0.80 | 0.81 | 0.76 | 0.51 | 0.69 |
| Leads (chance 0.33) | 0.17 | 0.42 | 0.60 | 0.41 | 0.14 | 0.26 |

A Pokémon's bring rate moves by a median 0.57 across slots; slot 1 is its lowest for 42 of 72. Win rate moves less: order standard deviation about 0.018 against listing noise 0.035, pooled shuffle test p = 0.057. Alphabetical order scored +0.014 above each team's six-order mean (range −0.027 to +0.054).

## The fix

`src/slot_order.py`: `ShuffledTeamBuilder` reorders the six Pokémon on every `yield_team()`, with its own `random.Random` so opponent draws (common random numbers) are unchanged. `shard.py` and `pair_shard.py` use it for both teams. The policy's slot preference is not removed; it is averaged, so every Pokémon gets the same mix of slots and the score no longer depends on how the file is ordered.

- Default `VGC_SLOT_ORDER=shuffled`. `VGC_SLOT_ORDER=listed` restores the written order to reproduce earlier results.
- `matchup_db.get_policy()` tags shuffled battles `+slots-shuffled`, so they get their own policy and never share a cell with listed-order battles. The shuffled policy starts with an empty cache.
- `matchup_db.check_slot_order()` (called by `ensure()`) refuses to add battles to a policy battled in the other order. Experiments pinned to policy 1 need `VGC_SLOT_ORDER=listed`.
- The fast runner (`team_completion_pipeline/fast_battle_runner.py`, in the vault) takes `slot_order='listed'` by default; the working-tree `matchup_db.fast_runner()` passes `SLOT_ORDER`.

Re-check with shuffling on (both teams shuffled every battle):

| Team | Listings | Before | After |
|---|---|---|---|
| 205 | rotation 3 vs 5 | 0.505 vs 0.337 | 0.409 vs 0.401, gap −0.009 [−0.048, +0.031] |
| 47355 | alphabetical vs Sylveon first | 0.695 vs 0.635 | 0.668 vs 0.649 over 2,352 each, gap −0.018 [−0.043, +0.008] |

Bring rates match across listings to within 0.03. Battle records: `results/slot_order_fixcheck_20261016/`; figure numbers: `results/slot_order_20261016.json`. Tests: `python scripts/test_slot_order.py`.

## Not covered

- Still listed order: `winrate.py` (the early scorer behind `winrate2.py` and `winrate3.py`), `scale_test.py` (throughput only) and `scripts/recover_temperature_challenge.py` (a one-off recovery).
- `plan_shard.py` forces team preview by species, so order does not decide the four. The policy may still index the team by slot when it switches during the battle; that was not tested.
- Every result recorded before this change was battled in listed order: the database's alphabetical paste for matchup-matrix cells, the file's order for `pool.score` labels.
