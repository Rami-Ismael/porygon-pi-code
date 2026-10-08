# Sparse matchup matrix — design (2026-09-30)

Code: `src/matchup_db.py` (database, cache, scoring, labels, cost), `src/pair_shard.py` (battle runner
with per-opponent results). Tests: `scripts/test_matchup_db.py`. Database: `~/vgc-data/matchup_regmb.sqlite`
(override with `MATCHUP_DB`).

## What it stores
One SQLite database per format (`gen9championsvgc2026regmb`). Rows and columns are legal teams. A
**cell** holds the record of one pair of teams under one fixed battle configuration: wins for each side,
ties, battles. A pair never battled has **no row** — unknown is never stored as 0.

| Table | Holds |
|---|---|
| `team` | one row per distinct team: canonical paste (the exact text battled), sha256, species set, Showdown validator result, origin |
| `team_source` | every file/record that maps to a team (VGCPastes id, path, event, placement, date); duplicates kept here |
| `team_set` | named column sets with weights; `top50` = the 50 top-50 entries (49 distinct teams; MB493 = MB494, weight 2) |
| `policy` | battle configuration: row policy, column policy (checkpoint hash), simulator revision. Cells never mix configurations |
| `cell` | unordered pair (lo < hi), `lo_wins, hi_wins, ties, battles`; `WITHOUT ROWID`, primary key (policy, lo, hi) |
| `batch`, `contribution` | provenance: every battle run, its seed and duration, and what it added to each cell |
| view `matchup` | directed read: `row_team, col_team, wins, losses, ties, battles, win_rate` (both directions) |

Win rate = wins / battles; a tie counts as not a win, the same definition as the objective so far.

## Team identity
Two pastes are the same team when every choice a player makes is the same: species, ability, item,
moves, Stat Points, alignment. Slot order, move order, nicknames, gender, level and the spelling of a
neutral alignment are dropped. The canonical paste (slots sorted, moves sorted) is stored and is the file
that is battled, so a cached result is always a result for exactly that text.

## Cache semantics (purpose a)
`ensure(rows, cols, per_cell)` plans the deficit of every (row, col) cell and battles only that. A pair
requested in both directions is planned once; a team is never battled against itself. A second identical
request runs 0 battles; raising `per_cell` runs only the top-up. `score_pastes(pastes)` is the drop-in
objective for search loops: add, validate, top up, return the score.

**Score** = the row averaged over the set's columns, with set weights (equal except MB493/MB494, which
counts twice), skipping the team's own column; reported with coverage and standard error.

## Labels (purpose b)
`labels` exports every team with full coverage against a set as JSON lines (score, coverage, battles,
standard error). Cell-level labels are the `matchup` view.

## How a battle is attributed to a cell
A job is one row team against an ordered schedule of column teams (one player pair per job, which keeps
throughput at ~51 battles/s instead of paying setup per cell). Each finished battle is assigned to the
column whose six species it showed at team preview. Columns sharing a species set (51 groups, 157 teams in
the VGCPastes pool; none in the top-50) are put in separate jobs. Battles that cannot be matched are counted
as `unattributed` and never written to a cell.

## Verification (2026-09-30)
- 9 unit tests pass: canonical identity, duplicate mapping, symmetric storage, unknown = no row, no re-battle
  with deficit-only top-up, pair planned once, weighted row average, species-collision split, and an open
  connection never blocking another writer (a bug caught while snapshotting: connect() left a write open).
- Seed: 695 VGCPastes files → 681 distinct teams, all 681 legal under Showdown's validator.
- Live: 16 rows × 49 top-50 columns × 8 battles = 6,272 battles, **0 unattributed**, every cell exactly 8;
  repeating the request ran 0 battles; a reordered paste of a scored team returned from cache in 0.1 s.

## Cost of coverage (measured: 48.5 bytes per cell + 17.2 per contribution; 50.9 battles/s on 7 servers)

| Scope | Cells | Storage | Hours at 8 / 24 / 96 battles per cell |
|---|---|---|---|
| One new team vs the top-50 | 50 | 3 kB | 0.002 / 0.007 / 0.03 |
| Seed pool vs the top-50 | 32,775 | 2.2 MB | 1.4 / 4.3 / 17 |
| Every pair in the seed pool (681 teams) | 231,540 | 15 MB | 10 / 30 / 121 |
| 100,000 search teams vs the top-50 | 5,000,000 | 328 MB | 218 / 655 / 2,620 |
| Every legal team (≈10^111.6) | ≈10^222.9 | — | ≈10^223.8 battles at 8: impossible |

Storage is never the constraint; battles are. Full coverage is reachable for the seed pool only.

## Findings while building it
1. **Five scripts scored against 36 opponents, not the top-50.** 14 top-50 teams live in
   `teams/reg_mb/featured/`; `interaction_order.py`, `ruggedness.py`, `ruggedness2.py`, `hps_eval.py`
   and `al_experiment.py` look only in the top folder and silently drop them (corrected 2026-09-30:
   `al_experiment.py` filters too; it never passed missing paths). The database resolves `featured/`;
   the scripts now use `src/top50.py`. Affected results: `docs/top50-36-opponents.md`.
2. **`corpus.parse_team_text` merges an item-less Pokémon into the slot before it** (MB77, MB502, MB661
   became five-slot teams), so `load_corpus` silently drops those teams. `matchup_db` parses block by block.
3. **The top-50 has 49 distinct teams:** MB493 and MB494 are the same team (2nd and 1st, Ranked Season M-3).

## Limits
- Results hold for the canonical slot order; a policy that is sensitive to slot order would score the
  original file slightly differently.
- Keep the database outside iCloud-synced folders (SQLite write-ahead log + sync = corruption risk);
  `python src/matchup_db.py backup --out <path>` writes a consistent snapshot.
