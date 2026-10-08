# Reuse the smoothness results from the matchup matrix

The source of truth is `/Users/ramiismael/vgc-data/matchup_regmb.sqlite`.
Building another visual, changing its design, or inspecting a stored difference does **not** require running the battle pipeline again.

On 2026-10-01 the database received:

- The complete original eight-team pilot: 96 panels and 4,800 battles.
- The completed member-coverage extension: 102 new panels and 5,100 new battles. It reuses the original baseline panels. All 48 displayed members have at least one measured edit; there are 50 distinct move-edit comparisons in total.
- A partial snapshot of the separate full 50+50 study, retaining every completed panel available at import time. Its active simulator processes were left running. Re-import it to archive later panels.
- Frozen team and opponent text, source identities, protocol manifests and hashes, seeds, panel counts, pointwise intervals, corrected estimates, group summaries, benchmark inputs and outcomes, and the visual's move names and available sprite assets.

The original two-edit-per-team group comparison is unchanged. Added member coverage is exploratory and marked separately in the visual.

## Read and serve without simulation

From the pilot project directory:

```sh
python3 src/smoothness_db.py list
python3 src/smoothness_db.py export \
  --run 47ca1e96fe27f322c8ad69d53095b44e0c5a45e3e5ca66021af7c152e1f047dd \
  --output /tmp/smoothness-results.json
```

Export uses only the database and Python's standard library. It does not read the original result directories, load a battle model, or start a simulator. `--javascript` writes the same records as `window.SMOOTHNESS_DATA` for a portable offline visual.

The local preview is served with:

```sh
python3 src/smoothness_server.py \
  --visual-root '/Users/ramiismael/Documents/yakumsi-vault/Personal Project/Using Advance method search in the whole search of possible vgc pokemon format to find the right counter to a pokemon team/Blog/visuals' \
  --run 47ca1e96fe27f322c8ad69d53095b44e0c5a45e3e5ca66021af7c152e1f047dd \
  --port 8769
```

The server binds only to `127.0.0.1`. It is read-only. `/smoothness/data.js` and `/smoothness/api/data` query SQLite on every page load; `/smoothness/api/assets/<filename>.png` serves archived image bytes. No HTTP endpoint starts battles or writes database records.

In Python:

```python
import sys
sys.path.insert(0, 'src')
import smoothness_db

with smoothness_db.open_db(readonly=True) as db:
    result = smoothness_db.snapshot(db)
    first_team = result['teams'][0]
    for edit in first_team['edges']:
        print(edit['candidate'], edit['delta'], edit['ci95'])
```

## Add later completed work without replaying it

Held items are exported from each original candidate's exact battled paste, retaining slot order. `heldItem` is the battle-start item name (or null); `itemSprite` references its archived PNG (or null if artwork is unavailable). `reference/item-sprites.json` holds source URLs and hashes; `visual/assets/item-*.png` holds image bytes. No item data depends on replaying a battle. To restore the visual's offline item files from the database, run:

```sh
python3 scripts/archive_smoothness_items.py --run 47ca1e96fe27f322c8ad69d53095b44e0c5a45e3e5ca66021af7c152e1f047dd --visual /path/to/visuals/smoothness
```

Only missing artwork is fetched from the public Pokémon Showdown sprite repository. Existing archived icons are reused without network requests. Consumable item use during a battle is not represented by these battle-start fields.

```sh
/tmp/vgc-pilot/.venv/bin/python src/smoothness_db.py import-run results/smoothness_top50_50x50
```

The importer validates the manifest and complete panel records, then imports only unseen records. Running it again adds no duplicate battles. Conflicting saved counts are rejected transactionally. Interrupted studies can be imported incrementally; partial or prepared runs remain explicitly labeled. Importing a subset later does not downgrade already saved completion status.

To fill the visual's member coverage again, the existing command resumes only missing panels; with the completed output directory, it launches no simulator:

```sh
/tmp/vgc-pilot/.venv/bin/python scripts/smoothness_member_coverage.py \
  --parent results/smoothness_pilot --output results/smoothness_member_coverage
```

This extension is bounded to the eight-team visual and at most 6,000 extra battles. Its frozen runtime must match the original baseline runtime. Do not create another output directory merely to rebuild a visual; use the database export instead.

## Tables and meaning

| Table | Reusable records |
|---|---|
| `smoothness_run` | Frozen protocol, manifest hash, analysis, benchmark data, completion state and optional parent experiment |
| `smoothness_anchor` | Original-team identity, source and cohort |
| `smoothness_candidate` | Exact battled paste, move edit and logical link to the existing matrix team |
| `smoothness_opponent` | Ordered frozen pool entries, including duplicate weights and exact pastes |
| `smoothness_panel` | Immutable candidate/replicate counts, seed and full result record |
| `smoothness_edge` | Edit effect, uncertainty and the exact baseline experiment/candidate it reuses |
| `smoothness_benchmark` | Reference points, edited inputs, outcomes, objective bounds and summaries |
| `smoothness_artifact` | Versioned source artifacts and archived images, with SHA-256 hashes |
| `smoothness_visual` | Versioned presentation metadata; numeric results are rebuilt from the saved experiment records |
| View `smoothness_coverage` | Queryable member-edit coverage and completed panel counts |

For example:

```sql
SELECT anchor_id, pokemon_slot, candidate_id, completed_panels, battles,
       100 * delta AS change_pp, 100 * ci_low AS low_pp, 100 * ci_high AS high_pp
FROM smoothness_coverage
WHERE run_id = '68d4ec7023530152c651bd886449d24528246e18a85b0a886184edef0bb396d6';
```

These are pool-level experiments, not per-opponent battle records. The importer therefore leaves `cell`, `contribution` and their scoring semantics untouched. In particular, the smoothness pool includes self matchups, whereas the existing matrix objective skips them. There is no invented per-opponent attribution and no mixing of the two objectives.

Matrix canonicalization sorts team and move order. The extension links to that logical identity for joins **and also preserves the exact original battled text**, its order and hash, so slot-sensitive evaluation can be distinguished. Legality of a newly canonicalized matrix identity is left unchecked until the existing validator handles it; an original validated paste does not silently certify a differently ordered paste.

## Verification and backup

Eight storage tests check idempotency, conflict rollback, exact pastes, duplicate opponents, database-only export, a valid backup, held items in battled slot order, and complete coverage with unchanged primary summaries. The browser checks every member and all 50 edit comparisons, saved counts, intervals, keyboard input, reduced motion, missing-data protection, and responsive layout. The held-item update was also inspected across all eight teams and at 320 pixels wide: all 48 icons loaded and the page had no horizontal overflow.

The database passed `PRAGMA quick_check` and `PRAGMA foreign_key_check` after import. A consistent SQLite backup was taken before the first import:

`/Users/ramiismael/vgc-data/backups/matchup_regmb.before-smoothness-20261001T180607535146Z.sqlite`

Use `import-run --backup` for another pre-import snapshot. Import is additive; source experiment files and existing matchup records are retained.
