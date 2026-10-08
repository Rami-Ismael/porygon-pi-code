"""Run: /tmp/vgc-pilot/.venv/bin/python scripts/test_matchup_db.py  (no battles; fake runner)."""
import random, sqlite3, sys, tempfile, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import matchup_db as M

POOL = sorted(M.POOL_DIR.glob("MB*.txt"))[:6]

def reorder(paste):
    blocks = paste.strip().split("\n\n")[::-1]
    out = []
    for b in blocks:
        lines = b.splitlines()
        moves = [l for l in lines if l.startswith("- ")][::-1]
        rest = [l for l in lines if not l.startswith("- ")]
        head = rest[0]
        sp, _, item = head.partition(" @ ")
        rest[0] = f"Nick ({sp.strip()}) @ {item}" if item else f"Nick ({sp.strip()})"
        out.append("\n".join(rest + moves))
    return "\n\n".join(out) + "\n"

def fake_runner(con, jobs, seed=None):
    rng = random.Random(seed)
    res = [(j["row"], c, *(lambda w: (w, 1 - w, 0))(int(rng.random() < .5))) for j in jobs for c in j["schedule"]]
    return res, 0, 0.0

class MatchupDBTest(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        M.PASTES = Path(self.d.name) / "pastes"
        self.con = M.connect(Path(self.d.name) / "t.sqlite")
        self.pid = M.get_policy(self.con, "p", "p", "sim")
        self.ids = [M.add_team(self.con, f.read_text(), "test")[0] for f in POOL]
        with self.con:
            for t in self.ids[:4]: self.con.execute("INSERT INTO team_set VALUES ('top50', ?, 1)", (t,))
    def tearDown(self):
        self.con.close(); self.d.cleanup()

    def test_canonical_form_ignores_order_nicknames_and_neutral_spelling(self):
        p = POOL[0].read_text()
        self.assertEqual(M.canonical_paste(p), M.canonical_paste(reorder(p)))
        first_move = p.split("\n- ", 1)[1].splitlines()[0]
        changed = p.replace(f"- {first_move}", "- Helping Hand", 1)
        self.assertNotEqual(changed, p)
        self.assertNotEqual(M.canon_hash(M.canonical_paste(p)), M.canon_hash(M.canonical_paste(changed)))

    def test_duplicate_paste_maps_to_the_same_team(self):
        tid, created = M.add_team(self.con, reorder(POOL[0].read_text()), "test")
        self.assertEqual((tid, created), (self.ids[0], False))

    def test_cells_store_one_pair_and_read_both_directions(self):
        a, b = self.ids[0], self.ids[1]
        M.record(self.con, self.pid, [(b, a, 3, 1, 0)])
        rows = dict(((r, c), (w, n)) for r, c, w, n in self.con.execute(
            "SELECT row_team, col_team, wins, battles FROM matchup"))
        self.assertEqual(rows[(b, a)], (3, 4)); self.assertEqual(rows[(a, b)], (1, 4))
        self.assertEqual(self.con.execute("SELECT COUNT(*) FROM cell").fetchone()[0], 1)
        with self.assertRaises(sqlite3.IntegrityError):
            with self.con: self.con.execute("INSERT INTO cell VALUES (?,?,?,0,0,0,0,'t')", (self.pid, b, a))

    def test_untested_pairs_have_no_row(self):
        self.assertEqual(M.battles_done(self.con, self.pid, self.ids[0], self.ids[5]), 0)
        self.assertIsNone(self.con.execute("SELECT 1 FROM cell").fetchone())

    def test_cache_never_battles_a_pair_twice_and_tops_up_only_the_deficit(self):
        rows, cols = [self.ids[4], self.ids[5]], self.ids[:4]
        n1 = M.ensure(self.con, self.pid, rows, cols, 8, seed=1, runner=fake_runner)
        self.assertEqual(n1, 2 * 4 * 8)
        self.assertEqual(M.ensure(self.con, self.pid, rows, cols, 8, seed=1, runner=fake_runner), 0)
        self.assertEqual(M.ensure(self.con, self.pid, rows, cols, 10, seed=2, runner=fake_runner), 2 * 4 * 2)
        self.assertEqual(M.battles_done(self.con, self.pid, rows[0], cols[0]), 10)

    def test_overlapping_rows_and_columns_plan_each_pair_once_and_skip_self(self):
        need = M.plan(self.con, self.pid, self.ids[:4], self.ids[:4], 8)
        self.assertEqual(len(need), 6)
        self.assertTrue(all(r != c for r, c, _ in need))

    def test_score_averages_the_row_over_set_columns_and_skips_itself(self):
        t = self.ids[0]
        M.record(self.con, self.pid, [(t, self.ids[1], 6, 2, 0), (t, self.ids[2], 2, 6, 0)])
        s = M.score(self.con, self.pid, t, "top50")
        self.assertAlmostEqual(s["score"], 0.5); self.assertEqual(s["coverage"], 2); self.assertEqual(s["columns"], 3)

    def test_an_open_connection_does_not_block_another_writer(self):
        other = sqlite3.connect(Path(self.d.name) / "t.sqlite", timeout=1)
        with other: other.execute("INSERT INTO meta VALUES ('probe', 'x')")
        other.close()

    def test_jobs_split_columns_that_share_a_species_set(self):
        clone = self.con.execute("SELECT species_key, base_key FROM team WHERE team_id=?", (self.ids[1],)).fetchone()
        self.con.execute("UPDATE team SET species_key=?, base_key=? WHERE team_id=?", (*clone, self.ids[2]))
        jobs = M.jobs_for(self.con, [(self.ids[0], self.ids[1], 2), (self.ids[0], self.ids[2], 2), (self.ids[0], self.ids[3], 2)])
        self.assertEqual(len(jobs), 2)
        for j in jobs: self.assertFalse({self.ids[1], self.ids[2]} <= set(j["schedule"]))

if __name__ == "__main__":
    unittest.main(verbosity=2)
