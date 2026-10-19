"""Run: /tmp/vgc-pilot/.venv/bin/python scripts/test_heldout_separation.py   (no battles).

Ranks 51-100 are an evaluation set: never trained on, searched against, or used to pick a candidate.
Identical means the same 48 fields (species, ability, item, moves, alignment; not Stat Points).
This checks the set itself (50 teams, the order that continues the top-50, disjoint from it), the
identity rule, and every place the separation is enforced: training rows, vocabulary stability,
labels, candidates, the matchup matrix (including deleting what is already in it), the battle pool."""
import ast, json, re, sqlite3, sys, tempfile, unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
import heldout as H
import matchup_db as M
import top50 as T50

POOL = [f for f in sorted(M.POOL_DIR.glob("MB*.txt")) if f.stem not in set(H.ids())][:6]
FAKE_BATTLES = []
CHANGED = ("matchup_db", "heldout", "pool", "corpus", "encode", "activesearch", "asked_vs_got", "lossdown",
           "lossdata", "gradguide", "cem_v3")

def src_first():
    """matchup_db puts /tmp/vgc-pilot/src (the laptop's copy) ahead of this tree. Until the two are synced
    the laptop's copy of a changed module has no filter: drop any such module and import this tree's."""
    if sys.path[0] != str(SRC): sys.path.insert(0, str(SRC))
    for name in CHANGED:
        m = sys.modules.get(name)
        if m is not None and Path(getattr(m, "__file__", "") or "").resolve().parent != SRC:
            del sys.modules[name]
    # H may be the laptop's heldout, dropped just above; the modules under test now raise this tree's
    # HeldoutLeak, so assertRaises(H.HeldoutLeak) must name that class.
    global H
    import heldout as H

def fake_runner(con, jobs, seed=None):
    out = []
    for j in jobs:
        for c in j["schedule"]:
            FAKE_BATTLES.append((j["row"], c)); out.append((j["row"], c, 1, 0, 0))
    return out, 0, 0.0

def reorder(paste):
    return "\n\n".join(paste.strip().split("\n\n")[::-1]) + "\n"

def respread(paste):
    """Same 48 fields, different Stat Points."""
    return re.sub(r"^EVs: .*$", "EVs: 1 HP / 1 Atk", paste, flags=re.M)


class TheSet(unittest.TestCase):
    def test_fifty_teams_continuing_the_top50(self):
        facts = H.verify()
        self.assertEqual((facts["n"], facts["distinct"]), (50, 50))
        top = json.load(open(T50.TOP50)); held = H.entries()
        self.assertEqual(len(top), 50)
        self.assertEqual(list(top[0]), list(held[0]))                 # same record format
        self.assertFalse({t["id"] for t in top} & {t["id"] for t in held})
        self.assertEqual(tuple(H.ids()), H.PINNED)

    def test_top50_is_untouched(self):
        self.assertEqual(len(T50.ids()), 50)
        self.assertEqual(len(T50.files()), 50)
        self.assertEqual(T50.ids()[0], "MB763")
        keys = {H.grid_key_text(Path(f).read_text()) for f in T50.files()}
        self.assertEqual(len(keys), 49)                                # MB493 = MB494, as before
        self.assertFalse(keys & H.identities())                        # no top-50 team is a held-out team

    def test_identical_means_the_same_48_fields(self):
        self.assertEqual(H.twins(), {"MB574": ["MB575"], "MB522": ["MB524", "MB540", "MB545"], "MB355": ["MB185", "MB80"]})
        for name in ("MB575", "MB524", "MB540", "MB545", "MB185", "MB80"):
            self.assertTrue(H.is_heldout_file(T50.resolve(name)), name)
        text = Path(H.files()[0]).read_text()
        self.assertTrue(H.is_heldout_text(reorder(text)))              # slot order is not identity
        self.assertTrue(H.is_heldout_text(respread(text)))            # Stat Points are not identity
        self.assertNotEqual(respread(text), text)
        self.assertFalse(H.is_heldout_text(POOL[0].read_text()))
        self.assertEqual(H.exclude_pastes([respread(text), POOL[0].read_text()]), [POOL[0].read_text()])


class Training(unittest.TestCase):
    def test_training_rows_lose_the_set_but_the_vocabulary_does_not(self):
        src_first()
        import corpus as C, activesearch as A, asked_vs_got as X
        whole = A.load_corpus_files(include_heldout=True)
        kept = A.load_corpus_files()
        names = {Path(f).stem for f, _ in kept}
        self.assertEqual(len(whole), 692)
        self.assertEqual(len(whole) - len(kept), 56)                           # 50 + MB575 + 5 same-48-field twins
        self.assertFalse(names & set(H.ids()))
        self.assertFalse(names & {"MB575", "MB524", "MB540", "MB545", "MB185", "MB80"})
        self.assertFalse(set(T50.ids()) - names)                               # the top-50 stays in
        self.assertEqual(len(C.load_corpus()[0]), len(kept))
        self.assertEqual(len(C.load_corpus(include_heldout=True)[0]), len(whole))
        old = A.Vocab([t for _, t in whole])
        self.assertEqual(A.Vocab([t for _, t in kept]).sizes, old.sizes)       # token ids must not move
        self.assertEqual(A.Vocab([t for _, t in kept]).itos, old.itos)
        self.assertEqual(old.sizes["species"], 190)
        corpus, V, Cn, look, spreads = X.setup()
        self.assertEqual(len(corpus), len(kept))
        self.assertEqual(V.itos, old.itos)

    def test_a_checkpoint_trained_with_the_set_fails_closed(self):
        src_first()
        import os, lossdown as L, asked_vs_got as X
        V = X.setup()[1]
        os.environ.pop("ALLOW_CONTAMINATED_CHECKPOINTS", None)
        for name in ("F1_final_medium", "G2_family_matrix_noprtrain", "A1_all_legal", "B2_family"):
            with self.assertRaises(H.HeldoutLeak): L.load_net(name, V)
        net, cfg, _ = L.load_net("F1_final_medium_noheldout", V)               # the retrain loads
        self.assertEqual(cfg["epochs"], 80)

    def test_only_allow_1_lets_a_contaminated_checkpoint_load(self):
        src_first()
        import os, lossdown as L
        from unittest import mock
        class Reached(Exception): pass
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(L, "CK", Path(tmp)), \
                mock.patch.object(L, "load_checkpoint", side_effect=Reached):
            (Path(tmp) / "F1_final_medium.pt").touch()
            for value in (None, "", "0", "false", "no", "true", "yes"):     # any value but "1" stays closed
                env = {k: v for k, v in os.environ.items() if k != "ALLOW_CONTAMINATED_CHECKPOINTS"}
                if value is not None: env["ALLOW_CONTAMINATED_CHECKPOINTS"] = value
                with mock.patch.dict(os.environ, env, clear=True), self.subTest(value=value):
                    with self.assertRaises(H.HeldoutLeak): L.load_net("F1_final_medium", None)
            with mock.patch.dict(os.environ, {"ALLOW_CONTAMINATED_CHECKPOINTS": "1"}):
                with self.assertRaises(Reached): L.load_net("F1_final_medium", None)  # past the guard

    def test_no_contaminated_checkpoint_is_left_in_the_load_path(self):
        import os
        home = Path.home()
        roots = [home / ".local/share/vgc-pilot-runtime/lossdown", home / "Documents/code/vgc-team-generator-pilot/results"]
        left = [str(p) for r in roots if r.exists() for p in r.rglob("*.pt")
                if "noheldout" not in p.name and "_contaminated_heldout" not in p.parts]
        self.assertEqual(left, [])
        quarantine = roots[0] / "_contaminated_heldout"
        self.assertTrue((quarantine / "F1_final_medium.pt").exists())
        backups = home / "vgc-data/backups"
        self.assertEqual([p.name for p in backups.glob("*.sqlite")], [])      # backups holding the entries sit in _contains_heldout

    def test_generators_point_at_clean_checkpoints(self):
        root = SRC.parent
        for rel, bad in (("scripts/dsame_experiment.py", "temperature_p0.pt"), ("scripts/mdts_experiment.py", "temperature_p0.pt"),
                         ("src/redundancy_loop.py", "temperature_p0.pt"), ("src/asked_vs_got.py", "temperature_p0.pt"),
                         ("src/lossdown_battle.py", '"F1_final_medium"')):
            self.assertNotIn(bad, (root / rel).read_text(), rel)
        self.assertIn("noheldout", (root / "src/asked_vs_got.py").read_text())

    def test_loss_data_and_labels_carry_no_heldout_row(self):
        src_first()
        import numpy as np
        import lossdown as L, asked_vs_got as X
        src_first()                                  # lossdown's imports put the laptop's copy back in front
        import gradguide as GG
        V = X.setup()[1]
        raw = {np.asarray(V.encode(__import__("corpus").parse_team(f)), np.int64).tobytes() for f in H.files()}
        full = L.load_data(include_heldout=True); d = L.load_data()
        self.assertGreater(len(full["X"]), len(d["X"]))
        for k in ("X", "W", "NB", "T", "S", "M"): self.assertEqual(len(d[k]), len(d["X"]))
        rows = lambda a: {r.astype(np.int64).tobytes() for r in a}
        self.assertFalse(rows(d["X"]) & raw)
        self.assertTrue(rows(full["X"]) & raw)
        self.assertTrue(H.row_mask(d["X"], V).all())
        self.assertEqual(len(d["top"]), len(full["top"]))
        teams, ys, srcs = GG.all_labels()
        self.assertEqual(Path(GG.__file__).resolve().parent, SRC)
        self.assertFalse(any(H.is_heldout_team(t) for t in teams))
        self.assertGreater(len(teams), 4000)                                   # the other ~4,200 labels are all still there


class Matrix(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        M.PASTES = Path(self.d.name) / "pastes"
        self.con = M.connect(Path(self.d.name) / "t.sqlite")
        self.pid = M.get_policy(self.con, "p", "p", "sim")
        add = lambda text: M.add_team(self.con, text, "test")[0]
        self.top = [add(f.read_text()) for f in POOL[:4]]
        self.cand = [add(f.read_text()) for f in POOL[4:6]]
        self.held = [add(Path(f).read_text()) for f in H.files()[:3]]
        self.dup = add(Path(T50.resolve("MB575")).read_text())                 # the same team as MB574
        self.mb574 = add(Path(H.files()[30]).read_text())
        self.variant = add(respread(Path(H.files()[0]).read_text()))           # held-out team, other Stat Points
        with self.con:
            for t in self.top: self.con.execute("INSERT INTO team_set VALUES ('top50', ?, 1)", (t,))
        for c in self.cand:
            for t in self.top: M.record(self.con, self.pid, [(c, t, 5, 3, 0)])
        for h in self.held + [self.variant]:                                   # entries that already exist in the matrix
            for t in self.top: M.record(self.con, self.pid, [(h, t, 8, 0, 0)])
        for c in self.cand:
            for h in self.held: M.record(self.con, self.pid, [(c, h, 0, 8, 0)])
        FAKE_BATTLES.clear()
    def tearDown(self):
        self.con.close(); self.d.cleanup()

    def test_identical_teams_are_one_held_out_team(self):
        self.assertEqual(self.dup, self.mb574)
        self.assertNotEqual(self.variant, self.held[0])                        # another paste, so another row ...
        held = H.team_ids(self.con)
        self.assertTrue({self.dup, self.variant, *self.held} <= held)          # ... and still a held-out team
        self.assertFalse(held & set(self.top + self.cand))

    def test_held_out_entries_never_reach_fitness_or_labels(self):
        before = [M.score(self.con, self.pid, c)["score"] for c in self.cand]
        self.assertTrue(all(abs(s - 5 / 8) < 1e-9 for s in before))
        with self.con:                                                         # even a held-out team listed in top50
            self.con.execute("INSERT INTO team_set VALUES ('top50', ?, 1)", (self.held[0],))
        self.assertEqual([M.score(self.con, self.pid, c)["score"] for c in self.cand], before)
        self.assertEqual(M.score(self.con, self.pid, self.cand[0])["columns"], len(self.top))
        got = {r["team_id"] for r in M.labels(self.con, self.pid, "top50", min_coverage=1)}
        self.assertTrue(set(self.cand) <= got)
        self.assertFalse(got & H.team_ids(self.con))
        with self.assertRaises(H.HeldoutLeak): M.score(self.con, self.pid, self.held[0])
        with self.assertRaises(H.HeldoutLeak): M.score(self.con, self.pid, self.variant)
        with self.assertRaises(H.HeldoutLeak): M.score(self.con, self.pid, self.cand[0], H.SET_NAME)
        with self.assertRaises(H.HeldoutLeak): M.labels(self.con, self.pid, H.SET_NAME)

    def test_no_search_battle_may_involve_the_set(self):
        for rows, cols in (([self.cand[0]], [self.held[0]]), ([self.held[1]], self.top), ([self.dup], self.top),
                           ([self.variant], self.top), ([self.cand[0]], [self.mb574])):
            with self.assertRaises(H.HeldoutLeak):
                M.ensure(self.con, self.pid, rows, cols, 99, runner=fake_runner)
        self.assertEqual(FAKE_BATTLES, [])

    def test_a_proposal_equal_to_a_heldout_team_is_not_a_candidate(self):
        class Valid:
            def __call__(self, text, fmt=None): return None
            def close(self): pass
        real, real_ensure = M.Validator, M.ensure
        ran = []
        M.Validator = Valid
        M.ensure = lambda con, pid, rows, cols, n, **kw: ran.append((tuple(rows), tuple(cols))) or 0
        try:
            pastes = [respread(Path(H.files()[0]).read_text()), Path(T50.resolve("MB575")).read_text(), POOL[0].read_text()]
            out = M.score_pastes(self.con, pastes, policy_id=self.pid)
        finally:
            M.Validator, M.ensure = real, real_ensure
        self.assertTrue(out[0].get("heldout") and out[1].get("heldout"))
        self.assertIsNone(out[0]["score"]); self.assertFalse(out[2].get("heldout"))
        self.assertFalse(set(ran[0][0]) & H.team_ids(self.con))

    def test_purge_deletes_what_is_in_the_matrix_and_nothing_else(self):
        with self.con:                                                         # the plan matrix is a matrix of the same teams
            self.con.execute("CREATE TABLE plan_cell (policy_id, lo, lo_plan, hi, hi_plan, lo_wins, hi_wins, ties, battles, updated_at)")
            self.con.execute("CREATE TABLE plan_contribution (batch_id, lo, lo_plan, hi, hi_plan, lo_wins, hi_wins, ties, battles)")
            for a, b in ((self.held[0], self.top[0]), (self.cand[0], self.top[1])):
                lo, hi = min(a, b), max(a, b)
                self.con.execute("INSERT INTO plan_cell VALUES (1,?,0,?,0,1,0,0,1,'t')", (lo, hi))
                self.con.execute("INSERT INTO plan_contribution VALUES (1,?,0,?,0,1,0,0,1)", (lo, hi))
        held = H.team_ids(self.con)
        cells = lambda: self.con.execute("SELECT COUNT(*) FROM cell").fetchone()[0]
        touching = self.con.execute("SELECT COUNT(*) FROM cell WHERE lo IN (%s) OR hi IN (%s)" % ((",".join("?" * len(held)),) * 2),
                                    sorted(held) * 2).fetchone()[0]
        self.assertGreater(touching, 0)
        keep = {(c, t) for c in self.cand for t in self.top}
        total = cells()
        with self.con: self.con.execute("INSERT INTO team_set VALUES ('top50', ?, 1)", (self.variant,))
        dry = M.purge_heldout(self.con, dry_run=True)
        self.assertEqual(cells(), total)                                       # a dry run deletes nothing
        out = M.purge_heldout(self.con)
        self.assertEqual((out["cells"], out["teams"]), (touching, len(held)))
        self.assertEqual(dry["cells"], out["cells"])
        self.assertEqual(cells(), total - touching)
        left = {tuple(sorted(r)) for r in self.con.execute("SELECT lo, hi FROM cell")}
        self.assertTrue(all(set(p).isdisjoint(held) for p in left))
        self.assertTrue(all(tuple(sorted(p)) in left for p in keep))           # candidate x top-50 cells survive
        self.assertEqual(self.con.execute("SELECT COUNT(*) FROM contribution WHERE lo IN (%s) OR hi IN (%s)" %
                                          ((",".join("?" * len(held)),) * 2), sorted(held) * 2).fetchone()[0], 0)
        self.assertEqual(self.con.execute("SELECT COUNT(*) FROM team WHERE team_id IN (%s)" % ",".join("?" * len(held)),
                                          sorted(held)).fetchone()[0], len(held))       # teams stay: they are how candidates are recognised
        self.assertEqual([self.con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ("plan_cell", "plan_contribution")], [1, 1])
        self.assertEqual(M.purge_heldout(self.con)["cells"], 0)                # idempotent
        self.assertEqual([M.score(self.con, self.pid, c)["score"] for c in self.cand], [5 / 8, 5 / 8])

    def test_an_old_database_gets_its_48_field_hash_on_connect(self):
        path = Path(self.d.name) / "old.sqlite"
        old = sqlite3.connect(path)
        old.executescript(M.SCHEMA.replace("  grid_hash   TEXT,                       -- sha1 of the 48 fields (no Stat Points): heldout.grid_key_text\n", ""))
        self.assertNotIn("grid_hash", [r[1] for r in old.execute("PRAGMA table_info(team)")])
        text = Path(H.files()[0]).read_text()
        canon = M.canonical_paste(text)
        old.execute("INSERT INTO team (format_id,canon_hash,paste,species_key,base_key,origin) VALUES (?,?,?,?,?,?)",
                    (M.FORMAT, M.canon_hash(canon), canon, "s", "b", "old"))
        old.commit(); old.close()
        con = M.connect(path)
        self.assertEqual(len(H.team_ids(con)), 1)
        con.close()

    def test_the_final_counter_is_evaluated_once_and_only_there(self):
        fit_before = M.score(self.con, self.pid, self.cand[0])
        res = M.evaluate_final(self.con, self.pid, [self.cand[0]], per_cell=4, runner=fake_runner, seed=1)
        self.assertEqual(res[0]["columns"], 50)
        self.assertEqual(res[0]["coverage"], 50)
        held = H.team_ids(self.con)
        self.assertTrue(all(c in held for _, c in FAKE_BATTLES))               # only held-out columns were battled
        self.assertEqual(M.score(self.con, self.pid, self.cand[0]), fit_before)          # fitness is untouched by evaluation
        self.assertFalse({r["team_id"] for r in M.labels(self.con, self.pid, "top50", min_coverage=1)} & held)
        for t in (self.held[0], self.dup, self.variant):
            with self.assertRaises(H.HeldoutLeak):
                M.evaluate_final(self.con, self.pid, [t], per_cell=1, runner=fake_runner)


class BattlePool(unittest.TestCase):
    def test_pool_refuses_the_set_as_candidate_or_opponent(self):
        src_first()
        import pool
        held, other = H.files()[0], str(POOL[0])
        real = pool.live_ports; pool.live_ports = lambda *a, **k: []          # never reach a server
        try:
            with self.assertRaises(H.HeldoutLeak): pool.score([(held, [other])])
            with self.assertRaises(H.HeldoutLeak): pool.score([(other, [other, held])])
            with self.assertRaises(H.HeldoutLeak): pool.score([(T50.resolve("MB575"), [other])])
            with self.assertRaises(H.HeldoutLeak): pool.score([(T50.resolve("MB524"), [other])])   # same 48 fields as MB522
            with self.assertRaises(H.HeldoutLeak): pool.score([(other, [other])], opp_schedule=[held])
            with self.assertRaises(RuntimeError) as c: pool.score([(other, [other])])
            self.assertNotIsInstance(c.exception, H.HeldoutLeak)                # a clean job reaches the server check
            with self.assertRaises(RuntimeError) as c: pool.score([(other, [held])], allow_heldout=True)
            self.assertNotIsInstance(c.exception, H.HeldoutLeak)
        finally:
            pool.live_ports = real

    def test_a_heldout_team_is_not_a_result(self):
        with self.assertRaises(H.HeldoutLeak): H.evaluate([H.files()[0]])


class Algorithms(unittest.TestCase):
    """The loops filter their candidates; their algorithms are untouched. Source tripwires."""
    def test_cem_and_map_elites_filter_candidates_and_start_teams(self):
        cem = (SRC / "cem_v3.py").read_text()
        self.assertIn("HO.is_heldout_text(p)", cem)
        self.assertIn("HO.evaluate(", cem)
        self.assertIn("noheldout", cem)                                         # the ridge cache fitted with the set is not reused
        self.assertIn("noheldout_regmb_v3", (SRC / "foundation.py").read_text())   # the members trained without it
        dsame = (SRC.parent / "scripts" / "dsame_experiment.py").read_text()
        for needle in ("HO.training_pairs", "HO.is_heldout_team", "not HO.is_heldout_text"):
            self.assertIn(needle, dsame)
        for f in ("cem_v3.py", "../scripts/dsame_experiment.py", "heldout.py", "matchup_db.py", "pool.py"):
            ast.parse((SRC / f).read_text())

if __name__ == "__main__":
    unittest.main(verbosity=2)
