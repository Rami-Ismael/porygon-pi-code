"""The scratch team writer against the code it replaces: foundation.Gen("v3") for scoring, sampling and
pastes (bit-for-bit, same device and seeds, except that the writer never samples a Pokémon with no
moves, which Showdown rejects and Gen writes in about 2% of teams), cem_v3.members and cem_dbbo.members for fine-tuning.
Training on the Mac's GPU is not bit-deterministic: two identical runs of cem_v3.members differ by about
1e-3 in log-probability after one epoch. So a fine-tune must start bit-identical (zero budget) and then
land within three times the old code's own run-to-run spread.
Needs the v3 member checkpoints; runs in about three minutes on the M4 Pro.

    /tmp/vgc-pilot/.venv/bin/python -m unittest discover -s scripts -p 'test_team_writer.py'
"""
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

import repo_paths  # noqa: F401  (src/ first on sys.path)
import foundation as FD
import lossdown as L
from team_writer import TeamWriter
from encode import NF

TAG = f"test_team_writer_{os.getpid()}"


def corpus_fixture(writer, n=96):
    """n encodable corpus teams, a label for each, and a half-masked batch (inputs, t, labels) to score."""
    rows = []
    for team in writer.corpus:
        try: rows.append(writer.V.encode(team))
        except ValueError: continue
        if len(rows) == n: break
    rows = np.stack(rows)
    labels = np.random.default_rng(0).uniform(0.3, 0.7, len(rows)).astype(np.float32)
    g = torch.Generator().manual_seed(1)
    x = torch.as_tensor(rows, device=L.DEV)
    batch = (x.masked_fill((torch.rand(x.shape, generator=g) < 0.5).to(L.DEV), 0),
             torch.full((len(x),), 0.5, device=L.DEV), torch.as_tensor(labels, device=L.DEV))
    return rows, labels, batch


class TeamWriterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gen = FD.Gen("v3")
        cls.writer = TeamWriter.load()
        cls.rows, cls.labels, (cls.xin, cls.t, cls.w) = corpus_fixture(cls.writer)
        cls.tmp = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()
        for f in L.CK.glob(f"{TAG}*"): f.unlink()

    def assert_same_logprobs(self, f_new, f_old):
        a, b = f_new(self.xin, self.t, self.w), f_old(self.xin, self.t, self.w)
        self.assertEqual(set(a), set(b))
        for k in a: self.assertTrue(torch.equal(a[k], b[k]), k)

    def gap(self, f1, f2):
        a, b = f1(self.xin, self.t, self.w), f2(self.xin, self.t, self.w)
        return max(float((a[k] - b[k]).abs().max()) for k in a)

    def assert_within_old_spread(self, new, old_twice):
        spread = self.gap(*old_twice)
        self.assertLessEqual(self.gap(new, old_twice[0]), max(3 * spread, 1e-4), f"run-to-run spread of the replaced fine-tune: {spread:.2e}")

    def test_logprobs_match_gen(self):
        self.assert_same_logprobs(self.writer.logprobs, self.gen.f)

    def test_sample_matches_gen(self):
        sur = FD.Surrogate(self.writer.V)
        settings = [dict(args=(8,), kw={}),
                    dict(args=(8, 0.5, 2.0, 0.9, 0.95), kw=dict(g_species=1.5, cads=0.2)),
                    dict(args=(8, 0.5, 2.0, 1.0, 1.0, sur, sur.lam), kw={})]
        # The writer forbids a slot with no moves (Showdown rejects it); Gen("v3") allows one. Each row draws
        # its own tokens, so every team must match Gen's except a team where Gen left a slot moveless.
        moveless = lambda r: any(int(r[NF * i + 3]) == self.writer.tables.EMV for i in range(6))
        for i, s in enumerate(settings):
            kw = dict(s["kw"]); torch.manual_seed(100 + i); old = self.gen.sample(*s["args"], **kw)
            if "g_species" in kw: kw["species_guidance"] = kw.pop("g_species")
            torch.manual_seed(100 + i); new = self.writer.sample(*s["args"], **kw)
            for a, b in zip(old, new):
                self.assertFalse(moveless(b), s)
                if not torch.equal(a, b): self.assertTrue(moveless(a), s)
        self.assertIsNone(L.COND_NOISE)

    def test_to_paste_matches_gen(self):
        for repair in (False, True):
            for r in self.rows[:8]:
                a = self.gen.to_paste(r, np.random.default_rng(3), repair)
                b = self.writer.to_paste(r, np.random.default_rng(3), repair)
                self.assertEqual(a, b)

    def test_epoch_fine_tune_matches_cem_v3_members(self):
        import cem_v3 as CV
        saved = CV.TAG, CV.FT_EPOCHS
        elites = [(r, float(y), "test") for r, y in zip(self.rows, self.labels)]
        old = {}
        try:
            for epochs, run in [(0, "z"), (2, "a"), (2, "b")]:
                CV.TAG, CV.FT_EPOCHS = f"{TAG}_cv{run}", epochs
                old[run] = CV.members(self.writer.V, self.writer.tables, elites, 0)
        finally:
            CV.TAG, CV.FT_EPOCHS = saved
        tune = lambda epochs: self.writer.fine_tuned(self.rows, self.labels, epochs=epochs, batch=CV.FT_BATCH,
                                                    lr=CV.FT_LR, seed=CV.SEED, cache_dir=self.tmp.name)
        self.assert_same_logprobs(tune(0).logprobs, old["z"])
        self.assert_within_old_spread(tune(2).logprobs, (old["a"], old["b"]))

    def test_mixed_step_fine_tune_matches_cem_dbbo_members(self):
        import cem_dbbo as CD
        saved = CD.TAG, CD.FT_STEPS
        lab = [(r, float(y)) for r, y in zip(self.rows, self.labels)]
        old = {}
        try:
            for run in "ab":                                   # cem_dbbo divides by FT_STEPS: no zero-budget run
                CD.TAG, CD.FT_STEPS = f"{TAG}_cd{run}", 3
                old[run] = CD.members(self.writer.V, self.writer.tables, lab, 0)
        finally:
            CD.TAG, CD.FT_STEPS = saved
        new = self.writer.fine_tuned(self.rows, self.labels, steps=3, batch=CD.FT_BATCH, lr=CD.FT_LR,
                                     mix_original=True, seed=CD.SEED, cache_dir=self.tmp.name)
        self.assert_within_old_spread(new.logprobs, (old["a"], old["b"]))

    def test_cache_key_covers_every_setting(self):
        d = Path(self.tmp.name) / "key"
        a = self.writer.fine_tuned(self.rows[:64], self.labels[:64], epochs=1, seed=5, cache_dir=d)
        self.assertEqual(len(list(d.glob("*.pt"))), 2)
        b = self.writer.fine_tuned(self.rows[:64], self.labels[:64], epochs=1, seed=5, cache_dir=d)
        self.assertEqual(len(list(d.glob("*.pt"))), 2)
        self.assert_same_logprobs(a.logprobs, b.logprobs)
        self.writer.fine_tuned(self.rows[:64], self.labels[:64], epochs=2, seed=5, cache_dir=d)
        self.writer.fine_tuned(self.rows[:64], self.labels[:64] + 0.01, epochs=1, seed=5, cache_dir=d)
        self.assertEqual(len(list(d.glob("*.pt"))), 6)
        self.assertNotEqual(a.key, self.writer.key)

    def test_fine_tune_leaves_the_writer_unchanged(self):
        before = self.writer.logprobs(self.xin, self.t, self.w)
        self.writer.fine_tuned(self.rows[:64], self.labels[:64], steps=2, seed=9, cache_dir=Path(self.tmp.name) / "keep")
        after = self.writer.logprobs(self.xin, self.t, self.w)
        for k in before: self.assertTrue(torch.equal(before[k], after[k]), k)

    def test_bad_budget_is_refused(self):
        with self.assertRaises(ValueError): self.writer.fine_tuned(self.rows, self.labels, cache_dir=self.tmp.name)
        with self.assertRaises(ValueError): self.writer.fine_tuned(self.rows, self.labels, epochs=1, mix_original=True,
                                                                   cache_dir=self.tmp.name)



class TeamWriterSpec(unittest.TestCase):
    """What the writer promises through its interface, checked against independent truths (the Showdown
    validator, the paste parser, the training loss). These outlive the equivalence tests above."""

    @classmethod
    def setUpClass(cls):
        cls.writer = TeamWriter.load()
        cls.rows, cls.labels, cls.batch = corpus_fixture(cls.writer)
        cls.tmp = tempfile.TemporaryDirectory(); cls.dir = Path(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_zero_budget_keeps_the_parent_weights(self):
        for budget in (dict(epochs=0), dict(steps=0)):
            child = self.writer.fine_tuned(self.rows, self.labels, seed=1, cache_dir=self.dir / "zero", **budget)
            parent, kept = self.writer.logprobs(*self.batch), child.logprobs(*self.batch)
            for k in parent: self.assertTrue(torch.equal(parent[k], kept[k]), (budget, k))

    def test_cache_hit_returns_without_training(self):
        d = self.dir / "hit"
        first = self.writer.fine_tuned(self.rows, self.labels, epochs=1, seed=2, cache_dir=d)
        stamps = {f.name: f.stat().st_mtime_ns for f in d.glob("*.pt")}
        again = self.writer.fine_tuned(self.rows, self.labels, epochs=1, seed=2, cache_dir=d)
        self.assertEqual({f.name: f.stat().st_mtime_ns for f in d.glob("*.pt")}, stamps)
        a, b = first.logprobs(*self.batch), again.logprobs(*self.batch)
        for k in a: self.assertTrue(torch.equal(a[k], b[k]), k)
        self.assertEqual(first.key, again.key)

    def test_fine_tuning_lowers_the_loss_on_its_rows(self):
        loss = lambda w: L.evaluate(w.logprobs, self.rows, self.labels, reps=4)[0]["elbo"]
        before = loss(self.writer)
        for budget in (dict(epochs=10), dict(steps=20), dict(steps=20, mix_original=True)):
            after = loss(self.writer.fine_tuned(self.rows, self.labels, seed=3, cache_dir=self.dir / "lower", **budget))
            self.assertLess(after, before, budget)

    def test_a_paste_parses_back_to_the_sampled_fields(self):
        from corpus import parse_team_text
        # A team is six slots in no particular order; the encoder sorts them, the sampler does not.
        slots = lambda r, keep: sorted(tuple(int(v) for v in r[8 * i:8 * i + 8][keep]) for i in range(6))
        torch.manual_seed(4); rows = self.writer.sample(200, 0.5, 2.0).cpu().numpy()
        rng = np.random.default_rng(4)
        for row in rows:
            for repair in (False, True):
                text, _ = self.writer.to_paste(row, rng, repair)
                parsed = parse_team_text(text, blank_ends_slot=True)
                self.assertEqual(len(parsed), 6, text)
                keep = np.ones(8, bool)
                if repair: keep[7] = False                           # repair may change the nature, nothing else
                self.assertEqual(slots(self.writer.V.encode(parsed), keep), slots(row, keep), text)

    def test_every_sampled_team_passes_the_showdown_validator(self):
        from showdown_validator import Validator
        rng = np.random.default_rng(5); failures = []
        v = Validator()
        try:
            for i, settings in enumerate([(64,), (64, 0.5, 2.0)]):
                torch.manual_seed(500 + i)
                for row in self.writer.sample(*settings).cpu().numpy():
                    text, _ = self.writer.to_paste(row, rng)
                    err = v(text)
                    if err: failures.append((settings, err[:160]))
        finally:
            v.close()
        self.assertEqual(failures, [], f"{len(failures)} of 128 sampled teams rejected")

if __name__ == "__main__":
    unittest.main()
