"""The loop decodes 0.3-asked and 0.5-asked teams in one batch (cem_dbbo.propose). Guard that every row is
conditioned on its own ask (2026-10-16 diagnosis: no leak found; the ask's weak effect inside the loop comes from
the ridge guidance choosing species, not from the code). Needs the v3 checkpoints; skipped without them.

    /tmp/vgc-pilot/.venv/bin/python -m unittest scripts/test_ask_rows.py   (from code/)
"""
import sys
import unittest
sys.path.insert(0, "/tmp/vgc-pilot/src")

try:
    import torch
    import foundation as FD
    import lossdown as L
    HAVE = all((L.CK / f"{m}.pt").exists() for m in FD.V3_MEMBERS)
except Exception:                                    # no runtime on this machine
    HAVE = False


@unittest.skipUnless(HAVE, "v3 checkpoints or the /tmp/vgc-pilot runtime are not available")
class AskReachesEveryRow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        G = FD.Gen("v3"); cls.f = staticmethod(G.f)       # a plain function stored on the class would bind as a method
        torch.manual_seed(0); n = 8
        x = G.sample(n, None, 1.0, 1.0, 1.0); x[:, 12:] = 0
        cls.x, cls.t = x, torch.full((n,), 0.75, device=L.DEV)
        cls.asks = torch.tensor([0.3, 0.5] * (n // 2), device=L.DEV)

    def test_the_ask_changes_the_output(self):
        with torch.no_grad():
            lo = self.f(self.x, self.t, torch.full_like(self.asks, 0.3))
            hi = self.f(self.x, self.t, torch.full_like(self.asks, 0.5))
        self.assertGreater(max(float((lo[k] - hi[k]).abs().max()) for k in lo), 1e-2)

    def test_rows_of_a_mixed_batch_do_not_leak(self):
        with torch.no_grad():
            mixed = self.f(self.x, self.t, self.asks)
            for i in range(len(self.asks)):
                alone = self.f(self.x[i:i + 1], self.t[i:i + 1], self.asks[i:i + 1])
                for k in mixed:
                    self.assertLess(float((mixed[k][i] - alone[k][0]).abs().max()), 1e-2, f"row {i}, {k}")


if __name__ == "__main__":
    unittest.main()
