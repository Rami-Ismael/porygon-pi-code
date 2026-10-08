"""Exercise real proposal-loop accounting with a stub decoder and validator."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
import gradguide as G


class SamplerTest(unittest.TestCase):
    def run_sampler(self, n, validator):
        sampling = {}
        def decode(model, constraints, guide, want, mode, strength):
            return torch.zeros((want, 48), dtype=torch.int64), []
        paste = "\n\n".join(f"Species{i} @ Item\nAbility: A\nBold Nature\n- Protect" for i in range(6))
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(G, "sample_guided", decode), \
             patch.object(G.A, "row_to_paste", return_value=paste):
            result = G.propose_guided(None, None, None, None, None, None,
                validator, "gloss", 1., n, Path(directory), 42, sampling_stats=sampling)
        return result, sampling

    def test_hard_cap_and_all_invalid(self):
        result, sampling = self.run_sampler(31, lambda _: "invalid")
        self.assertEqual(sampling, dict(attempts=248, accepted=0, rejected=248, attempt_limit=248))
        self.assertEqual(result[2], [])

    def test_counts_rejections_and_keeps_duplicate_accepted_draws(self):
        calls = []
        def validate(paste):
            calls.append(paste)
            return None if len(calls) % 2 else "invalid"
        result, sampling = self.run_sampler(3, validate)
        self.assertEqual(sampling["attempts"], 5)
        self.assertEqual(sampling["rejected"], 2)
        self.assertEqual(len(result[2]), 3)
        self.assertEqual(len(set(result[2])), 1)
        self.assertEqual(result[3], 3/5)


if __name__ == "__main__":
    unittest.main()
