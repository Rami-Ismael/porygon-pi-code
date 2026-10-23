"""Run every scripts/test_*.py except those that need data or tools only the laptop has.

CI calls this; locally, `python scripts/ci_tests.py` shows what CI will see.
A new test file runs in CI by default. Add it to NEEDS_LOCAL only with the reason.
"""
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent

NEEDS_LOCAL = {
    "test_dibo": "/tmp/vgc-pilot/learnset_true.json",
    "test_dibo_diversity": "/tmp/vgc-pilot/learnset_true.json",
    "test_diversity_sampler": "/tmp/vgc-pilot/learnset_true.json",
    "test_regmb_vocab": "/tmp/vgc-pilot/learnset_true.json",
    "test_team_writer": "/tmp/vgc-pilot/learnset_true.json",
    "test_showdown_pinned": "/tmp/vgc-pilot/learnset_true.json and the pinned Showdown checkout",
    "test_showdown_validator": "/tmp/vgc-pilot/learnset_true.json and the pinned Showdown checkout",
    "test_heldout_separation": "results/heldout_next50_evs.json, outside git",
    "test_regmb_training_policy": "results/heldout_next50_evs.json and Node",
    "test_shape_checks": "results/heldout_next50_evs.json",
    "test_strict_encode": "results/heldout_next50_evs.json",
    "test_smoothness_db": "results/smoothness_pilot/, outside git",
    "test_smoothness_experiment": "matplotlib, not in requirements.txt",
    "test_matchup_db": "the matchup matrix database in ~/vgc-data",
    "test_pair_disconnect": "poke_env, the battle client",
    "test_dspy_structure": "blog/mask-lab/data.json from the monorepo",
}


def main():
    sys.path.insert(0, str(SCRIPTS))
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for path in sorted(SCRIPTS.glob("test_*.py")):
        if path.stem in NEEDS_LOCAL:
            print(f"skip {path.stem}: needs {NEEDS_LOCAL[path.stem]}")
            continue
        suite.addTests(loader.loadTestsFromName(path.stem))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
