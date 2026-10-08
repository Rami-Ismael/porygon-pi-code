"""Fault-injection checks against one actually completed experimental run.

Usage: python scripts/test_temperature_protocol_audit.py RESULTS.json
The supplied results are read only. Mutated copies live in a temporary directory.
"""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

from audit_temperature_protocol import audit


def test_suite(results):
    data = json.loads(Path(results).read_text())
    key = next(k for k, r in data["runs"].items() if "scores" in r.get("final", {}))
    fixture = dict(status="running", manifest=data["manifest"], runs={key: data["runs"][key]})

    class ProtocolAuditTests(unittest.TestCase):
        def audit_copy(self, mutate=None, allow_incomplete=True):
            trial = copy.deepcopy(fixture)
            if mutate:
                mutate(trial["runs"][key])
            with tempfile.TemporaryDirectory(prefix="temperature-audit-test-") as directory:
                path = Path(directory) / "results.json"
                path.write_text(json.dumps(trial))
                return audit(path, allow_incomplete=allow_incomplete, verify_inputs=False)

        def test_real_completed_run_passes_as_partial_campaign(self):
            receipt = self.audit_copy()
            self.assertEqual(receipt["status"], "partial_checks_passed")
            self.assertFalse(receipt["study_complete"])
            self.assertEqual(receipt["measured_battles"], 61440)
            self.assertEqual(receipt["audited_generations"], 11)

        def test_partial_campaign_cannot_pass_full_gate(self):
            with self.assertRaisesRegex(ValueError, "campaign is not complete"):
                self.audit_copy(allow_incomplete=False)

        def test_rejects_wrong_temperature(self):
            with self.assertRaisesRegex(ValueError, "temperature schedule|adaptive concentration"):
                self.audit_copy(lambda r: r["generations"]["1"]["training"].update(temperature=.25))

        def test_rejects_nonzero_entropy_bonus(self):
            with self.assertRaisesRegex(ValueError, "loss entropy bonus"):
                self.audit_copy(lambda r: r["generations"]["1"]["training"].update(beta=.1))

        def test_rejects_changed_training_budget(self):
            with self.assertRaisesRegex(ValueError, "training steps"):
                self.audit_copy(lambda r: r["generations"]["1"]["training"].update(refit_steps=81))

        def test_rejects_wrong_battle_denominator(self):
            with self.assertRaisesRegex(ValueError, "score denominator"):
                self.audit_copy(lambda r: r["generations"]["1"]["scores"][0].update(battles=23))

        def test_rejects_changed_acquisition(self):
            with self.assertRaisesRegex(ValueError, "surrogate top-128"):
                self.audit_copy(lambda r: r["generations"]["1"]["selected"].__setitem__(0, "changed team"))

        def test_rejects_reported_diversity_not_supported_by_samples(self):
            with self.assertRaisesRegex(ValueError, "effective species sets"):
                self.audit_copy(lambda r: r["generations"]["1"]["proposal"]["diversity"].update(effective_species_sets=1))

        def test_rejects_reused_search_seed_in_fresh_evaluation(self):
            with self.assertRaisesRegex(ValueError, "reused phase seed"):
                self.audit_copy(lambda r: r["holdout"].update(battle_seed=r["generations"]["1"]["battle_seed"]))

        def test_rejects_finalist_reranking(self):
            with self.assertRaisesRegex(ValueError, "top unique search-scored teams"):
                self.audit_copy(lambda r: r["final"]["teams"].reverse())

    return unittest.defaultTestLoader.loadTestsFromTestCase(ProtocolAuditTests)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    result = unittest.TextTestRunner(verbosity=2).run(test_suite(sys.argv[1]))
    raise SystemExit(not result.wasSuccessful())
