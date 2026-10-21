"""showdown_validator.Validator through its interface: one `node validate-teams-batch.js`
process in the pinned checkout, one JSON line per team.

The node process is the only thing replaced: it is the external dependency at this seam.
"""
import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

import repo_paths  # puts src/ first on sys.path
import showdown_validator as SV

PINNED = Path.home() / ".local/share/vgc-pilot-runtime/validator-913da36"
TEAM = "Garchomp @ Choice Scarf\nAbility: Rough Skin\nLevel: 50\n- Earthquake\n"


def validator_answering(reply, **kwargs):
    """A Validator whose node process answers every team with `reply`; returns (validator, process)."""
    with mock.patch.object(SV.subprocess, "Popen") as popen:
        process = popen.return_value
        process.stdout.readline.return_value = json.dumps(reply) + "\n"
        return SV.Validator(**kwargs), process, popen


class ValidatorInterface(unittest.TestCase):
    def test_a_legal_team_returns_none(self):
        validator, _, _ = validator_answering({"valid": True})
        self.assertIsNone(validator(TEAM))

    def test_an_illegal_team_returns_every_error_joined(self):
        validator, _, _ = validator_answering({"valid": False, "errors": ["Garchomp can't learn X", "Level must be 50"]})
        self.assertEqual(validator(TEAM), "Garchomp can't learn X; Level must be 50")

    def test_each_team_is_one_json_line_with_the_reg_mb_format(self):
        validator, process, _ = validator_answering({"valid": True})
        validator(TEAM)
        line = process.stdin.write.call_args.args[0]
        self.assertTrue(line.endswith("\n"))
        self.assertEqual(json.loads(line), {"format": "gen9championsvgc2026regmb", "team": TEAM})

    def test_it_runs_in_the_pinned_checkout_by_default(self):
        _, _, popen = validator_answering({"valid": True})
        self.assertEqual(popen.call_args.args[0], ["node", "validate-teams-batch.js"])
        self.assertEqual(popen.call_args.kwargs["cwd"], str(PINNED))

    def test_it_runs_in_the_checkout_it_is_given(self):
        _, _, popen = validator_answering({"valid": True}, showdown=Path("/elsewhere/pokemon-showdown"))
        self.assertEqual(popen.call_args.kwargs["cwd"], "/elsewhere/pokemon-showdown")

    def test_close_ends_the_process(self):
        validator, process, _ = validator_answering({"valid": True})
        validator.close()
        process.stdin.close.assert_called_once()
        process.wait.assert_called_once()


class OneValidator(unittest.TestCase):
    def test_every_module_hands_out_the_same_validator(self):
        # A fresh process: other test modules put /tmp/vgc-pilot/src (older copies) on sys.path.
        probe = ("import repo_paths, showdown_validator as s, hps_generate, propose, matchup_db\n"
                 "print(all(m.Validator is s.Validator for m in (hps_generate, propose, matchup_db)))")
        out = subprocess.run([sys.executable, "-c", probe], cwd=Path(__file__).parent,
                             capture_output=True, text=True, timeout=300)
        self.assertEqual(out.returncode, 0, out.stderr[-2000:])
        self.assertEqual(out.stdout.strip().splitlines()[-1], "True")


if __name__ == "__main__":
    unittest.main()
