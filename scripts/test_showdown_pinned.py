"""The pinned Showdown checkout is written in one place: showdown_validator.SHOWDOWN.

Before, hps_generate and matchup_db defaulted to the /tmp staging copy and six
files patched them to the pinned checkout; a script that forgot to patch
validated against /tmp without saying so. These tests keep that from coming back.
"""
import re
import sys
import unittest
from pathlib import Path
from unittest import mock

import repo_paths  # puts src/ first on sys.path
import showdown_validator as SV

ROOT = Path(__file__).resolve().parents[1]
# report_regmb prints the shell command that was run to export the vocabulary:
# a record of what happened, so it keeps the literal path.
RECORDS = {"showdown_validator.py", "report_regmb.py", Path(__file__).name}


def sources():
    for folder in ("src", "scripts"):
        for path in sorted((ROOT / folder).glob("*.py")):
            yield path, path.read_text(encoding="utf-8")


class PinnedCheckout(unittest.TestCase):
    def test_the_checkout_path_is_written_once(self):
        found = [p.name for p, text in sources() if p.name not in RECORDS and SV.SHOWDOWN.name in text]
        self.assertEqual(found, [], f"{SV.SHOWDOWN.name} is written outside showdown_validator.py")

    def test_no_module_sets_its_own_showdown(self):
        own = re.compile(r"^SHOWDOWN\s*=", re.M)          # a module-level default
        patch = re.compile(r"\b\w+\.SHOWDOWN\s*=")          # one module patching another
        found = [p.name for p, text in sources()
                 if p.name not in RECORDS and not p.name.startswith("test_") and (own.search(text) or patch.search(text))]
        self.assertEqual(found, [])

    def test_runtime_is_gone(self):
        self.assertFalse(hasattr(SV, "runtime"))

    def test_matchup_db_uses_the_pinned_checkout(self):
        import matchup_db
        self.assertEqual(matchup_db.SHOWDOWN, SV.SHOWDOWN)
        self.assertEqual(matchup_db.FORMAT, SV.FORMAT)


class ValidatorCheckout(unittest.TestCase):
    def setUp(self):
        try:
            import hps_generate
        except (FileNotFoundError, ImportError) as error:   # needs the /tmp/vgc-pilot data files
            self.skipTest(f"hps_generate does not import here: {error}")
        self.h = hps_generate

    def test_validator_runs_in_the_pinned_checkout_by_default(self):
        with mock.patch.object(self.h.subprocess, "Popen") as popen:
            self.h.Validator()
        self.assertEqual(popen.call_args.kwargs["cwd"], str(SV.SHOWDOWN))

    def test_validator_takes_another_checkout(self):
        with mock.patch.object(self.h.subprocess, "Popen") as popen:
            self.h.Validator(showdown=Path("/elsewhere"))
        self.assertEqual(popen.call_args.kwargs["cwd"], "/elsewhere")


if __name__ == "__main__":
    unittest.main()
