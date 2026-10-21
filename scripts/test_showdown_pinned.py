"""The pinned Showdown checkout is written in one place: showdown_validator.SHOWDOWN.

Before, hps_generate and matchup_db defaulted to the /tmp staging copy and six
files patched them to the pinned checkout; a script that forgot to patch
validated against /tmp without saying so. These tests keep that from coming back.
"""
import re
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

import repo_paths  # puts src/ first on sys.path
import showdown_validator as SV

ROOT = Path(__file__).resolve().parents[1]
# The pinned checkout and format, written out: the independent source the tests compare against.
PINNED = Path.home() / ".local/share/vgc-pilot-runtime/validator-913da36"
REG_MB = "gen9championsvgc2026regmb"
# report_regmb prints the shell command that was run to export the vocabulary:
# a record of what happened, so it keeps the literal path.
RECORDS = {"showdown_validator.py", "report_regmb.py", Path(__file__).name}


class TheConstants(unittest.TestCase):
    def test_showdown_is_the_pinned_checkout(self):
        self.assertEqual(SV.SHOWDOWN, PINNED)

    def test_format_is_reg_mb(self):
        self.assertEqual(SV.FORMAT, REG_MB)


def sources():
    for folder in ("src", "scripts"):
        for path in sorted((ROOT / folder).glob("*.py")):
            yield path, path.read_text(encoding="utf-8")


class PinnedCheckout(unittest.TestCase):
    def test_the_checkout_path_is_written_once(self):
        found = [p.name for p, text in sources() if p.name not in RECORDS and PINNED.name in text]
        self.assertEqual(found, [], f"{PINNED.name} is written outside showdown_validator.py")

    def test_no_module_sets_its_own_showdown(self):
        own = re.compile(r"^SHOWDOWN\s*=", re.M)          # a module-level default
        patch = re.compile(r"\b\w+\.SHOWDOWN\s*=")          # one module patching another
        found = [p.name for p, text in sources()
                 if p.name not in RECORDS and not p.name.startswith("test_") and (own.search(text) or patch.search(text))]
        self.assertEqual(found, [])

    def test_runtime_is_gone(self):
        self.assertFalse(hasattr(SV, "runtime"))



class ConsumerDefaults(unittest.TestCase):
    """Each module that runs Showdown defaults to the pinned checkout with no caller setting it."""
    def test_matchup_db_defaults_to_the_pinned_checkout(self):
        import matchup_db
        self.assertEqual(matchup_db.SHOWDOWN, PINNED)
        self.assertEqual(matchup_db.FORMAT, REG_MB)

    def test_hps_generate_validator_runs_in_the_pinned_checkout(self):
        h = importable(self, "hps_generate")
        with mock.patch.object(h.subprocess, "Popen") as popen:
            h.Validator()
        self.assertEqual(popen.call_args.kwargs["cwd"], str(PINNED))

    def test_propose_validator_runs_in_the_pinned_checkout(self):
        p = importable(self, "propose")
        with mock.patch.object(p.subprocess, "Popen") as popen:
            p.Validator()
        self.assertEqual(popen.call_args.kwargs["cwd"], str(PINNED))


class ValidatorTakesACheckout(unittest.TestCase):
    def test_validator_runs_in_the_checkout_it_is_given(self):
        h = importable(self, "hps_generate")
        with mock.patch.object(h.subprocess, "Popen") as popen:
            h.Validator(showdown=Path("/elsewhere/pokemon-showdown"))
        self.assertEqual(popen.call_args.kwargs["cwd"], "/elsewhere/pokemon-showdown")


class CallersImportLazily(unittest.TestCase):
    """hps_generate reads /tmp/vgc-pilot data when imported, so callers import it inside the
    function that needs it. Importing a caller must not import hps_generate."""
    def test_importing_a_caller_does_not_import_hps_generate(self):
        probe = ("import sys, repo_paths, battles, completion_study, llm_completion\n"
                 "print('hps_generate' in sys.modules)")
        out = subprocess.run([sys.executable, "-c", probe], cwd=Path(__file__).parent,
                             capture_output=True, text=True, timeout=300)
        self.assertEqual(out.returncode, 0, out.stderr[-2000:])
        self.assertEqual(out.stdout.strip().splitlines()[-1], "False")


def importable(case, name):
    """The module, or a skip naming why it does not import here (it needs the /tmp/vgc-pilot data)."""
    try:
        return __import__(name)
    except (FileNotFoundError, ImportError) as error:
        case.skipTest(f"{name} does not import here: {error}")



if __name__ == "__main__":
    unittest.main()
