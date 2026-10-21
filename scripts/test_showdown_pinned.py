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

import repo_paths  # puts src/ first on sys.path
import showdown_validator as SV

ROOT = Path(__file__).resolve().parents[1]
# The pinned checkout and format, written out: the independent source the tests compare against.
PINNED = Path.home() / ".local/share/vgc-pilot-runtime/validator-913da36"
REG_MB = "gen9championsvgc2026regmb"
# The pinned vgc-bench code checkout. Not the /tmp/vgc-pilot/vgc-bench staging copy, whose team pool differs.
PINNED_VGC_BENCH = Path.home() / ".local/share/vgc-pilot-runtime/unpacked/vgc-bench-d79f9532947ac114dce1dda2456a590afcd375b2"
# report_regmb prints the shell command that was run to export the vocabulary:
# a record of what happened, so it keeps the literal path. The two test files write
# the path out on purpose: it is the independent value they compare against.
RECORDS = {"showdown_validator.py", "report_regmb.py", Path(__file__).name, "test_showdown_validator.py"}


class TheConstants(unittest.TestCase):
    def test_vgc_bench_is_the_pinned_checkout(self):
        self.assertEqual(SV.VGC_BENCH, PINNED_VGC_BENCH)

    def test_showdown_is_the_pinned_checkout(self):
        self.assertEqual(SV.SHOWDOWN, PINNED)

    def test_format_is_reg_mb(self):
        self.assertEqual(SV.FORMAT, REG_MB)


def sources():
    for folder in ("src", "scripts"):
        for path in sorted((ROOT / folder).glob("*.py")):
            yield path, path.read_text(encoding="utf-8")


class PinnedCheckout(unittest.TestCase):
    def test_the_vgc_bench_path_is_written_once(self):
        found = [p.name for p, text in sources() if p.name not in RECORDS and PINNED_VGC_BENCH.name in text]
        self.assertEqual(found, [], f"{PINNED_VGC_BENCH.name} is written outside showdown_validator.py")

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



def fresh(case, code):
    """Run code in a new Python with this tree's src/ first; return its printed lines.
    A fresh process matters: other test modules put /tmp/vgc-pilot/src (an older copy of
    these modules) on sys.path, so an in-process import can return the wrong file."""
    out = subprocess.run([sys.executable, "-c", "import repo_paths\n" + code], cwd=Path(__file__).parent,
                         capture_output=True, text=True, timeout=300)
    case.assertEqual(out.returncode, 0, out.stderr[-2000:])
    return out.stdout.strip().splitlines()


VALIDATOR_CWD = """
import {module} as m
from unittest import mock
from pathlib import Path
with mock.patch.object(m.subprocess, "Popen") as popen:
    m.Validator({args})
print(m.__file__)
print(popen.call_args.kwargs["cwd"])
"""


class ConsumerDefaults(unittest.TestCase):
    """Each module that runs Showdown defaults to the pinned checkout with no caller setting it."""
    def test_matchup_db_defaults_to_the_pinned_checkout(self):
        file, showdown, fmt = fresh(self, "import matchup_db as m\nprint(m.__file__); print(m.SHOWDOWN); print(m.FORMAT)")
        self.assertEqual(Path(file).parent, ROOT / "src")
        self.assertEqual((Path(showdown), fmt), (PINNED, REG_MB))

    def test_hps_generate_validator_runs_in_the_pinned_checkout(self):
        file, cwd = fresh(self, VALIDATOR_CWD.format(module="hps_generate", args=""))[-2:]
        self.assertEqual(Path(file).parent, ROOT / "src")
        self.assertEqual(cwd, str(PINNED))

    def test_propose_validator_runs_in_the_pinned_checkout(self):
        file, cwd = fresh(self, VALIDATOR_CWD.format(module="propose", args=""))[-2:]
        self.assertEqual(Path(file).parent, ROOT / "src")
        self.assertEqual(cwd, str(PINNED))


class ValidatorTakesACheckout(unittest.TestCase):
    def test_validator_runs_in_the_checkout_it_is_given(self):
        args = 'showdown=Path("/elsewhere/pokemon-showdown")'
        cwd = fresh(self, VALIDATOR_CWD.format(module="hps_generate", args=args))[-1]
        self.assertEqual(cwd, "/elsewhere/pokemon-showdown")


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


if __name__ == "__main__":
    unittest.main()
