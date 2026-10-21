"""The pinned Showdown validator checkout, the format, and a team written as Showdown paste text.

SHOWDOWN and VGC_BENCH are the only places the pinned checkouts' paths are written. hps_generate and
matchup_db take their default from it, so validation and the battle records use
the same Showdown without any caller setting it. Validator, the one class that talks to
that checkout, lives here too; hps_generate, propose and matchup_db hand out this class.
"""
import json
import subprocess
from pathlib import Path

FORMAT = "gen9championsvgc2026regmb"
SHOWDOWN = Path.home() / ".local/share/vgc-pilot-runtime/validator-913da36"
# The pinned vgc-bench code checkout. Not /tmp/vgc-pilot/vgc-bench, the staging copy, whose team pool differs.
VGC_BENCH = Path.home() / ".local/share/vgc-pilot-runtime/unpacked/vgc-bench-d79f9532947ac114dce1dda2456a590afcd375b2"



class Validator:
    """Showdown's own TeamValidator through the project's batch script: one
    `node validate-teams-batch.js` process in `showdown` (default SHOWDOWN).
    Calling it with a team's paste text returns None if the team is legal, else every
    error joined with "; ", untruncated: callers parse each error."""
    def __init__(self, showdown=None):
        self.p = subprocess.Popen(["node", "validate-teams-batch.js"],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  text=True, cwd=str(showdown or SHOWDOWN))

    def __call__(self, text, fmt=FORMAT):
        self.p.stdin.write(json.dumps({"format": fmt, "team": text}) + "\n")
        self.p.stdin.flush()
        r = json.loads(self.p.stdout.readline())
        return None if r["valid"] else "; ".join(r["errors"])

    def close(self):
        self.p.stdin.close(); self.p.wait()


def paste(team, h):
    return "\n\n".join(h.slot_to_text(dict(s,moves=[m for m in s['moves'] if m])) for s in team) + "\n"
