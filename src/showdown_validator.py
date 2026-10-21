"""The pinned Showdown validator checkout, the format, and a team written as Showdown paste text.

SHOWDOWN is the only place the checkout's path is written. hps_generate and
matchup_db take their default from it, so validation and the battle records use
the same Showdown without any caller setting it.
"""
from pathlib import Path

FORMAT = "gen9championsvgc2026regmb"
SHOWDOWN = Path.home() / ".local/share/vgc-pilot-runtime/validator-913da36"


def paste(team, h):
    return "\n\n".join(h.slot_to_text(dict(s,moves=[m for m in s['moves'] if m])) for s in team) + "\n"
