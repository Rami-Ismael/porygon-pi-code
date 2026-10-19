"""The pinned Showdown validator checkout, and a team written as Showdown paste text."""
from pathlib import Path

FORMAT = "gen9championsvgc2026regmb"


def runtime():
    """hps_generate and matchup_db, both pointed at the frozen validator checkout."""
    import hps_generate as h
    import matchup_db as db
    h.SHOWDOWN = Path.home() / ".local/share/vgc-pilot-runtime/validator-913da36"
    db.SHOWDOWN = h.SHOWDOWN
    return h, db


def paste(team, h):
    return "\n\n".join(h.slot_to_text(dict(s,moves=[m for m in s['moves'] if m])) for s in team) + "\n"
