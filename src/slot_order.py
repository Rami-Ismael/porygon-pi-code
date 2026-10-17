"""Shuffle the order of the six Pokémon before every battle.

The behaviour-cloning policy picks its four at team preview by slot number
(vgc_bench policy_player.teampreview returns "/team 1234"), so the order a team is
written in changes who it brings. Measured 2026-10-16 over 12 teams in 6 rotated
orders, 13,848 battles: slot 1 brought 0.43, slots 2-3 0.80, slot 5 0.51 (chance 0.67).
Win rate moved less (order standard deviation about 0.018).

Shuffling every battle gives each Pokémon the policy's average treatment over slots,
so a team's score no longer depends on how its file happens to be ordered.
VGC_SLOT_ORDER=listed restores the written order to reproduce earlier results.
Battles in the two modes are different configurations: get_policy() tags shuffled
battles so they never share a matchup-database cell with listed-order battles.
"""
import os
import random

try:
    from vgc_bench.src.teams import RandomTeamBuilder
except ImportError:          # matchup_db reads MODE and POLICY_TAG without vgc_bench on the path
    RandomTeamBuilder = object

MODE = os.environ.get("VGC_SLOT_ORDER", "shuffled")
if MODE not in ("shuffled", "listed"):
    raise ValueError(f"VGC_SLOT_ORDER must be shuffled or listed, not {MODE!r}")
POLICY_TAG = "+slots-shuffled" if MODE == "shuffled" else ""


def shuffle_paste(paste, rng):
    """The same Showdown export with its Pokémon blocks in a random order."""
    blocks = [b.strip() for b in paste.strip().split("\n\n") if b.strip()]
    rng.shuffle(blocks)
    return "\n\n".join(blocks) + "\n"


class ShuffledTeamBuilder(RandomTeamBuilder):
    """RandomTeamBuilder that reorders the six Pokémon on every yield_team().

    Uses its own random.Random, so the shuffle never moves the global random state
    that picks opponents (common random numbers across candidates stay intact)."""

    def __init__(self, *args, order_seed=None, **kw):
        super().__init__(*args, **kw)
        self._order_rng = random.Random(order_seed)

    def _load_team(self, path):
        mons = self.parse_showdown_team(path.read_text())
        if MODE == "shuffled":
            self._order_rng.shuffle(mons)
        return self.join_team(mons)
