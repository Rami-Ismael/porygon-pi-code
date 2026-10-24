"""The Mask Lab's paired statistics: a difference per mask, the starting teams weighted equally, and a 95% bootstrap
over the starting teams (4,000 resamples, seed 20261008), as in mask_lab_textdiff.py report and scripts/text_vs_ling.py."""
import random

SEED = 20261008


def cluster(pairs):
    """The mean of (starting team, value) pairs with each starting team weighted equally, and its 95% bootstrap
    interval over the starting teams; None when there are no pairs."""
    by = {}
    for s, v in pairs: by.setdefault(s, []).append(v)
    if not by: return None
    means = [sum(v) / len(v) for v in by.values()]
    rng = random.Random(SEED); draws = sorted(sum(rng.choice(means) for _ in means) / len(means) for _ in range(4000))
    return dict(mean=sum(means) / len(means), lo=draws[100], hi=draws[-100], n=sum(len(v) for v in by.values()))


def fmt(c):
    """A cluster() result in percentage points, "+1.2 [-0.3, +2.6]", or a dash for no pairs."""
    return "—" if not c else f"{100*c['mean']:+.1f} [{100*c['lo']:+.1f}, {100*c['hi']:+.1f}]"
