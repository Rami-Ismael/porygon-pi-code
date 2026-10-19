"""Diversity interventions for DiBO; no imports or mutations of historical loops.

Pure selection/state helpers are also used by the battle-free regression tests.
"""
from collections import Counter
import math
import numpy as np
from boltzmann_selection import boltz_weights, selection_temperature
from temperature_experiment import paste_species

ARMS = ('control', 'boltz', 'fixed_010', 'fixed_020', 'fixed_030', 'adaptive',
        'niche', 'refit', 'mmr30', 'mmr60', 'taboo', 'peel', 'island', 'big',
        'dpp', 'negative', 'random', 'no_guidance')


def composition(row):
    return paste_species(row['paste'])


def distinct(rows, seen=()):
    used = set(seen)
    result = []
    for row in rows:
        if row['key'] not in used:
            result.append(row)
            used.add(row['key'])
    return result


def mmr(rows, k, penalty, score='y'):
    """Greedy quality minus maximum shared-species/6 redundancy."""
    quality = np.array([r[score] for r in rows], dtype=float)
    species = [set(composition(r)) for r in rows]
    redundancy = np.zeros(len(rows))
    picked = []
    for _ in range(min(k, len(rows))):
        values = quality - penalty * redundancy
        values[picked] = -np.inf
        j = int(values.argmax())
        picked.append(j)
        redundancy = np.maximum(redundancy, [len(s & species[j]) / 6 for s in species])
    return [rows[i] for i in picked]


def training_rows(rows, arm, budget, seed):
    rows = distinct(rows)
    if arm in ('mmr30', 'mmr60'):
        return mmr(rows, budget, .3 if arm == 'mmr30' else .6)
    if arm == 'random':
        ix = np.random.default_rng(seed).permutation(len(rows))[:budget]
        return [rows[i] for i in ix]
    return sorted(rows, key=lambda r: r['y'], reverse=arm != 'negative')[:budget]


def prior_weights(rows, arm, generation, rounds):
    """Temperature remains in measured win-rate units, never density units."""
    y = np.array([r['y'] for r in rows], float)
    if not len(y) or not np.isfinite(y).all():
        raise ValueError('finite observations required')
    info = {}
    if arm in ('boltz', 'fixed_010', 'fixed_020', 'fixed_030', 'adaptive', 'niche'):
        temp, info = selection_temperature('annealed' if arm in ('boltz', 'niche') else arm,
                                           generation + 1, y, generations=rounds)
        if arm == 'niche':
            best = {}
            for i, row in enumerate(rows):
                c = composition(row)
                if c not in best or y[i] > y[best[c]]:
                    best[c] = i
            reps = sorted(best.values())
            w = np.zeros(len(y)); w[reps] = boltz_weights(y[reps], temp)
        else:
            w = boltz_weights(y, temp)
        info = dict(info, temperature=temp)
    elif arm == 'random':
        w = np.full(len(y), 1 / len(y))
    else:
        scale = max(float(y.std(ddof=1)), 1e-7) if len(y) > 1 else 1e-7
        w = boltz_weights(-y if arm == 'negative' else y, scale)
    p = w[w > 0]
    return w, dict(info, ess=float(1 / (w @ w)), nonzero=int(len(p)),
                   entropy=float(-(p * np.log(p)).sum()))


class Taboo:
    def __init__(self, previous=(), cap=2):
        self.previous = set(previous)
        self.counts = Counter()
        self.cap = cap

    def accept(self, row):
        c = composition(row)
        if c in self.previous or self.counts[c] >= self.cap:
            return False
        self.counts[c] += 1
        return True


def peel_bans(rounds, generation, count=3):
    """At g1/3/5/... peel top-16 species from the preceding two-round epoch."""
    banned = []
    for start in range(1, generation + 1, 2):
        source = [r for g in range(max(0, start - 2), start)
                  for r in rounds.get(str(g), {}).get('evaluated', [])]
        frequency = Counter(s for r in sorted(source, key=lambda r: -r['y'])[:16]
                            for s in composition(r) if s not in banned)
        banned.extend(s for s, _ in sorted(frequency.items(), key=lambda kv: (-kv[1], kv[0]))[:count])
    return banned


def island_seeds(anchors, count):
    candidates = distinct(sorted(anchors, key=lambda r: -r['y']))
    if len(candidates) < count:
        raise ValueError('not enough distinct anchors for islands')
    selected = [candidates.pop(0)]
    def distance(a, b):
        a, b = set(composition(a)), set(composition(b))
        return 1 - len(a & b) / len(a | b)
    while len(selected) < count:
        row = max(candidates, key=lambda a: min(distance(a, b) for b in selected))
        selected.append(row); candidates.remove(row)
    return selected


def island_observations(initial, rounds, seeds, generation):
    """Reconstruct private histories and simultaneous migration from saved rounds."""
    homes = [[] for _ in seeds]
    for row in initial:
        j = max(range(len(seeds)), key=lambda i: len(set(composition(row)) & set(composition(seeds[i]))))
        homes[j].append(row)
    for i, seed in enumerate(seeds):
        homes[i] = distinct([seed] + homes[i])
    for g in range(generation + 1):
        if g >= 3 and g % 2 == 1:
            snapshot = [list(rows) for rows in homes]
            for i, home in enumerate(snapshot):
                common = {s for s, _ in Counter(s for r in home for s in composition(r)).most_common(6)}
                for j, other in enumerate(snapshot):
                    if i == j:
                        continue
                    choices = [r for r in other if r.get('src', '').startswith('round')
                               and len(set(composition(r)) & common) <= 2]
                    if choices:
                        homes[i] = distinct(homes[i] + [max(choices, key=lambda r: r['y'])])
        if g < generation:
            for row in rounds.get(str(g), {}).get('evaluated', []):
                homes[row['island']].append(row)
    return homes


def select_dpp(rows, k, tau=.05):
    """Species Gram DPP; quality is predicted win rate, not log density.

    Rank exhaustion is explicit; remaining slots fall back to the DiBO target.
    """
    species = [set(composition(r)) for r in rows]
    vocab = {s: i for i, s in enumerate(sorted(set().union(*species)))}
    phi = np.zeros((len(rows), len(vocab)))
    mu = np.array([r['mu'] for r in rows])
    for i, sp in enumerate(species):
        phi[i, [vocab[s] for s in sp]] = np.exp((mu[i] - mu.max()) / tau) / math.sqrt(6)
    residual = (phi * phi).sum(1)
    factors = np.zeros((min(k, len(rows)), len(rows)))
    picked = []
    while len(picked) < min(k, len(rows)):
        j = int(residual.argmax())
        if residual[j] <= 1e-10:
            break
        i = len(picked)
        factors[i] = (phi @ phi[j] - factors[:i].T @ factors[:i, j]) / math.sqrt(residual[j])
        picked.append(j)
        residual -= factors[i] ** 2
        residual[picked] = -np.inf
    rank = len(picked)
    used = set(picked)
    picked += sorted((i for i in range(len(rows)) if i not in used),
                     key=lambda i: -rows[i]['log_target'])[:k - rank]
    return [rows[i] for i in picked], rank


def metrics(rows):
    counts = Counter(composition(r) for r in rows)
    n = len(rows)
    p = np.array(list(counts.values()), float) / n if n else np.array([])
    return dict(n=n, distinct_teams=len({r['key'] for r in rows}),
                compositions=len(counts), effective_compositions=float(np.exp(-(p * np.log(p)).sum())) if n else 0,
                largest_share=float(p.max()) if n else None)
