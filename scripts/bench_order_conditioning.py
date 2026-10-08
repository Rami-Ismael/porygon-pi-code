"""Higher-order interaction and ill-conditioning of the collected BO benchmarks, measured with the
same designs as the VGC side, for Version 3 of the vault's benchmark comparison note.

  interaction order : the 2^4 Walsh cube of scripts/interaction_order.py (4 factors = 4 random coordinates
                      each set to a fixed other value; 24 random + 24 hill-climbed anchors; 2 replicates,
                      one simulation seed shared by the 16 points of a replicate; cross-replicate product
                      estimator; shares of order 1 / 2 / 3+, bootstrap 95% CI over anchors)
  conditioning      : per-variable noise-corrected effect size d*_j = sqrt(E[d_j^2] - E[r^2]) from single
                      edits of variable j at 24 random anchors (noisy tasks: 2 replicates for E[r^2]);
                      reported as largest / median d*_j and the share of variables with d*_j = 0.
                      VGC's analogue (results/ruggedness2.json) is per edit type, not per field.

No battles. Molecule tasks are skipped: no fixed coordinates for either design.
Run: /tmp/vgc-pilot/.venv/bin/python scripts/bench_order_conditioning.py
"""
import importlib.util, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
spec = importlib.util.spec_from_file_location("bf", HERE / "bench_features.py"); BF = importlib.util.module_from_spec(spec); spec.loader.exec_module(BF)
TASKS = ["contamination", "ising", "pest_combo", "pest_bounce", "maxsat60", "labs50", "ackley53"]
K, N_ANCH, CLIMB, BOOT = 4, 24, 100, 2000
CUBE = np.array([[(m >> i) & 1 for i in range(K)] for m in range(2 ** K)])
SUBSETS = [tuple(i for i in range(K) if (s >> i) & 1) for s in range(2 ** K)]
CHI = np.array([[np.prod([2 * b[i] - 1 for i in S]) if S else 1 for b in CUBE] for S in SUBSETS])
ORDER = np.array([len(S) for S in SUBSETS])

def other_value(x, j, card, rng):
    return rng.random() if card[j] == 0 else rng.choice([v for v in range(card[j]) if v != x[j]])

def shares(cubes):
    p = np.sum([(CHI @ np.asarray(c[0]) / 16) * (CHI @ np.asarray(c[1]) / 16) for c in cubes], axis=0)
    tot = p[ORDER > 0].sum()
    return [p[ORDER == 1].sum() / tot, p[ORDER == 2].sum() / tot, p[ORDER >= 3].sum() / tot] if tot > 0 else [np.nan] * 3

def run(name):
    f, card, noisy = BF.TASKS[name]()
    rng = np.random.default_rng(11); seeds = iter(range(10 ** 7))
    ev = lambda x, s=None: f(x, np.random.default_rng(s if s is not None else 900_000 + next(seeds)))
    out = {}
    # ---- interaction order
    for start in ("random", "search"):
        cubes = []
        for _ in range(N_ANCH):
            x = BF.rand_point(card, rng)
            if start == "search":
                fx = ev(x)
                for _ in range(CLIMB):
                    y = BF.edit(x, card, rng); fy = ev(y)
                    if fy < fx: x, fx = y, fy
            coords = rng.choice(len(card), K, replace=False); vals = [other_value(x, j, card, rng) for j in coords]
            reps = []
            for _ in range(2):
                s = next(seeds)
                ys = []
                for b in CUBE:
                    y = x.copy()
                    for j, v, on in zip(coords, vals, b):
                        if on: y[j] = v
                    ys.append(ev(y, s))
                reps.append(ys)
            cubes.append(reps)
        sh = shares(cubes)
        bs = np.array([shares([cubes[i] for i in rng.integers(0, len(cubes), len(cubes))]) for _ in range(BOOT)])
        out[f"order_{start}"] = dict(o1=sh[0], o2=sh[1], o3plus=sh[2],
                                     o3plus_ci95=np.nanpercentile(bs[:, 2], [2.5, 97.5]).tolist())
    # ---- conditioning: per-variable effect size at random anchors
    D = len(card); d2 = np.zeros(D); r2 = []
    for _ in range(N_ANCH):
        x = BF.rand_point(card, rng); f0 = ev(x)
        if noisy: r2.append((ev(x) - f0) ** 2)
        for j in range(D):
            y = x.copy(); y[j] = other_value(x, j, card, rng); d2[j] += (ev(y) - f0) ** 2
    d2 /= N_ANCH; noise = float(np.mean(r2)) if r2 else 0.0
    dstar = np.sqrt(np.maximum(d2 - noise, 0.0))
    med = float(np.median(dstar))
    out["conditioning"] = dict(variables=D, largest_over_median=float(dstar.max() / med) if med > 0 else None,
                               share_zero=float(np.mean(dstar <= 1e-12)), dstar=dstar.tolist(), noise_var=noise)
    print(name, {k: (v if k != "conditioning" else {kk: vv for kk, vv in v.items() if kk != "dstar"}) for k, v in out.items()}, flush=True)
    return out

def vgc():
    r2 = json.load(open(RES / "ruggedness2.json"))["vgc"]
    io = json.load(open(RES / "interaction_order.json"))["vgc"]["primary"]
    cond = {}
    for start in ("meta", "search"):
        ds = {op: v["delta_star"] for op, v in r2[start].items() if isinstance(v, dict) and "delta_star" in v}
        med = float(np.median(list(ds.values())))
        cond[start] = dict(per_edit_type=ds, largest_over_median=max(ds.values()) / med if med > 0 else None,
                           share_zero=float(np.mean([v <= 1e-12 for v in ds.values()])),
                           beyond_noise=[op for op, v in r2[start].items() if isinstance(v, dict) and v.get("T1")])
    return dict(order=dict(o1=io["o1"]["est"], o2=io["o2"]["est"], o3plus=io["o3plus"]["est"], o3plus_ci95=io["o3plus"]["ci95"]),
                conditioning=cond)

if __name__ == "__main__":
    res = {"design": dict(k=K, anchors=N_ANCH, climb=CLIMB, boot=BOOT), "tasks": {}, "vgc": vgc()}
    for t in TASKS: res["tasks"][t] = run(t)
    json.dump(res, open(RES / "bench_order_conditioning.json", "w"), indent=1)
    print(json.dumps(res["vgc"], indent=1))
