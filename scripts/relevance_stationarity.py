"""Non-stationary relevance: does WHICH variable matters depend on the starting point? Battle-free.

At every starting point (anchor), each variable gets a relevance profile entry = mean |Δf| of single edits to
that variable. Two independent edit sets per anchor give profiles A and B.
  within  : mean over anchors of Spearman(A_a, B_a)          -> agreement allowed by edit-to-edit spread + noise
  between : mean over anchor pairs a != b of Spearman(A_a, B_b) -> agreement across starting points
  gap     : between - within (< 0 means relevance changes with the starting point beyond what noise explains)
Bootstrap over anchors (2,000).
VGC: variables = the six edit types of the ruggedness runs (3 edits per type per anchor: A = edits 1-2, B = edit 3).
Benchmarks: variables = coordinates; 24 random anchors; two independent edits per coordinate.
Run: /tmp/vgc-pilot/.venv/bin/python scripts/relevance_stationarity.py
"""
import importlib.util, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; RES = HERE.parent / "results"
def load(name, f):
    s = importlib.util.spec_from_file_location(name, HERE / f); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
BF = load("bf", "bench_features.py")
TASKS = ["contamination", "ising", "pest_combo", "pest_bounce", "maxsat60", "labs50", "ackley53"]
OPS = ["A_move_swap", "B_ability_swap", "C_item_swap", "D_alignment_swap", "E_spread_copy", "F_candidate_copy"]

def ranks(v):
    v = np.asarray(v, float); o = np.argsort(v, kind="mergesort"); r = np.empty(len(v)); r[o] = np.arange(len(v))
    for x in np.unique(v):
        m = v == x
        if m.sum() > 1: r[m] = r[m].mean()
    return r
def spear(a, b):
    ra, rb = ranks(a), ranks(b)
    if ra.std() == 0 or rb.std() == 0: return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])

def stats(A, B, rng):
    n = len(A)
    def calc(idx):
        w = np.nanmean([spear(A[i], B[i]) for i in idx])
        bt = np.nanmean([spear(A[i], B[j]) for i in idx for j in idx if i != j])
        return w, bt
    w, bt = calc(range(n))
    boot = [np.subtract(*calc(rng.integers(0, n, n))[::-1]) for _ in range(300)]
    return dict(anchors=n, variables=len(A[0]), within=w, between=bt, gap=bt - w,
                gap_ci95=[float(x) for x in np.nanpercentile(boot, [2.5, 97.5])])

def bench(name, rng):
    f, card, noisy = BF.TASKS[name]()
    sim = iter(range(10 ** 7))
    ev = lambda x: f(x, np.random.default_rng(700_000 + next(sim)))
    A, B = [], []
    for _ in range(24):
        x = BF.rand_point(card, rng); f0 = ev(x); pa, pb = [], []
        for j in range(len(card)):
            d = []
            for _ in range(2):
                y = x.copy()
                y[j] = rng.random() if card[j] == 0 else rng.choice([v for v in range(card[j]) if v != x[j]])
                d.append(abs(ev(y) - f0))
            pa.append(d[0]); pb.append(d[1])
        A.append(pa); B.append(pb)
    return stats(A, B, rng)

def vgc(rng):
    R2 = load("rug2", "ruggedness2.py"); V = R2.vgc_anchors(); out = {}
    for start, anchors in V.items():
        A, B = [], []
        for a in anchors:
            pa, pb = [], []
            for op in OPS:
                ds = [abs(e["f"] - a["f"]) for e in a["edits"] if e["op"] == op]
                if len(ds) < 3: break
                pa.append(np.mean(ds[:2])); pb.append(ds[2])
            else:
                A.append(pa); B.append(pb)
        out[start] = stats(A, B, rng)
    return out

if __name__ == "__main__":
    rng = np.random.default_rng(3)
    res = {"vgc": vgc(rng), "tasks": {}}
    for t in TASKS:
        res["tasks"][t] = bench(t, rng); print(t, {k: round(v, 3) if isinstance(v, float) else v for k, v in res["tasks"][t].items()}, flush=True)
    for k, v in res["vgc"].items(): print("vgc", k, {kk: round(vv, 3) if isinstance(vv, float) else vv for kk, vv in v.items()})
    json.dump(res, open(RES / "relevance_stationarity.json", "w"), indent=1)
