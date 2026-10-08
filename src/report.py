"""Aggregate the pilot's seeds into the table that answers the experiment."""
import json, glob
import numpy as np

files = sorted(glob.glob("/tmp/vgc-pilot/results_s*.json"))
R = [json.load(open(f)) for f in files]
print(f"seeds: {len(R)}  ({', '.join(f.split('_')[-1][:-5] for f in files)})")
r0 = R[0]
print(f"model: {r0['params']:,} params, {r0['epochs']} epochs, "
      f"{r0['n_train']} train / {r0['n_heldout']} held-out teams, "
      f"{r0['train_time_s']}s per seed on {r0['device']}")
h = r0["loss_history"]
print(f"loss: train {h[0]['train']:.2f} -> {h[-1]['train']:.2f}, "
      f"held-out {h[0]['heldout']:.2f} -> {h[-1]['heldout']:.2f}")

def agg(path, key):
    vals = []
    for r in R:
        d = r
        for p in path: d = d[p]
        vals.append(d[key] if key in d else d["memorization"][key])
    return np.mean(vals), np.std(vals)

for regime in ["scatter", "slot"]:
    arms = list(R[0]["regimes"][regime].keys())
    print(f"\n=== regime: {regime} ===")
    print(f"  {'arm':10s} {'legal':>14s} {'novel':>8s} {'legal&novel':>14s} {'unique':>8s}")
    for a in arms:
        l = agg(["regimes", regime, a], "legal")
        n = agg(["regimes", regime, a], "novel")
        ln = agg(["regimes", regime, a], "legal_and_novel")
        u = agg(["regimes", regime, a], "unique")
        print(f"  {a:10s} {l[0]:.3f} +/- {l[1]:.3f} {n[0]:8.3f} {ln[0]:8.3f} +/- {ln[1]:.3f} {u[0]:8.3f}")
    print("  top violations (seed 0):")
    for a in arms:
        print(f"    {a:10s} {R[0]['regimes'][regime][a]['violations']}")

print("\n=== unconditional (all 48 fields sampled from the model) ===")
for k in ["legal", "novel", "unique"]:
    m, s = agg(["unconditional"], k)
    print(f"  {k:8s} {m:.3f} +/- {s:.3f}")
for k in ["memorized", "exact_copy", "mean_nn_dist"]:
    m, s = agg(["unconditional"], k)
    print(f"  {k:12s} {m:.3f} +/- {s:.3f}")
print("  violations (seed 0):", R[0]["unconditional"]["violations"])
