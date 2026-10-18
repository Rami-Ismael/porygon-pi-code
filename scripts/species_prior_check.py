"""Is the generation-0 species prior uniform? Dataset frequency vs model marginal vs sampled frequency (2026-10-17)."""
import json, math, sys
from collections import Counter
import numpy as np, torch
sys.path.insert(0, "/tmp/vgc-pilot/src")
import lossdown as L, lossdown_battle as LB, sampler_battle as SB

SB._setup(); V, tables, f = SB.V, SB.tables, SB.f
names = V.itos["species"]; SP = L.COLS_BY_KEY["species"]
GAR = next(i for i, n in enumerate(names) if n.lower().replace(" ", "") == "garchomp"); print("vocab sample", names[1:6], "garchomp idx", GAR)
d = L.load_data()
X = d["X"]; print("rows", len(X), "keys", list(d.keys()))
def stats(cnt, label, n_slots):
    tot = sum(cnt.values()); p = np.array(sorted(cnt.values(), reverse=True)) / tot
    H = -(p * np.log(p)).sum()
    top = cnt.most_common(12)
    g = [i for i, (k, v) in enumerate(cnt.most_common()) if k == GAR]
    print(f"\n{label}: slots {n_slots:,}  distinct species {len(cnt)}  exp(H) {math.exp(H):.1f}  top share {p[0]:.3f}  "
          f"top-6 share {p[:6].sum():.3f}  Garchomp rank {g[0]+1 if g else 'absent'} share {cnt[GAR]/tot:.3f}")
    print("  " + ", ".join(f"{names[k]} {v/tot:.3f}" for k, v in top))
    return {names[k]: v / tot for k, v in cnt.most_common()}
out = {}
# 1) dataset, all rows and real rows only
cnt_all = Counter(int(v) for v in X[:, SP].ravel() if v != 0)
out["dataset_all"] = stats(cnt_all, "DATASET all 147k rows", int((X[:, SP] != 0).sum()))
if "S" in d:
    S = d["S"]; fams = L.FAM
    u, c = np.unique(S.astype(str), return_counts=True); print("  S dtype", S.dtype, "values", dict(zip(u.tolist()[:12], c.tolist()[:12])))
    key = "real" if "real" in u else str(fams.index("real"))
    real = X[S.astype(str) == key]
    if len(real):
        cnt_real = Counter(int(v) for v in real[:, SP].ravel() if v != 0)
        out["dataset_real"] = stats(cnt_real, f"DATASET real rows only ({len(real)} teams)", int((real[:, SP] != 0).sum()))
# 2) model marginal at the all-masked state, slot 0
with torch.no_grad():
    x = torch.zeros(1, L.COLS, dtype=torch.long, device=L.DEV); t = torch.ones(1, device=L.DEV)
    lu = f(x, t, torch.full((1,), float("nan"), device=L.DEV))["species"][0, 0]
    lc = f(x, t, torch.full((1,), 0.5, device=L.DEV))["species"][0, 0]
    for label, lg in (("MODEL marginal, all masked, unconditional", lu), ("MODEL marginal, all masked, ask 0.5 guidance 2", 2 * lc - lu)):
        lg = lg.clone(); lg[0] = -1e9
        p = torch.softmax(lg, -1).cpu().numpy()
        cnt = Counter({i: float(p[i]) for i in range(len(p)) if p[i] > 1e-6})
        out[label] = stats(cnt, label, 1)
# 3) sampled teams, today's sampler, 512 each
for label, (ask, g) in {"SAMPLED 512 teams, unconditional": (None, 1.0), "SAMPLED 512 teams, ask 0.5 guidance 2": (0.5, 2.0)}.items():
    torch.manual_seed(1)
    rows = torch.cat([LB.sample(f, tables, 128, ask, g) for _ in range(4)]).cpu().numpy()
    cnt = Counter(int(v) for v in rows[:, SP].ravel())
    out[label] = stats(cnt, label, rows.shape[0] * 6)
    out[label + " line-ups"] = len({tuple(sorted(r[SP])) for r in rows}); print(f"  distinct line-ups {out[label + ' line-ups']}/512")
json.dump(out, open("/Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/species_prior_check.json", "w"), indent=1)
