"""Does more training data lower the asked-vs-got diffusion loss? Battle-free check (2026-10-04).

Same model, recipe and held-out 500 as asked_vs_got.train; the labelled training teams are
subsampled to 12.5 / 25 / 50 / 100%, the 692 real corpus teams (null condition) always kept.
Every arm gets the SAME number of gradient steps as the full run (smaller sets see more epochs,
so overfitting shows up if it exists). Scored by plain cross-entropy on the held-out teams.

    python data_scaling.py        (run with /tmp/vgc-pilot/.venv/bin/python)
"""
import json, sys, time
import numpy as np, torch

sys.path.insert(0, "/tmp/vgc-pilot/src")
sys.path.insert(0, "/Users/ramiismael/Documents/code/vgc-team-generator-pilot/src")
import asked_vs_got as X
import loss_diagnosis as L
import activesearch as A
import diffusion as D

OUT = X.RES / "data_scaling.json"
FRACS = [0.125, 0.25, 0.5, 1.0]
BATCH = 128

def main():
    V, xtr, wtr, xte, wte, Xc = L.load_split()
    steps = X.EPOCHS * int(np.ceil((len(xtr) + len(Xc)) / BATCH))
    out = json.load(open(OUT)) if OUT.exists() else {}
    for f in FRACS:
        key = f"{f:g}"
        if key in out: continue
        torch.manual_seed(X.SEED); rng = np.random.default_rng(X.SEED)
        keep = rng.permutation(len(xtr))[:int(round(f * len(xtr)))]
        xs = torch.tensor(np.concatenate([xtr[keep], Xc]), device=D.DEV)
        ws = torch.tensor(np.concatenate([wtr[keep], np.full(len(Xc), np.nan, np.float32)]), device=D.DEV)
        m = X.ContinuousWR(V).to(D.DEV)
        m.load_state_dict(torch.load(X.P0, map_location=D.DEV)["sd"], strict=False)
        ep = max(1, round(steps / int(np.ceil(len(xs) / BATCH))))
        opt = torch.optim.AdamW(m.parameters(), lr=2e-4, weight_decay=.01)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ep)
        t0 = time.perf_counter()
        for _ in range(ep):
            m.train(); A._epoch(m, opt, xs, ws, BATCH); sch.step()
        m.eval()
        sub = np.random.default_rng(1).choice(len(keep), min(2000, len(keep)), replace=False)
        ho = L.ce_at(m, xte, wte, "random", reps=8)
        tr = L.ce_at(m, xtr[keep][sub], wtr[keep][sub], "random", reps=4)
        out[key] = dict(labelled=len(keep), epochs=ep, heldout_ce=ho["all"], train_ce=tr["all"],
                        heldout_elbo=ho["elbo_weighted"], heldout_by_type={k: ho[k] for k in L.TYPES},
                        minutes=(time.perf_counter() - t0) / 60)
        json.dump(out, open(OUT, "w"), indent=1)
        print(f"  {f:6.3f} ({len(keep):5d} teams, {ep:3d} epochs): held-out CE {ho['all']:.3f} · train CE {tr['all']:.3f} "
              f"· gap {ho['all'] - tr['all']:+.3f} · reported loss {ho['elbo_weighted']:.3f} "
              f"({(time.perf_counter()-t0)/60:.1f} min)", flush=True)

if __name__ == "__main__":
    main()
