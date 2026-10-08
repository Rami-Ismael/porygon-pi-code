"""Does the DISTRIBUTION of labelled teams matter more than their number?

Owner's hypothesis (2026-08-27): another session saw win rate grow linearly with
data; the flat curve here came from how the pool was sampled. The audit agrees:
39% of the labelling budget went to teams under 0.2, which carry no information
about winning.

Arms (d=96 for speed, 200 teams evaluated at the top bin, guidance 4):
  uniform_f25/f50/f100 x seeds 1,2   -> replicate the data curve (seed 0 exists in scale_results.json)
  cut02  -> all teams with win rate >= 0.2  (~2,970 teams)
  cut04  -> all teams with win rate >= 0.4  (~1,570 teams)
If cut02/cut04 match or beat uniform_f100 with fewer labels, the mixture is the limit.
"""
import json, os, sys, time, random
import numpy as np, torch
sys.path.insert(0, "/tmp/vgc-pilot/src")
import diffusion as D, wrdiffusion as W, scale as SC
from encode import NF, NSLOT

OUT = "/tmp/vgc-pilot/sampling_results.json"
ARMS = [
    # equal SIZE (the winning half's size), different DISTRIBUTIONS
    ("half_uniform_s0",  "uniform", 0.5, 0),     # the exact half that beat the full set
    ("half_uniform_s1",  "uniform", 0.5, 1),     # another random half
    ("half_top",         "top",     0.5, 0),     # the highest-win-rate half
    ("half_bottom",      "bottom",  0.5, 0),     # the lowest-win-rate half (control)
    # full set replicate, and the same size with the bottom removed
    ("full_s1",          "uniform", 1.0, 1),
    ("cut02",            "cut", 0.2, 0),         # all teams with win rate >= 0.2
    ("cut04",            "cut", 0.4, 0),         # all teams with win rate >= 0.4
]

def train_arm(tag, mode, val, seed, d=96, epochs=300):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    V, L, corpus, teams, wrs = W.build_vocab()
    X = np.stack([V.encode(t) for t in teams]); Y = np.array([D.style_of(t) for t in teams]); Wb = np.array([W.wr_bin(w) for w in wrs]); wr = np.array(wrs)
    rng = np.random.default_rng(seed); idx = rng.permutation(len(X)); nte = int(len(X) * .1)
    te = idx[:nte]; tr = idx[nte:]
    if mode == "uniform": tr = tr[: int(len(tr) * val)]
    elif mode == "cut":   tr = tr[wr[tr] >= val]
    elif mode == "top":   tr = tr[np.argsort(-wr[tr])][: int(len(tr) * val)]
    elif mode == "bottom": tr = tr[np.argsort(wr[tr])][: int(len(tr) * val)]
    m = W.TeamDiffusionWR(V, d=d, nhead=4).to(W.DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=.01); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    T = lambda a, i: torch.tensor(a[i], device=W.DEV)
    xtr, ytr, wtr = T(X, tr), T(Y, tr), T(Wb, tr); xte, yte, wte = T(X, te), T(Y, te), T(Wb, te)
    print(f"[{tag}] train {len(tr)} ({mode} {val}, seed {seed}) / held-out {len(te)} · mean label {float(wr[tr].mean()):.3f} · top-bin {int((wr[tr]>=.5).sum())}", flush=True)
    t0 = time.perf_counter()
    for ep in range(epochs):
        m.train(); perm = torch.randperm(xtr.shape[0], device=W.DEV)
        for i in range(0, xtr.shape[0], 64):
            b = perm[i:i + 64]; xb = xtr[b].view(-1, NSLOT, NF)
            xb = torch.stack([r[torch.randperm(NSLOT)] for r in xb]).view(-1, W.COLS)
            loss = m.loss(xb, ytr[b], wtr[b]); opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.); opt.step()
        sch.step()
    m.eval()
    with torch.no_grad(): vl = float(np.mean([float(m.loss(xte, yte, wte)) for _ in range(8)]))
    return m, V, L, corpus, teams, len(tr), vl, time.perf_counter() - t0

def main():
    done = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for tag, mode, val, seed in ARMS:
        if tag in done: continue
        m, V, L, corpus, teams, ntr, vl, secs = train_arm(tag, mode, val, seed)
        r = SC.eval_arm(tag, m, V, L, corpus, teams)
        r.update(dict(mode=mode, val=val, seed=seed, n_train=ntr, heldout_loss=vl, train_min=secs / 60))
        done[tag] = r; json.dump(done, open(OUT, "w"), indent=1)
        print(f"[{tag}] DONE  n_train {ntr}  valid {r['valid']:.0%}  win rate {r['win_rate']:.4f} ± {r['se']:.4f}  held-out {vl:.3f}", flush=True)
    print("ALL_DONE", flush=True)

if __name__ == "__main__":
    main()
