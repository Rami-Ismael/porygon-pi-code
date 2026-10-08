"""Does balancing the win-rate brackets, or growing the top bracket, help the
win-rate-conditioned generator?  (Owner plan 2026-08-28, items 4.1.2 "balance the
number of teams per bracket" and 7.1.2 "bigger dataset, balanced".)

Every arm fine-tunes the SAME shipped checkpoint (wrdiffusion.pt, d=192) for the SAME
number of optimiser steps (filterrate.py's recipe: 4,560 steps, lr 1e-4), then decodes 150 Showdown-valid teams with the best
known scheme (dependency order, temp 0.7, top-p 0.9, guidance 2, top bin) and battles
each 24 times against the top-50 meta.  Only the training pool, the bin edges and the
per-example sampling weights differ:

  base            4,850 labelled teams · 6 bins (top = [0.5, 1])   · uniform sampling
  balanced6       same pool · 6 bins · each bin sampled equally often (DDOM-style reweighting)
  base_fine       same pool · 8 bins (top = [0.7, 1])              · uniform
  grown           4,850 + 1,600 copy-paste proposals (cp_labels.json) · 6 bins · uniform
  grown_fine      grown pool · 8 bins (top = [0.7, 1])             · uniform
  grown_fine_bal  grown pool · 8 bins · balanced

base_fine vs grown_fine separates "a finer top bracket" from "more teams in it";
balanced6 vs base isolates the reweighting the owner asked for.  Results go to
/tmp/vgc-pilot/brackets_results.json (resumable by arm); generated teams to
/tmp/vgc-pilot/brackets_gen/<arm>/.

    python brackets.py --arms base,balanced6,base_fine
    python brackets.py --arms grown,grown_fine,grown_fine_bal   # needs cp_labels.json complete
"""
import argparse, json, math, os, random, sys, time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/tmp/vgc-pilot/src")
import diffusion as D
import wrdiffusion as W
import schemes as S
from corpus import norm, load_corpus
from encode import Vocab, Legality, NF, NSLOT
from loop import PMI, opponents
from propose import Validator
from wrdiffusion import parse_team

DEV = W.DEV
OUT = "/tmp/vgc-pilot/brackets_results.json"
GEN = Path("/tmp/vgc-pilot/brackets_gen")
CP = "/tmp/vgc-pilot/cp_labels.json"
STEPS = 60 * math.ceil(4850 / 64)                # = filterrate.py's budget (4,560 steps, ~10 min)
BATCH, LR = 64, 1e-4                             # filterrate.py's fine-tuning recipe
INIT = "/tmp/vgc-pilot/wrdiffusion.pt"            # every arm starts from the shipped checkpoint

EDGES6 = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.01]
EDGES8 = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.01]

ARMS = {
    "base":           dict(pool="base",  edges=EDGES6, balanced=False),
    "balanced6":      dict(pool="base",  edges=EDGES6, balanced=True),
    "base_fine":      dict(pool="base",  edges=EDGES8, balanced=False),
    "grown":          dict(pool="grown", edges=EDGES6, balanced=False),
    "grown_fine":     dict(pool="grown", edges=EDGES8, balanced=False),
    "grown_fine_bal": dict(pool="grown", edges=EDGES8, balanced=True),
}


def set_bins(edges):
    """The bin table lives in module globals that the model, wr_bin and the guided
    decoder all read at call time, so switching it here switches it everywhere."""
    W.EDGES = list(edges); W.NBIN = len(edges) - 1; W.WNULL = W.NBIN


def load_pool(kind):
    lab = {}
    for f in ["/tmp/vgc-pilot/ladder_labels.json", "/tmp/vgc-pilot/strata_labels.json"]:
        for p, v in json.load(open(f)).items(): lab[p] = v["win_rate"]
    n_base = len(lab)
    if kind == "grown":
        cp = json.load(open(CP))
        for p, v in cp.items(): lab[p] = v["win_rate"]
    teams, wrs = [], []
    for p, w in lab.items():
        t = parse_team(p)
        if t and len(t) == 6: teams.append(t); wrs.append(w)
    return teams, np.array(wrs), n_base


def train(V, teams, wrs, balanced, seed=0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    X = np.stack([V.encode(t) for t in teams])
    Y = np.array([D.style_of(t) for t in teams])
    B = np.array([W.wr_bin(w) for w in wrs])
    rng = np.random.default_rng(seed); idx = rng.permutation(len(X)); nte = int(len(X) * .1)
    tr, te = idx[nte:], idx[:nte]
    counts = np.bincount(B[tr], minlength=W.NBIN)
    if balanced:
        wgt = 1.0 / counts[B[tr]]                     # every bin gets the same share of draws
    else:
        wgt = np.ones(len(tr))
    wgt = torch.tensor(wgt / wgt.sum(), device=DEV, dtype=torch.float)
    m = W.TeamDiffusionWR(V).to(DEV)
    # filterrate.py's protocol: fine-tune the shipped checkpoint. The win-rate embedding
    # is re-initialised when the bin count differs (8-bin arms); everything else loads.
    ck = torch.load(INIT, map_location=DEV)
    sd = {k: v for k, v in ck["sd"].items() if not k.startswith("wr.")}
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if any(not k.startswith("wr.") for k in missing) or unexpected:
        raise SystemExit(f"checkpoint mismatch: missing {missing[:3]} unexpected {unexpected[:3]}")
    if W.NBIN == 6: m.wr.weight.data.copy_(ck["sd"]["wr.weight"])
    print(f"    initialised from {INIT} (wr embedding {'kept' if W.NBIN == 6 else 'reset'})", flush=True)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    xtr, ytr, btr = (torch.tensor(a[tr], device=DEV) for a in (X, Y, B))
    xte, yte, bte = (torch.tensor(a[te], device=DEV) for a in (X, Y, B))
    print(f"    train {len(tr)} / held-out {len(te)} · bins {counts.tolist()} · "
          f"balanced={balanced} · {STEPS} steps", flush=True)
    t0 = time.perf_counter(); tot = nb = 0
    for step in range(STEPS):
        m.train()
        b = torch.multinomial(wgt, BATCH, replacement=True)
        xb = xtr[b].view(-1, NSLOT, NF)
        xb = torch.stack([r[torch.randperm(NSLOT)] for r in xb]).view(-1, W.COLS)
        loss = m.loss(xb, ytr[b], btr[b])
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(), 1.); opt.step(); sch.step()
        tot += float(loss); nb += 1
        if (step + 1) % 2000 == 0 or step == 0:
            m.eval()
            with torch.no_grad():
                vl = float(np.mean([float(m.loss(xte, yte, bte)) for _ in range(6)]))
            print(f"    step {step+1:6d} train {tot/nb:7.2f}  held-out {vl:7.2f}  "
                  f"{(time.perf_counter()-t0)/60:.1f} min", flush=True)
            tot = nb = 0
    m.eval()
    with torch.no_grad():
        vl = float(np.mean([float(m.loss(xte, yte, bte)) for _ in range(10)]))
    return m, dict(train_n=int(len(tr)), heldout_n=int(len(te)), bins_train=counts.tolist(),
                   heldout_loss=vl, train_minutes=(time.perf_counter() - t0) / 60)


def make_ctx(V, L, model, corpus, labelled):
    C = D.Constraints(V, L); A = S.Arc(C, V)
    look, spreads = {}, defaultdict(list)
    for t in corpus + labelled:
        for s in t:
            look[norm(s["species"])] = s["species"]; look[norm(s["ability"])] = s["ability"]
            if s["item"]: look[norm(s["item"])] = s["item"]
            for mv in s["moves"]: look[norm(mv)] = mv
            if s["nature"]: look[norm(s["nature"])] = s["nature"]
            spreads[norm(s["species"])].append(dict(s["evs"]))
    return dict(V=V, L=L, C=C, A=A, model=model, look=look, spreads=spreads, corpus=corpus,
                pmi=PMI(corpus), opp=opponents(), val=Validator(),
                Xc=np.stack([V.encode(t) for t in corpus]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--per", type=int, default=150)
    ap.add_argument("--battles", type=int, default=24)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    global STEPS
    if a.smoke: STEPS = 30
    per, battles = (3, 2) if a.smoke else (a.per, a.battles)
    outfile = "/tmp/vgc-pilot/brackets_smoke.json" if a.smoke else OUT
    res = json.load(open(outfile)) if os.path.exists(outfile) else {}
    corpus, _ = load_corpus(); corpus = [t for t in corpus if len(t) == 6]
    for name in a.arms.split(","):
        if name in res: print(f"[{name}] already done, skipping", flush=True); continue
        cfg = ARMS[name]
        if cfg["pool"] == "grown":
            while not os.path.exists(CP) or len(json.load(open(CP))) < 1600:
                print(f"[{name}] waiting for {CP} to reach 1600 labels", flush=True); time.sleep(60)
        set_bins(cfg["edges"])
        teams, wrs, n_base = load_pool(cfg["pool"])
        V = Vocab(corpus + teams); L = Legality(corpus)
        print(f"[{name}] pool {len(teams)} teams ({n_base} base) · {W.NBIN} bins · "
              f"top bin = [{W.EDGES[-2]}, 1]", flush=True)
        t0 = time.perf_counter()
        model, tinfo = train(V, teams, wrs, cfg["balanced"])
        torch.save({"sd": model.state_dict(), "itos": V.itos, "edges": W.EDGES},
                   f"/tmp/vgc-pilot/brackets_{name}.pt")
        ctx = make_ctx(V, L, model, corpus, teams)
        dcfg = dict(S.DEFAULTS); dcfg.update(order="dep", cmode="full", temp=0.7, top_p=0.9,
                                             guidance=2.0, wbin=W.NBIN - 1)
        S.GEN = GEN
        r = S.run_arm(name, dcfg, ctx, per, battles, max_attempts=6000, batch=48)
        ctx["val"].close()
        r.update(tinfo, pool=cfg["pool"], pool_n=len(teams), edges=W.EDGES,
                 balanced=cfg["balanced"], top_bin_edge=W.EDGES[-2],
                 minutes=(time.perf_counter() - t0) / 60)
        res[name] = r
        json.dump(res, open(outfile, "w"), indent=1)
        wr = f"{r['win_rate']:.4f} +/- {r['se']:.4f}" if r.get("win_rate") is not None else "n/a"
        print(f"[{name}] DONE  win rate {wr}  valid {r['valid']}/{r['attempts']}  "
              f"p90 {r.get('p90')}  PMI {r.get('surprise_mean')}  held-out {tinfo['heldout_loss']:.2f}  "
              f"{r['minutes']:.1f} min", flush=True)
    print("\narm             pool  bins top   bal   n    mean     se     p90   >=.458  valid  PMI")
    for k, r in res.items():
        if r.get("win_rate") is None: continue
        print(f"{k:15s} {r['pool_n']:5d}  {len(r['edges'])-1}   {r['top_bin_edge']:.1f}  "
              f"{'Y' if r['balanced'] else 'N'}  {r['n']:3d}  {r['win_rate']:.3f}  {r['se']:.3f}  "
              f"{r['p90']:.3f}  {r['above_real_median']:.2f}  {r['acceptance']:.2f}  {r['surprise_mean']:.2f}")
    print("BRACKETS_DONE", flush=True)


if __name__ == "__main__":
    main()
