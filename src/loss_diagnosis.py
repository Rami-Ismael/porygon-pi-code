"""Why is the asked-vs-got diffusion loss ~5.0? Battle-free decomposition (2026-10-04).

1. The reported loss is the MDLM ELBO: cross-entropy weighted by 1/t, averaged over
   masked fields. Report it next to the plain (unweighted) cross-entropy at the same masks.
2. Plain cross-entropy per field type at fixed mask rates, held-out vs train (fit gap).
3. Baselines on the same held-out teams: field frequency alone, and frequency given species.
4. Real corpus teams vs generated (labelled) teams: the labelled pool was sampled from
   earlier models at temperature 1, so its own entropy is a floor no model can beat.
5. Slot-order cost: all six species masked, the model cannot know which slot holds which.

    python loss_diagnosis.py     (run with /tmp/vgc-pilot/.venv/bin/python)
"""
import json, math, sys
from collections import Counter, defaultdict
import numpy as np
import torch, torch.nn.functional as F

sys.path.insert(0, "/tmp/vgc-pilot/src")
sys.path.insert(0, "/Users/ramiismael/Documents/code/vgc-team-generator-pilot/src")
import asked_vs_got as X
import diffusion as D
from corpus import parse_team_text
from encode import NF, NSLOT, FIELDS

OUT = X.RES / "loss_diagnosis.json"
TYPES = ["species", "ability", "item", "move", "nature"]
def ftype(c): k = FIELDS[c % NF]; return "move" if k.startswith("m") and k != "nature" else k

def load_split():
    corpus, V, *_ = X.setup()
    rows = json.load(open(X.DATA))["rows"]
    Xs, W = [], []
    for r in rows:
        t = parse_team_text(r["paste"])
        if len(t) != 6: continue
        e = V.encode(t)
        if (e == 0).any(): continue
        Xs.append(e); W.append(r["y"])
    Xs, W = np.stack(Xs), np.array(W, np.float32)
    idx = np.random.default_rng(X.SEED).permutation(len(Xs)); te, tr = idx[:500], idx[500:]   # train()'s split
    Xc = np.stack([V.encode(t) for t in corpus])
    return V, Xs[tr], W[tr], Xs[te], W[te], Xc

@torch.no_grad()
def ce_at(m, x, w, t, reps=4, seed=0):
    """Plain CE per masked field at mask rate t (t=None: one random field masked = leave-one-out).
    Also returns the 1/t-weighted ELBO number on the same masks when t is random."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    out = defaultdict(list)
    xt = torch.tensor(x, device=D.DEV); wt = torch.tensor(w, device=D.DEV)
    for _ in range(reps):
        B = xt.shape[0]
        ps = torch.rand(B, NSLOT, generator=g).argsort(1).to(D.DEV)         # same slot shuffling as training
        xb = torch.gather(xt.view(B, NSLOT, NF), 1, ps.unsqueeze(-1).expand(-1, -1, NF)).reshape(B, -1)
        if t is None:
            mask = torch.zeros(B, D.COLS, dtype=torch.bool)
            mask[torch.arange(B), torch.randint(0, D.COLS, (B,), generator=g)] = True
            tv = torch.full((B,), 1.0 / D.COLS)
        elif t == "random":
            tv = torch.rand(B, generator=g)
            mask = torch.rand(B, D.COLS, generator=g) < tv.view(-1, 1)
        else:
            tv = torch.full((B,), float(t))
            mask = torch.rand(B, D.COLS, generator=g) < t
        empty = ~mask.any(1)
        mask[empty, torch.randint(0, D.COLS, (int(empty.sum()),), generator=g)] = True
        mask, tv = mask.to(D.DEV), tv.to(D.DEV)
        xin = xb.clone(); xin[mask] = 0
        h = m(xin, tv, wt)
        for c in range(D.COLS):
            mc = mask[:, c]
            if not mc.any(): continue
            ce = F.cross_entropy(m.logits(h, c)[mc], xb[mc, c], reduction="none")
            out[ftype(c)] += ce.tolist()
            out["_w"] += (ce / tv[mc].clamp(min=1e-3)).tolist()
    res = {k: float(np.mean(v)) for k, v in out.items() if k != "_w"}
    allce = [v for k, vs in out.items() if k != "_w" for v in vs]
    res["all"] = float(np.mean(allce)); res["elbo_weighted"] = float(np.mean(out["_w"]))
    return res

def baselines(V, xtr, xte):
    """Held-out CE of (a) field frequency, (b) field frequency given the slot's species. Add-0.5 smoothing."""
    res = {}
    for k in TYPES:
        cols = [c for c in range(D.COLS) if ftype(c) == k]
        nv = V.sizes[k]
        marg = Counter(int(v) for c in cols for v in xtr[:, c])
        tot = sum(marg.values())
        res[f"{k}_freq"] = float(np.mean([-math.log((marg[int(v)] + .5) / (tot + .5 * nv)) for c in cols for v in xte[:, c]]))
        if k == "species": continue
        cond = defaultdict(Counter)
        for c in cols:
            sc = (c // NF) * NF
            for s, v in zip(xtr[:, sc], xtr[:, c]): cond[int(s)][int(v)] += 1
        lp = []
        for c in cols:
            sc = (c // NF) * NF
            for s, v in zip(xte[:, sc], xte[:, c]):
                cs = cond[int(s)]; n = sum(cs.values())
                p = (cs[int(v)] + .5 * (marg[int(v)] + .5) / (tot + .5 * nv) * 20) / (n + 10)   # back off to marginal
                lp.append(-math.log(p))
        res[f"{k}_given_species"] = float(np.mean(lp))
    return res

def main():
    V, xtr, wtr, xte, wte, Xc = load_split()
    m = X.load(V)
    nan = np.full(len(Xc), np.nan, np.float32)
    sub = np.random.default_rng(1).choice(len(xtr), 2000, replace=False)
    r = {"n_train": len(xtr), "n_heldout": len(xte)}
    r["elbo_random_t"] = {"heldout": ce_at(m, xte, wte, "random"), "train": ce_at(m, xtr[sub], wtr[sub], "random"),
                          "real_corpus": ce_at(m, Xc, nan, "random")}
    r["by_mask_rate"] = {}
    for t in [None, 0.15, 0.5, 0.85, 1.0]:
        key = "leave_one_out" if t is None else f"t{t}"
        r["by_mask_rate"][key] = {"heldout": ce_at(m, xte, wte, t), "real_corpus": ce_at(m, Xc, nan, t)}
    r["baselines_heldout"] = baselines(V, xtr, xte)
    r["slot_order_floor_nats_per_species_col_all_masked"] = math.log(math.factorial(6)) / 6
    json.dump(r, open(OUT, "w"), indent=1)

    e = r["elbo_random_t"]
    print(f"reported (1/t-weighted) loss  held-out {e['heldout']['elbo_weighted']:.3f} · train {e['train']['elbo_weighted']:.3f} · real corpus {e['real_corpus']['elbo_weighted']:.3f}")
    print(f"plain CE, same masks          held-out {e['heldout']['all']:.3f} · train {e['train']['all']:.3f} · real corpus {e['real_corpus']['all']:.3f}")
    print("\nplain CE per field type (held-out generated | real corpus)")
    print(f"{'mask':14s}" + "".join(f"{k:>16s}" for k in TYPES + ['all']))
    for key, v in r["by_mask_rate"].items():
        print(f"{key:14s}" + "".join(f"{v['heldout'].get(k, float('nan')):7.2f} | {v['real_corpus'].get(k, float('nan')):5.2f}" for k in TYPES + ['all']))
    b = r["baselines_heldout"]
    print("\nbaselines on held-out (plain CE):")
    for k in TYPES:
        print(f"  {k:8s} frequency {b[f'{k}_freq']:.2f}" + (f" · given species {b[f'{k}_given_species']:.2f}" if f"{k}_given_species" in b else ""))
    print(f"\nslot-order floor, species column with all six species masked: {r['slot_order_floor_nats_per_species_col_all_masked']:.2f} nats")

if __name__ == "__main__":
    main()
