"""Pilot: does a masked-field generator trained on the Reg M-B meta-team corpus
memorise or generalise, and does its mask-and-resample move propose better
candidates than random mutation at equal budget?

Battle-free by construction. Win rate is NOT measured: this project has no battle
policy yet, so f cannot be evaluated. What is measured is everything that must hold
BEFORE spending battles on a proposal.

Two move regimes:
  scatter - k fields chosen anywhere in the team (the naive 'random mutation' move)
  slot    - all 8 fields of one slot (the natural 'replace one team member' move)

Four arms per regime (identical sources, identical masked columns):
  model         - mask the fields, resample them from the masked-field model
  uniform       - replace each field with a uniform draw from its vocabulary
  marginal      - draw each field from the corpus marginal for that column
  slotcopy      - (slot regime only) paste a whole slot from a random corpus team
"""
import argparse, json, random, time
from collections import Counter, defaultdict
import numpy as np
import torch
from corpus import load_corpus
from encode import Vocab, Legality, team_fields, FIELDS, NF, NSLOT
from model import MaskedFieldModel, random_mask, permute_slots, resample_fields

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
COLS = NSLOT * NF

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

def hamming(a, B):
    return (a[None, :] != B).sum(1)

def memorization_stats(gen_X, train_X):
    """Gu et al. (TMLR 2025): memorised if nearest-neighbour distance < 1/3 of the
    second-nearest. Computed on the canonical field grid (slots sorted by species)."""
    mem, exact, d1s = 0, 0, []
    for i in range(gen_X.shape[0]):
        d = hamming(gen_X[i], train_X)
        srt = np.sort(d)
        d1, d2 = float(srt[0]), float(srt[1])
        d1s.append(d1)
        if d1 == 0: exact += 1
        if d2 > 0 and d1 < d2 / 3.0: mem += 1
    n = gen_X.shape[0]
    return dict(memorized=mem / n, exact_copy=exact / n, mean_nn_dist=float(np.mean(d1s)))

def decode_fields(V, row):
    return [V.decode_field(c, int(row[c])) for c in range(COLS)]

def evaluate(V, L, X_new, X_src, train_X):
    n = X_new.shape[0]
    legal, viol = [], Counter()
    for i in range(n):
        v = L.violations(decode_fields(V, X_new[i]))
        legal.append(not v)
        for x in v: viol[x] += 1
    novel = [bool(np.min(hamming(X_new[i], train_X)) > 0) for i in range(n)]
    changed = [int((X_new[i] != X_src[i]).sum()) > 0 for i in range(n)]
    ln = [a and b and c for a, b, c in zip(legal, novel, changed)]
    return dict(legal=float(np.mean(legal)), novel=float(np.mean(novel)),
                changed=float(np.mean(changed)), legal_and_novel=float(np.mean(ln)),
                unique=len({tuple(r) for r in X_new}) / n,
                violations=dict(viol.most_common(6)))

def op_uniform(X_src, cols, V, rng):
    X = X_src.copy()
    for b in range(X.shape[0]):
        for c in cols[b]:
            X[b, c] = rng.integers(1, V.sizes[V.key(c)])
    return X

def op_marginal(X_src, cols, V, rng, colvals):
    """Independent draw from the corpus marginal for that column type."""
    X = X_src.copy()
    for b in range(X.shape[0]):
        for c in cols[b]:
            pool = colvals[V.key(c)]
            X[b, c] = pool[rng.integers(0, len(pool))]
    return X

def op_slotcopy(X_src, slot_of, train_X, rng):
    """Paste a whole slot from a random corpus team: the cheap recombination baseline
    (the review's SMOTE analogue) that a learned generator has to beat."""
    X = X_src.copy()
    for b in range(X.shape[0]):
        s = slot_of[b]
        src = train_X[rng.integers(0, len(train_X))]
        s2 = rng.integers(0, NSLOT)
        X[b, s*NF:(s+1)*NF] = src[s2*NF:(s2+1)*NF]
    return X

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--nprop", type=int, default=256)
    ap.add_argument("--holdout", type=float, default=0.15)
    ap.add_argument("--out", default="/tmp/vgc-pilot/results.json")
    a = ap.parse_args()
    set_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    teams, _ = load_corpus()
    teams = [t for t in teams if len(t) == 6]
    V, L = Vocab(teams), Legality(teams)
    X = np.stack([V.encode(t) for t in teams])
    idx = rng.permutation(len(X))
    ntest = int(len(X) * a.holdout)
    Xte, Xtr = X[idx[:ntest]], X[idx[ntest:]]
    print(f"device={DEV} teams={len(X)} train={len(Xtr)} heldout={len(Xte)} cols={COLS}")

    colvals = defaultdict(list)
    for c in range(COLS):
        colvals[V.key(c)] += list(Xtr[:, c])
    colvals = {k: np.array(v) for k, v in colvals.items()}

    model = MaskedFieldModel(V).to(DEV)
    nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    xtr = torch.tensor(Xtr, device=DEV); xte = torch.tensor(Xte, device=DEV)
    hist, t0 = [], time.time()
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(xtr.shape[0], device=DEV)
        tot, nb = 0.0, 0
        for i in range(0, xtr.shape[0], 64):
            xb = permute_slots(xtr[perm[i:i+64]])
            loss = model.loss(xb, random_mask(xb))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tot += float(loss); nb += 1
        sched.step()
        if (ep + 1) % 50 == 0 or ep == 0:
            model.eval()
            with torch.no_grad():
                g = torch.Generator(device=DEV); g.manual_seed(1234)
                vl = float(np.mean([float(model.loss(xte, random_mask(xte, g))) for _ in range(8)]))
            hist.append(dict(epoch=ep+1, train=tot/nb, heldout=vl))
            print(f"  ep{ep+1:4d} train {tot/nb:.3f}  held-out {vl:.3f}")
    train_time = time.time() - t0
    model.eval()

    res = dict(device=DEV, params=nparam, epochs=a.epochs, seed=a.seed,
               n_teams=len(X), n_train=len(Xtr), n_heldout=len(Xte),
               train_time_s=round(train_time, 1), loss_history=hist, regimes={})

    for regime in ["scatter", "slot"]:
        src_i = rng.integers(0, len(Xtr), a.nprop)
        X_src = Xtr[src_i]
        if regime == "scatter":
            cols = [list(rng.choice(COLS, a.k, replace=False)) for _ in range(a.nprop)]
            slot_of = None
        else:
            slot_of = rng.integers(0, NSLOT, a.nprop)
            cols = [list(range(s*NF, (s+1)*NF)) for s in slot_of]
        g = torch.Generator(device=DEV); g.manual_seed(a.seed)
        Xm = resample_fields(model, torch.tensor(X_src, device=DEV), cols, gen=g).cpu().numpy()
        arms = {
            "model": Xm,
            "uniform": op_uniform(X_src, cols, V, rng),
            "marginal": op_marginal(X_src, cols, V, rng, colvals),
        }
        if regime == "slot":
            arms["slotcopy"] = op_slotcopy(X_src, slot_of, Xtr, rng)
        out = {}
        for name, Xa in arms.items():
            out[name] = evaluate(V, L, Xa, X_src, Xtr)
            out[name]["memorization"] = memorization_stats(Xa, Xtr)
        res["regimes"][regime] = out
        print(f"\n== regime {regime} (k={a.k if regime=='scatter' else NF} fields) ==")
        for name, r in out.items():
            print(f"  {name:9s} legal {r['legal']:.3f}  novel {r['novel']:.3f}  "
                  f"legal&novel {r['legal_and_novel']:.3f}  unique {r['unique']:.3f}  "
                  f"memorised {r['memorization']['memorized']:.3f}")

    # unconditional sampling from all-mask: the 'proposal prior' use
    g = torch.Generator(device=DEV); g.manual_seed(a.seed + 7)
    xz = torch.zeros(a.nprop, COLS, dtype=torch.long, device=DEV)
    Xu = resample_fields(model, xz, [list(range(COLS))]*a.nprop, gen=g).cpu().numpy()
    res["unconditional"] = evaluate(V, L, Xu, np.full_like(Xu, -1), Xtr)
    res["unconditional"]["memorization"] = memorization_stats(Xu, Xtr)
    u = res["unconditional"]
    print(f"\n== unconditional (all 48 fields sampled) ==\n  legal {u['legal']:.3f}  "
          f"novel {u['novel']:.3f}  unique {u['unique']:.3f}  "
          f"memorised {u['memorization']['memorized']:.3f}  "
          f"mean NN dist {u['memorization']['mean_nn_dist']:.1f}\n  top violations {u['violations']}")
    json.dump(res, open(a.out, "w"), indent=2)
    print("\nwrote", a.out)

if __name__ == "__main__":
    main()
