"""Adversarial re-run: oracle strictness, decoding, edit-size-matched baselines."""
import json, random, time, sys
from collections import Counter, defaultdict
import numpy as np, torch
from corpus import (load_corpus, norm, legal_moves, legal_abilities,
                    MOVE_NAMES, ABIL_NAMES, ITEM_NAMES, NATURES)
from encode import Vocab, Legality, team_fields, FIELDS, NF, NSLOT
from model import MaskedFieldModel, random_mask, permute_slots
from pilot import decode_fields, evaluate, op_uniform, op_marginal, op_slotcopy, set_seed, COLS, memorization_stats, hamming

DEV = "mps" if torch.backends.mps.is_available() else "cpu"

class StrictLegality(Legality):
    """Showdown learnsets/abilities only: no corpus widening."""
    def __init__(self, teams):
        super().__init__(teams)
        self.corpus_moves = defaultdict(set); self.corpus_abils = defaultdict(set)
        self._lm, self._la = {}, {}

class GlobalLegality(Legality):
    """No per-species check at all: a move/ability only has to exist in the game."""
    def moves_for(self, sp): return MOVE_NAMES
    def abils_for(self, sp): return ABIL_NAMES

@torch.no_grad()
def resample(model, x, cols_to_mask, temp=1.0, gen=None, sweeps=1):
    x = x.clone()
    for b in range(x.shape[0]):
        cols = list(cols_to_mask[b]); random.shuffle(cols)
        for c in cols: x[b, c] = 0
        for s in range(sweeps):
            order = list(cols); random.shuffle(order)
            for c in order:
                if s > 0: x[b, c] = 0
                h = model(x[b:b+1]); lg = model.logits_at(h, c)[0] / temp
                lg[0] = -1e9
                if temp <= 0.0:
                    x[b, c] = int(lg.argmax())
                else:
                    x[b, c] = torch.multinomial(torch.softmax(lg, -1), 1, generator=gen).item()
    return x

def op_condcopy(X_src, cols, V, rng, Xtr):
    """Trivial species-conditional table lookup, no learning: for each masked column,
    copy the value from a random corpus SLOT of the same species. If species itself
    is masked, draw a species from the corpus marginal then fill its fields."""
    by_sp = defaultdict(list)
    for r in Xtr:
        for s in range(NSLOT):
            by_sp[int(r[s*NF])].append(r[s*NF:(s+1)*NF])
    all_sl = [v for vs in by_sp.values() for v in vs]
    X = X_src.copy()
    for b in range(X.shape[0]):
        for c in sorted(cols[b]):
            s = c // NF; j = c % NF
            if j == 0:
                donor = all_sl[rng.integers(0, len(all_sl))]
                X[b, c] = donor[0]
            else:
                pool = by_sp.get(int(X[b, s*NF]))
                if not pool: continue
                donor = pool[rng.integers(0, len(pool))]
                X[b, c] = donor[j]
    return X

def canon_rows(X):
    """Re-sort slots by species index (== the canonical order used at encode time)."""
    Y = X.reshape(len(X), NSLOT, NF).copy()
    for i in range(len(Y)):
        Y[i] = Y[i][np.argsort(Y[i][:, 0], kind="stable")]
    return Y.reshape(len(X), NSLOT*NF)

def report(tag, V, oracles, Xa, X_src, Xtr, extra=""):
    r = {}
    for on, L in oracles.items():
        r[on] = float(np.mean([not L.violations(decode_fields(V, Xa[i])) for i in range(len(Xa))]))
    ch = (Xa != X_src).sum(1) if X_src is not None else np.full(len(Xa), NF*NSLOT)
    novel = np.array([bool(np.min(hamming(Xa[i], Xtr)) > 0) for i in range(len(Xa))])
    ok = {on: 0 for on in oracles}
    for on, L in oracles.items():
        ok[on] = float(np.mean([(not L.violations(decode_fields(V, Xa[i]))) and novel[i] and ch[i] > 0
                                for i in range(len(Xa))]))
    print(f"{tag:34s} edits {ch.mean():5.2f}  noop {float(np.mean(ch==0)):.3f}  " +
          "  ".join(f"{on} {r[on]:.3f}/{ok[on]:.3f}" for on in oracles) + " " + extra)
    return dict(legal={k: v for k, v in r.items()}, legal_novel_changed=ok,
                mean_edits=float(ch.mean()), noop=float(np.mean(ch == 0)))

def main():
    set_seed(0); rng = np.random.default_rng(0)
    teams, _ = load_corpus(); teams = [t for t in teams if len(t) == 6]
    V = Vocab(teams)
    oracles = {"WIDE": Legality(teams), "STRICT": StrictLegality(teams), "GLOBAL": GlobalLegality(teams)}
    X = np.stack([V.encode(t) for t in teams])
    idx = rng.permutation(len(X)); ntest = int(len(X)*0.15)
    Xte, Xtr = X[idx[:ntest]], X[idx[ntest:]]
    colvals = defaultdict(list)
    for c in range(COLS): colvals[V.key(c)] += list(Xtr[:, c])
    colvals = {k: np.array(v) for k, v in colvals.items()}

    print("== corpus itself under each oracle ==")
    for on, L in oracles.items():
        print("  ", on, "train teams legal:", float(np.mean([not L.violations(decode_fields(V, r)) for r in Xtr])),
              " heldout legal:", float(np.mean([not L.violations(decode_fields(V, r)) for r in Xte])))

    model = MaskedFieldModel(V).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 600)
    xtr = torch.tensor(Xtr, device=DEV)
    t0 = time.time()
    for ep in range(600):
        model.train(); perm = torch.randperm(xtr.shape[0], device=DEV)
        for i in range(0, xtr.shape[0], 64):
            xb = permute_slots(xtr[perm[i:i+64]])
            loss = model.loss(xb, random_mask(xb))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sched.step()
        if (ep+1) % 200 == 0: print(f"  ep{ep+1} {float(loss):.3f} {time.time()-t0:.0f}s", flush=True)
    model.eval(); torch.save(model.state_dict(), "/tmp/vgc-pilot/refute_model.pt")
    out = {}

    N = 256
    # ---------- reproduce pilot rng order ----------
    for regime in ["scatter", "slot"]:
        src_i = rng.integers(0, len(Xtr), N); X_src = Xtr[src_i]
        if regime == "scatter":
            cols = [list(rng.choice(COLS, 3, replace=False)) for _ in range(N)]; slot_of = None
        else:
            slot_of = rng.integers(0, NSLOT, N)
            cols = [list(range(s*NF, (s+1)*NF)) for s in slot_of]
        print(f"\n== regime {regime}  (legal / legal&novel&changed per oracle) ==")
        g = torch.Generator(device=DEV); g.manual_seed(0)
        res = {}
        Xm = resample(model, torch.tensor(X_src, device=DEV), cols, gen=g).cpu().numpy()
        res["model(t=1,1sweep)"] = report("model t=1.0 1sweep", V, oracles, Xm, X_src, Xtr)
        res["uniform"] = report("uniform", V, oracles, op_uniform(X_src, cols, V, rng), X_src, Xtr)
        res["marginal"] = report("marginal", V, oracles, op_marginal(X_src, cols, V, rng, colvals), X_src, Xtr)
        if regime == "slot":
            res["slotcopy"] = report("slotcopy", V, oracles, op_slotcopy(X_src, slot_of, Xtr, rng), X_src, Xtr)
        res["condcopy"] = report("condcopy (table lookup)", V, oracles, op_condcopy(X_src, cols, V, rng, Xtr), X_src, Xtr)
        for temp, sw in [(0.5, 1), (0.0, 1), (1.0, 3), (0.5, 3)]:
            g = torch.Generator(device=DEV); g.manual_seed(0)
            Xv = resample(model, torch.tensor(X_src, device=DEV), cols, temp=temp, gen=g, sweeps=sw).cpu().numpy()
            res[f"model(t={temp},{sw}sweep)"] = report(f"model t={temp} {sw}sweep", V, oracles, Xv, X_src, Xtr)
        out[regime] = res

    # ---------- edit-size-matched uniform in the scatter regime ----------
    print("\n== scatter, uniform at matched realised edit size ==")
    em = {}
    for k in [1, 2, 3]:
        src = Xtr[rng.integers(0, len(Xtr), N)]
        cols = [list(rng.choice(COLS, k, replace=False)) for _ in range(N)]
        em[f"uniform_k{k}"] = report(f"uniform k={k}", V, oracles, op_uniform(src, cols, V, rng), src, Xtr)
        em[f"condcopy_k{k}"] = report(f"condcopy k={k}", V, oracles, op_condcopy(src, cols, V, rng, Xtr), src, Xtr)
    out["edit_matched"] = em

    # ---------- unconditional ----------
    print("\n== unconditional ==")
    g = torch.Generator(device=DEV); g.manual_seed(7)
    xz = torch.zeros(N, COLS, dtype=torch.long, device=DEV)
    Xu = resample(model, xz, [list(range(COLS))]*N, gen=g).cpu().numpy()
    u = report("uncond t=1 1sweep", V, oracles, Xu, None, Xtr)
    perslot = []
    for i in range(N):
        f = decode_fields(V, Xu[i])
        vs = oracles["WIDE"].violations(f)
        perslot.append(len([v for v in vs if v != "speciesclause"]))
    print("  mean non-speciesclause violations per generated team:", float(np.mean(perslot)))
    raw = memorization_stats(Xu, Xtr); can = memorization_stats(canon_rows(Xu), Xtr)
    print("  NN dist raw", raw, "\n  NN dist RE-CANONICALISED", can)
    # calibration comparison, both canonicalised
    print("  heldout (already canonical):", memorization_stats(Xte, Xtr))
    ru = np.stack([[rng.integers(1, V.sizes[V.key(c)]) for c in range(COLS)] for _ in range(N)])
    print("  uniform-random teams:", memorization_stats(canon_rows(ru), Xtr))
    out["unconditional"] = dict(summary=u, mean_viol=float(np.mean(perslot)), raw_nn=raw, canon_nn=can)
    json.dump(out, open("/tmp/vgc-pilot/refute_lens.json", "w"), indent=2)
    print("\nwrote /tmp/vgc-pilot/refute_lens.json")

if __name__ == "__main__":
    main()
