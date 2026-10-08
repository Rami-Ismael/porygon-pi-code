"""Follow-up: (a) what does each operator REACH, (b) does the local move survive
when the source team is HELD-OUT rather than a memorised training team?"""
import numpy as np, torch, random, json
from corpus import load_corpus
from encode import Vocab, Legality, NF, NSLOT
from model import MaskedFieldModel
from pilot import op_slotcopy, op_uniform, op_marginal, set_seed, COLS, decode_fields
from refute_lens import resample, op_condcopy, StrictLegality
from collections import defaultdict
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
set_seed(0); rng = np.random.default_rng(0)
teams, _ = load_corpus(); teams = [t for t in teams if len(t) == 6]
V = Vocab(teams); L = Legality(teams)
X = np.stack([V.encode(t) for t in teams])
idx = rng.permutation(len(X)); nte = int(len(X)*.15); Xte, Xtr = X[idx[:nte]], X[idx[nte:]]
model = MaskedFieldModel(V).to(DEV); model.load_state_dict(torch.load("/tmp/vgc-pilot/refute_model.pt")); model.eval()
slots_train = {tuple(r[s*NF:(s+1)*NF]) for r in Xtr for s in range(NSLOT)}
print("distinct corpus slots (the entire reach of slotcopy):", len(slots_train))
N = 256
out = {}

# ---- (a) reach, slot regime ----
src_i = rng.integers(0, len(Xtr), N); X_src = Xtr[src_i]
slot_of = rng.integers(0, NSLOT, N)
cols = [list(range(s*NF, (s+1)*NF)) for s in slot_of]
g = torch.Generator(device=DEV); g.manual_seed(0)
arms = {"model": resample(model, torch.tensor(X_src, device=DEV), cols, gen=g).cpu().numpy(),
        "slotcopy": op_slotcopy(X_src, slot_of, Xtr, rng),
        "condcopy": op_condcopy(X_src, cols, V, rng, Xtr)}
for name, Xa in arms.items():
    ns = [tuple(Xa[b, slot_of[b]*NF:(slot_of[b]+1)*NF]) for b in range(N)]
    verb = float(np.mean([s in slots_train for s in ns]))
    legal = np.array([not L.violations(decode_fields(V, Xa[i])) for i in range(N)])
    ln = float(np.mean([(s not in slots_train) and legal[i] for i, s in enumerate(ns)]))
    print(f"{name:9s} verbatim-corpus-slot {verb:.3f}  legal {legal.mean():.3f}  legal AND slot-not-in-corpus {ln:.3f}")
    out[name] = dict(verbatim=verb, legal=float(legal.mean()), legal_and_new_slot=ln)

# ---- (b) scatter local move: train sources vs held-out sources ----
print("\nscatter k=3, model local move, by source pool:")
for tag, pool in [("TRAIN src", Xtr), ("HELDOUT src", Xte)]:
    accs = []
    for rep in range(3):
        r = np.random.default_rng(100 + rep)
        src = pool[r.integers(0, len(pool), N)]
        cl = [list(r.choice(COLS, 3, replace=False)) for _ in range(N)]
        gg = torch.Generator(device=DEV); gg.manual_seed(rep)
        Xa = resample(model, torch.tensor(src, device=DEV), cl, gen=gg).cpu().numpy()
        legal = np.array([not L.violations(decode_fields(V, Xa[i])) for i in range(N)])
        ch = (Xa != src).sum(1)
        nov = np.array([bool(np.min((Xa[i][None, :] != Xtr).sum(1)) > 0) for i in range(N)])
        Xu = op_uniform(src, cl, V, r)
        ul = np.mean([not L.violations(decode_fields(V, Xu[i])) for i in range(N)])
        Xc = op_condcopy(src, cl, V, r, Xtr)
        cll = np.mean([not L.violations(decode_fields(V, Xc[i])) for i in range(N)])
        accs.append((legal.mean(), float(np.mean(legal & nov & (ch > 0))), ch.mean(), float(np.mean(ch == 0)), ul, cll))
    a = np.array(accs).mean(0)
    print(f"  {tag:12s} model legal {a[0]:.3f}  legal&novel&changed {a[1]:.3f}  "
          f"edits {a[2]:.2f}  noop {a[3]:.3f} | uniform legal {a[4]:.3f}  condcopy legal {a[5]:.3f}")
    out[tag] = dict(model_legal=float(a[0]), model_lnc=float(a[1]), edits=float(a[2]),
                    noop=float(a[3]), uniform_legal=float(a[4]), condcopy_legal=float(a[5]))
json.dump(out, open("/tmp/vgc-pilot/refute2.json", "w"), indent=2)
print("wrote refute2.json")

# ---- (c) unconditional with a less naive decoder ----
print("\nunconditional, alternative decoders (N=96):")
M = 96
for temp, sw in [(1.0, 1), (0.5, 3), (0.7, 5)]:
    gg = torch.Generator(device=DEV); gg.manual_seed(7)
    xz = torch.zeros(M, COLS, dtype=torch.long, device=DEV)
    Xu = resample(model, xz, [list(range(COLS))]*M, temp=temp, gen=gg, sweeps=sw).cpu().numpy()
    lg = np.array([not L.violations(decode_fields(V, Xu[i])) for i in range(M)])
    nv = float(np.mean([len([x for x in L.violations(decode_fields(V, Xu[i])) if x != "speciesclause"]) for i in range(M)]))
    print(f"  t={temp} sweeps={sw}: legal {lg.mean():.3f}  mean per-team violations {nv:.1f}")
