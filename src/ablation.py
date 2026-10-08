"""Where does the local move stop working?

Seed 0 showed the masked-field model beats random mutation when 3 fields are
resampled and loses badly when a whole candidate (8 fields) is. This sweeps k to
find the crossover, and separates two cases the headline number confounds:
  scatter  - k fields anywhere in the team
  within   - k fields inside ONE candidate's set (never the species)
  species  - the species field plus k-1 other fields of that same candidate
The third is the case that matters: swapping a candidate strands its old set.
"""
import json, random
import numpy as np, torch
from corpus import load_corpus
from encode import Vocab, Legality, FIELDS, NF, NSLOT
from model import MaskedFieldModel, random_mask, permute_slots, resample_fields
from pilot import decode_fields, evaluate, op_uniform, set_seed, COLS

DEV = "mps" if torch.backends.mps.is_available() else "cpu"

def build_cols(mode, k, n, rng):
    cols, slots = [], rng.integers(0, NSLOT, n)
    for b in range(n):
        s = int(slots[b])
        base = s * NF
        if mode == "scatter":
            cols.append(list(rng.choice(COLS, k, replace=False)))
        elif mode == "within":          # never touch the species field
            opts = [base + j for j in range(1, NF)]
            cols.append(list(rng.choice(opts, min(k, len(opts)), replace=False)))
        elif mode == "species":         # species + k-1 others in the same slot
            opts = [base + j for j in range(1, NF)]
            pick = list(rng.choice(opts, min(k - 1, len(opts)), replace=False))
            cols.append([base] + pick)
    return cols, slots

def main():
    set_seed(0); rng = np.random.default_rng(0)
    teams, _ = load_corpus(); teams = [t for t in teams if len(t) == 6]
    V, L = Vocab(teams), Legality(teams)
    X = np.stack([V.encode(t) for t in teams])
    idx = rng.permutation(len(X)); ntest = int(len(X) * .15)
    Xte, Xtr = X[idx[:ntest]], X[idx[ntest:]]
    model = MaskedFieldModel(V).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 600)
    xtr = torch.tensor(Xtr, device=DEV)
    for ep in range(600):
        model.train()
        perm = torch.randperm(xtr.shape[0], device=DEV)
        for i in range(0, xtr.shape[0], 64):
            xb = permute_slots(xtr[perm[i:i+64]])
            loss = model.loss(xb, random_mask(xb))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sched.step()
        if (ep+1) % 200 == 0: print(f"  ep{ep+1} loss {float(loss):.3f}")
    model.eval()

    N = 192
    out = {}
    for mode in ["scatter", "within", "species"]:
        out[mode] = {}
        ks = range(1, 9) if mode != "species" else range(1, 9)
        for k in ks:
            cols, _ = build_cols(mode, k, N, rng)
            src = Xtr[rng.integers(0, len(Xtr), N)]
            g = torch.Generator(device=DEV); g.manual_seed(k)
            Xm = resample_fields(model, torch.tensor(src, device=DEV), cols, gen=g).cpu().numpy()
            Xu = op_uniform(src, cols, V, rng)
            em = evaluate(V, L, Xm, src, Xtr); eu = evaluate(V, L, Xu, src, Xtr)
            out[mode][k] = dict(model=em["legal"], model_ln=em["legal_and_novel"],
                                uniform=eu["legal"], uniform_ln=eu["legal_and_novel"],
                                model_novel=em["novel"])
            print(f"{mode:8s} k={k}  model legal {em['legal']:.3f} (novel {em['novel']:.3f})  "
                  f"uniform legal {eu['legal']:.3f}")
    json.dump(out, open("/tmp/vgc-pilot/ablation.json", "w"), indent=2)
    print("wrote /tmp/vgc-pilot/ablation.json")

if __name__ == "__main__":
    main()
