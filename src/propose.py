"""Generate team proposals as real Showdown pastes and validate them with
Showdown's own TeamValidator (the same gate the corpus passed).

Two operators, each replacing ONE candidate slot of a source team:
  model    - resample the slot's 8 categorical fields from the masked-field model
  slotcopy - paste another corpus team's slot wholesale

Stat Points and Level for the model arm are inherited from the source slot: a
spread is legal under the 66-point budget regardless of which candidate holds it.
"""
import json, random, subprocess, sys
from pathlib import Path
import numpy as np, torch
from corpus import load_corpus, parse_team, norm, STATS, TEAMS
from encode import Vocab, team_fields, canon, FIELDS, NF, NSLOT
from model import MaskedFieldModel, random_mask, permute_slots, resample_fields

SHOWDOWN = Path("/tmp/vgc-pilot/vgc-bench/pokemon-showdown")
DEV = "mps" if torch.backends.mps.is_available() else "cpu"

class Validator:
    def __init__(self):
        self.p = subprocess.Popen(["node", "validate-teams-batch.js"],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  text=True, cwd=str(SHOWDOWN))
    def __call__(self, text, fmt="gen9championsvgc2026regmb"):
        self.p.stdin.write(json.dumps({"format": fmt, "team": text}) + "\n")
        self.p.stdin.flush()
        r = json.loads(self.p.stdout.readline())
        return None if r["valid"] else "; ".join(r["errors"])  # NOT truncated: callers parse every error
    def close(self):
        self.p.stdin.close(); self.p.wait()

def slot_to_text(sl):
    """Render one parsed slot back to Showdown paste form."""
    head = sl["species"] + (f" @ {sl['item']}" if sl["item"] else "")
    lines = [head, f"Ability: {sl['ability']}", "Level: 50"]
    ev = " / ".join(f"{sl['evs'][s]} {s}" for s in STATS if sl["evs"].get(s))
    if ev: lines.append(f"EVs: {ev}")
    if sl.get("nature"): lines.append(f"{sl['nature']} Nature")
    lines += [f"- {m}" for m in sl["moves"]]
    return "\n".join(lines)

def team_to_text(team):
    return "\n\n".join(slot_to_text(s) for s in team) + "\n"

def pretty(V, col, val_idx, lookup):
    """Map a normalised vocabulary string back to a display name Showdown parses."""
    v = V.decode_field(col, int(val_idx))
    return lookup.get(v, v)

def build_lookup(teams):
    """normalised -> the display spelling seen in the corpus."""
    L = {}
    for t in teams:
        for s in t:
            L[norm(s["species"])] = s["species"]
            L[norm(s["ability"])] = s["ability"]
            if s["item"]: L[norm(s["item"])] = s["item"]
            for m in s["moves"]: L[norm(m)] = m
            if s["nature"]: L[norm(s["nature"])] = s["nature"]
    return L

def main():
    n_per_arm = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    teams, names = load_corpus()
    keep = [(t, n) for t, n in zip(teams, names) if len(t) == 6]
    teams = [t for t, _ in keep]; names = [n for _, n in keep]
    V = Vocab(teams); LOOK = build_lookup(teams)
    X = np.stack([V.encode(t) for t in teams])
    idx = rng.permutation(len(X)); ntest = int(len(X) * .15)
    tr_i = idx[ntest:]
    Xtr = X[tr_i]

    model = MaskedFieldModel(V).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 600)
    xtr = torch.tensor(Xtr, device=DEV)
    print("training the masked-field model ...")
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
    torch.save(model.state_dict(), "/tmp/vgc-pilot/propose_model.pt")

    val = Validator()
    out = Path("/tmp/vgc-pilot/proposals"); out.mkdir(exist_ok=True)
    for f in out.glob("*.txt"): f.unlink()
    rec = {"model": [], "slotcopy": [], "control": []}
    tries = {"model": 0, "slotcopy": 0}
    src_pool = list(tr_i)
    rng.shuffle(src_pool)

    # control arm: unmodified source teams (must validate, they are corpus teams)
    for k in range(n_per_arm):
        si = src_pool[k]
        txt = team_to_text(canon(teams[si]))
        err = val(txt)
        if err is None:
            p = out/f"control_{k}.txt"; p.write_text(txt)
            rec["control"].append({"file": p.name, "src": names[si]})
    print(f"control: {len(rec['control'])}/{n_per_arm} valid")

    # model + slotcopy arms, paired on the same source team and same slot
    k, ptr = 0, 0
    while (len(rec["model"]) < n_per_arm or len(rec["slotcopy"]) < n_per_arm) and ptr < 4000:
        si = src_pool[ptr % len(src_pool)]; ptr += 1
        base = canon(teams[si])
        s = int(rng.integers(0, NSLOT))
        # ---- model arm
        if len(rec["model"]) < n_per_arm:
            tries["model"] += 1
            cols = [list(range(s*NF, (s+1)*NF))]
            g = torch.Generator(device=DEV); g.manual_seed(int(rng.integers(0, 1 << 30)))
            xm = resample_fields(model, torch.tensor(X[si][None, :], device=DEV), cols, gen=g).cpu().numpy()[0]
            newslot = dict(base[s])
            newslot["species"]  = pretty(V, s*NF+0, xm[s*NF+0], LOOK)
            newslot["ability"]  = pretty(V, s*NF+1, xm[s*NF+1], LOOK)
            newslot["item"]     = pretty(V, s*NF+2, xm[s*NF+2], LOOK)
            newslot["moves"]    = [pretty(V, s*NF+3+j, xm[s*NF+3+j], LOOK) for j in range(4)]
            newslot["moves"]    = [m for m in newslot["moves"] if m]
            newslot["nature"]   = pretty(V, s*NF+7, xm[s*NF+7], LOOK)
            cand = [dict(x) for x in base]; cand[s] = newslot
            txt = team_to_text(cand); err = val(txt)
            if err is None:
                p = out/f"model_{len(rec['model'])}.txt"; p.write_text(txt)
                rec["model"].append({"file": p.name, "src": names[si], "slot": s,
                                     "new": newslot["species"], "old": base[s]["species"]})
        # ---- slotcopy arm
        if len(rec["slotcopy"]) < n_per_arm:
            tries["slotcopy"] += 1
            di = int(rng.integers(0, len(teams)))
            ds = int(rng.integers(0, NSLOT))
            cand = [dict(x) for x in base]; cand[s] = dict(canon(teams[di])[ds])
            txt = team_to_text(cand); err = val(txt)
            if err is None:
                p = out/f"slotcopy_{len(rec['slotcopy'])}.txt"; p.write_text(txt)
                rec["slotcopy"].append({"file": p.name, "src": names[si], "slot": s,
                                        "new": cand[s]["species"], "old": base[s]["species"]})
    val.close()
    summary = {a: dict(n=len(rec[a]), tries=tries.get(a, len(rec[a])),
                       accept=round(len(rec[a]) / max(tries.get(a, len(rec[a])), 1), 4)) for a in rec}
    print(json.dumps(summary, indent=2))
    json.dump({"records": rec, "summary": summary}, open("/tmp/vgc-pilot/proposals.json", "w"), indent=2)
    print("wrote /tmp/vgc-pilot/proposals -> ", {a: len(rec[a]) for a in rec})

if __name__ == "__main__":
    main()
