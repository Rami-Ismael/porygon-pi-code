"""Scaling experiment: does a bigger network, or more data, raise the generator's win rate?

Model: the win-rate-conditioned masked diffusion (48 columns, spreads copied — the best
configuration found). Outcome: win rate of teams generated at the top win-rate bin,
guidance 4, against the top-50 pool, 200 teams per arm, errors clustered by team.

  network scale : width d in {96, 192, 384} at 100% of the data (params ~0.5M / 2.1M / 8M)
  data scale    : 25% / 50% / 100% of the 4,850 labelled teams at d=192

Pre-registered expectations (written before any run):
  - held-out loss falls with width; win rate stays flat — the bottleneck is label
    information, not capacity (train and held-out loss already track each other).
  - win rate rises with data, modestly, because the win-rate token gets sharper; the
    slope of that curve says whether labelling more teams is worth the battles.
"""
import json, os, sys, time, random
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, "/tmp/vgc-pilot/src")
import diffusion as D, wrdiffusion as W
from corpus import norm, STATS
from encode import NF, NSLOT
from propose import Validator, slot_to_text
import pool

ARMS = [("d96_f100", 96, 1.0), ("d192_f100", 192, 1.0), ("d384_f100", 384, 1.0),
        ("d192_f25", 192, 0.25), ("d192_f50", 192, 0.5)]
OUT = "/tmp/vgc-pilot/scale_results.json"

def train_arm(tag, d, frac, epochs=300, seed=0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    V, L, corpus, teams, wrs = W.build_vocab()
    X = np.stack([V.encode(t) for t in teams]); Y = np.array([D.style_of(t) for t in teams]); Wb = np.array([W.wr_bin(w) for w in wrs])
    rng = np.random.default_rng(seed); idx = rng.permutation(len(X)); nte = int(len(X) * .1)
    te = idx[:nte]; tr = idx[nte:]; tr = tr[: int(len(tr) * frac)]
    nhead = 6 if d % 6 == 0 else 4
    m = W.TeamDiffusionWR(V, d=d, nhead=nhead).to(W.DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=.01); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    T = lambda a, i: torch.tensor(a[i], device=W.DEV)
    xtr, ytr, wtr = T(X, tr), T(Y, tr), T(Wb, tr); xte, yte, wte = T(X, te), T(Y, te), T(Wb, te)
    nparam = sum(p.numel() for p in m.parameters())
    print(f"[{tag}] train {len(tr)} / held-out {len(te)} · d={d} · {nparam:,} params", flush=True)
    t0 = time.perf_counter(); vl = None
    for ep in range(epochs):
        m.train(); perm = torch.randperm(xtr.shape[0], device=W.DEV)
        for i in range(0, xtr.shape[0], 64):
            b = perm[i:i + 64]; xb = xtr[b].view(-1, NSLOT, NF)
            xb = torch.stack([r[torch.randperm(NSLOT)] for r in xb]).view(-1, W.COLS)
            loss = m.loss(xb, ytr[b], wtr[b]); opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.); opt.step()
        sch.step()
        if (ep + 1) % 100 == 0:
            m.eval()
            with torch.no_grad(): vl = float(np.mean([float(m.loss(xte, yte, wte)) for _ in range(8)]))
            print(f"[{tag}]   ep{ep+1} held-out {vl:.3f}  ({(time.perf_counter()-t0)/60:.0f} min)", flush=True)
    m.eval(); return m, V, L, corpus, teams, nparam, vl, time.perf_counter() - t0

def eval_arm(tag, m, V, L, corpus, teams, n=200):
    C = D.Constraints(V, L); look, spreads = {}, {}
    for t in corpus + teams:
        for s in t:
            look[norm(s["species"])] = s["species"]; look[norm(s["ability"])] = s["ability"]
            if s["item"]: look[norm(s["item"])] = s["item"]
            for mv in s["moves"]: look[norm(mv)] = mv
            if s["nature"]: look[norm(s["nature"])] = s["nature"]
            spreads.setdefault(norm(s["species"]), []).append(dict(s["evs"]))
    rng = np.random.default_rng(0); torch.manual_seed(0)
    out = Path(f"/tmp/vgc-pilot/scale_gen/{tag}"); out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.txt"): f.unlink()
    val = Validator(); kept = tries = 0; files = []
    while kept < n and tries < n * 4:
        row = W.sample_constrained(m, C, 1, 5, "none", 4.0).cpu().numpy()[0]; tries += 1
        slots = []
        for i in range(NSLOT):
            b = i * NF; g = lambda j: V.decode_field(b + j, int(row[b + j])); sp = g(0)
            mv = [look.get(g(3 + j), g(3 + j)) for j in range(4) if g(3 + j) and g(3 + j) != "[MASK]"]
            pl = spreads.get(sp) or [{s: 0 for s in STATS}]
            slots.append(dict(species=look.get(sp, sp), item=look.get(g(2), g(2)), ability=look.get(g(1), g(1)),
                              nature=look.get(g(7), g(7)), moves=mv, evs=pl[int(rng.integers(0, len(pl)))]))
        txt = "\n\n".join(slot_to_text(s) for s in slots) + "\n"
        if val(txt) is None:
            p = out / f"{kept:04d}.txt"; p.write_text(txt); files.append(str(p)); kept += 1
    val.close()
    root = Path("/tmp/vgc-pilot/vgc-bench/teams/reg_mb"); opp = []
    for i in [t["id"] for t in json.load(open("/tmp/vgc-pilot/top50_evs.json"))]:
        for c in (root / f"{i}.txt", root / "featured" / f"{i}.txt"):
            if c.exists(): opp.append(str(c)); break
    if not files:                        # a model that yields no valid team gets a recorded zero, not a crash
        return dict(valid=0.0, n=0, win_rate=None, se=None, p90=None, max=None)
    res = pool.score([(f, opp) for f in files], battles=24, conc=50, quiet=True)
    w = np.array([res[f]["win_rate"] for f in files])
    return dict(valid=kept / max(tries, 1), n=len(w), win_rate=float(w.mean()), se=float(w.std(ddof=1) / np.sqrt(len(w))),
                p90=float(np.percentile(w, 90)), max=float(w.max()))

def main():
    done = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for tag, d, frac in ARMS:
        if tag in done: continue
        m, V, L, corpus, teams, nparam, vl, secs = train_arm(tag, d, frac)
        r = eval_arm(tag, m, V, L, corpus, teams)
        r.update(dict(d=d, frac=frac, params=nparam, heldout_loss=vl, train_min=secs / 60))
        done[tag] = r; json.dump(done, open(OUT, "w"), indent=1)
        print(f"[{tag}] DONE  valid {r['valid']:.0%}  win rate {r['win_rate']:.4f} ± {r['se']:.4f}  held-out {vl:.3f}  params {nparam:,}", flush=True)
    print("ALL_DONE", flush=True)

if __name__ == "__main__":
    main()
