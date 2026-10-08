"""Closing the loop: the cross-entropy method with two proposal distributions.

Two loops, identical budget, same starting pool (4,850 labelled + 692 corpus teams):
  diffusion : retrain the win-rate-conditioned model on the pool, propose at the top
              bin (guidance 2), PMI-filter, battle, add ALL results to the pool
  copypaste : paste one candidate between top-20% teams, PMI-filter, battle, add all

PMI-surprise of a team = -mean over its 15 species pairs of
  log p(a,b) / (p(a) p(b))   (estimated on the corpus, add-alpha smoothed)
so meta cores score low and unseen pairings score high. Each round proposes 400
Showdown-valid teams and battles the 200 most surprising (decision Q13a).

Verdict after round 3 (decision Q12b): the 90th percentile of each loop's final-round
proposals, against the median real team (0.458) and against each other.
"""
import json, os, sys, time, random, itertools, math
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np, torch
sys.path.insert(0, "/tmp/vgc-pilot/src")
import diffusion as D, wrdiffusion as W
from corpus import load_corpus, parse_team, norm, STATS
from encode import Vocab, Legality, canon, NF, NSLOT
from propose import Validator, slot_to_text, team_to_text
import pool

ROOT = Path("/tmp/vgc-pilot/loop"); ROOT.mkdir(exist_ok=True)
OUT = ROOT / "loop_results.json"
PROPOSE, BATTLE, BATTLES = 400, 200, 24
ROUNDS = 3

# ---------- PMI over species pairs, from the corpus ----------
class PMI:
    def __init__(self, corpus, alpha=0.5):
        self.n = len(corpus); self.uni = Counter(); self.pair = Counter()
        for t in corpus:
            S = sorted({norm(s["species"]) for s in t})
            for a in S: self.uni[a] += 1
            for a, b in itertools.combinations(S, 2): self.pair[(a, b)] += 1
        self.alpha = alpha; self.V = len(self.uni)
    def surprise(self, team):
        S = sorted({norm(s["species"]) for s in team}); vals = []
        for a, b in itertools.combinations(S, 2):
            pab = (self.pair[(a, b)] + self.alpha) / (self.n + self.alpha * self.V * self.V / 2)
            pa = (self.uni[a] + self.alpha) / (self.n + self.alpha * self.V)
            pb = (self.uni[b] + self.alpha) / (self.n + self.alpha * self.V)
            vals.append(math.log(pab / (pa * pb)))
        return -float(np.mean(vals)) if vals else 0.0

def opponents():
    root = Path("/tmp/vgc-pilot/vgc-bench/teams/reg_mb"); opp = []
    for i in [t["id"] for t in json.load(open("/tmp/vgc-pilot/top50_evs.json"))]:
        for c in (root / f"{i}.txt", root / "featured" / f"{i}.txt"):
            if c.exists(): opp.append(str(c)); break
    return opp

def battle(files, opp):
    res = pool.score([(f, opp) for f in files], battles=BATTLES, conc=50, quiet=True)
    return {f: res[f]["win_rate"] for f in files}

# ---------- diffusion proposer ----------
def retrain(model, V, teams, wrs, epochs=60, lr=1e-4):
    X = np.stack([V.encode(t) for t in teams]); Y = np.array([D.style_of(t) for t in teams]); B = np.array([W.wr_bin(w) for w in wrs])
    xtr, ytr, btr = (torch.tensor(a, device=W.DEV) for a in (X, Y, B))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.01); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(X), device=W.DEV)
        for i in range(0, len(X), 64):
            b = perm[i:i + 64]; xb = xtr[b].view(-1, NSLOT, NF)
            xb = torch.stack([r[torch.randperm(NSLOT)] for r in xb]).view(-1, W.COLS)
            loss = model.loss(xb, ytr[b], btr[b]); opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); opt.step()
        sch.step()
    model.eval(); return model

def propose_diffusion(model, V, L, C, look, spreads, val, n_valid, rng, outdir):
    kept, tries, files, slots_of = 0, 0, [], {}
    while kept < n_valid and tries < n_valid * 4:
        row = W.sample_constrained(model, C, 1, 5, "none", 2.0).cpu().numpy()[0]; tries += 1
        slots = []
        for i in range(NSLOT):
            b = i * NF; g = lambda j: V.decode_field(b + j, int(row[b + j])); sp = g(0)
            mv = [look.get(g(3 + j), g(3 + j)) for j in range(4) if g(3 + j) and g(3 + j) != "[MASK]"]
            pl = spreads.get(sp) or [{s: 0 for s in STATS}]
            slots.append(dict(species=look.get(sp, sp), item=look.get(g(2), g(2)), ability=look.get(g(1), g(1)),
                              nature=look.get(g(7), g(7)), moves=mv, evs=pl[int(rng.integers(0, len(pl)))]))
        txt = "\n\n".join(slot_to_text(s) for s in slots) + "\n"
        if val(txt) is None:
            p = outdir / f"{kept:04d}.txt"; p.write_text(txt); files.append(str(p)); slots_of[str(p)] = slots; kept += 1
    return files, slots_of, kept / max(tries, 1)

# ---------- copy-paste proposer ----------
def propose_copypaste(pool_teams, pool_wr, val, n_valid, rng, outdir):
    order = np.argsort(-np.array(pool_wr)); elite = [pool_teams[i] for i in order[: max(20, len(order) // 5)]]
    kept, tries, files, slots_of = 0, 0, [], {}
    while kept < n_valid and tries < n_valid * 6:
        tries += 1
        a = [dict(s) for s in canon(elite[int(rng.integers(0, len(elite)))])]; b = canon(elite[int(rng.integers(0, len(elite)))])
        a[int(rng.integers(0, 6))] = dict(b[int(rng.integers(0, 6))])
        sp = [norm(s["species"]) for s in a]; it = [norm(s["item"]) for s in a if s["item"]]
        if len(set(sp)) != 6 or len(set(it)) != len(it): continue
        txt = team_to_text(a)
        if val(txt) is None:
            p = outdir / f"{kept:04d}.txt"; p.write_text(txt); files.append(str(p)); slots_of[str(p)] = a; kept += 1
    return files, slots_of, kept / max(tries, 1)

def main(smoke=False):
    global PROPOSE, BATTLE, BATTLES, ROUNDS
    if smoke: PROPOSE, BATTLE, BATTLES, ROUNDS = 6, 3, 2, 1
    rng = np.random.default_rng(0); torch.manual_seed(0); random.seed(0)
    corpus, _ = load_corpus(); corpus = [t for t in corpus if len(t) == 6]
    lab_teams, lab_wrs = W.load_labelled()
    V = Vocab(corpus + lab_teams); L = Legality(corpus); C = D.Constraints(V, L); pmi = PMI(corpus)
    look, spreads = {}, defaultdict(list)
    for t in corpus + lab_teams:
        for s in t:
            look[norm(s["species"])] = s["species"]; look[norm(s["ability"])] = s["ability"]
            if s["item"]: look[norm(s["item"])] = s["item"]
            for mv in s["moves"]: look[norm(mv)] = mv
            if s["nature"]: look[norm(s["nature"])] = s["nature"]
            spreads[norm(s["species"])].append(dict(s["evs"]))
    real_surprise = float(np.mean([pmi.surprise(t) for t in corpus]))
    opp = opponents(); val = Validator()
    results = json.load(open(OUT)) if OUT.exists() and not smoke else {"real_surprise": real_surprise, "rounds": {}}
    # separate pools per loop
    pools = {"diffusion": (list(lab_teams), list(lab_wrs)), "copypaste": (list(lab_teams), list(lab_wrs))}
    model = W.TeamDiffusionWR(V).to(W.DEV)
    ck = torch.load("/tmp/vgc-pilot/wrdiffusion.pt", map_location=W.DEV)
    try: model.load_state_dict(ck["sd"]); print("diffusion: initialised from wrdiffusion.pt", flush=True)
    except Exception as e: print("diffusion: vocab mismatch, training from scratch this round:", str(e)[:80], flush=True)
    model.eval()
    for r in range(1, ROUNDS + 1):
        for loop in ["diffusion", "copypaste"]:
            key = f"{loop}_r{r}"
            if key in results["rounds"]: continue
            teams, wrs = pools[loop]; outdir = ROOT / key; outdir.mkdir(exist_ok=True)
            for f in outdir.glob("*.txt"): f.unlink()
            t0 = time.perf_counter()
            if loop == "diffusion":
                model = retrain(model, V, teams, wrs, epochs=(3 if smoke else 60)); trained = time.perf_counter() - t0
                files, slots_of, acc = propose_diffusion(model, V, L, C, look, spreads, val, PROPOSE, rng, outdir)
            else:
                trained = 0.0; files, slots_of, acc = propose_copypaste(teams, wrs, val, PROPOSE, rng, outdir)
            sur = {f: pmi.surprise(slots_of[f]) for f in files}
            chosen = sorted(files, key=lambda f: -sur[f])[:BATTLE]
            wr = battle(chosen, opp)
            for f in chosen:
                t = parse_team(f)
                if t and len(t) == 6: teams.append(t); wrs.append(wr[f])
            w = np.array([wr[f] for f in chosen]); s = np.array([sur[f] for f in chosen])
            results["rounds"][key] = dict(loop=loop, round=r, proposed=len(files), valid_rate=acc, battled=len(chosen),
                mean=float(w.mean()), se=float(w.std(ddof=1) / np.sqrt(len(w))) if len(w) > 1 else None,
                p90=float(np.percentile(w, 90)), max=float(w.max()), above_real_median=float((w >= .458).mean()),
                surprise_mean=float(s.mean()), pool_size=len(teams), minutes=(time.perf_counter() - t0) / 60, retrain_min=trained / 60)
            if not smoke: json.dump(results, open(OUT, "w"), indent=1)
            print(f"[{key}] proposed {len(files)} (valid {acc:.0%}) · battled {len(chosen)} · mean {w.mean():.4f} · p90 {np.percentile(w,90):.3f} · "
                  f"max {w.max():.3f} · ≥0.458: {(w>=.458).mean():.1%} · surprise {s.mean():.2f} (real {real_surprise:.2f}) · {(time.perf_counter()-t0)/60:.0f} min", flush=True)
    val.close(); print("LOOP_DONE", flush=True)

if __name__ == "__main__":
    main(smoke=("smoke" in sys.argv))
