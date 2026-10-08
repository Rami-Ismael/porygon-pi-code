"""Predictor guidance from a closed-form surrogate.

Today's structure test: a ridge on species indicators + pairs explains 60% of held-out
win rate. Because it is additive, the surrogate's contribution to choosing species v for
a slot is closed form -- w[v] plus the pair terms with the already-decided slots -- which
is exactly the per-value shape the discrete sampler needs. So we can add

    scale * ( w_species[v] + sum_{u decided} w_pair[u,v] )

to the species logits during constrained decoding. This is predictor guidance
(Nisonoff et al. ICLR 2025 call it that because the conditioning quantity is real-valued),
with no neural classifier: the "predictor" is 189 + 581 numbers.

Arms, all with the win-rate bin 5 condition at guidance 2 (the current best configuration):
    ridge0  scale 0  -- control, identical to the existing generator
    ridge2  scale 2
    ridge6  scale 6
    ridge15 scale 15
plus a free best-of-n baseline: generate 4x, keep the quarter the ridge scores highest.
"""
import json, sys, time, itertools
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "/tmp/vgc-pilot/src")
import diffusion as D, wrdiffusion as W
from corpus import load_corpus, parse_team, norm, STATS
from encode import Vocab, Legality, NF, NSLOT
from propose import Validator, slot_to_text
import pool

MIN_PAIR = 20

def fit_ridge(V, lam_s=10.0, lam_p=30.0):
    lab = {}
    for f in ["/tmp/vgc-pilot/ladder_labels.json", "/tmp/vgc-pilot/strata_labels.json"]:
        for p, v in json.load(open(f)).items(): lab[p] = v["win_rate"]
    teams, ys = [], []
    for p, w in lab.items():
        t = parse_team(p)
        if t and len(t) == 6: teams.append(sorted({norm(s["species"]) for s in t})); ys.append(w)
    y = np.array(ys)
    sp = V.itos["species"]; si = {s: i for i, s in enumerate(sp)}      # index 0 is [MASK]
    pc = Counter()
    for t in teams:
        for a, b in itertools.combinations(t, 2): pc[(a, b)] += 1
    pairs = [p for p, c in pc.items() if c >= MIN_PAIR]; pi = {p: i for i, p in enumerate(pairs)}
    X = np.zeros((len(teams), len(sp) + len(pairs)))
    for r, t in enumerate(teams):
        for s in t:
            if s in si: X[r, si[s]] = 1
        for a, b in itertools.combinations(t, 2):
            if (a, b) in pi: X[r, len(sp) + pi[(a, b)]] = 1
    Xb = np.hstack([X, np.ones((len(X), 1))])
    lam = np.concatenate([np.full(len(sp), lam_s), np.full(len(pairs), lam_p), [0.0]])
    w = np.linalg.solve(Xb.T @ Xb + np.diag(lam), Xb.T @ y)
    ws = w[:len(sp)]; wp = {p: w[len(sp) + i] for p, i in pi.items()}
    # held-out check
    rng = np.random.default_rng(0); idx = rng.permutation(len(y)); nte = len(y) // 5
    te, tr = idx[:nte], idx[nte:]
    w2 = np.linalg.solve(Xb[tr].T @ Xb[tr] + np.diag(lam), Xb[tr].T @ y[tr]); pred = Xb[te] @ w2
    r2 = 1 - ((y[te] - pred) ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum()
    print(f"ridge surrogate: {len(sp)} species + {len(pairs)} pairs · held-out R² {r2:.3f}", flush=True)
    return ws, wp, r2

def team_score(ws, wp, V, row):
    """Surrogate prediction for a decoded team."""
    sp = sorted({V.decode_field(i * NF, int(row[i * NF])) for i in range(NSLOT)})
    s = sum(ws[V.stoi["species"].get(x, 0)] for x in sp)
    for a, b in itertools.combinations(sp, 2): s += wp.get((a, b), 0.0) + wp.get((b, a), 0.0)
    return float(s)

@torch.no_grad()
def sample_ridge(model, C, V, ws, wp, n, scale, wbin=5, g_wr=2.0, temp=1.0, device=D.DEV):
    """Constrained decoding + CFG on the win-rate bin + closed-form ridge guidance on species."""
    y = torch.full((n,), D.NULL, device=device, dtype=torch.long)
    b5 = torch.full((n,), wbin, device=device, dtype=torch.long); wN = torch.full((n,), W.WNULL, device=device, dtype=torch.long)
    x = torch.zeros(n, W.COLS, dtype=torch.long, device=device)
    sp_names = V.itos["species"]; nsp = len(sp_names)
    ridge_vec = torch.tensor(ws, dtype=torch.float32, device=device)
    for step, c in enumerate(D.ORDER):
        t_now = 1.0 - step / len(D.ORDER); tt = torch.full((n,), t_now, device=device)
        lu = F.log_softmax(model.logits(model(x, tt, y, wN), c), -1)
        lc = F.log_softmax(model.logits(model(x, tt, y, b5), c), -1)
        l = lu + g_wr * (lc - lu)
        if scale > 0 and c % NF == 0:                       # species column: add the surrogate's term
            slot = c // NF
            bonus = ridge_vec.unsqueeze(0).repeat(n, 1)
            for bi in range(n):
                decided = [V.decode_field(s2 * NF, int(x[bi, s2 * NF])) for s2 in range(NSLOT) if s2 != slot and int(x[bi, s2 * NF]) != 0]
                if decided:
                    add = torch.zeros(nsp, device=device)
                    for vi in range(1, nsp):
                        v = sp_names[vi]
                        add[vi] = sum(wp.get((min(v, u), max(v, u)), 0.0) for u in decided)
                    bonus[bi] = bonus[bi] + add
            l = l + scale * bonus
        for bi in range(n):
            ok = C.mask_for(c, x[bi], device); lb = l[bi].clone(); lb[~ok] = -1e9
            x[bi, c] = torch.multinomial(torch.softmax(lb / temp, -1), 1).item()
    return x

def main():
    per = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    corpus, _ = load_corpus(); corpus = [t for t in corpus if len(t) == 6]
    lab_teams, _ = W.load_labelled()
    V = Vocab(corpus + lab_teams); L = Legality(corpus); C = D.Constraints(V, L)
    model = W.TeamDiffusionWR(V).to(D.DEV)
    model.load_state_dict(torch.load("/tmp/vgc-pilot/wrdiffusion.pt", map_location=D.DEV)["sd"]); model.eval()
    ws, wp, r2 = fit_ridge(V)
    look, spreads = {}, defaultdict(list)
    for t in corpus + lab_teams:
        for s in t:
            look[norm(s["species"])] = s["species"]; look[norm(s["ability"])] = s["ability"]
            if s["item"]: look[norm(s["item"])] = s["item"]
            for mv in s["moves"]: look[norm(mv)] = mv
            if s["nature"]: look[norm(s["nature"])] = s["nature"]
            spreads[norm(s["species"])].append(dict(s["evs"]))
    rng = np.random.default_rng(0); torch.manual_seed(0)
    root = Path("/tmp/vgc-pilot/ridge_gen"); root.mkdir(exist_ok=True)
    for f in root.glob("*.txt"): f.unlink()
    val = Validator(); meta = {}; stats = {}
    def emit(tag, row, k):
        slots = []
        for i in range(NSLOT):
            b = i * NF; g = lambda j: V.decode_field(b + j, int(row[b + j])); s0 = g(0)
            mv = [look.get(g(3 + j), g(3 + j)) for j in range(4) if g(3 + j) and g(3 + j) != "[MASK]"]
            pl = spreads.get(s0) or [{s: 0 for s in STATS}]
            slots.append(dict(species=look.get(s0, s0), item=look.get(g(2), g(2)), ability=look.get(g(1), g(1)),
                              nature=look.get(g(7), g(7)), moves=mv, evs=pl[int(rng.integers(0, len(pl)))]))
        txt = "\n\n".join(slot_to_text(s) for s in slots) + "\n"
        if val(txt) is None:
            p = root / f"{tag}_{k:04d}.txt"; p.write_text(txt); meta[str(p)] = tag; return str(p)
        return None
    for tag, scale in [("ridge0", 0.0), ("ridge2", 2.0), ("ridge6", 6.0), ("ridge15", 15.0)]:
        kept = tries = 0; t0 = time.perf_counter()
        while kept < per and tries < per * 4:
            nb = min(10, per * 4 - tries); tries += nb
            X = sample_ridge(model, C, V, ws, wp, nb, scale).cpu().numpy()
            for row in X:
                if kept < per and emit(tag, row, kept): kept += 1
        stats[tag] = dict(scale=scale, valid=kept / max(tries, 1), gen_min=(time.perf_counter() - t0) / 60)
        print(f"  {tag:8s} scale {scale:4.1f} · Showdown-valid {kept}/{tries} ({stats[tag]['valid']:.0%}) · {stats[tag]['gen_min']:.0f} min", flush=True)
    # best-of-4 by the surrogate, at scale 0 (free filter, no guidance)
    cand = []; tries = 0
    while len(cand) < per * 4 and tries < per * 12:
        nb = min(10, per * 12 - tries); tries += nb
        X = sample_ridge(model, C, V, ws, wp, nb, 0.0).cpu().numpy()
        for row in X: cand.append((team_score(ws, wp, V, row), row))
    cand.sort(key=lambda z: -z[0]); kept = 0
    for _, row in cand:
        if kept >= per: break
        if emit("bestof4", row, kept): kept += 1
    stats["bestof4"] = dict(scale=None, valid=kept / max(tries, 1), gen_min=0)
    print(f"  bestof4  surrogate-filtered top quarter · kept {kept}", flush=True)
    val.close()
    rootd = Path("/tmp/vgc-pilot/vgc-bench/teams/reg_mb"); opp = []
    for i in [t["id"] for t in json.load(open("/tmp/vgc-pilot/top50_evs.json"))]:
        for c in (rootd / f"{i}.txt", rootd / "featured" / f"{i}.txt"):
            if c.exists(): opp.append(str(c)); break
    print(f"\nbattling {len(meta)} teams x 24 vs {len(opp)} meta teams", flush=True)
    res = pool.score([(t, opp) for t in sorted(meta)], battles=24, conc=50)
    for t, v in res.items(): v["arm"] = meta[t]
    json.dump({"results": res, "stats": stats, "ridge_r2": r2}, open("/tmp/vgc-pilot/ridge_eval.json", "w"), indent=1)
    print(f"\n{'arm':9s} {'valid':>6s} {'n':>4s} {'win rate':>9s} {'clustered SE':>13s} {'p90':>6s} {'max':>6s} {'≥0.458':>7s}")
    for tag in ["ridge0", "ridge2", "ridge6", "ridge15", "bestof4"]:
        w = np.array([v["win_rate"] for v in res.values() if v["arm"] == tag])
        if len(w): print(f"{tag:9s} {stats[tag]['valid']:6.0%} {len(w):4d} {w.mean():9.4f} {w.std(ddof=1)/np.sqrt(len(w)):13.4f} {np.percentile(w,90):6.3f} {w.max():6.3f} {(w>=.458).mean():7.1%}")
    print("reference: unconditioned 0.128 · bin5 g2 0.168 · bin5 g4 0.193 · copy-paste loop 0.494 · real team 0.470")
    print("RIDGE_DONE", flush=True)

if __name__ == "__main__":
    main()
