"""Cross-entropy loop on the version-3 foundation (2026-10-08). Pre-registration: docs/cem-v3-loop.md.

Per generation: re-steer (fine-tune the ORIGINAL v3 members on the elites) -> refit the ridge with the
loop's labels -> propose 512 valid teams (ask 0.5 g2, temperature 1, ridge guidance, nature repair)
-> battle the 128 the ridge ranks highest (matchup matrix, one battle per top-50 column) -> elites =
every labelled team >= 0.531 (min 64). Resumable at every step; run under battle_supervisor.

    python battle_supervisor.py cem_v3.py run --then final
    python cem_v3.py report
(run with /tmp/vgc-pilot/.venv/bin/python)
"""
import itertools, json, math, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "/tmp/vgc-pilot/src")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from showdown_validator import VGC_BENCH
from checkpoints import load_checkpoint
import foundation as FD
import lossdown as L
import lossdown_battle as LB
import asked_vs_got as X
import activesearch as A
import matchup_db as MDB
import heldout as HO               # ranks 51-100: not proposed, not labelled, scored once at the end
import pool
from hps_generate import Validator
from corpus import parse_team_text, norm
from encode import Vocab, team_fields

import os
TAG = os.environ.get("CEM_TAG", "cem_v3")        # CEM_TAG=cem_v3_noheldout: own result file, checkpoints, matrix origin
OUT = L.RES / f"{TAG}.json"
BASE = L.RES / "cem_v3_ridge_base_noheldout.npz"   # the unsuffixed cache was fitted with ranks 51-100 in it
GENS, PROPOSE, BATTLE, CUT, ELITE_MIN = 10, 512, 128, 0.531, 64
FT_EPOCHS, FT_LR, FT_BATCH, LOOP_W = 40, 1e-4, 64, 10.0
# Elite rule (2026-10-16). "cumulative" (default) = every labelled team so far + the top-50 anchors at >= CUT,
# topped up to ELITE_MIN. "topfrac" = the owner's intended rule: the top ELITE_FRAC of the PREVIOUS generation's
# battled teams by label, no anchors, nothing carried over. CEM_FT_EPOCHS / CEM_GENS override for that arm.
ELITE_RULE, ELITE_FRAC = os.environ.get("CEM_ELITE", "cumulative"), float(os.environ.get("CEM_ELITE_FRAC", 0.2))
assert ELITE_RULE in ("cumulative", "topfrac")
FT_EPOCHS = int(os.environ.get("CEM_FT_EPOCHS", FT_EPOCHS)); GENS = int(os.environ.get("CEM_GENS", GENS))
SAT_EPS, SAT_PATIENCE, COPY_MAX, SETS_FRAC = 0.01, 3, 0.10, 0.5   # sets guardrail: half of generation 0's
SEED, FINAL_N, FINAL_BATTLES = 4040, 16, 192
# Species guidance inside the loop (2026-10-16, docs/species-guidance-loop.md): CEM_SEED = a new search seed,
# CEM_G_SPECIES = classifier-free guidance weight on the six species columns at every generation's proposer
# (unset = the usual 2 everywhere). Unset, the loop runs unchanged.
SEED = int(os.environ.get("CEM_SEED", SEED))
G_SPECIES = float(os.environ["CEM_G_SPECIES"]) if os.environ.get("CEM_G_SPECIES") else None
LAM_S, LAM_P = 10.0, 30.0
SMOKE = bool(os.environ.get("CEM_SMOKE"))
if SMOKE:                                     # plumbing check: tiny sizes, own file, own origin tag
    TAG += "_smoke"; OUT = L.RES / f"{TAG}.json"
    GENS, PROPOSE, BATTLE, ELITE_MIN, FT_EPOCHS, FINAL_N, FINAL_BATTLES = 2, 16, 4, 8, 1, 2, 8
LEGACY = set(json.load(open(Path.home() / ".local/share/vgc-pilot-runtime/regmb-v3/data-audit.json"))["legacy_vocabulary"]["species"])


def save(out): json.dump(out, open(OUT, "w"))


# ---------------------------------------------------------------- ridge with the foundation feature map
class Ridge:
    def __init__(self):
        z = np.load(FD.RIDGE)
        self.keys, self.pairs_l = list(z["keys"]), list(z["pairs"])
        self.idx = {tuple(k.split("|", 1)): i for i, k in enumerate(self.keys)}
        off = len(self.keys)
        self.pairs = {tuple(p.split("|", 1)): off + i for i, p in enumerate(self.pairs_l)}
        self.dim = off + len(self.pairs) + 1
        self.lam = np.concatenate([np.full(off, LAM_S), np.full(len(self.pairs), LAM_P), [0.0]])

    def vec(self, team):
        x = np.zeros(self.dim); x[-1] = 1.0
        sps = sorted({norm(s["species"]) for s in team})
        for a, b in itertools.combinations(sps, 2):
            j = self.pairs.get((a, b))
            if j is not None: x[j] = 1.0
        for s in team:
            for k, v in (("species", s["species"]), ("ability", s["ability"]), ("item", s["item"]), ("nature", s["nature"])):
                j = self.idx.get((k, norm(v))) if v else None
                if j is not None: x[j] = 1.0
            for m in s["moves"]:
                j = self.idx.get(("move", norm(m)))
                if j is not None: x[j] = 1.0
        return x

    def base(self):
        """X'X and X'y of the 44,887 foundation labels under this exact feature map (cached)."""
        if BASE.exists():
            z = np.load(BASE); return z["XtX"], z["Xty"]
        corpus, V, *_ = X.setup(); d = L.load_data()
        lab = np.where(np.isfinite(d["W"]))[0]
        XtX = np.zeros((self.dim, self.dim)); Xty = np.zeros(self.dim)
        for i in range(0, len(lab), 4000):
            Xb = np.stack([self.vec(FD.row_team(V, r)) for r in d["X"][lab[i:i + 4000]]])
            XtX += Xb.T @ Xb; Xty += Xb.T @ d["W"][lab[i:i + 4000]].astype(float)
        np.savez(BASE, XtX=XtX, Xty=Xty); return XtX, Xty

    def fit(self, teams, ys, path):
        XtX, Xty = self.base()
        XtX, Xty = XtX.copy(), Xty.copy()
        if teams:
            Xe = np.stack([self.vec(t) for t in teams]); ye = np.asarray(ys, float)
            XtX += LOOP_W * Xe.T @ Xe; Xty += LOOP_W * Xe.T @ ye
        w = np.linalg.solve(XtX + np.diag(self.lam), Xty)
        z = np.load(FD.RIDGE)
        np.savez(path, w=w, keys=z["keys"], pairs=z["pairs"], lam=z["lam"], rho=z["rho"], n=z["n"])
        return path


# ---------------------------------------------------------------- the model: v3 members, re-steered from the original
def members(V, tables, elites, g):
    """Fine-tuned copies of the ORIGINAL v3 members (cached per generation)."""
    nets = []
    for name in FD.V3_MEMBERS:
        ck = L.CK / f"{TAG}_g{g}_{name}.pt"
        net, cfg, _ = L.load_net(name, V)
        if ck.exists():
            net.load_state_dict(load_checkpoint(ck, map_location=L.DEV)["sd"])
        elif elites is not None:
            torch.manual_seed(SEED + g)
            Xe = torch.as_tensor(np.stack([e[0] for e in elites]), device=L.DEV)
            We = torch.as_tensor(np.array([e[1] for e in elites], np.float32), device=L.DEV)
            if cfg.get("mrow"): We = torch.cat([We[:, None], torch.full((len(We), 49), float("nan"), device=L.DEV)], 1)
            sched = L.sched_tensor(cfg["sched"]).to(L.DEV); rules = tuple(cfg["rules"].split(","))
            opt = torch.optim.AdamW(net.parameters(), lr=FT_LR, weight_decay=0.01)
            net.train()
            for _ in range(FT_EPOCHS):
                perm = torch.randperm(len(Xe), device=L.DEV)
                for i in range(0, len(Xe), FT_BATCH):
                    b = perm[i:i + FT_BATCH]
                    loss = L.batch_loss(net, tables, rules, Xe[b], We[b], None, sched, cfg)
                    opt.zero_grad(set_to_none=True); loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            net.eval()
            torch.save({"sd": net.state_dict(), "cfg": ck.name}, ck)
        nets.append(L.logprobs_fn(net, tables, tuple(cfg["rules"].split(","))))
    return L.ensemble_fn(nets)


# ---------------------------------------------------------------- diversity (docs/species-guidance-loop.md)
def vendi(lineups):
    """Vendi score (Friedman & Dieng, TMLR 2023), K_ij = shared species / 6 — as guidance_timing_cads.vendi."""
    vocab = {s: i for i, s in enumerate(sorted({s for l in lineups for s in l}))}
    B = np.zeros((len(lineups), len(vocab)))
    for i, l in enumerate(lineups):
        for s in l: B[i, vocab[s]] += 1 / math.sqrt(6)
    lam = np.linalg.eigvalsh(B @ B.T / len(lineups)); lam = lam[lam > 1e-12]
    return float(np.exp(-(lam * np.log(lam)).sum()))


def elite_lineups(el, V):
    """Distinct species line-ups among encoded elites (rows of V-indices)."""
    return {tuple(sorted(V.decode_field(s * FD.NF, int(e[0][s * FD.NF])) for s in range(FD.NSLOT))) for e in el}


# ---------------------------------------------------------------- labelled pool
def anchors(con):
    """The 49 distinct top-50 meta teams with their matrix win rates against the other columns."""
    rows = []
    for tid in MDB.set_members(con, "top50"):
        (p,) = con.execute("SELECT paste FROM team WHERE team_id=?", (tid,)).fetchone()
        s = MDB.score(con, LB.POLICY, tid)
        if s["score"] is not None: rows.append(dict(paste=p, y=s["score"], src="anchor"))
    return rows


def labelled(out):
    rows = list(out["anchors"])
    for g, c in sorted(out["gens"].items(), key=lambda kv: int(kv[0])):
        if "y" in c: rows += [dict(paste=p, y=y, src=f"g{g}") for p, y in zip(c["battled"], c["y"]) if y is not None]
    return rows


def elites_from(rows, V):
    ok = [r for r in rows if r["y"] >= CUT]
    if len(ok) < ELITE_MIN: ok = sorted(rows, key=lambda r: -r["y"])[:ELITE_MIN]
    out = []
    for r in ok:
        t = parse_team_text(r["paste"])
        if len(t) != 6: continue                                  # parse_team_text merges item-less slots (known): skip
        try: out.append((V.encode(t), float(r["y"]), r["src"]))
        except ValueError: pass                                   # outside the regulation vocabulary: skipped
    return out


# ---------------------------------------------------------------- one run
def run():
    LB._stack()
    con = MDB.connect()
    out = json.load(open(OUT)) if OUT.exists() else {"config": dict(gens=GENS, propose=PROPOSE, battle=BATTLE, cut=CUT,
                         elite_min=ELITE_MIN, elite_rule=ELITE_RULE, elite_frac=ELITE_FRAC, ft_epochs=FT_EPOCHS, ft_lr=FT_LR, loop_weight=LOOP_W, seed=SEED,
                         g_species=G_SPECIES), "gens": {}}
    if "anchors" not in out:
        out["anchors"] = anchors(con); save(out)
    G = FD.Gen("v3"); V, tables = G.V, G.tables
    corpus = G.corpus
    known = {tuple(team_fields(parse_team_text(r["paste"]))) for r in out["anchors"]}
    cV = X.setup()[1]; d = L.load_data()
    known |= {tuple(cV.decode_field(c, int(r[c])) for c in range(L.COLS)) for r in d["X"]}
    ridge = Ridge()
    best = []
    for g in range(GENS):
        c = out["gens"].setdefault(str(g), {})
        rows = labelled({**out, "gens": {k: v for k, v in out["gens"].items() if int(k) < g}})
        # 1. model
        if g > 0 and ELITE_RULE == "topfrac":
            prev = sorted([r for r in rows if r["src"] == f"g{g - 1}"], key=lambda r: -r["y"])
            el = elites_from(prev[:math.ceil(ELITE_FRAC * len(prev))], V)
        else:
            el = elites_from(rows, V) if g > 0 else None
        if g > 0 and "elites" not in c:
            c["elites"] = dict(n=len(el), anchors=sum(e[2] == "anchor" for e in el), cut=min(e[1] for e in el),
                               qualified=sum(r["y"] >= CUT for r in rows), sets=len(elite_lineups(el, V)))
        G.f = members(V, tables, el, g)
        # 2. ridge
        loop_rows = [r for r in rows if r["src"] != "anchor"]
        FD.RIDGE = ridge.fit([parse_team_text(r["paste"]) for r in loop_rows], [r["y"] for r in loop_rows],
                             L.RES / f"{TAG}_ridge_g{g}.npz") if loop_rows else FD.RIDGE
        sur = FD.Surrogate(V)
        # 3. propose
        if "proposals" not in c:
            torch.manual_seed(SEED + 100 * g); rng = np.random.default_rng(SEED + 100 * g)
            val = Validator(); keep, tried, t0 = [], 0, time.perf_counter()
            try:
                while len(keep) < PROPOSE and tried < PROPOSE * 8:
                    for r in G.sample(96, 0.5, 2.0, 1.0, 1.0, sur, sur.lam, g_species=G_SPECIES).cpu().numpy():
                        tried += 1
                        p, slots = G.to_paste(r, rng, repair=True)
                        if val(p) is None and not HO.is_heldout_text(p): keep.append(p)
                        if len(keep) >= PROPOSE: break
            finally:
                val.close()
            teams = [parse_team_text(p) for p in keep]
            c.update(proposals=keep, mu=[float(sur.score(t)) for t in teams], tried=tried, valid=len(keep) / tried,
                     sets=len({tuple(sorted(norm(s["species"]) for s in t)) for t in teams}),
                     vendi=vendi([tuple(sorted(norm(s["species"]) for s in t)) for t in teams]),
                     copies=sum(tuple(team_fields(t)) in known for t in teams) / max(len(teams), 1),
                     new_species=sum(any(norm(s["species"]) not in LEGACY for s in t) for t in teams),
                     propose_s=time.perf_counter() - t0)
            save(out)
            print(f"[g{g}] proposed {len(keep)}/{tried} valid · {c['sets']} species sets · copies {c['copies']:.1%} · "
                  f"new-species teams {c['new_species']}", flush=True)
        # 4. acquire + battle
        if "y" not in c:
            order = np.argsort(c["mu"])[::-1][:BATTLE]
            c["battled"] = [c["proposals"][i] for i in order]; c["battled_mu"] = [c["mu"][i] for i in order]
            v = MDB.Validator()
            try: ids = [MDB.add_team(con, p, f"{TAG}:g{g}", v)[0] for p in c["battled"]]
            finally: v.close()
            legal = {t for (t,) in con.execute(f"SELECT team_id FROM team WHERE legal=1 AND team_id IN ({','.join('?' * len(ids))})", ids)}
            MDB.ensure(con, LB.POLICY, sorted(legal), MDB.set_members(con, "top50"), LB.PER_CELL, seed=SEED + g, note=f"{TAG} g{g}")
            sc = [MDB.score(con, LB.POLICY, t) if t in legal else None for t in ids]
            c["team_id"] = ids; c["y"] = [s["score"] if s else None for s in sc]
            y = np.array([v for v in c["y"] if v is not None], float)
            bt = [parse_team_text(p) for p in c["battled"]]
            nsp = [any(norm(s["species"]) not in LEGACY for s in t) for t in bt]
            c["stats"] = dict(mean=float(y.mean()), se=float(y.std(ddof=1) / math.sqrt(len(y))), p90=float(np.quantile(y, .9)),
                              max=float(y.max()), above_cut=int((y >= CUT).sum()), n=len(y),
                              battled_sets=len({tuple(sorted(norm(s["species"]) for s in t)) for t in bt}),
                              new_species_battled=int(sum(nsp)),
                              new_species_mean=float(np.mean([v for v, f in zip(c["y"], nsp) if f and v is not None])) if any(nsp) else None)
            save(out)
        known |= {tuple(team_fields(parse_team_text(p))) for p in c["battled"]}
        s = c["stats"]; best.append(s["mean"])
        print(f"[g{g}] battled {s['n']}: mean {s['mean']:.3f} ± {s['se']:.3f} · p90 {s['p90']:.3f} · max {s['max']:.3f} · "
              f"≥{CUT}: {s['above_cut']} · new-species teams {s['new_species_battled']}", flush=True)
        # 5. stopping
        stop = None
        if c["copies"] > COPY_MAX: stop = f"copy rate {c['copies']:.1%} > {COPY_MAX:.0%}"
        elif g > 0 and c["sets"] < SETS_FRAC * out["gens"]["0"]["sets"]:
            stop = f"{c['sets']} species sets < half of generation 0's {out['gens']['0']['sets']}"
        elif len(best) > SAT_PATIENCE:
            run_best = [max(best[:i + 1]) for i in range(len(best))]
            if run_best[-1] - run_best[-1 - SAT_PATIENCE] < SAT_EPS: stop = f"saturated: no +{SAT_EPS} in {SAT_PATIENCE} generations"
        if stop:
            out["stopped"] = dict(generation=g, reason=stop); save(out)
            print(f"[g{g}] STOP: {stop}", flush=True); return
    out["stopped"] = dict(generation=GENS - 1, reason="max generations"); save(out)


def final():
    """The 16 best distinct battled teams by loop label, re-battled fresh: 192 battles vs the top-50 schedule."""
    out = json.load(open(OUT))
    if "final" in out: return heldout_eval(out)
    rows = [r for r in labelled(out) if r["src"] != "anchor"]
    rows.sort(key=lambda r: -r["y"])
    seen, top = set(), []
    for r in rows:
        k = tuple(team_fields(parse_team_text(r["paste"])))
        if k in seen: continue
        seen.add(k); top.append(r)
        if len(top) >= FINAL_N: break
    pool.REPO = str(VGC_BENCH)
    d = Path(f"/tmp/vgc-pilot/{TAG}_final"); d.mkdir(parents=True, exist_ok=True)
    files = []
    for i, r in enumerate(top):
        f = d / f"{i:02d}.txt"; f.write_text(r["paste"]); files.append(str(f))
    res = pool.score([(f, A.opponents()) for f in files], battles=FINAL_BATTLES, conc=50)
    out["final"] = [dict(paste=r["paste"], label=r["y"], src=r["src"], fresh=res.get(f, {}).get("win_rate"),
                         battles=res.get(f, {}).get("battles")) for r, f in zip(top, files)]
    save(out)
    heldout_eval(out)
    report()


def heldout_eval(out):
    """The only place ranks 51-100 are battled: the final selected teams, once, after selection.
    Selection above used the top-50 only; nothing here feeds back into a generation."""
    if "heldout" in out or "final" not in out: return
    pool.REPO = str(VGC_BENCH)
    d = Path(f"/tmp/vgc-pilot/{TAG}_final"); d.mkdir(parents=True, exist_ok=True); files = []
    for i, r in enumerate(out["final"]):
        f = d / f"{i:02d}.txt"; f.write_text(r["paste"]); files.append(str(f))
    res = HO.evaluate(files, battles=FINAL_BATTLES, conc=50)
    out["heldout"] = [dict(fresh_top50=r["fresh"], heldout=res.get(f, {}).get("win_rate"), battles=res.get(f, {}).get("battles"))
                      for r, f in zip(out["final"], files)]
    save(out)


def report():
    out = json.load(open(OUT))
    print(f"{'gen':>4s} {'mean':>13s} {'p90':>6s} {'max':>6s} {'>=cut':>6s} {'sets/512':>9s} {'copies':>7s} {'new-sp':>7s} {'elites':>7s}")
    for g, c in sorted(out["gens"].items(), key=lambda kv: int(kv[0])):
        if "stats" not in c: continue
        s = c["stats"]; e = c.get("elites", {})
        print(f"{g:>4s} {s['mean']:.3f}±{s['se']:.3f} {s['p90']:6.3f} {s['max']:6.3f} {s['above_cut']:6d} {c['sets']:9d} "
              f"{c['copies']:7.1%} {s['new_species_battled']:7d} {e.get('n', '-'):>7}")
    if out.get("stopped"): print("stopped:", out["stopped"])
    if out.get("heldout"):
        h = [r["heldout"] for r in out["heldout"] if r["heldout"] is not None]
        print(f"held-out ranks 51-100: {len(h)} final teams x {FINAL_BATTLES}: mean {np.mean(h):.3f} · best {max(h):.3f}")
    if out.get("final"):
        f = [r for r in out["final"] if r["fresh"] is not None]
        print(f"final: {len(f)} teams re-battled x {FINAL_BATTLES}: fresh mean {np.mean([r['fresh'] for r in f]):.3f} "
              f"(loop labels {np.mean([r['label'] for r in f]):.3f}) · best fresh {max(r['fresh'] for r in f):.3f} · "
              f"≥0.531 fresh: {sum(r['fresh'] >= CUT for r in f)}")


if __name__ == "__main__":
    {"run": run, "final": final, "report": report}[sys.argv[1]]()
