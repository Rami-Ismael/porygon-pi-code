"""Trust-Region Noise Search (TRS) ported to the HPS diffusion generator.

Source: Schweiger, Cremers, Ram — "Trust-Region Noise Search for Black-Box
Alignment of Diffusion and Flow Models" (arXiv 2603.14504v2), Algorithm 1 with
the released code (github.com/niklasschweiger/trust-region-noise-search) as the
tiebreaker on ambiguities. Full mapping, budget arithmetic, and pre-registered
comparison: docs/trs-port.md — written before this file.

The noise is x in [0,1]^M (M = 10,542): per column, a slice of the unit cube
mapped to Gumbel noise by -log(-log(u)), decoded by argmax over the feasible
set of (logits/temp + G) — the Gumbel-max trick makes decode(x) for
x ~ U[0,1]^M an EXACT sample from the constrained decoder, and the map
noise -> team fully deterministic (CPU decode; the legality projection is an
in-step mask, not accept/reject). Six extra coordinates pick the Stat Points
spread from the per-species corpus pool. See docs/trs-port.md section 2.

Adaptations forced by the domain, all disclosed in the doc:
  - the reward is a 96-battle Monte Carlo estimate on a FIXED opponent schedule
    (common random numbers through pool.score(opp_schedule=, seed=));
  - candidates entering the global top-k are re-battled on an independent fixed
    schedule before promotion; rankings use the pooled estimate;
  - identical decoded teams are label-cached, never re-battled;
  - legality is a hard constraint (Showdown validator), never a reward term;
  - budgets are counted in battles and capped per arm (budget-matched arms).

  geom   : decode-only tests — determinism + plateau curve + validity. No battles.
  run    : one arm's campaign (--arm trs|rands|bofn), battle-budget capped.
  report : re-evaluate every arm's top-8 at 192 fresh battles, print the table.
"""
import argparse, hashlib, inspect, json, math, os, sys, time
from collections import Counter
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, "/tmp/vgc-pilot/src")
import hpsdiffusion as H
import diffusion as D
from encode import NF, NSLOT
from corpus import STATS
from hps_generate import Validator, slot_to_text
from hps_eval import build_decode_tables, row_to_paste
import pool
from activesearch import opponents

OUTDIR = Path("/tmp/vgc-pilot/trs")
RESULTS = "/Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/trs.json"
COLS = D.COLS

# ---- pre-registered constants (docs/trs-port.md sections 3 and 5) ----------
BATTLES = 96            # per reward call, schedule A
BATTLES_F = 192         # final report, schedule F
CAP = 50_000            # battles per arm, hard
SEARCH_CAP = 46_000     # stop searching here; the rest is the report's reserve
BATCH = 20              # B  (paper T2I)
NREGION = 15            # k  (paper T2I)
WARMUP = 80             # 20% of the paper's 400-call budget
L_INIT, L_MAX, ALPHA = 0.8, 2.4, 1.5
C_SUCC = C_FAIL = 3
P_MIN, P_MAX = 0.1, 0.9
EPS = 1e-6
SEED = 7000             # shard reseed, shared by every candidate in every arm

def schedules(opp):
    """Fixed opponent schedules A (search), B (promotion re-eval), F (report)."""
    a = np.random.default_rng(0xA).choice(opp, BATTLES).tolist()
    b = np.random.default_rng(0xB).choice(opp, BATTLES).tolist()
    f = np.random.default_rng(0xF).choice(opp, BATTLES_F).tolist()
    return a, b, f

# ---- noise -> team --------------------------------------------------------
class NoiseSpace:
    """x in [0,1]^M  ->  a decoded, spread-assigned team. Deterministic on CPU."""
    def __init__(self, V, spreads):
        self.V = V
        self.spreads = spreads
        self.sizes = [V.sizes[V.key(c)] for c in range(COLS)]
        self.off = np.concatenate([[0], np.cumsum(self.sizes)]).astype(int)
        self.M = int(self.off[-1]) + NSLOT          # + 6 spread coordinates
    def gumbel(self, X, c):
        u = np.clip(X[:, self.off[c]:self.off[c+1]], EPS, 1 - EPS)
        return torch.tensor(-np.log(-np.log(u)), dtype=torch.float32)
    def spread_idx(self, x_row, slot, npool):
        u = min(max(float(x_row[self.off[-1] + slot]), EPS), 1 - EPS)
        return min(int(u * npool), npool - 1)

@torch.no_grad()
def decode(model, C, ns, X, wbin=H.WNULL, guidance=1.0, temp=1.0):
    """`hpsdiffusion.sample_constrained` with argmax(logits/temp + Gumbel(x))
    in place of torch.multinomial — same distribution, deterministic in X."""
    n = X.shape[0]
    dev = "cpu"
    w = torch.full((n,), wbin, dtype=torch.long)
    x = torch.zeros(n, COLS, dtype=torch.long)
    for step, c in enumerate(D.ORDER):
        t_now = 1.0 - step / COLS
        lg = H._step_logits(model, x, t_now, w, guidance)[c]
        G = ns.gumbel(X, c)
        for b in range(n):
            ok = C.mask_for(c, x[b], dev)
            l = lg[b].clone(); l[~ok] = -1e9
            x[b, c] = int(torch.argmax(l / temp + G[b]))
    return x

def row_to_paste_det(V, ns, row, x_row, look, spreads):
    """hps_eval.row_to_paste with the spread indexed by the noise, not an RNG."""
    slots = []
    for i in range(NSLOT):
        b = i * NF
        g = lambda j: V.decode_field(b + j, int(row[b + j]))
        sp = g(0)
        mv = [look.get(g(3+j), g(3+j)) for j in range(4) if g(3+j) and g(3+j) != "[MASK]"]
        pool_ = spreads.get(sp) or [{s: 0 for s in STATS}]
        evs = pool_[ns.spread_idx(x_row, i, len(pool_))]
        slots.append(dict(species=look.get(sp, sp), item=look.get(g(2), g(2)),
                          ability=look.get(g(1), g(1)), nature=look.get(g(7), g(7)),
                          moves=mv, evs=dict(evs)))
    return "\n\n".join(slot_to_text(s) for s in slots) + "\n"

def teamhash(txt):
    return hashlib.sha1(txt.encode()).hexdigest()[:16]

# ---- label store ----------------------------------------------------------
class Labels:
    """Pooled win-rate estimates per unique team, block A (search) + block B
    (promotion re-eval). Persisted so a killed run never re-buys battles, and
    the final ranking is rebuilt from THIS file, never from in-memory state."""
    def __init__(self, path):
        self.path = Path(path)
        self.d = json.load(open(path)) if self.path.exists() else {}
        self.spent = sum(v.get("nA", 0) + v.get("nB", 0) + v.get("nF", 0)
                         for v in self.d.values())
    def save(self):
        tmp = str(self.path) + ".tmp"
        json.dump(self.d, open(tmp, "w")); os.replace(tmp, self.path)
    def block_a(self, h):
        v = self.d[h]; return v["wA"] / max(v["nA"], 1)
    def pooled(self, h):
        v = self.d[h]
        n = v.get("nA", 0) + v.get("nB", 0)
        return (v.get("wA", 0) + v.get("wB", 0)) / max(n, 1)
    def has_b(self, h):
        return self.d.get(h, {}).get("nB", 0) > 0

def _crn_guard():
    """The runtime tree is /tmp/vgc-pilot/src; refuse to run against stale
    staged copies that would silently drop common random numbers."""
    assert "opp_schedule" in inspect.signature(pool.score).parameters, \
        "stale pool.py at /tmp/vgc-pilot/src — sync from the repo first"
    assert "ScheduledTeamBuilder" in open(pool.SHARD).read(), \
        "stale shard.py at /tmp/vgc-pilot/src — sync from the repo first"

def battle(labels, hashes, sched, block, files_of, n_battles):
    """One pool.score call for the given team hashes on a fixed schedule.
    CAP is enforced HERE: a call never starts battles it has no budget for."""
    todo = [h for h in dict.fromkeys(hashes)
            if labels.d.get(h, {}).get("n" + block, 0) == 0]
    room = max(0, (CAP - labels.spent) // n_battles)
    todo = todo[:room]
    if not todo:
        return 0
    opp = opponents()
    jobs = [(files_of[h], opp) for h in todo]
    res = pool.score(jobs, battles=n_battles, conc=50, opp_schedule=sched, seed=SEED)
    spent = 0
    for h in todo:
        r = res.get(files_of[h])
        if not r or r["battles"] <= 0:
            continue
        v = labels.d.setdefault(h, {})
        v["w" + block] = r["wins"]; v["n" + block] = r["battles"]
        spent += r["battles"]
    labels.spent += spent
    labels.save()
    return spent

def _ranked(labels, hashes):
    """Deterministic ranking: pooled estimate, hash as the tiebreak (set
    iteration order is PYTHONHASHSEED-dependent — never let it decide)."""
    return sorted(set(hashes), key=lambda h: (labels.pooled(h), h), reverse=True)

def promote(labels, hashes, files_of, sched_b, k, cap=SEARCH_CAP):
    """Re-battle would-be centers on schedule B until the top-k is all pooled.
    Ranking uses the pooled estimate; a lucky block A gets caught by block B.
    Retries per hash are bounded so a persistently failing shard cannot spin."""
    tried = Counter()
    while True:
        ranked = _ranked(labels, hashes)
        need = [h for h in ranked[:k] if not labels.has_b(h) and tried[h] < 2]
        if not need or labels.spent >= cap:
            return ranked[:k]
        for h in need:
            tried[h] += 1
        if battle(labels, need, sched_b, "B", files_of, BATTLES) == 0:
            return ranked[:k]

# ---- shared candidate pipeline -------------------------------------------
class Pipeline:
    def __init__(self, arm):
        t0 = time.perf_counter()
        _crn_guard()
        teams, _ = H.load_hps()
        self.V, L = H.build_vocab(teams)
        self.look, self.spreads = build_decode_tables(teams)
        self.model = H.load_model(self.V).to("cpu")
        self.C = D.Constraints(self.V, L)
        self.ns = NoiseSpace(self.V, self.spreads)
        self.val = Validator()
        self.dir = OUTDIR / arm
        self.dir.mkdir(parents=True, exist_ok=True)
        self.files = {}
        for f in self.dir.glob("*.txt"):        # resume: re-index existing teams
            self.files[f.stem] = str(f)
        print(f"[{arm}] setup {time.perf_counter()-t0:.0f}s, M={self.ns.M}", flush=True)
    def realize(self, X):
        """Decode a noise batch -> (hash, valid?) per row; write valid team files."""
        rows = decode(self.model, self.C, self.ns, X)
        out = []
        for i in range(X.shape[0]):
            txt = row_to_paste_det(self.V, self.ns, rows[i], X[i], self.look, self.spreads)
            h = teamhash(txt)
            if h in self.files:
                out.append((h, True)); continue
            ok = self.val(txt) is None
            if ok:
                p = self.dir / f"{h}.txt"
                p.write_text(txt); self.files[h] = str(p)
            out.append((h, ok))
        return out, rows

# ---- the TRS arm ----------------------------------------------------------
def run_trs(l_min):
    pipe = Pipeline("trs")
    labels = Labels(OUTDIR / "trs_labels.json")
    sched_a, sched_b, _ = schedules(opponents())
    eng = torch.quasirandom.SobolEngine(pipe.ns.M, scramble=True, seed=SEED)
    rng = np.random.default_rng(SEED)
    stats = []

    # warm-up: Sobol over the unit cube IS the generator prior here
    Xw = eng.draw(WARMUP, dtype=torch.float64).numpy()
    real, _ = pipe.realize(Xw)
    valid = [(Xw[i], h) for i, (h, ok) in enumerate(real) if ok]
    battle(labels, [h for _, h in valid], sched_a, "A", pipe.files, BATTLES)
    archive = [(x, h) for x, h in valid if h in labels.d]
    print(f"[trs] warm-up: {len(archive)}/{WARMUP} valid+labelled, "
          f"{labels.spent} battles", flush=True)

    top = promote(labels, [h for _, h in archive], pipe.files, sched_b, NREGION)
    centers = []                                 # (x, hash) in pooled order
    for h in top:
        centers.append((next(x for x, hh in archive if hh == h), h))
    k = len(centers)          # frozen for the run, as in the reference solver
    if k < NREGION:
        print(f"[trs] only {k} centers after warm-up; num_regions reduced", flush=True)
    length = [L_INIT] * k
    succ = [0] * k; fail = [0] * k
    rengines = [torch.quasirandom.SobolEngine(pipe.ns.M, scramble=True, seed=SEED + 1 + j)
                for j in range(k)]

    it = 0
    while labels.spent < SEARCH_CAP:
        it += 1
        alloc = [BATCH // k + (1 if j < BATCH % k else 0) for j in range(k)]
        X, owner = [], []
        for j, nj in enumerate(alloc):
            p = rng.uniform(P_MIN, P_MAX)        # one mask rate per region per iter
            u = rengines[j].draw(nj, dtype=torch.float64).numpy()
            offset = length[j] * (u - 0.5)
            mask = rng.random((nj, pipe.ns.M)) < p
            for r in range(nj):
                if not mask[r].any():
                    mask[r, rng.integers(pipe.ns.M)] = True
            cand = np.clip(centers[j][0][None, :] + offset * mask, EPS, 1 - EPS)
            X.append(cand); owner += [j] * nj
        X = np.concatenate(X)
        real, _ = pipe.realize(X)
        fresh = [h for (h, ok) in real if ok]
        battle(labels, fresh, sched_a, "A", pipe.files, BATTLES)

        plateau_r = [0] * k
        for i, (h, ok) in enumerate(real):
            if ok and h == centers[owner[i]][1]:
                plateau_r[owner[i]] += 1
        plateau = sum(plateau_r)
        best_a = [-1.0] * k
        for i, (h, ok) in enumerate(real):
            if ok and h in labels.d:
                best_a[owner[i]] = max(best_a[owner[i]], labels.block_a(h))
                archive.append((X[i], h))
        for j in range(k):
            if best_a[j] > labels.block_a(centers[j][1]):
                succ[j] += 1; fail[j] = 0
            else:
                fail[j] += 1; succ[j] = 0
            if succ[j] >= C_SUCC:
                length[j] = min(length[j] * ALPHA, L_MAX); succ[j] = 0
            elif fail[j] >= C_FAIL:
                if length[j] <= l_min + 1e-12:
                    length[j] = L_INIT           # restart (paper Appendix B)
                else:
                    length[j] = max(length[j] / ALPHA, l_min)
                fail[j] = 0

        top = promote(labels, [h for _, h in archive], pipe.files, sched_b, k)
        by_hash = {}
        for x, h in archive:
            by_hash.setdefault(h, x)
        centers = [(by_hash[h], h) for h in top]

        nvalid = len(fresh)
        row = dict(iter=it, spent=labels.spent, valid=nvalid, batch=len(real),
                   plateau=plateau, plateau_by_region=plateau_r,
                   distinct=len(set(fresh)),
                   best_pooled=round(labels.pooled(top[0]), 4),
                   lengths=[round(l, 3) for l in length])
        stats.append(row)
        print(f"[trs] it{it:02d} spent {labels.spent:6d}  validity {nvalid}/{len(real)}"
              f"  plateau {plateau}  best(pooled) {row['best_pooled']}"
              f"  l[:5] {row['lengths'][:5]}", flush=True)
    _finish("trs", labels, pipe.files, sched_b, stats)

# ---- baselines ------------------------------------------------------------
def run_random_search():
    """The paper's own baseline: i.i.d. prior noise, same protocol, same cap."""
    pipe = Pipeline("rands")
    labels = Labels(OUTDIR / "rands_labels.json")
    sched_a, sched_b, _ = schedules(opponents())
    rng = np.random.default_rng(SEED + 500)
    archive, stats, it = [], [], 0
    while labels.spent < SEARCH_CAP:
        it += 1
        X = rng.random((BATCH, pipe.ns.M))
        real, _ = pipe.realize(X)
        fresh = [h for (h, ok) in real if ok]
        battle(labels, fresh, sched_a, "A", pipe.files, BATTLES)
        for i, (h, ok) in enumerate(real):
            if ok and h in labels.d:
                archive.append((X[i], h))
        top = promote(labels, [h for _, h in archive], pipe.files, sched_b, NREGION)
        stats.append(dict(iter=it, spent=labels.spent, valid=len(fresh),
                          batch=len(real), best_pooled=round(labels.pooled(top[0]), 4)))
        print(f"[rands] it{it:02d} spent {labels.spent:6d}  validity {len(fresh)}/{len(real)}"
              f"  best(pooled) {stats[-1]['best_pooled']}", flush=True)
    _finish("rands", labels, pipe.files, sched_b, stats)

def run_best_of_n():
    """Best-of-N through the EXISTING multinomial decode path — identical in
    distribution to rands by the Gumbel-max construction; run as an A/A check."""
    pipe = Pipeline("bofn")
    labels = Labels(OUTDIR / "bofn_labels.json")
    sched_a, sched_b, _ = schedules(opponents())
    rng = np.random.default_rng(SEED + 900)
    torch.manual_seed(SEED + 900)
    archive, stats, it = [], [], 0
    while labels.spent < SEARCH_CAP:
        it += 1
        rows = H.sample_constrained(pipe.model, pipe.C, BATCH, H.WNULL, device="cpu")
        fresh, batch = [], 0
        for i in range(BATCH):
            batch += 1
            # spread drawn like hps_eval.row_to_paste (rng), then frozen in the file
            txt = row_to_paste(pipe.V, rows[i], pipe.look, pipe.spreads, rng)
            h = teamhash(txt)
            if h not in pipe.files:
                if pipe.val(txt) is not None:
                    continue
                p = pipe.dir / f"{h}.txt"; p.write_text(txt); pipe.files[h] = str(p)
            fresh.append(h)
        battle(labels, fresh, sched_a, "A", pipe.files, BATTLES)
        archive += [(None, h) for h in fresh if h in labels.d]
        top = promote(labels, [h for _, h in archive], pipe.files, sched_b, NREGION)
        stats.append(dict(iter=it, spent=labels.spent, valid=len(fresh), batch=batch,
                          best_pooled=round(labels.pooled(top[0]), 4)))
        print(f"[bofn] it{it:02d} spent {labels.spent:6d}  validity {len(fresh)}/{batch}"
              f"  best(pooled) {stats[-1]['best_pooled']}", flush=True)
    _finish("bofn", labels, pipe.files, sched_b, stats)

def _finish(arm, labels, files_of, sched_b, stats):
    """Final ranking, rebuilt from the PERSISTED labels (survives kills and
    reruns). One last promotion pass pools the final iteration's block-A-only
    entrants, with the report's schedule-F budget held in reserve; the top-16
    then prefers pooled (A+B) teams so no winner's-curse A-only score leads."""
    all_h = [h for h, v in labels.d.items() if v.get("nA", 0) > 0]
    promote(labels, all_h, files_of, sched_b, 16, cap=CAP - 8 * BATTLES_F)
    ranked = sorted(all_h, key=lambda h: (labels.has_b(h), labels.pooled(h), h),
                    reverse=True)
    if not stats:                       # rerun after completion: keep the log
        old = json.load(open(RESULTS)) if Path(RESULTS).exists() else {}
        stats = old.get(arm, {}).get("iters", [])
    out = dict(arm=arm, spent=labels.spent, iters=stats,
               top16=[dict(hash=h, pooled=round(labels.pooled(h), 4),
                           nA=labels.d[h].get("nA", 0), nB=labels.d[h].get("nB", 0))
                      for h in ranked[:16]])
    _write_results(arm, out)
    print(f"[{arm}] done: {labels.spent} battles, best pooled "
          f"{out['top16'][0]['pooled'] if out['top16'] else None} -> {RESULTS}", flush=True)

def _write_results(key, value):
    res = json.load(open(RESULTS)) if Path(RESULTS).exists() else {}
    res[key] = value
    tmp = RESULTS + ".tmp"
    json.dump(res, open(tmp, "w"), indent=1); os.replace(tmp, RESULTS)

# ---- geometry (decode-only, zero battles) ---------------------------------
def run_geom():
    pipe = Pipeline("geom")
    eng = torch.quasirandom.SobolEngine(pipe.ns.M, scramble=True, seed=1)
    rng = np.random.default_rng(1)

    X = eng.draw(20, dtype=torch.float64).numpy()
    r1, _ = pipe.realize(X)
    r2, _ = pipe.realize(X)
    r3 = [pipe.realize(X[i:i+1])[0][0] for i in range(5)]
    det_batch = all(a[0] == b[0] for a, b in zip(r1, r2))
    det_size = all(r1[i][0] == r3[i][0] for i in range(5))
    print(f"determinism: repeat={det_batch}  batchsize={det_size}", flush=True)

    Xv = eng.draw(200, dtype=torch.float64).numpy()
    rv, _ = pipe.realize(Xv)
    print(f"prior validity: {sum(ok for _, ok in rv)}/200", flush=True)

    grid = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 2.4]
    ncenter, nprop = 8, 64
    Xc = eng.draw(ncenter, dtype=torch.float64).numpy()
    rc, crows = pipe.realize(Xc)
    curve = {}
    for l in grid:
        same_cols = same_hash = changed = valid = distinct = 0
        fields = []
        for ci in range(ncenter):
            u = rng.random((nprop, pipe.ns.M))
            p = rng.uniform(P_MIN, P_MAX, nprop)
            mask = rng.random((nprop, pipe.ns.M)) < p[:, None]
            cand = np.clip(Xc[ci][None, :] + l * (u - 0.5) * mask, EPS, 1 - EPS)
            rr, rows = pipe.realize(cand)
            hs = set()
            for i, (h, ok) in enumerate(rr):
                nchanged = int((rows[i] != crows[ci]).sum())
                if nchanged == 0:
                    # the section-2c event: the 48-column decode is unmoved
                    # (spread-coordinate churn is tracked via the hash, below)
                    same_cols += 1
                else:
                    changed += 1; valid += ok
                    fields.append(nchanged)
                if h == rc[ci][0]:
                    same_hash += 1
                hs.add(h)
            distinct += len(hs)
        n = ncenter * nprop
        curve[l] = dict(identical=round(same_cols / n, 3),
                        identical_hash=round(same_hash / n, 3),
                        fields_changed=round(float(np.mean(fields)), 2) if fields else 0.0,
                        validity_changed=round(valid / max(changed, 1), 3),
                        distinct_per_batch=round(distinct / ncenter, 1))
        print(f"l={l:4.2f}  identical(48col) {curve[l]['identical']:.0%}  "
              f"(hash {curve[l]['identical_hash']:.0%})  "
              f"fields changed {curve[l]['fields_changed']:5.2f}  "
              f"validity {curve[l]['validity_changed']:.0%}  "
              f"distinct/batch {curve[l]['distinct_per_batch']}", flush=True)
    _write_results("geom", dict(det_repeat=det_batch, det_batchsize=det_size,
                                prior_validity=sum(ok for _, ok in rv) / 200,
                                curve=curve))
    print("wrote", RESULTS, flush=True)

# ---- final report ---------------------------------------------------------
def run_report():
    _crn_guard()
    res = json.load(open(RESULTS))
    _, _, sched_f = schedules(opponents())
    meta_sets, meta_fields = _meta_reference()
    for arm in ("trs", "rands", "bofn"):
        if arm not in res or "top16" not in res[arm]:
            continue
        labels = Labels(OUTDIR / f"{arm}_labels.json")
        files = {p.stem: str(p) for p in (OUTDIR / arm).glob("*.txt")}
        top8 = [t["hash"] for t in res[arm]["top16"][:8]]
        battle(labels, top8, sched_f, "F", files, BATTLES_F)
        battle(labels, top8, sched_f, "F", files, BATTLES_F)   # retry shard losses
        rows, failed = [], []
        for h in top8:
            v = labels.d.get(h, {})
            if v.get("nF", 0) == 0:
                failed.append(h)          # NEVER report a fake 0.0 for a lost shard
                continue
            wr = v["wF"] / v["nF"]
            se = math.sqrt(max(wr * (1 - wr), 1e-9) / v["nF"])
            tf = _team_fields_of(files[h])
            sp = frozenset(tf[i * NF] for i in range(NSLOT))
            rows.append(dict(hash=h, wr_f=round(wr, 4), se=round(se, 4),
                             pooled_selection=round(labels.pooled(h), 4),
                             species=sorted(sp), copy=sp in meta_sets,
                             max_field_overlap=max((sum(a == b for a, b in zip(tf, mf))
                                                    for mf in meta_fields), default=0)))
        battled = [h for h, v in labels.d.items() if v.get("nA", 0) > 0 and h in files]
        bf = [_team_fields_of(files[h]) for h in battled]
        wrs = [r["wr_f"] for r in rows]
        res[arm]["report"] = dict(
            top8=rows, failed_f=failed,
            mean_top8=round(float(np.mean(wrs)), 4) if wrs else None,
            se_top8=round(float(np.std(wrs) / max(len(wrs), 1) ** .5), 4) if wrs else None,
            best=round(max(wrs), 4) if wrs else None,
            copy_rate=round(float(np.mean([r["copy"] for r in rows])), 3) if rows else None,
            distinct_species_sets_top8=len({tuple(r["species"]) for r in rows}),
            battled_teams=len(battled),
            distinct_species_sets_battled=len({frozenset(t[i * NF] for i in range(NSLOT))
                                               for t in bf}),
            mean_nn_hamming_battled=_mean_nn_hamming(bf),
            spent_total=labels.spent)
        r = res[arm]["report"]
        print(f"[{arm}] top-8 mean {r['mean_top8']} (se {r['se_top8']}), best {r['best']}, "
              f"copies {r['copy_rate']}, NN-Hamming {r['mean_nn_hamming_battled']}, "
              f"total battles {labels.spent}"
              + (f", FAILED F: {failed}" if failed else ""), flush=True)
    for other in ("rands", "bofn"):
        key = f"diff_trs_vs_{other}"
        a = res.get("trs", {}).get("report", {}).get("top8", [])
        b = res.get(other, {}).get("report", {}).get("top8", [])
        if a and b:
            res[key] = _boot_diff([r["wr_f"] for r in a], [r["wr_f"] for r in b])
            print(f"trs - {other}: {res[key]}", flush=True)
    tmp = RESULTS + ".tmp"
    json.dump(res, open(tmp, "w"), indent=1); os.replace(tmp, RESULTS)

def _boot_diff(a, b, n=10000):
    """Team-level bootstrap CI of mean(a) - mean(b) — the section-5 endpoint."""
    rng = np.random.default_rng(0)
    a, b = np.array(a), np.array(b)
    d = [float(np.mean(rng.choice(a, len(a))) - np.mean(rng.choice(b, len(b))))
         for _ in range(n)]
    return dict(diff=round(float(a.mean() - b.mean()), 4),
                ci95=[round(float(np.percentile(d, 2.5)), 4),
                      round(float(np.percentile(d, 97.5)), 4)])

def _team_fields_of(path):
    from corpus import parse_team_text
    from encode import team_fields
    return team_fields(parse_team_text(Path(path).read_text()))

def _mean_nn_hamming(fields_list):
    """Mean nearest-neighbour Hamming distance over the canonical 48 fields."""
    n = len(fields_list)
    if n < 2:
        return None
    best = []
    for i in range(n):
        ti = fields_list[i]
        best.append(min(sum(a != b for a, b in zip(ti, fields_list[j]))
                        for j in range(n) if j != i))
    return round(float(np.mean(best)), 2)

def _meta_reference():
    from corpus import parse_team_text
    from encode import team_fields
    sets, fields = set(), []
    root = Path("/Users/ramiismael/Documents/code/vgc-team-generator-pilot/teams/reg_mb")
    for p in list(root.glob("*.txt")) + list((root / "featured").glob("*.txt")):
        try:
            tf = team_fields(parse_team_text(p.read_text()))
            if len(tf) == NSLOT * NF:
                sets.add(frozenset(tf[i * NF] for i in range(NSLOT)))
                fields.append(tf)
        except Exception:
            pass
    return sets, fields

def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["geom", "run", "report"])
    ap.add_argument("--arm", choices=["trs", "rands", "bofn"], default="trs")
    ap.add_argument("--lmin", type=float, default=0.05,
                    help="set from the geom plateau curve, docs/trs-port.md 2c")
    a = ap.parse_args()
    if a.cmd == "geom":
        run_geom()
    elif a.cmd == "report":
        run_report()
    elif a.arm == "trs":
        run_trs(a.lmin)
    elif a.arm == "rands":
        run_random_search()
    else:
        run_best_of_n()

if __name__ == "__main__":
    cli()
