"""The self-training loop: sample the diffusion model, admit novel legal teams,
retrain, evaluate, stop — per docs/self-training-loop-protocol.md.

Loop invariants (the four blocking fixes from the protocol):
  A. admission key is spread-free and slot/move-order-invariant, checked against
     a persistent index of every team ever in the dataset (rebuilt from files,
     so a crashed generation can rerun without poisoning the index)
  B. admission and evaluation sampling are unbiased (temp 1.0, no top-p,
     guidance 1.0, unconditional) — biased decoding never feeds the dataset
  C. every admitted row carries provenance; the diffusion-sourced fraction of
     each retraining mix is capped at half the real-row count (~1/3 of the mix)
  D. one checkpoint and one admissions file per generation, nothing overwritten

Frozen references, built once by `init`:
  - holdout: the 2,000 original lines train(seed=0) held out (recomputed from
    the same rng stream, written to disk) — excluded from every retrain, so the
    shipped hpsdiffusion.pt is a clean generation 0 that never saw them
  - the generation-0 checkpoint (the shipped hpsdiffusion.pt, copied)
  - a fresh 100k HPS reference never trained on (E7's yardstick) + a 5k batch
    that calibrates E7's Hamming radius

  init       build indices, grids, references (runs hps_generate for the ref)
  calibrate  two eval draws from generation 0 -> noise bands
  run        drive generations until a stop rule fires or --max_gens
  status     print the per-generation table so far
  speedtest  time 64 unbiased samples, extrapolate the admission batch

Results: results/selftrain_results.json (+ per-gen files under teams/selftrain/).
"""
import argparse, json, shutil, subprocess, sys, time
from collections import Counter
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F

import hpsdiffusion as H
import diffusion as D
from encode import NF, NSLOT, team_fields
from corpus import parse_team_text
from hps_generate import Validator
from hps_eval import build_decode_tables, row_to_paste

DEV = H.DEV
COLS = D.COLS
REPO = Path("/Users/ramiismael/Documents/code/vgc-team-generator-pilot")
TMP = Path("/tmp/vgc-pilot")
STDIR = REPO / "teams" / "selftrain"
RESULTS = REPO / "results" / "selftrain_results.json"
REF_JSONL = REPO / "teams" / "hps_ref_100k.jsonl"
CAL_JSONL = REPO / "teams" / "hps_refcal_5k.jsonl"
REF_GRIDS = REPO / "results" / "selftrain_ref_grids.npy"
REF_SS = REPO / "results" / "selftrain_ref_species_sets.txt"
KEY_INDEX = STDIR / "orig_keys.txt"          # the original 100k only; admissions
HOLDOUT = REPO / "results" / "selftrain_holdout_idx.json"   # live in their gen files

def ckpt_path(g):
    return TMP / f"selftrain_gen{g}.pt"

def gen_file(g):
    return STDIR / f"admitted_gen{g}.jsonl"

# ---- identity ---------------------------------------------------------------
def fields_of_row(V, row):
    """Decoded 48 normalised strings, '[MASK]' rendered as ''. Same space as
    encode.team_fields, so keys from pastes and keys from grids agree."""
    f = [V.decode_field(c, int(row[c])) for c in range(COLS)]
    return ["" if v == "[MASK]" else v for v in f]

def key_from_fields(f):
    """Spread-free, slot-order- and move-order-invariant team identity."""
    slots = []
    for i in range(NSLOT):
        s = f[i * NF:(i + 1) * NF]
        slots.append((s[0], s[1], s[2]) + tuple(sorted(s[3:7])) + (s[7],))
    return "||".join(";".join(sl) for sl in sorted(slots))

def canon_grid(V, row):
    """Canonicalise a sampled grid the way encode.team_fields canonicalises a
    paste (slots sorted by species, moves sorted within a slot) so Hamming
    distances between sampled and encoded-corpus grids are real."""
    slots = []
    for i in range(NSLOT):
        s = [int(v) for v in row[i * NF:(i + 1) * NF]]
        mv = sorted(s[3:7], key=lambda ix: V.decode_field(3, ix))
        slots.append((V.decode_field(0, s[0]), [s[0], s[1], s[2], *mv, s[7]]))
    out = []
    for _, s in sorted(slots, key=lambda x: x[0]):
        out += s
    return np.array(out, dtype=np.int16)

def species_set(f):
    return ";".join(sorted({f[i * NF] for i in range(NSLOT)}))

# ---- shared state -----------------------------------------------------------
class Ctx:
    """Everything derived from the frozen original 100k, loaded once per process."""
    def __init__(self):
        t0 = time.time()
        self.teams, self.W = H.load_hps()
        self.V, self.L = H.build_vocab(self.teams)
        self.C = D.Constraints(self.V, self.L)
        self.look, self.spreads = build_decode_tables(self.teams)
        self.X = np.stack([self.V.encode(t) for t in self.teams])
        self.holdout = np.array(json.load(open(HOLDOUT))) if HOLDOUT.exists() else None
        self.orig_fields = [team_fields(t) for t in self.teams]
        self.orig_keys = {key_from_fields(f) for f in self.orig_fields}
        self.orig_ss = {species_set(f) for f in self.orig_fields}
        self.pair_ref = pair_dist(self.orig_fields)
        print(f"ctx ready in {time.time()-t0:.0f}s ({len(self.teams)} teams)", flush=True)

def load_model_at(ctx, g):
    m = H.TeamDiffusionHPS(ctx.V).to(DEV)
    m.load_state_dict(torch.load(ckpt_path(g), map_location=DEV)["sd"]); m.eval()
    return m

def admitted_rows(up_to=None):
    rows = []
    for p in sorted(STDIR.glob("admitted_gen*.jsonl")):
        g = int(p.stem.replace("admitted_gen", ""))
        if up_to is not None and g > up_to:
            continue
        rows += [json.loads(l) for l in open(p)]
    return rows

def current_key_index(ctx):
    """Original keys plus every admitted key — rebuilt from files each time, so
    a rerun after a crash cannot leave orphaned keys blocking admission."""
    return ctx.orig_keys | {r["key"] for r in admitted_rows()}

def load_results():
    return json.load(open(RESULTS)) if RESULTS.exists() else {"generations": []}

def save_results(r):
    RESULTS.write_text(json.dumps(r, indent=1))

# ---- unbiased batched sampling ---------------------------------------------
@torch.no_grad()
def sample_unbiased(model, C, n, seed, batch=512):
    """Constrained dependency-order decode, unconditional, temp 1.0 — the
    admission/eval protocol. Batched: one forward per column, masks + draws on
    CPU (per-item .item() syncs on MPS are what made the original slow)."""
    torch.manual_seed(seed)
    out = []
    for i in range(0, n, batch):
        b = min(batch, n - i)
        x = torch.zeros(b, COLS, dtype=torch.long)
        w = torch.full((b,), H.WNULL, dtype=torch.long, device=DEV)
        for step, c in enumerate(D.ORDER):
            t_now = 1.0 - step / len(D.ORDER)
            tt = torch.full((b,), t_now, device=DEV)
            h = model(x.to(DEV), tt, w)
            lg = model.logits(h, c).float().cpu()
            masks = torch.stack([C.mask_for(c, x[j], "cpu") for j in range(b)])
            lg[~masks] = -1e9
            x[:, c] = torch.multinomial(torch.softmax(lg, -1), 1).squeeze(1)
        out.append(x.numpy())
    return np.concatenate(out)

# ---- metrics ----------------------------------------------------------------
def pair_dist(fields_list):
    """Unordered species-pair frequency distribution over teams."""
    cnt = Counter()
    for f in fields_list:
        sp = sorted(f[i * NF] for i in range(NSLOT))
        for a in range(NSLOT):
            for b in range(a + 1, NSLOT):
                cnt[(sp[a], sp[b])] += 1
    tot = sum(cnt.values())
    return {k: v / tot for k, v in cnt.items()}

def pair_tv(p, q):
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)

def species_entropy(fields_list):
    cnt = Counter(f[i * NF] for f in fields_list for i in range(NSLOT))
    tot = sum(cnt.values())
    ps = np.array([v / tot for v in cnt.values()])
    return float(-(ps * np.log(ps)).sum())

def nn_hamming(A, B, chunk=64):
    """Per-row-of-A min Hamming distance to B (both k x 48 int16)."""
    out = np.empty(len(A), dtype=np.int32)
    for i in range(0, len(A), chunk):
        d = (A[i:i + chunk, None, :] != B[None, :, :]).sum(2)
        out[i:i + chunk] = d.min(1)
    return out

@torch.no_grad()
def holdout_loss(model, ctx, seed, reps=16):
    """E0: masked-prediction loss on the frozen holdout, averaged over `reps`
    random maskings under a fixed seed so generations are comparable."""
    torch.manual_seed(seed)
    xte = torch.tensor(ctx.X[ctx.holdout], device=DEV)
    wte = torch.tensor(ctx.W[ctx.holdout], device=DEV)
    vals = [float(model.loss(xte, wte)) for _ in range(reps)]
    return float(np.mean(vals)), float(np.std(vals) / np.sqrt(reps))

@torch.no_grad()
def per_sample_loss(model, X, seed, K=4, batch=1024):
    """E5: mean masked cross-entropy per cell, per sample, under `model`
    (used with the frozen generation-0 checkpoint)."""
    torch.manual_seed(seed)
    n = len(X)
    tot = np.zeros(n); cnt = np.zeros(n)
    for i in range(0, n, batch):
        x = torch.tensor(X[i:i + batch].astype(np.int64), device=DEV)
        b = x.shape[0]
        w = torch.full((b,), H.WNULL, dtype=torch.long, device=DEV)
        for _ in range(K):
            t = torch.rand(b, device=DEV).clamp(min=0.15)
            m = torch.rand(b, COLS, device=DEV) < t.view(-1, 1)
            empty = ~m.any(1)
            if empty.any():
                m[empty, torch.randint(0, COLS, (int(empty.sum()),), device=DEV)] = True
            xin = x.clone(); xin[m] = 0
            h = model(xin, t, w)
            ce = torch.zeros(b, COLS, device=DEV)
            for k, cols in model.cols_by_key.items():
                lg = model.head[k](h[:, cols])
                ce[:, cols] = F.cross_entropy(
                    lg.reshape(-1, lg.shape[-1]), x[:, cols].reshape(-1),
                    reduction="none").view(b, len(cols))
            ce = ce * m
            tot[i:i + b] += ce.sum(1).cpu().numpy()
            cnt[i:i + b] += m.sum(1).cpu().numpy()
    return tot / np.maximum(cnt, 1)

def evaluate_checkpoint(ctx, g, seed, n_eval, model0, key_index, label=""):
    """The per-generation evaluation: E0..E5 on `n_eval` unbiased samples."""
    model = load_model_at(ctx, g)
    t0 = time.time()
    rows = sample_unbiased(model, ctx.C, n_eval, seed)
    t_sample = time.time() - t0
    fields = [fields_of_row(ctx.V, r) for r in rows]
    grids = np.stack([canon_grid(ctx.V, r) for r in rows])
    keys = [key_from_fields(f) for f in fields]
    ss = [species_set(f) for f in fields]

    val = Validator()
    rng = np.random.default_rng(seed)
    valid = np.array([val(row_to_paste(ctx.V, r, ctx.look, ctx.spreads, rng)) is None
                      for r in rows])
    val.close()

    e0_mean, e0_se = holdout_loss(model, ctx, seed=97)
    ho_grids = ctx.X[ctx.holdout].astype(np.int16)
    e4_nn = float(nn_hamming(ho_grids, grids).mean())
    e5 = per_sample_loss(model0, grids, seed=53)
    m = {
        "gen": g, "label": label, "n_eval": n_eval, "seed": seed,
        "E0_holdout_loss": round(e0_mean, 4), "E0_se": round(e0_se, 4),
        "E1_validity": round(float(valid.mean()), 4),
        "E2_novel_vs_orig": round(float(np.mean([k not in ctx.orig_keys for k in keys])), 4),
        "E2_novel_vs_current": round(float(np.mean([k not in key_index for k in keys])), 4),
        "E3_distinct_species_sets": len(set(ss)),
        "E3_species_sets_in_orig": round(float(np.mean([s in ctx.orig_ss for s in ss])), 4),
        "E4_species_entropy": round(species_entropy(fields), 4),
        "E4_pair_tv_vs_orig": round(pair_tv(pair_dist(fields), ctx.pair_ref), 4),
        "E4_mean_nn_hamming_holdout": round(e4_nn, 2),
        "E5_gen0_loss_mean": round(float(e5.mean()), 4),
        "E5_gen0_loss_p90": round(float(np.percentile(e5, 90)), 4),
        "sample_seconds": round(t_sample, 1),
    }
    print(f"  eval gen{g} {label}: " + " ".join(f"{k}={v}" for k, v in m.items()
          if k not in ("gen", "label", "n_eval", "seed")), flush=True)
    return m

# ---- admission --------------------------------------------------------------
def admit(ctx, g, attempts, seed, key_index, ref_grids, ref_ss, seen_ss, radius):
    """Sample generation g-1's model, keep valid AND novel, write the gen file.
    Also measures E7 (beyond-HPS reach) and the E3 yield on the admitted teams."""
    model = load_model_at(ctx, g - 1)
    t0 = time.time()
    rows = sample_unbiased(model, ctx.C, attempts, seed)
    print(f"  sampled {attempts} in {time.time()-t0:.0f}s", flush=True)

    val = Validator()
    rng = np.random.default_rng(seed)
    admitted, kept_rows, batch_seen = [], [], set()
    n_valid = 0
    for r in rows:
        f = fields_of_row(ctx.V, r)
        k = key_from_fields(f)
        paste = row_to_paste(ctx.V, r, ctx.look, ctx.spreads, rng)
        if val(paste) is not None:
            continue
        n_valid += 1
        if k in key_index or k in batch_seen:
            continue
        batch_seen.add(k)
        admitted.append({"key": k, "gen": g, "source": f"diffusion_gen{g}",
                         "team": paste, "species_set": species_set(f)})
        kept_rows.append(r)
    val.close()

    STDIR.mkdir(parents=True, exist_ok=True)
    with open(gen_file(g), "w") as fh:
        for a in admitted:
            fh.write(json.dumps(a) + "\n")

    if kept_rows:
        ad_grids = np.stack([canon_grid(ctx.V, r) for r in kept_rows])
        nn = nn_hamming(ad_grids, ref_grids)
        e7_beyond = float((nn > radius).mean())
        ad_ss = {a["species_set"] for a in admitted}
        e7_newss = int(sum(s not in ref_ss for s in ad_ss))
        new_ss = len(ad_ss - seen_ss)
        seen_ss |= ad_ss
    else:
        e7_beyond, e7_newss, new_ss = 0.0, 0, 0

    stats = {
        "attempts": attempts, "valid": n_valid, "admitted": len(admitted),
        "validity": round(n_valid / attempts, 4),
        "novelty_of_valid": round(len(admitted) / max(n_valid, 1), 4),
        "E3_new_species_sets_per_1k_admitted":
            round(1000 * new_ss / max(len(admitted), 1), 2),
        "E7_beyond_hps_frac": round(e7_beyond, 4),
        "E7_species_sets_not_in_ref": e7_newss,
    }
    print(f"  admit gen{g}: " + json.dumps(stats), flush=True)
    return stats

# ---- retraining -------------------------------------------------------------
def retrain(ctx, g, epochs=30, seed=0, batch=256, cap=0.5):
    """From-scratch retrain (identical recipe to hpsdiffusion.train) on the
    accumulated dataset: original 100k minus the frozen holdout, plus every
    admitted generation up to g, synthetic capped at `cap` x real rows."""
    import torch.nn as nn
    import random as _random
    _random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    keep = np.setdiff1d(np.arange(len(ctx.X)), ctx.holdout)
    n_real = len(keep)
    Xs, Ws = [ctx.X[keep]], [ctx.W[keep]]
    syn = admitted_rows(up_to=g)
    if syn:
        cap_n = int(cap * n_real)
        if len(syn) > cap_n:
            rng = np.random.default_rng(seed)
            syn = [syn[i] for i in rng.choice(len(syn), cap_n, replace=False)]
            print(f"  synthetic capped at {cap_n}", flush=True)
        Xsyn = np.stack([ctx.V.encode(parse_team_text(r["team"])) for r in syn])
        assert (Xsyn == 0).sum() == 0, "out-of-vocab field in an admitted team"
        Xs.append(Xsyn); Ws.append(np.full(len(Xsyn), H.WNULL, dtype=np.int64))
    X = np.concatenate(Xs); W = np.concatenate(Ws)
    lam = (len(X) - n_real) / n_real
    print(f"  retrain gen{g}: {len(X)} rows ({n_real} real, {len(X)-n_real} synthetic, "
          f"lambda={lam:.3f}), {epochs} epochs", flush=True)

    model = H.TeamDiffusionHPS(ctx.V).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    xtr = torch.tensor(X, device=DEV); wtr = torch.tensor(W, device=DEV)
    curve = []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(xtr.shape[0], device=DEV)
        tot = nb = 0
        for i in range(0, xtr.shape[0], batch):
            b = perm[i:i + batch]
            xb = xtr[b].view(-1, NSLOT, NF)
            ps = torch.rand(xb.shape[0], NSLOT, device=DEV).argsort(1)
            xb = torch.gather(xb, 1, ps.unsqueeze(-1).expand(-1, -1, NF)).reshape(-1, COLS)
            loss = model.loss(xb, wtr[b])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.); opt.step()
            tot += float(loss); nb += 1
        sch.step()
        model.eval()
        vl, _ = holdout_loss(model, ctx, seed=97, reps=4)
        curve.append(round(vl, 4))
        print(f"    ep{ep+1:3d} train {tot/nb:8.3f}  frozen-holdout {vl:8.3f}", flush=True)
    torch.save({"sd": model.state_dict()}, ckpt_path(g))
    shutil.copy(ckpt_path(g), REPO / "results" / f"selftrain_gen{g}.pt")
    return {"rows": len(X), "n_real": n_real, "lambda": round(lam, 4),
            "epochs": epochs, "holdout_curve": curve}

# ---- stopping ---------------------------------------------------------------
def check_stop(res):
    """S0 plateau / S1 value / S2 collapse, per the protocol. Fires only after
    2 completed generations with calibrated bands."""
    bands = res.get("bands")
    gens = sorted([g for g in res["generations"] if "eval" in g and g["gen"] >= 1],
                  key=lambda g: g["gen"])
    if not bands or len(gens) < 2:
        return None
    g0 = next(g for g in res["generations"] if g["gen"] == 0)["eval"]
    a, b = gens[-2], gens[-1]
    verdicts = []

    e0_flat = abs(a["eval"]["E0_holdout_loss"] - b["eval"]["E0_holdout_loss"]) \
        <= bands["E0_holdout_loss"]
    y1 = gens[0]["admission"]["E3_new_species_sets_per_1k_admitted"]
    ya = a["admission"]["E3_new_species_sets_per_1k_admitted"]
    yb = b["admission"]["E3_new_species_sets_per_1k_admitted"]
    yield_flat = all(y <= 50 or abs(y - y1) <= 0.1 * max(y1, 1) for y in (ya, yb))
    if e0_flat and yield_flat:
        verdicts.append("S0 plateau: holdout loss and species-set yield flat "
                        "for 2 consecutive generations")

    if all(g["admission"]["E7_beyond_hps_frac"] <= 0.005 for g in (a, b)):
        verdicts.append("S1 value exit: beyond-HPS reach ~0 for 2 generations "
                        "— the loop is a slow HPS")

    for key, sign in [("E4_species_entropy", -1), ("E4_pair_tv_vs_orig", +1),
                      ("E4_mean_nn_hamming_holdout", +1)]:
        d_a = sign * (a["eval"][key] - g0[key])
        d_b = sign * (b["eval"][key] - g0[key])
        if d_a > bands[key] and d_b > bands[key]:
            verdicts.append(f"S2 collapse tripwire: {key} beyond its band vs "
                            f"generation 0 twice — roll back to gen {a['gen'] - 1}")

    v0, va, vb = g0["E1_validity"], a["eval"]["E1_validity"], b["eval"]["E1_validity"]
    if bands.get("E1_validity") and (abs(va - v0) > 3 * bands["E1_validity"]
                                     or abs(vb - v0) > 3 * bands["E1_validity"]):
        print(f"  S3 anomaly (not a stop): validity moved {v0} -> {va} -> {vb}",
              flush=True)
    return verdicts or None

# ---- commands ---------------------------------------------------------------
def cmd_init(args):
    STDIR.mkdir(parents=True, exist_ok=True)
    if not HOLDOUT.exists():
        # exactly the split train(seed=0) used, so the shipped checkpoint is a
        # clean generation 0 that never saw these lines
        rng = np.random.default_rng(0)
        idx = rng.permutation(100000)[:2000]
        HOLDOUT.write_text(json.dumps(sorted(int(i) for i in idx)))
        print(f"frozen holdout written ({len(idx)} lines)", flush=True)
    if not ckpt_path(0).exists():
        shutil.copy(TMP / "hpsdiffusion.pt", ckpt_path(0))
        shutil.copy(TMP / "hpsdiffusion.pt", REPO / "results" / "selftrain_gen0.pt")
        print("generation-0 checkpoint frozen", flush=True)
    for out, n, seed in [(REF_JSONL, 100000, 777), (CAL_JSONL, 5000, 778)]:
        if not out.exists():
            print(f"generating {out.name} (n={n}, seed={seed})…", flush=True)
            subprocess.run([sys.executable, str(REPO / "src" / "hps_generate.py"),
                            "--n", str(n), "--seed", str(seed), "--out", str(out)],
                           check=True, cwd=str(REPO / "src"))
    ctx = Ctx()
    if not KEY_INDEX.exists():
        KEY_INDEX.write_text("\n".join(sorted(ctx.orig_keys)) + "\n")
        print(f"original key index written ({len(ctx.orig_keys)} distinct)", flush=True)
    if not REF_GRIDS.exists() or not REF_SS.exists():
        grids, sss = [], []
        for l in open(REF_JSONL):
            t = parse_team_text(json.loads(l)["team"])
            grids.append(ctx.V.encode(t))
            sss.append(species_set(team_fields(t)))
        np.save(REF_GRIDS, np.stack(grids).astype(np.int16))
        REF_SS.write_text("\n".join(sorted(set(sss))) + "\n")
        print(f"reference grids + species sets cached ({len(grids)})", flush=True)
    res = load_results()
    if "radius" not in res:
        ref_grids = np.load(REF_GRIDS)
        cal = np.stack([ctx.V.encode(parse_team_text(json.loads(l)["team"]))
                        for l in open(CAL_JSONL)]).astype(np.int16)
        nn = nn_hamming(cal, ref_grids)
        res["radius"] = int(np.percentile(nn, 99))
        res["radius_note"] = ("p99 nearest-neighbour Hamming of 5k fresh HPS teams "
                              "to the 100k HPS reference — beyond this is beyond "
                              "HPS's own typical spacing")
        save_results(res)
        print(f"E7 radius calibrated: {res['radius']}", flush=True)
    print("init done", flush=True)

def cmd_calibrate(args):
    ctx = Ctx()
    res = load_results()
    model0 = load_model_at(ctx, 0)
    key_index = current_key_index(ctx)
    evs = [evaluate_checkpoint(ctx, 0, seed, args.n_eval, model0, key_index,
                               label=f"cal{seed}") for seed in (101, 102)]
    bands = {k: round(3 * abs(evs[0][k] - evs[1][k]) + 1e-6, 5)
             for k in ("E1_validity", "E4_species_entropy",
                       "E4_pair_tv_vs_orig", "E4_mean_nn_hamming_holdout",
                       "E5_gen0_loss_mean", "E3_distinct_species_sets")}
    # both draws score the same checkpoint on the same fixed masking seed, so
    # their E0 difference is exactly 0 — the honest E0 band is the repeat-masking
    # standard error (training-run-to-run noise is not calibrated here, so S0's
    # E0 term errs conservative)
    bands["E0_holdout_loss"] = round(3 * evs[0]["E0_se"], 5)
    res["bands"] = bands
    res["calibration"] = evs
    res["generations"] = ([g for g in res["generations"] if g["gen"] != 0]
                          + [{"gen": 0, "eval": evs[0]}])
    res["generations"].sort(key=lambda g: g["gen"])
    save_results(res)
    print("bands:", json.dumps(bands), flush=True)

def cmd_run(args):
    res = load_results()
    if "radius" not in res or "bands" not in res:
        sys.exit("run `init` and `calibrate` first")
    ctx = Ctx()
    model0 = load_model_at(ctx, 0)
    ref_grids = np.load(REF_GRIDS)
    ref_ss = set(REF_SS.read_text().splitlines())
    seen_ss = set(ctx.orig_ss) | {r["species_set"] for r in admitted_rows()}
    done = {g["gen"] for g in res["generations"] if "eval" in g}
    g = max(done | {0}) + 1
    while g <= args.max_gens:
        print(f"=== generation {g} ===", flush=True)
        t0 = time.time()
        adm = admit(ctx, g, args.attempts, seed=1000 + g,
                    key_index=current_key_index(ctx), ref_grids=ref_grids,
                    ref_ss=ref_ss, seen_ss=seen_ss, radius=res["radius"])
        tr = retrain(ctx, g, epochs=args.epochs)
        ev = evaluate_checkpoint(ctx, g, seed=2000 + g, n_eval=args.n_eval,
                                 model0=model0, key_index=current_key_index(ctx))
        res = load_results()
        res["generations"] = [x for x in res["generations"] if x["gen"] != g]
        res["generations"].append({"gen": g, "admission": adm, "retrain": tr,
                                   "eval": ev,
                                   "minutes": round((time.time() - t0) / 60, 1)})
        res["generations"].sort(key=lambda x: x["gen"])
        stop = check_stop(res)
        if stop:
            res["stop"] = {"at_gen": g, "rules": stop}
            save_results(res)
            print("STOP:", "; ".join(stop), flush=True)
            return
        save_results(res)
        g += 1
    res["stop"] = {"at_gen": args.max_gens, "rules": ["S4 budget cap reached"]}
    save_results(res)
    print("STOP: S4 budget cap reached", flush=True)

def cmd_status(args):
    res = load_results()
    for g in res["generations"]:
        row = {"gen": g["gen"]}
        row.update({k: v for k, v in g.get("eval", {}).items()
                    if k.startswith(("E0_h", "E1", "E2", "E3_d", "E4", "E5_gen0_loss_mean"))})
        if "admission" in g:
            row.update({k: g["admission"][k] for k in
                        ("admitted", "validity", "E3_new_species_sets_per_1k_admitted",
                         "E7_beyond_hps_frac")})
        if "retrain" in g:
            row["lambda"] = g["retrain"]["lambda"]
        print(json.dumps(row))
    if "stop" in res:
        print("stop:", json.dumps(res["stop"]))

def cmd_speedtest(args):
    ctx = Ctx()
    model = load_model_at(ctx, 0)
    sample_unbiased(model, ctx.C, 8, seed=0)          # warm the MPS graph
    t0 = time.time(); sample_unbiased(model, ctx.C, 256, seed=1)
    dt = time.time() - t0
    per = dt / 256
    print(f"256 samples in {dt:.1f}s -> {per*1000:.0f} ms/team; "
          f"admission ({args.attempts}) ~{args.attempts*per/60:.0f} min, "
          f"eval ({args.n_eval}) ~{args.n_eval*per/60:.0f} min", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["init", "calibrate", "run", "status", "speedtest"])
    ap.add_argument("--attempts", type=int, default=12000)
    ap.add_argument("--n_eval", type=int, default=8000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--max_gens", type=int, default=5)
    a = ap.parse_args()
    {"init": cmd_init, "calibrate": cmd_calibrate, "run": cmd_run,
     "status": cmd_status, "speedtest": cmd_speedtest}[a.cmd](a)

if __name__ == "__main__":
    main()
