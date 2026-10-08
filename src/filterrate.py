"""What does the ELITE FRACTION rho buy you in the cross-entropy method?

THE QUESTION
------------
Owner, 2026-08-27: "If you are doing a cross entropy method, I would like to see
different filter rate base on the elite against the meta." And from the TODO:
"Will there be a filter method to on the n number of generated pokemon team base
on the win rate againts the meta."

rho is the one free parameter of the cross-entropy method nobody here has swept.
`loop.py` hardcodes rho = 0.2 (`max(20, len//5)`) for its copy-paste proposer and
never varies it. This file varies it, and only it.

TWO READINGS OF "FILTER RATE", BOTH MEASURED
--------------------------------------------
The loop already run was NOT classic cross-entropy method. It refit the density on
the WHOLE labelled pool and asked for quality at sampling time through the win-rate
condition token. Classic cross-entropy method instead refits the density on the
ELITES ONLY and samples it unconditioned. Both are legitimate readings of "filter
rate on the n generated teams by win rate against the meta", so both are arms here.

  Group 1 - copy-paste proposer (cheap, no training, runs first)
    The donor pool for slot-pasting is the top-rho of the 4,850 labelled teams,
    ranked by win rate against the top-50 meta. One candidate of elite team A is
    replaced by one candidate of elite team B. rho in {0.02, 0.05, 0.10, 0.20,
    0.50, 1.00}; rho = 1.00 is the no-filter control and rho = 0.20 reproduces
    `loop.py`. Unlike `loop.py` there is NO second PMI selection step - every
    Showdown-valid proposal is battled, so rho is the only filter in the pipeline.

  Group 2 - diffusion proposer (expensive, ~10 min refit per arm, runs second)
    cond_all    : the current form. Refit on all 4,850 WITH the win-rate bin token,
                  sample at bin 5, guidance 2.0. The rho = 1.0 reference for the
                  CONDITIONAL family, i.e. `loop.py`'s diffusion arm.
    cem_rho05/20/50 : classic cross-entropy method. Refit on the top-rho elites
                  ONLY, with no win-rate token and no style token (both pinned to
                  their null value), and sample unconditioned at guidance 1.0.
                  This is the honest cross-entropy method density.
    cem_rho100  : the same unconditional density refit on the WHOLE pool. This arm
                  is not in the original ask and is added on purpose: without it,
                  cem_rho05/20/50 have no same-family baseline and any difference
                  between them and cond_all confounds "elite filtering" with
                  "dropped the conditioning tokens". cem_rho100 is the rho = 1.0
                  point of the cross-entropy-method curve.

WHY THIS IS NOT A REPEAT OF `sampling.py`
-----------------------------------------
`sampling.py` already showed that for the CONDITIONAL model, cutting losers HURTS:
cut >= 0.2 -> 0.122, top-half -> 0.089, cut >= 0.4 -> 0.075, against uniform-full
0.197. That is a fact about the conditional model, and it has a mechanism -
classifier-free guidance needs losing examples to contrast against, so deleting
them deletes the signal the guidance rides on. The cem_* arms are UNCONDITIONAL
densities. No guidance, no contrast, nothing for the losers to be useful for. If
elite-only refitting works anywhere, it works there. That is why the sweep is worth
its battles even though `sampling.py` exists.

PRE-REGISTERED EXPECTATIONS (written before the run, do not edit after)
----------------------------------------------------------------------
  1. The copy-paste sweep shows a SHALLOW optimum at a small rho, somewhere in
     0.05-0.20. Tighter elite = better donors, but fewer distinct donors = less
     novelty and more duplication, so the curve turns over rather than rising to
     rho -> 0. rho = 1.00 is worst of the copy-paste arms.
  2. Every copy-paste arm lands near the measured reference band 0.420-0.494 -
     the operator, not rho, sets the scale.
  3. All cem_* arms land far BELOW every copy-paste arm, in the generator's band
     (0.128-0.195). Elite-only refitting does not rescue the generator.
  4. Among the cem_* arms, cem_rho05 <= cem_rho20 <= cem_rho50 <= cem_rho100 is
     NOT expected; the honest guess is that they are flat within error, i.e. the
     filter rate does not matter for a density this far from the target.
  5. PMI surprise FALLS monotonically as rho tightens for copy-paste, because a
     small elite is a small set of species cores. A rho that raises win rate by
     collapsing onto four known-good donors is not a win, and expectations 1 and 5
     together are the trade this sweep exists to expose.

WHAT IS RECORDED PER ARM, AND WHY
---------------------------------
  n, valid_rate_showdown  : acceptance under Showdown's real TeamValidator
  mean, se_clustered      : SE = std over PER-TEAM win rates / sqrt(n_teams).
                            NEVER binomial over battles - battle-level errors
                            understate the uncertainty by roughly 3x.
  p90, max, frac_ge_0458  : the tail, and the fraction beating the median real
                            team (0.458)
  surprise_mean           : mean PMI surprise (loop.PMI, corpus-fit). Real-team
                            reference 1.844. This is the novelty half of the trade.
  distinct_sources        : how many DISTINCT labelled-pool teams survived the
                            filter into the battled set. Copy-paste: distinct pool
                            teams used as base or donor. Diffusion: distinct pool
                            species-sets that a battled proposal reproduces, i.e.
                            how many known teams the density collapsed onto.
  distinct_species_sets   : distinct 6-species sets among the battled proposals.
                            Same meaning in both groups; the collapse detector.
  matches_real            : fraction of battled proposals whose 6-species set
                            already exists in the labelled pool. `loop.py` found
                            66% for copy-paste against 1-2% for diffusion.

KNOWN CONFOUNDS, STATED UP FRONT
--------------------------------
  a. The diffusion arms FINE-TUNE from /tmp/vgc-pilot/wrdiffusion.pt, exactly as
     `loop.retrain` does, so a cem_rho05 model has still seen the losers during
     pre-training. This is a refit, not a from-scratch density. Training d=192
     from scratch for a 60-epoch budget would be dominated by undertraining rather
     than by rho, which is a worse confound. Recorded as `init_from_ckpt`.
  b. `pool.score` writes FIXED paths /tmp/vgc-pilot/_shard_{i}.json, so two scripts
     battling at once overwrite each other's job specs and results. This was seen
     live on 2026-08-27 (a concurrent alignment.py run ate 2 of 3 teams from one
     arm). `wait_for_pool` therefore blocks until no other shard.py is running, and
     `battle` re-battles anything that comes back empty; `n_lost_to_battle_errors`
     records what never returned. Still: do not start this while another battling
     script is live, and do not start a training while this one is training.
  c. "60 epochs" over a 20x smaller elite set is 20x fewer gradient steps, which
     would make small-rho arms look bad for a reason that is not rho. The default
     `--epochs-mode steps` therefore equalises the GRADIENT-STEP budget across arms
     (60 epochs' worth of full-pool steps) and reports the epoch count it used.
     `--epochs-mode epochs` gives the literal 60 for every arm.

CLI
---
  python filterrate.py [--group copypaste|diffusion|all] [--per 150]
                       [--battles 24] [--epochs-mode steps|epochs] [--smoke]
  --smoke : 2 arms (one per group), 4 teams, 2 battles, 3 epochs -> a couple of
            minutes. Writes filterrate_smoke.json, never the real results file.
  --out   : override the results file (used to exercise the resume path).

Output: /tmp/vgc-pilot/filterrate_results.json (resumable - written after EVERY
arm, and a re-run skips arms already in it).
"""
import argparse, binascii, json, math, os, random, subprocess, sys, time
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, "/tmp/vgc-pilot/src")
import diffusion as D, wrdiffusion as W
import loop as LP                      # PMI, opponents
import pool
from corpus import load_corpus, norm, STATS
from encode import Vocab, Legality, canon, NF, NSLOT
from propose import Validator, slot_to_text, team_to_text

ROOT = Path("/tmp/vgc-pilot/filterrate"); ROOT.mkdir(exist_ok=True)
OUT = "/tmp/vgc-pilot/filterrate_results.json"
OUT_SMOKE = "/tmp/vgc-pilot/filterrate_smoke.json"
REAL_MEDIAN = 0.458                    # median real corpus team, measured
BATCH = 64
LR = 1e-4
BASE_EPOCHS = 60                       # matches loop.retrain
SAMPLE_BATCH = 16                      # decode this many teams per forward pass

# tag, rho  -- the donor pool is the top-rho of the labelled pool by win rate
CP_ARMS = [("cp_rho0005", 0.005), ("cp_rho001", 0.01),
           ("cp_rho002", 0.02), ("cp_rho005", 0.05), ("cp_rho010", 0.10),
           ("cp_rho020", 0.20), ("cp_rho050", 0.50), ("cp_rho100", 1.00)]
# tag, rho, conditional?  -- conditional arms keep the win-rate/style tokens
DF_ARMS = [("cond_all", 1.00, True), ("cem_rho05", 0.05, False),
           ("cem_rho20", 0.20, False), ("cem_rho50", 0.50, False),
           ("cem_rho100", 1.00, False)]
SMOKE_ARMS = ["cp_rho020", "cem_rho20"]


# ---------------------------------------------------------------- the filter
def elite_indices(wrs, rho, rng, floor=20):
    """Top-rho of the labelled pool by win rate against the top-50 meta.

    Win rates come from 24 battles, so there are only ~25 distinct values and the
    cut usually lands inside a tie group. Shuffle before the (stable) argsort so
    ties break uniformly at random rather than by file order.
    """
    n = len(wrs)
    k = n if rho >= 1.0 else max(floor, int(round(rho * n)))
    k = min(k, n)
    perm = rng.permutation(n)
    order = perm[np.argsort(-np.asarray(wrs)[perm], kind="stable")]
    return order[:k]


# ---------------------------------------------------------- shared bookkeeping
def species_set(team):
    return frozenset(norm(s["species"]) for s in team)


def summarise(tag, files, slots_of, battled_wr, pmi, pool_sets, src_of, extra):
    """Every arm reports the same row: quality, tail, novelty, collapse."""
    battled = [f for f in files if f in battled_wr]
    row = dict(arm=tag, n_proposed=len(files), n_battled=len(battled),
               n_lost_to_battle_errors=len(files) - len(battled), **extra)
    if not battled:
        row.update(mean=None, se_clustered=None, p90=None, max=None,
                   frac_ge_0458=None, surprise_mean=None, distinct_sources=None,
                   distinct_species_sets=None, matches_real=None)
        return row
    w = np.array([battled_wr[f] for f in battled])
    s = np.array([pmi.surprise(slots_of[f]) for f in battled])
    sets = [species_set(slots_of[f]) for f in battled]
    # distinct labelled-pool teams that survived the filter into the battled set
    if src_of is not None:                       # copy-paste: exact provenance
        srcs = set()
        for f in battled: srcs |= set(src_of[f])
        n_src = len(srcs)
    else:                                        # diffusion: species-set trace
        n_src = len({x for x in sets if x in pool_sets})
    row.update(
        mean=float(w.mean()),
        se_clustered=float(w.std(ddof=1) / math.sqrt(len(w))) if len(w) > 1 else None,
        p90=float(np.percentile(w, 90)), max=float(w.max()),
        frac_ge_0458=float((w >= REAL_MEDIAN).mean()),
        surprise_mean=float(s.mean()),
        distinct_sources=n_src,
        distinct_species_sets=len(set(sets)),
        matches_real=float(np.mean([x in pool_sets for x in sets])),
    )
    return row


def others_battling():
    """Count shard.py processes that are NOT ours (we have none while we poll)."""
    try:
        out = subprocess.run(["ps", "-Ao", "pid,command"], capture_output=True,
                             text=True, check=False).stdout
    except Exception:
        return 0
    return sum(1 for L in out.splitlines() if "src/shard.py" in L and "grep" not in L)


def wait_for_pool(poll=20, limit=3600):
    """`pool.score` writes FIXED paths /tmp/vgc-pilot/_shard_{i}.json, so two
    scripts using it at once silently overwrite each other's job specs AND each
    other's results. Observed live on 2026-08-27: a concurrent alignment.py run
    ate 2 of 3 teams out of one arm. Wait our turn instead of corrupting both runs.
    """
    t0 = time.time(); warned = False
    while others_battling() and time.time() - t0 < limit:
        if not warned:
            print("  waiting: another pool.score run holds /tmp/vgc-pilot/_shard_*.json",
                  flush=True)
            warned = True
        time.sleep(poll)
    if warned:
        print(f"  pool free after {time.time()-t0:.0f}s", flush=True)


def battle(files, opp, battles, retries=2):
    """Battle every file, and re-battle any that came back with no result at all."""
    got, todo = {}, list(files)
    for attempt in range(retries + 1):
        if not todo: break
        wait_for_pool()
        res = pool.score([(f, opp) for f in todo], battles=battles, conc=50, quiet=True)
        for f in todo:
            if f in res and res[f]["battles"] > 0:
                got[f] = res[f]["win_rate"]
        todo = [f for f in todo if f not in got]
        if todo and attempt < retries:
            print(f"  {len(todo)} team(s) returned no result, retrying", flush=True)
    if todo:
        print(f"  WARNING {len(todo)} team(s) never returned a result", flush=True)
    return got


def arm_dir(tag):
    d = ROOT / tag; d.mkdir(exist_ok=True)
    for f in d.glob("*.txt"): f.unlink()
    return d


# ------------------------------------------------------ group 1: copy-paste
def propose_copypaste(teams, elite_idx, val, n_valid, rng, outdir):
    """Paste one candidate from one elite team into another. `loop.py`'s operator,
    with the donor pool parameterised by rho instead of hardcoded to the top 20%."""
    elite = [teams[i] for i in elite_idx]
    kept = tries = submitted = 0
    files, slots_of, src_of = [], {}, {}
    while kept < n_valid and tries < n_valid * 8:
        tries += 1
        ia, ib = int(rng.integers(0, len(elite))), int(rng.integers(0, len(elite)))
        a = [dict(s) for s in canon(elite[ia])]
        b = canon(elite[ib])
        a[int(rng.integers(0, 6))] = dict(b[int(rng.integers(0, 6))])
        sp = [norm(s["species"]) for s in a]
        it = [norm(s["item"]) for s in a if s["item"]]
        if len(set(sp)) != 6 or len(set(it)) != len(it):
            continue                                   # Species / Item Clause
        submitted += 1
        txt = team_to_text(a)
        if val(txt) is None:
            p = outdir / f"{kept:04d}.txt"; p.write_text(txt)
            files.append(str(p)); slots_of[str(p)] = a
            src_of[str(p)] = (int(elite_idx[ia]), int(elite_idx[ib]))
            kept += 1
    return files, slots_of, src_of, dict(
        tries=tries, submitted=submitted,
        valid_rate_showdown=kept / max(submitted, 1),
        accept_rate_overall=kept / max(tries, 1))


# -------------------------------------------------------- group 2: diffusion
def refit(V, teams, wrs, idx, conditional, epochs, seed):
    """Refit the density on the selected slice. `loop.retrain`'s recipe: fine-tune
    the pre-trained checkpoint, batch 64, lr 1e-4, AdamW + cosine, grad clip 1.0,
    slot order permuted every batch (a team is a set, slot order is a nuisance).

    conditional=False pins BOTH conditioning tokens to their null value for every
    training example, so the fitted density carries no label at all and the elite
    filter is doing all of the work. That is the classic cross-entropy method.
    """
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    sub = [teams[i] for i in idx]
    X = np.stack([V.encode(t) for t in sub])
    if conditional:
        Y = np.array([D.style_of(t) for t in sub])
        B = np.array([W.wr_bin(wrs[i]) for i in idx])
    else:
        Y = np.full(len(sub), D.NULL)
        B = np.full(len(sub), W.WNULL)
    m = W.TeamDiffusionWR(V, d=192, nhead=6).to(W.DEV)
    init_ok = False
    try:
        ck = torch.load("/tmp/vgc-pilot/wrdiffusion.pt", map_location=W.DEV)
        m.load_state_dict(ck["sd"]); init_ok = True
    except Exception as e:
        print(f"    checkpoint unusable, training from scratch: {str(e)[:80]}", flush=True)
    xtr, ytr, btr = (torch.tensor(a, device=W.DEV) for a in (X, Y, B))
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    steps = 0
    for ep in range(epochs):
        m.train(); perm = torch.randperm(len(X), device=W.DEV)
        for i in range(0, len(X), BATCH):
            b = perm[i:i + BATCH]
            xb = xtr[b].view(-1, NSLOT, NF)
            xb = torch.stack([r[torch.randperm(NSLOT)] for r in xb]).view(-1, W.COLS)
            loss = m.loss(xb, ytr[b], btr[b])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.); opt.step(); steps += 1
        sch.step()
    m.eval()
    return m, init_ok, steps, float(loss.detach())


def propose_diffusion(m, V, C, look, spreads, val, n_valid, rng, outdir,
                      wbin, style, guidance):
    """Constrained decode -> real Showdown paste -> Showdown's own TeamValidator.
    Lifted from `loop.propose_diffusion`; the win-rate bin, style token and
    guidance are arm parameters here rather than hardcoded to (5, none, 2.0)."""
    kept = tries = 0
    files, slots_of = [], {}
    while kept < n_valid and tries < n_valid * 4:
        n = min(SAMPLE_BATCH, max(1, n_valid - kept), n_valid * 4 - tries)
        if n <= 0: break
        rows = W.sample_constrained(m, C, n, wbin, style, guidance).cpu().numpy()
        for row in rows:
            tries += 1
            slots = []
            for i in range(NSLOT):
                b = i * NF
                g = lambda j: V.decode_field(b + j, int(row[b + j]))
                sp = g(0)
                mv = [look.get(g(3 + j), g(3 + j)) for j in range(4)
                      if g(3 + j) and g(3 + j) != "[MASK]"]
                pl = spreads.get(sp) or [{s: 0 for s in STATS}]
                slots.append(dict(species=look.get(sp, sp), item=look.get(g(2), g(2)),
                                  ability=look.get(g(1), g(1)), nature=look.get(g(7), g(7)),
                                  moves=mv, evs=pl[int(rng.integers(0, len(pl)))]))
            txt = "\n\n".join(slot_to_text(s) for s in slots) + "\n"
            if val(txt) is None:
                p = outdir / f"{kept:04d}.txt"; p.write_text(txt)
                files.append(str(p)); slots_of[str(p)] = slots; kept += 1
            if kept >= n_valid: break
    return files, slots_of, dict(tries=tries, submitted=tries,
                                 valid_rate_showdown=kept / max(tries, 1),
                                 accept_rate_overall=kept / max(tries, 1))


# ------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=["copypaste", "diffusion", "all"], default="all")
    ap.add_argument("--per", type=int, default=150)
    ap.add_argument("--battles", type=int, default=24)
    ap.add_argument("--epochs-mode", choices=["steps", "epochs"], default="steps",
                    dest="epochs_mode")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="results file (default filterrate_results.json)")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    per, nbat, epochs_fixed = a.per, a.battles, None
    outfile = OUT
    if a.smoke:
        per, nbat, epochs_fixed, outfile = 4, 2, 3, OUT_SMOKE
        if a.out is None and os.path.exists(outfile): os.remove(outfile)
    if a.out: outfile = a.out

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    corpus, _ = load_corpus(); corpus = [t for t in corpus if len(t) == 6]
    lab_teams, lab_wrs = W.load_labelled()
    V = Vocab(corpus + lab_teams); L = Legality(corpus); C = D.Constraints(V, L)
    pmi = LP.PMI(corpus)
    real_surprise = float(np.mean([pmi.surprise(t) for t in corpus]))
    pool_sets = {species_set(t) for t in lab_teams}
    look, spreads = {}, {}
    for t in corpus + lab_teams:
        for s in t:
            look[norm(s["species"])] = s["species"]; look[norm(s["ability"])] = s["ability"]
            if s["item"]: look[norm(s["item"])] = s["item"]
            for mv in s["moves"]: look[norm(mv)] = mv
            if s["nature"]: look[norm(s["nature"])] = s["nature"]
            spreads.setdefault(norm(s["species"]), []).append(dict(s["evs"]))
    opp = LP.opponents()
    full_steps_per_epoch = math.ceil(len(lab_teams) / BATCH)
    target_steps = BASE_EPOCHS * full_steps_per_epoch
    print(f"pool {len(lab_teams)} labelled · mean {lab_wrs.mean():.3f} · corpus {len(corpus)} · "
          f"{len(opp)} opponents · real surprise {real_surprise:.3f} (ref 1.844) · "
          f"{per} teams/arm x {nbat} battles · step budget {target_steps}", flush=True)

    done = json.load(open(outfile)) if os.path.exists(outfile) else {}
    done.setdefault("_meta", dict(real_surprise=real_surprise, real_median=REAL_MEDIAN,
                                  pool=len(lab_teams), per=per, battles=nbat,
                                  epochs_mode=a.epochs_mode, seed=a.seed))
    todo = []
    if a.group in ("copypaste", "all"): todo += [("copypaste",) + x for x in CP_ARMS]
    if a.group in ("diffusion", "all"): todo += [("diffusion",) + x[:2] + (x[2],) for x in DF_ARMS]
    if a.smoke: todo = [t for t in todo if t[1] in SMOKE_ARMS]

    val = Validator()
    try:
        for spec in todo:
            group, tag, rho = spec[0], spec[1], spec[2]
            if tag in done: print(f"[{tag}] already done, skipping", flush=True); continue
            t0 = time.perf_counter()
            arng = np.random.default_rng(a.seed + int(binascii.crc32(tag.encode())) % 100000)
            idx = elite_indices(lab_wrs, rho, arng)
            ewr = np.asarray(lab_wrs)[idx]
            extra = dict(group=group, rho=rho, elite_n=int(len(idx)),
                         elite_mean_wr=float(ewr.mean()), wr_cut=float(ewr.min()))
            print(f"[{tag}] rho={rho} · elite {len(idx)}/{len(lab_teams)} · "
                  f"mean label {ewr.mean():.3f} · cut {ewr.min():.3f}", flush=True)
            out = arm_dir(tag)

            if group == "copypaste":
                files, slots_of, src_of, acc = propose_copypaste(
                    lab_teams, idx, val, per, arng, out)
                extra.update(acc, train_min=0.0)
            else:
                conditional = spec[3]
                spe = max(1, math.ceil(len(idx) / BATCH))
                if epochs_fixed is not None: ep = epochs_fixed
                elif a.epochs_mode == "steps": ep = max(1, int(round(target_steps / spe)))
                else: ep = BASE_EPOCHS
                t1 = time.perf_counter()
                m, init_ok, steps, last = refit(V, lab_teams, lab_wrs, idx, conditional,
                                                ep, a.seed)
                tmin = (time.perf_counter() - t1) / 60
                wbin = 5 if conditional else W.WNULL
                style = "none" if conditional else "__uncond__"   # unknown -> D.NULL
                guid = 2.0 if conditional else 1.0
                print(f"[{tag}]   refit {ep} epochs / {steps} steps · conditional={conditional} · "
                      f"init_from_ckpt={init_ok} · loss {last:.2f} · {tmin:.1f} min", flush=True)
                files, slots_of, acc = propose_diffusion(
                    m, V, C, look, spreads, val, per, arng, out, wbin, style, guid)
                src_of = None
                extra.update(acc, conditional=conditional, epochs=ep, steps=steps,
                             init_from_ckpt=init_ok, final_loss=last, train_min=tmin,
                             sample_bin=wbin, sample_style=style, guidance=guid)
                del m
                try: torch.mps.empty_cache()
                except Exception: pass

            wr = battle(files, opp, nbat)
            row = summarise(tag, files, slots_of,
                            wr, pmi, pool_sets,
                            src_of if group == "copypaste" else None, extra)
            row["minutes"] = (time.perf_counter() - t0) / 60
            done[tag] = row
            json.dump(done, open(outfile, "w"), indent=1)
            if row["mean"] is None:
                print(f"[{tag}] DONE  no battled team", flush=True)
            else:
                se = row['se_clustered']
                print(f"[{tag}] DONE  n {row['n_battled']} · valid {row['valid_rate_showdown']:.0%} · "
                      f"mean {row['mean']:.4f} ± {(se if se is not None else float('nan')):.4f} · "
                      f"p90 {row['p90']:.3f} · max {row['max']:.3f} · "
                      f">={REAL_MEDIAN}: {row['frac_ge_0458']:.1%} · "
                      f"surprise {row['surprise_mean']:.2f} (real {real_surprise:.2f}) · "
                      f"sources {row['distinct_sources']} · sets {row['distinct_species_sets']} · "
                      f"matches_real {row['matches_real']:.0%} · {row['minutes']:.1f} min", flush=True)
    finally:
        val.close()

    rows = [v for k, v in done.items() if k != "_meta" and v.get("mean") is not None]
    if rows:
        print("\narm          rho   n   valid  mean    se     p90    >=.458  surprise  src  sets  matches", flush=True)
        for r in sorted(rows, key=lambda r: (r["group"], r["rho"])):
            print(f"{r['arm']:<12} {r['rho']:<5} {r['n_battled']:<3} "
                  f"{r['valid_rate_showdown']:<6.0%} {r['mean']:<7.4f} "
                  f"{(r['se_clustered'] if r['se_clustered'] is not None else 0.0):<6.4f} {r['p90']:<6.3f} "
                  f"{r['frac_ge_0458']:<7.1%} {r['surprise_mean']:<9.2f} "
                  f"{r['distinct_sources']:<4} {r['distinct_species_sets']:<5} "
                  f"{r['matches_real']:.0%}", flush=True)
    print(f"\nwrote {outfile}", flush=True)
    print("FILTERRATE_DONE", flush=True)


if __name__ == "__main__":
    main()
