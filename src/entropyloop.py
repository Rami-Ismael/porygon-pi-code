"""Entropy in the elite step of the guidance loop - does the term help?

Owner question 2026-09-02: Sanokowski, Hochreiter & Lehner (ICML 2024, Reading
List 45) keep their diffusion sampler from collapsing with the per-step Shannon
entropy term of their Eq. 6 - the variational free energy of the reverse
Kullback-Leibler divergence to a Boltzmann distribution - and anneal T from
T_start to zero ("highly beneficial").  Their loss needs the energy in closed
form; ours is a 24-battle win rate.  What transfers is the entropy term and the
schedule, and the 2026-09-01 combined loop (gradloop.py) says the place it has
to act is ELITE SELECTION: the re-steer restarts from p0 every generation, but
the elite set went 100 real / 0 own proposals -> 108 / 294 by generation 11,
distinct species sets among proposals 485 -> 176, and the diversity guardrail
stopped the run with the win rate still climbing.

Three arms, each the cached `combined` loop (gloss@108 guidance + value
acquisition + re-steer from p0, 512 proposed / 128 battled x 24) with ONE change:

    boltz : the entropy-regularised selection.  Elite weights over the whole
            labelled pool maximise  E_w[y] + T * H(w), i.e. w_i ∝ exp(y_i / T);
            the refit multiset is k draws from w (sampling-importance-resampling)
            with k = the baseline's own elite count that generation, so the
            finetune sees the same number of optimizer steps.  The paper's term,
            literally: entropy over teams.
    niche : the same, with the entropy taken over SPECIES SETS - the quantity
            that actually collapsed.  One representative per species set (its
            best-labelled team), weights ∝ exp(y*_s / T) across sets, k draws.
    refit : hard top-RHO cut exactly as the baseline, but the finetune loss
            carries the paper's term where it literally lives: masked-position
            cross-entropy minus beta * Shannon entropy of the predicted
            categorical (the reverse kernel), beta on the same schedule.

Schedule: T (and beta) linear from T_START to T_END over the GENS generations.
T_END is the label noise (SE ~0.10 at 24 battles): below it the weights rank
coin flips, and at T = 0.05 the effective elite set is ~20 teams, past the
diversity knee of the 2026-08-27 filter-rate sweep.  Battle-free check on the
baseline's own pools: at T=0.30 the generation-1 effective elite size is 224
(baseline hard cut: 100); at T=0.10 generation 11 it is 276 (baseline: 402).

Fixed budget: every arm runs all GENS generations (no win-rate stall rule -
an annealed arm is SUPPOSED to be flat while T is high); only the novelty
guardrails stop an arm early (copy rate > 0.10, < 200 distinct species sets
in 512 proposals), and tripping one is a result.  After each arm, its top 16
labelled teams are re-battled at 192 battles (SE ~0.035) - the deliverable is
the best team, and 24-battle labels are winner's-cursed.

    python entropyloop.py smoke            # plumbing, tiny, nothing persisted
    python entropyloop.py                  # the run; resumable, flushes per arm-gen
    python entropyloop.py --arms niche     # one arm
    python entropyloop.py compare --help   # matched selection-temperature study
"""
import json, os, shutil, sys, time
from pathlib import Path
from diversity_metrics import Reference, measure

# Comparison preflight/offline diagnostics must work without the battle runtime.
if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "compare":
    from temperature_experiment import main as compare_main
    compare_main(sys.argv[2:])
    sys.exit(0)

import numpy as np
from boltzmann_selection import (boltz_weights, select_boltz, linear_temperature,
                                 selection_temperature, adaptive_temperature)
import torch
import torch.nn.functional as F

sys.path.insert(0, "/tmp/vgc-pilot/src")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus import parse_team, parse_team_text, norm
from encode import NF, NSLOT
import diffusion as D
import hpsdiffusion as H
import activesearch as A
import gradguide as GG
from hps_generate import Validator

ROOT = Path("/tmp/vgc-pilot/entropyloop")
# The repo lives under ~/Documents (iCloud); on 2026-09-02 it was evicted to
# placeholders and every read stalled.  Inputs are read from the local staging
# copy when it exists; the results file is written locally and mirrored to the repo.
REPO_RESULTS = "/Users/ramiismael/Documents/code/vgc-team-generator-pilot/results"
LOCAL_RESULTS = "/tmp/vgc-pilot/local/results"
def _in(name):
    loc = os.path.join(LOCAL_RESULTS, name)
    return loc if os.path.exists(loc) else os.path.join(REPO_RESULTS, name)
RESULTS = os.path.join(LOCAL_RESULTS if os.path.isdir(LOCAL_RESULTS) else REPO_RESULTS,
                       "entropyloop.json")
RESULTS_MIRROR = os.path.join(REPO_RESULTS, "entropyloop.json")
BASELINE = _in("gradloop.json")
BASELINE_REBATTLE = _in("rebattle_top.json")
ANCHOR_SRC = _in("activesearch.json")

MODE, LAM = "gloss", 108.0      # the baseline's guidance, unchanged
PROPOSE = 512
BATTLE = 128
BATTLES = 24
GENS = 11                       # the baseline's record length
RHO, ELITE_MIN = A.RHO, A.ELITE_MIN
REAL_MEAN = 0.4625
T_START, T_END = 0.30, 0.10
COPY_MAX = 0.10
SETS_MIN = 200
REBATTLE_TOP, REBATTLE_N = 16, 192
ARMS = ["boltz", "niche", "refit"]


ANNEAL_GENS = 11                # the schedule reaches T_END here; later generations hold it


def temperature(g):
    """Linear T_START -> T_END over ANNEAL_GENS generations, then held at T_END
    (an extension run past the anneal must not keep cooling: T would cross
    zero and invert the weights)."""
    return linear_temperature(g, ANNEAL_GENS, T_START, T_END)


# ------------------------------------------------------------ elite selection
def select_niche(lab_teams, lab_y, k, T, rng):
    best = {}
    for i, (t, v) in enumerate(zip(lab_teams, lab_y)):
        s = A.species_set(t)
        if s not in best or v > lab_y[best[s]]:
            best[s] = i
    reps = np.array(sorted(best.values()))
    ws = boltz_weights([lab_y[i] for i in reps], T)
    idx = reps[rng.choice(len(reps), size=k, replace=True, p=ws)]
    w = np.zeros(len(lab_y)); w[reps] = ws
    return idx, w


def select_cut(lab_y, k):
    return np.argsort(-np.asarray(lab_y))[:k]


def selection_stats(idx, w, lab_teams, lab_y, n_anchor):
    """What the refit sees: composition, effective sizes, species-set entropy."""
    idx = np.asarray(idx)
    y = np.asarray(lab_y, dtype=float)
    if w is None:                              # hard cut: uniform over idx
        w = np.zeros(len(y)); w[idx] = 1.0 / len(idx)
    ess = float(1.0 / (w ** 2).sum())
    real_share = float(w[:n_anchor].sum())
    sets_w = {}
    for i in np.nonzero(w)[0]:
        s = A.species_set(lab_teams[i]); sets_w[s] = sets_w.get(s, 0.0) + w[i]
    p = np.array(list(sets_w.values()))
    h_sets = float(-(p * np.log(p)).sum())
    return dict(k=int(len(idx)),
                distinct_teams=int(len(set(idx.tolist()))),
                distinct_species_sets=int(len({A.species_set(lab_teams[i]) for i in idx})),
                real=int((idx < n_anchor).sum()),
                proposals=int((idx >= n_anchor).sum()),
                ess=ess, real_weight_share=real_share,
                species_set_entropy=h_sets, effective_species_sets=float(np.exp(h_sets)),
                y_mean=float(y[idx].mean()), y_min=float(y[idx].min()))


# ------------------------------------------------------------ the refit
def loss_entropy(m, x, w, beta, p_uncond=0.15):
    """hpsdiffusion.TeamDiffusionHPS.loss, plus the Eq. 6 per-step entropy bonus.

    Masked-position cross-entropy (MDLM 1/t weighting) minus beta times the
    Shannon entropy of the predicted categorical at the same positions - the
    entropy of the reverse kernel q(x_{t-1} | x_t), which for masked diffusion
    is the unmasking distribution.  beta = 0 reproduces the original loss with
    the same random draws, so the boltz/niche arms match A.resteer exactly.
    """
    B = x.shape[0]
    t = torch.rand(B, device=x.device)
    mk = torch.rand(B, D.COLS, device=x.device) < t.view(-1, 1)
    empty = ~mk.any(1)
    if empty.any():
        mk[empty, torch.randint(0, D.COLS, (int(empty.sum()),), device=x.device)] = True
    ww = w.clone()
    ww[torch.rand(B, device=x.device) < p_uncond] = H.WNULL
    xin = x.clone(); xin[mk] = 0
    h = m(xin, t, ww)
    wt_all = (1.0 / t.clamp(min=1e-3))
    tot, ent, n = 0., 0., 0
    for k, cols in m.cols_by_key.items():
        mkk = mk[:, cols]
        if not mkk.any():
            continue
        lg = m.head[k](h[:, cols])[mkk]
        ce = F.cross_entropy(lg, x[:, cols][mkk], reduction="none")
        wt = wt_all.view(-1, 1).expand(-1, len(cols))[mkk]
        tot = tot + (wt * ce).sum()
        if beta:
            lp = F.log_softmax(lg, -1)
            ent = ent + (wt * (-(lp.exp() * lp).sum(-1))).sum()
        n += int(mkk.sum())
    return (tot - beta * ent) / max(n, 1)


def resteer(V, teams, beta=0.0, seed=0, batch=64, epochs=None, checkpoint=None):
    """p_g = finetune(p0, multiset) - A.resteer with the loss swapped in."""
    torch.manual_seed(seed)
    m = H.TeamDiffusionHPS(V).to(D.DEV)
    m.load_state_dict(torch.load(checkpoint or A.P0_CKPT, map_location=D.DEV)["sd"])
    xt, wt = A._tensors(V, teams)
    opt = torch.optim.AdamW(m.parameters(), lr=A.FT_LR, weight_decay=.01)
    epochs = A.FT_EPOCHS if epochs is None else epochs
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    steps = 0
    for _ in range(epochs):
        m.train()
        perm = torch.randperm(xt.shape[0], device=D.DEV)
        for i in range(0, xt.shape[0], batch):
            b = perm[i:i + batch]
            xb = xt[b].view(-1, NSLOT, NF)
            ps = torch.rand(xb.shape[0], NSLOT, device=xb.device).argsort(1)
            xb = torch.gather(xb, 1, ps.unsqueeze(-1).expand(-1, -1, NF)).reshape(-1, D.COLS)
            loss = loss_entropy(m, xb, wt[b], beta)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.); opt.step()
            steps += 1
        sch.step()
    m.eval()
    return m, steps


@torch.no_grad()
def pred_entropy(m, V, teams, seed=0, frac=0.5):
    """Mean entropy (nats) of the model's unmasking distribution on a fixed
    probe set at 50% masking - species columns and the rest, separately."""
    g = torch.Generator().manual_seed(seed)
    X = torch.tensor(np.stack([V.encode(t) for t in teams]), device=D.DEV)
    B = X.shape[0]
    mk = (torch.rand(B, D.COLS, generator=g) < frac).to(D.DEV)
    t = torch.full((B,), frac, device=D.DEV)
    w = torch.full((B,), H.WNULL, device=D.DEV, dtype=torch.long)
    xin = X.clone(); xin[mk] = 0
    h = m(xin, t, w)
    acc = {"species": [0., 0], "other": [0., 0]}
    for k, cols in m.cols_by_key.items():
        mkk = mk[:, cols]
        if not mkk.any():
            continue
        lp = F.log_softmax(m.head[k](h[:, cols])[mkk], -1)
        hh = -(lp.exp() * lp).sum(-1)
        key = "species" if all(c % NF == 0 for c in cols) else "other"
        acc[key][0] += float(hh.sum()); acc[key][1] += int(mkk.sum())
    return {k: v[0] / max(v[1], 1) for k, v in acc.items()}


# ------------------------------------------------------------ guardrails
def broken(st):
    p = st.get("proposal") or {}
    if (p.get("copy_rate") or 0) > COPY_MAX:
        return f"copy rate {p['copy_rate']:.2f} > {COPY_MAX}"
    if (p.get("distinct_species_sets") or 512) < SETS_MIN:
        return f"{p['distinct_species_sets']} distinct sets < {SETS_MIN}"
    return None


def rebattle(arm, out, opp, smoke):
    """Re-battle the arm's top labelled teams at REBATTLE_N battles."""
    rows = []
    for tag, g in out["gens"].items():
        if tag.rsplit("_g", 1)[0] == arm:
            rows += [(float(y), tag, p) for p, y in zip(g["pastes"], g["y"])]
    rows.sort(key=lambda r: -r[0])
    seen, top = set(), []
    for y, tag, p in rows:
        if p.strip() in seen:
            continue
        seen.add(p.strip()); top.append((y, tag, p))
        if len(top) >= (2 if smoke else REBATTLE_TOP):
            break
    d = ROOT / f"rebattle_{arm}"; d.mkdir(parents=True, exist_ok=True)
    files = []
    for i, (_, _, p) in enumerate(top):
        f = d / f"{i:02d}.txt"; f.write_text(p); files.append(str(f))
    n = 2 if smoke else REBATTLE_N
    print(f"\n=== {arm} · re-battling top {len(files)} at {n} battles ===", flush=True)
    quiet()
    res = A.pool.score([(f, opp) for f in files], battles=n, conc=50)
    recs = []
    for i, (y24, tag, p) in enumerate(top):
        r = res.get(files[i])
        if not r or not r["battles"]:
            continue
        team = parse_team_text(p)
        recs.append(dict(label24=y24, source=tag, win_rate=r["win_rate"], se=r["se"],
                         battles=r["battles"],
                         species=[norm(s["species"]) for s in team], paste=p))
    recs.sort(key=lambda d: -d["win_rate"])
    for i, r in enumerate(recs):
        r["rank"] = i + 1
        print(f"  {r['rank']:2d} {r['win_rate']:.4f} ± {r['se']:.4f} (label {r['label24']:.3f}, "
              f"{r['source']}) {', '.join(r['species'])}", flush=True)
    return dict(battles=n, teams=recs)


# ------------------------------------------------------------ the run
NO_WAIT = False                 # --no-wait: share the servers with another campaign instead of waiting


def quiet():
    if not NO_WAIT:
        GG.wait_for_quiet()


def main(smoke=False, only_arms=None, gens=None):
    global PROPOSE, BATTLE, BATTLES, GENS
    if gens:
        GENS = gens
    if smoke:
        PROPOSE, BATTLE, BATTLES, GENS = 8, 4, 2, 1

    print("loading corpus + the 2026-09-01 anchor/gen0 labels …", flush=True)
    cf = A.load_corpus_files()
    corpus_teams = [t for _, t in cf]
    diversity_reference = Reference(corpus_teams, scope="corpus", format_id="reg_mb")
    corpus_sets = {A.species_set(t) for t in corpus_teams}
    V = A.Vocab(corpus_teams); L = A.Legality(corpus_teams); C = D.Constraints(V, L)
    corpus_grid = np.stack([V.encode(t) for t in corpus_teams])
    look, spreads = A.decode_tables(corpus_teams)
    opp = A.opponents()

    as1 = json.load(open(ANCHOR_SRC))
    anchor_teams = [parse_team(f) for f in as1["anchor"]]
    anchor_y = [float(y) for y in as1["anchor"].values()]
    gen0 = as1["gens"]["gen0"]
    gen0_teams = [parse_team_text(p) for p in gen0["pastes"]]
    gen0_y = [float(y) for y in gen0["y"]]
    n_anchor = len(anchor_teams)
    print(f"  {len(corpus_teams)} corpus teams · anchor {n_anchor} "
          f"(mean {np.mean(anchor_y):.4f}) · gen0 {len(gen0_y)} "
          f"(mean {np.mean(gen0_y):.4f}) · {len(opp)} meta opponents", flush=True)

    p0 = H.TeamDiffusionHPS(V).to(D.DEV)
    p0.load_state_dict(torch.load(A.P0_CKPT, map_location=D.DEV)["sd"]); p0.eval()
    ent0 = pred_entropy(p0, V, anchor_teams)
    print(f"  p0 unmasking entropy on the anchors: species {ent0['species']:.3f} · "
          f"other {ent0['other']:.3f} nats", flush=True)
    val = Validator()

    out = (json.load(open(RESULTS)) if os.path.exists(RESULTS) and not smoke
           else {"config": {}, "gens": {}, "rebattle": {}, "stopped": {}})
    if "diversity_reference" in out and out["diversity_reference"]["metadata"] != diversity_reference.metadata:
        raise ValueError("diversity reference changed; use a new experiment output")
    out.setdefault("diversity_reference", dict(metadata=diversity_reference.metadata,
                                               normalized_records=diversity_reference.snapshot))
    out["config"] = dict(mode=MODE, lam=LAM, propose=PROPOSE, battle=BATTLE,
                         battles=BATTLES, gens=GENS, rho=RHO, elite_min=ELITE_MIN,
                         t_start=T_START, t_end=T_END, ft_epochs=A.FT_EPOCHS,
                         ft_lr=A.FT_LR, rebattle_top=REBATTLE_TOP, rebattle_n=REBATTLE_N,
                         p0=A.P0_CKPT, p0_entropy=ent0,
                         baseline="combined arm of gradloop.json (2026-09-01)",
                         gen0="reused from activesearch.json 2026-09-01")

    def flush():
        if smoke:
            return
        tmp = RESULTS + ".tmp"
        json.dump(out, open(tmp, "w"), indent=1); os.replace(tmp, RESULTS)
        if RESULTS_MIRROR != RESULTS:
            try:
                shutil.copyfile(RESULTS, RESULTS_MIRROR)
            except OSError as e:
                print(f"    (mirror to repo failed: {e})", flush=True)

    for arm in (only_arms or ARMS):
        lab_teams = list(anchor_teams) + list(gen0_teams)
        lab_y = list(anchor_y) + list(gen0_y)
        ran_new = False
        for g in range(1, GENS + 1):
            tag = f"{arm}_g{g}"
            if tag in out["gens"]:
                r = out["gens"][tag]
                lab_teams += [parse_team_text(p) for p in r["pastes"]]
                lab_y += list(r["y"])
                print(f"[{tag}] cached", flush=True)
                continue
            if arm in out.get("stopped", {}):
                break
            t0 = time.perf_counter()
            T = temperature(g)
            k = max(ELITE_MIN, int(RHO * len(lab_y)))
            rng = np.random.default_rng(100 + g)

            F_, wv = GG.fit_ridge(lab_teams, lab_y, corpus_teams)
            G = GG.Guide(F_, wv, V, D.DEV)

            beta = 0.0
            if arm == "boltz":
                idx, w = select_boltz(lab_y, k, T, rng)
            elif arm == "niche":
                idx, w = select_niche(lab_teams, lab_y, k, T, rng)
            else:
                idx, w = select_cut(lab_y, k), None
                beta = T
            ss = selection_stats(idx, w, lab_teams, lab_y, n_anchor)
            print(f"\n=== {tag} · T {T:.3f} · beta {beta:.3f} · refit on {ss['k']} draws: "
                  f"{ss['distinct_teams']} distinct teams / {ss['distinct_species_sets']} sets, "
                  f"{ss['real']} real, {ss['proposals']} proposals · ESS {ss['ess']:.0f} · "
                  f"real weight {ss['real_weight_share']:.2f} · effective sets "
                  f"{ss['effective_species_sets']:.0f} · ridge on {len(lab_y)} ===", flush=True)
            model, steps = resteer(V, [lab_teams[i] for i in idx], beta=beta)
            ent = pred_entropy(model, V, anchor_teams)
            print(f"    refit {steps} steps · unmasking entropy species {ent['species']:.3f} "
                  f"· other {ent['other']:.3f} (p0 {ent0['species']:.3f} / {ent0['other']:.3f})",
                  flush=True)

            sampling = {}
            files, teams, pastes, validity, diag = GG.propose_guided(
                model, C, G, V, look, spreads, val, MODE, LAM, PROPOSE,
                ROOT / tag, 1000 + g, sampling_stats=sampling)
            mu = F_.mat(teams) @ wv
            sel = [int(i) for i in np.argsort(-mu)[:BATTLE]]

            if not smoke:
                quiet()
            res = A.pool.score([(files[i], opp) for i in sel],
                               battles=BATTLES, conc=50)
            keep = [i for i in sel if files[i] in res and res[files[i]]["battles"] > 0]
            y = [res[files[i]]["win_rate"] for i in keep]
            st = A.batch_stats(y, [teams[i] for i in keep], corpus_sets, REAL_MEAN)
            st.update(validity=validity, acquisition="value", mode=MODE, lam=LAM,
                      n_labels_fit=len(lab_y),
                      surrogate_spearman=A.spearman([mu[i] for i in keep], y),
                      mu_batch=float(np.mean([mu[i] for i in keep])),
                      mu_all=float(np.mean(mu)),
                      proposal=A.memorisation(V, teams, corpus_grid),
                      team_diversity=measure(teams, diversity_reference,
                                             expected_n=PROPOSE, attempts=sampling["attempts"]),
                      n_proposed=len(teams), diag=diag,
                      temperature=T, beta=beta, refit_steps=steps,
                      selection=ss, model_entropy=ent)
            st["minutes"] = (time.perf_counter() - t0) / 60
            out["gens"][tag] = dict(stats=st, pastes=[pastes[i] for i in keep], y=y,
                                    proposal_pastes=pastes)
            ran_new = True
            flush()
            print(A.fmt(tag, st) + f" · novel {st['novel_species_set']:.2f}", flush=True)
            lab_teams += [teams[i] for i in keep]; lab_y += y
            why = broken(st)
            if why:
                out.setdefault("stopped", {})[arm] = dict(gen=g, reason=why)
                flush()
                print(f"[{arm}] STOPPED at generation {g}: {why}", flush=True)
                break
        if ran_new or arm not in out.get("rebattle", {}):
            old = out.get("rebattle", {}).get(arm)
            if old:                             # keep the earlier re-battle, named by its last generation
                last = max(int(t["source"].rsplit("_g", 1)[1]) for t in old["teams"])
                out["rebattle"][f"{arm}_through_g{last}"] = old
            out.setdefault("rebattle", {})[arm] = rebattle(arm, out, opp, smoke)
            flush()

    val.close()

    # ---- headline beside the baseline ---------------------------------------
    base = json.load(open(BASELINE))["gens"]
    print(f"\n{'arm-gen':10s} {'mean':>7s} {'se':>6s} {'max':>6s} {'valid':>6s} {'sets':>5s} "
          f"{'novel':>6s} {'NNham':>6s} {'k':>4s} {'real':>5s} {'effS':>5s} {'Hsp':>6s}", flush=True)
    def row(tag, s):
        p = s.get("proposal") or {}; ss = s.get("selection") or {}; me = s.get("model_entropy") or {}
        real = ss.get("real_weight_share", s.get("elite_real", 0) / max(s.get("elite_n", 1), 1))
        print(f"{tag:10s} {s['mean']:7.4f} {s['se'] or 0:6.4f} {s['max']:6.3f} {s['validity']:6.2f} "
              f"{p.get('distinct_species_sets', 0):5d} {s.get('novel_species_set', 0):6.3f} "
              f"{p.get('nn_hamming', 0):6.2f} {ss.get('k', s.get('elite_n', 0)):4d} "
              f"{real:5.2f} {ss.get('effective_species_sets', float('nan')):5.0f} "
              f"{me.get('species', float('nan')):6.3f}", flush=True)
    for g in range(1, 12):
        if f"combined_g{g}" in base:
            row(f"base_g{g}", base[f"combined_g{g}"]["stats"])
    for arm in (only_arms or ARMS):
        for g in range(1, GENS + 1):
            if f"{arm}_g{g}" in out["gens"]:
                row(f"{arm}_g{g}", out["gens"][f"{arm}_g{g}"]["stats"])
    print(flush=True)
    for arm, rb in out.get("rebattle", {}).items():
        wr = [t["win_rate"] for t in rb["teams"]]
        if wr:
            print(f"rebattle {arm:6s}: best {max(wr):.4f} · top-{len(wr)} mean {np.mean(wr):.4f} "
                  f"· ≥ real mean {sum(v >= REAL_MEAN for v in wr)}/{len(wr)}", flush=True)
    if os.path.exists(BASELINE_REBATTLE):
        rb = [t for t in json.load(open(BASELINE_REBATTLE))["teams"] if t["source"].startswith("combined_g")]
        wr = [t["win_rate"] for t in rb]
        print(f"rebattle base  : best {max(wr):.4f} · top-{len(wr)} mean {np.mean(wr):.4f} "
              f"· ≥ real mean {sum(v >= REAL_MEAN for v in wr)}/{len(wr)}  (rebattle_top.json)", flush=True)
    flush()
    print("\nENTROPYLOOP_DONE", flush=True)


if __name__ == "__main__":
    def opt(flag, cast):
        return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else None
    NO_WAIT = "--no-wait" in sys.argv
    main(smoke=("smoke" in sys.argv), only_arms=opt("--arms", lambda s: s.split(",")),
         gens=opt("--gens", int))
