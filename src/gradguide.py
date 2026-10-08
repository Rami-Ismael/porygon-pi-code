"""Decode-time gradient guidance - the talk's method vs the naive gradient, on p0.

Mengdi Wang's INI talk (2024-07-18, "Guiding Diffusion Models Towards Generative
Optimization") presents two ways to steer a PRE-TRAINED diffusion model with the
gradient of an objective f:

  naive : add the raw gradient of f at the current noisy state to the score.
          Talk, 40:10: this "jeopardizes the low-dimensional structure that was
          learned from pre-training ... this does not work."
  gloss : the gradient of a look-ahead quadratic loss (y_target - f(E[x0|xt]))^2
          computed THROUGH the pre-trained denoiser (autograd of a function of the
          score network, 41:49), which preserves the latent structure, with the
          tilt scaled by the gap between target reward and the current expected
          reward at xt (39:27).

This repo has NEITHER. activesearch.py uses the ridge surrogate only OUTSIDE the
sampler (acquisition: battle the 128 best-ranked of 512 blind decodes). The talk
puts the surrogate INSIDE the decode loop. This experiment ports both guidance
forms to the discrete constrained decoder and measures, against the unguided
baseline (the activesearch gen0 protocol, fresh replicate):

  structure : Showdown validity of the raw decode stream, copy rate, NN-Hamming,
              distinct species sets   (the discrete stand-ins for "latent structure")
  objective : win rate, 128 teams x 24 battles vs the top-50 meta pool

Discrete port. The decoder reveals one column at a time in dependency order; at
the reveal of column c the model gives p(x_c | partial team). Guidance multiplies
it by exp(scale * tilt_c[v]) with tilt_c[v] = df/dx_{c=v}. The ridge f is linear
in field indicators + species-pair indicators, so its gradient in the token
coordinate (c, v) is  w_single[c,v] + sum over other species slots of w_pair:

  naive : the pair sum runs over already-revealed slots only (the gradient at the
          partial state xt: masked coordinates are zero), scale = lambda.
  gloss : still-masked species slots contribute their pair term under the model's
          own posterior for that slot - the look-ahead E[x0|xt] - and
          scale = lambda * max(0, y_target - f_hat(E[x0|xt])), annealing to zero
          as the look-ahead completion reaches the target.

  gjac  : the faithful form of the talk's fix (added 2026-09-01 on the owner's
          request - gloss above stops short of it). The one-hot input of the
          column being decoded is relaxed, f_hat is computed at the look-ahead
          completion the TRANSFORMER predicts from that relaxed input, and the
          tilt is autograd of f_hat back to the input: gloss's direct term plus
          the Jacobian term "how does the model re-predict the other slots when
          this slot leans toward v". Gap-scaled like gloss. One extra
          forward+backward per decode step. The diag records the indirect/direct
          norm ratio and their cosine, i.e. how much the Jacobian term changes
          the direction gloss already had.

With a LINEAR surrogate the singles part of the gradient is constant in x, so the
naive and gloss arms can only differ through the species-pair term and the gap annealing -
the surrogate family bounds how much structure the guidance can see. Both arms
decode under the same per-column legality constraints as every other campaign
(the constrained decoder already projects each step onto the feasible set, which
is exactly the protection the continuous naive form lacks - so this also tests
whether the talk's warning binds in the discrete setting at all).

f is the activesearch ridge fitted once on every battle label this repo has
produced (activesearch anchor + gens, activesearch2 gens; win_rate key). Cells
share seeds (common random numbers): at lambda=0 all modes are the same decode.

    python gradguide.py smoke    # plumbing, tiny, no persistent results
    python gradguide.py calib    # stage 1, battle-free: structure + tilt scale
    python gradguide.py          # stage 2: battle the cells (resumable)
"""
import json, os, subprocess, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/tmp/vgc-pilot/src")
from corpus import parse_team, parse_team_text
from encode import NF, NSLOT
import diffusion as D
import hpsdiffusion as H
import activesearch as A
from hps_generate import Validator

ROOT = Path("/tmp/vgc-pilot/gradguide")
RESULTS = "/Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/gradguide.json"
AS1 = "/Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/activesearch.json"
AS2 = "/Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/activesearch2.json"

LAMBDAS = [4.0, 12.0, 36.0]     # judged by `calib`: tilt scale printed per cell
TARGET_Y = 0.60                 # gloss gap target; anchor max was 0.583-0.667
N_BATTLE = 128                  # teams battled per cell (no acquisition step)
BATTLES = 24                    # per team, vs the top-50 meta pool
N_CALIB = 192                   # raw decodes per battle-free calibration cell
CHUNK = 48

SPECIES_COLS = [c for c in range(D.COLS) if c % NF == 0]


# ---------------------------------------------------------------- labels -> f
def all_labels():
    """(team, win_rate) for everything this repo ever battled at 24x vs top-50."""
    teams, ys, srcs = [], [], []
    r = json.load(open(AS1))
    for f, y in r["anchor"].items():
        teams.append(parse_team(f)); ys.append(float(y)); srcs.append("as1_anchor")
    for tag, g in r["gens"].items():
        for p, y in zip(g["pastes"], g["y"]):
            teams.append(parse_team_text(p)); ys.append(float(y)); srcs.append("as1")
    try:
        r2 = json.load(open(AS2))
        for f, y in (r2.get("anchor") or {}).items():
            teams.append(parse_team(f))
            ys.append(float(y["win_rate"] if isinstance(y, dict) else y))
            srcs.append("as2_anchor")
        for tag, g in r2["gens"].items():
            for p, l in zip(g["pastes"], g["lab"]):
                teams.append(parse_team_text(p)); ys.append(float(l["win_rate"]))
                srcs.append("as2")
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"  activesearch2 labels skipped ({e})", flush=True)
    return teams, ys, srcs


def fit_ridge(lab_teams, lab_y, corpus_teams):
    """One fit for the whole experiment. The Feats map is built over labelled +
    corpus teams so every value p0 can emit has an indicator; values never seen
    in a labelled team get weight exactly 0 (their design column is all zero)."""
    F = A.Feats(lab_teams + corpus_teams)
    Xl = F.mat(lab_teams)
    w = np.linalg.solve(Xl.T @ Xl + np.diag(F.lam),
                        Xl.T @ np.asarray(lab_y, dtype=float))
    return F, w


class Guide:
    """The ridge gradient, laid out over the decoder's vocabularies."""

    def __init__(self, F, w, V, device):
        self.V, self.device = V, device
        self.single = {}
        for k in ["species", "ability", "item", "move", "nature"]:
            v = np.zeros(len(V.itos[k]))
            for i, s in enumerate(V.itos[k]):
                j = F.idx.get((k, s))
                if j is not None:
                    v[i] = w[j]
            self.single[k] = torch.tensor(v, dtype=torch.float32, device=device)
        S = len(V.itos["species"])
        Wp = np.zeros((S, S))
        sidx = {s: i for i, s in enumerate(V.itos["species"])}
        self.n_pairs_mapped = 0
        for (a, b), j in F.pairs.items():
            ia, ib = sidx.get(a), sidx.get(b)
            if ia and ib:
                Wp[ia, ib] = Wp[ib, ia] = w[j]
                self.n_pairs_mapped += 1
        self.Wp = torch.tensor(Wp, dtype=torch.float32, device=device)
        self.intercept = float(w[-1])

    def _post(self, model, h, cols):
        """Per-column posteriors p(x_c | partial), [MASK] zeroed, renormalised."""
        out = {}
        for c in cols:
            p = torch.softmax(model.logits(h, c), -1)
            p[:, 0] = 0
            out[c] = p / p.sum(-1, keepdim=True).clamp(min=1e-9)
        return out

    def tilt(self, model, h, x, c, mode):
        """d f / d x_{c=v} for every candidate v of column c.  n x V_c."""
        k = self.V.key(c)
        t = self.single[k].unsqueeze(0).expand(x.shape[0], -1).clone()
        if k == "species":
            others = [c2 for c2 in SPECIES_COLS if c2 != c]
            post = self._post(model, h, others) if mode == "gloss" else {}
            for c2 in others:
                dec = x[:, c2] > 0
                if dec.any():
                    t[dec] += self.Wp[x[dec, c2]]
                if mode == "gloss" and (~dec).any():
                    t[~dec] += post[c2][~dec] @ self.Wp
        return t

    def fhat(self, model, h, x):
        """f at the look-ahead completion E[x0|xt], under independent per-column
        posteriors (decided columns are one-hot).  Returns a length-n tensor."""
        n = x.shape[0]
        f = torch.full((n,), self.intercept, device=self.device)
        undec = [c for c in range(D.COLS) if bool((x[:, c] == 0).any())]
        post = self._post(model, h, undec)
        q = []                                   # species occupancy, n x S
        for c in range(D.COLS):
            k = self.V.key(c)
            dec = x[:, c] > 0
            pc = torch.zeros(n, self.single[k].shape[0], device=self.device)
            if dec.any():
                pc[dec] = torch.nn.functional.one_hot(
                    x[dec, c], self.single[k].shape[0]).float()
            if (~dec).any() and c in post:
                pc[~dec] = post[c][~dec]
            f = f + pc @ self.single[k]
            if k == "species":
                q.append(pc)
        for i in range(len(q)):
            for j in range(i + 1, len(q)):
                f = f + (q[i] @ self.Wp * q[j]).sum(-1)
        return f

    def gap(self, model, h, x):
        return (TARGET_Y - self.fhat(model, h, x)).clamp(min=0.0)

    # ---- the faithful form: autograd THROUGH the denoiser (owner, 2026-09-01) ----
    def _forward_soft(self, model, x, tt, w, c, q_c):
        """model.forward with column c's embedding replaced by the soft mix
        q_c @ E, so the transformer's predictions for every other masked column
        become a differentiable function of q_c."""
        B, d = x.shape[0], model.pos.shape[1]
        h = torch.empty(B, D.COLS, d, device=x.device)
        for k, cols in model.cols_by_key.items():
            h[:, cols] = model.emb[k](x[:, cols])
        h = h.clone()
        h[:, c] = q_c @ model.emb[model.keys[c]].weight
        h = h + model.pos.unsqueeze(0)
        h = h + model.tproj(tt.view(-1, 1).float()).unsqueeze(1)
        h = h + model.wemb(w).unsqueeze(1)
        return model.ln(model.tr(h))

    def tilt_jac(self, model, x, tt, w, c):
        """gloss's direction PLUS the Jacobian term: d f_hat / d q_c where the
        look-ahead completion E[x0|xt] itself is re-predicted by the denoiser
        as q_c moves.  Linearised at the model's current posterior for c.
        Returns (total tilt, f_hat, direct-only tilt) - the last for diagnostics."""
        n = x.shape[0]
        with torch.no_grad():
            h0 = model(x, tt, w)
            p_c = torch.softmax(model.logits(h0, c), -1)
            p_c[:, 0] = 0
            p_c = p_c / p_c.sum(-1, keepdim=True).clamp(min=1e-9)
            direct = self.tilt(model, h0, x, c, "gloss")
        q_c = p_c.clone().requires_grad_(True)
        h = self._forward_soft(model, x, tt, w, c, q_c)
        f = torch.full((n,), self.intercept, device=self.device)
        q = []
        for c2 in range(D.COLS):
            k = self.V.key(c2)
            if c2 == c:
                pc = q_c
            else:
                dec = x[:, c2] > 0
                pc = torch.zeros(n, self.single[k].shape[0], device=self.device)
                if dec.any():
                    pc = pc.clone()
                    pc[dec] = torch.nn.functional.one_hot(
                        x[dec, c2], self.single[k].shape[0]).float()
                if (~dec).any():
                    p = torch.softmax(model.logits(h, c2), -1)
                    p = p * (torch.arange(p.shape[1], device=p.device) > 0).float()
                    p = p / p.sum(-1, keepdim=True).clamp(min=1e-9)
                    pc = torch.where(dec.unsqueeze(1), pc, p)
            f = f + pc @ self.single[k]
            if k == "species":
                q.append(pc)
        for i in range(len(q)):
            for j in range(i + 1, len(q)):
                f = f + (q[i] @ self.Wp * q[j]).sum(-1)
        (g,) = torch.autograd.grad(f.sum(), q_c)
        return g.detach(), f.detach(), direct


# ---------------------------------------------------------------- the sampler
@torch.no_grad()
def sample_guided(model, C, G, n, mode, lam, device=D.DEV):
    """H.sample_constrained with a per-step gradient tilt on the logits."""
    w = torch.full((n,), H.WNULL, device=device, dtype=torch.long)
    x = torch.zeros(n, D.COLS, dtype=torch.long, device=device)
    seq = list(D.ORDER)
    diag = []
    for step, c in enumerate(seq):
        t_now = 1.0 - step / max(len(seq), 1)
        tt = torch.full((n,), t_now, device=device)
        h = model(x, tt, w)
        lg = model.logits(h, c)
        if mode != "none" and lam > 0:
            extra = {}
            if mode == "gjac":
                with torch.enable_grad():
                    t, fh, direct = G.tilt_jac(model, x, tt, w, c)
                s = lam * (TARGET_Y - fh).clamp(min=0.0).unsqueeze(1)
                ind = t - direct
                extra = dict(indirect_over_direct=float(
                    (ind.norm(dim=-1) / direct.norm(dim=-1).clamp(min=1e-9)).mean()),
                    cos_direct=float(torch.nn.functional.cosine_similarity(
                        t, direct, dim=-1).mean()))
            else:
                t = G.tilt(model, h, x, c, mode)
                s = lam if mode == "naive" else lam * G.gap(model, h, x).unsqueeze(1)
            lg = lg + s * t
            if step in (0, 6, len(seq) - 1):
                spread = (t.max(-1).values - t.min(-1).values)
                sm = float(s) if mode == "naive" else float(s.mean())
                diag.append(dict(step=step, col=int(c),
                                 tilt_spread=float(spread.mean()), scale=sm, **extra))
        for b in range(n):
            ok = C.mask_for(c, x[b], device)
            l = lg[b].clone(); l[~ok] = -1e9
            p = torch.softmax(l, -1)
            x[b, c] = torch.multinomial(p, 1).item()
    return x, diag


def propose_guided(model, C, G, V, look, spreads, val, mode, lam, n_valid,
                   outdir, seed, *, sampling_stats=None):
    """Draw until `n_valid` Showdown-valid teams; the raw stream is the validity
    measurement, so every attempt is validated and counted."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    outdir.mkdir(parents=True, exist_ok=True)
    for f in outdir.glob("*.txt"):
        f.unlink()
    files, teams, pastes, attempts, diag = [], [], [], 0, None
    t0 = time.perf_counter()
    while len(files) < n_valid and attempts < n_valid * 8:
        want = min(n_valid * 8 - attempts, max(2, min(CHUNK, 2 * (n_valid - len(files)))))
        x, dg = sample_guided(model, C, G, want, mode, lam)
        diag = diag or dg
        x = x.cpu().numpy()
        for i in range(want):
            attempts += 1
            txt = A.row_to_paste(V, x[i], look, spreads, rng)
            if val(txt) is None:
                p = outdir / f"{len(files):04d}.txt"
                p.write_text(txt)
                files.append(str(p)); pastes.append(txt)
                teams.append(parse_team_text(txt))
                if len(files) >= n_valid:
                    break
    mins = (time.perf_counter() - t0) / 60
    if sampling_stats is not None:
        sampling_stats.update(attempts=attempts, accepted=len(files),
                              rejected=attempts - len(files), attempt_limit=n_valid * 8)
    print(f"    [{mode}@{lam}] {len(files)} valid / {attempts} sampled "
          f"({len(files)/max(attempts,1):.0%}) in {mins:.1f} min", flush=True)
    return files, teams, pastes, len(files) / max(attempts, 1), diag


def other_battle_runs():
    """PIDs of shard drivers that are not ours - the two live campaigns."""
    # full path: a plain "src/shard.py" pattern also matches any monitoring shell
    # whose command text mentions the shard script (measured 2026-09-01: it made
    # this check see a phantom battle run and wait the full 20 min per cell)
    r = subprocess.run(["pgrep", "-f", "vgc-pilot/src/shard.py"],
                       capture_output=True, text=True)
    return [p for p in r.stdout.split() if p]


def wait_for_quiet(max_wait=1200):
    t0 = time.time()
    while time.time() - t0 < max_wait:
        if not other_battle_runs():
            return
        print("    another battle run is on the servers; waiting 30 s …", flush=True)
        time.sleep(30)
    print("    proceeding despite server load (waited 20 min)", flush=True)


# ---------------------------------------------------------------- the run
def cells():
    out = [("none", 0.0)]
    for m in ["naive", "gloss"]:
        for l in LAMBDAS:
            out.append((m, l))
    # gloss's tilt is self-annealed by the gap factor (~0.3 at the start of a
    # decode), so its effective strength at a shared lambda is ~1/3 of naive's;
    # this cell lets the two arms overlap at matched effective strength.
    out.append(("gloss", 108.0))
    # the faithful form (autograd through the denoiser), owner-requested
    # 2026-09-01; lambda grid bracketing gloss@108's effective strength
    for l in [36.0, 108.0, 324.0]:
        out.append(("gjac", l))
    return out


def main(stage):
    smoke = stage == "smoke"
    n_battle, battles, n_calib = (6, 2, 8) if smoke else (N_BATTLE, BATTLES, N_CALIB)

    print("loading corpus + labels …", flush=True)
    cf = A.load_corpus_files()
    corpus_teams = [t for _, t in cf]
    corpus_sets = {A.species_set(t) for t in corpus_teams}
    V = A.Vocab(corpus_teams); L = A.Legality(corpus_teams); C = D.Constraints(V, L)
    corpus_grid = np.stack([V.encode(t) for t in corpus_teams])
    look, spreads = A.decode_tables(corpus_teams)
    lab_teams, lab_y, srcs = all_labels()
    print(f"  {len(lab_teams)} labels "
          f"({', '.join(f'{s}:{srcs.count(s)}' for s in dict.fromkeys(srcs))}), "
          f"mean {np.mean(lab_y):.3f}", flush=True)
    F, wv = fit_ridge(lab_teams, lab_y, corpus_teams)
    G = Guide(F, wv, V, D.DEV)
    mu_lab = F.mat(lab_teams) @ wv
    print(f"  ridge: dim {F.dim} ({len(F.pairs)} pairs, {G.n_pairs_mapped} on the "
          f"decoder vocab) · in-sample Spearman {A.spearman(mu_lab, lab_y):+.3f}",
          flush=True)

    p0 = H.TeamDiffusionHPS(V).to(D.DEV)
    p0.load_state_dict(torch.load(A.P0_CKPT, map_location=D.DEV)["sd"]); p0.eval()
    val = Validator()
    out = (json.load(open(RESULTS)) if os.path.exists(RESULTS) and not smoke
           else {"config": {}, "calib": {}, "cells": {}})
    out["config"] = dict(lambdas=LAMBDAS, target_y=TARGET_Y, n_battle=N_BATTLE,
                         battles=BATTLES, n_calib=N_CALIB, p0=A.P0_CKPT,
                         n_labels=len(lab_teams))

    def flush():
        if smoke:
            return
        tmp = RESULTS + ".tmp"
        json.dump(out, open(tmp, "w"), indent=1); os.replace(tmp, RESULTS)

    if stage in ("calib", "smoke"):
        print(f"\n== stage 1: battle-free, {n_calib} raw decodes per cell ==",
              flush=True)
        print(f"{'cell':12s} {'valid':>6s} {'copy':>6s} {'NN-Ham':>7s} "
              f"{'sets':>5s} {'mu':>7s} {'mu_p90':>7s} {'tilt0':>13s}", flush=True)
        for mode, lam in cells():
            tag = f"{mode}@{lam:g}"
            if tag in out["calib"]:
                print(f"{tag:12s} cached", flush=True); continue
            torch.manual_seed(4200)
            x, diag = sample_guided(p0, C, G, n_calib, mode, lam)
            x = x.cpu().numpy()
            rng = np.random.default_rng(4200)
            pastes = [A.row_to_paste(V, r, look, spreads, rng) for r in x]
            okv = [val(p) is None for p in pastes]
            teams = [parse_team_text(p) for p in pastes]
            mem = A.memorisation(V, teams, corpus_grid)
            mu = F.mat(teams) @ wv
            rec = dict(validity=float(np.mean(okv)), **mem,
                       mu_mean=float(mu.mean()), mu_p90=float(np.percentile(mu, 90)),
                       diag=diag)
            out["calib"][tag] = rec; flush()
            d0 = diag[0] if diag else {}
            print(f"{tag:12s} {rec['validity']:6.2f} {rec['copy_rate']:6.2f} "
                  f"{rec['nn_hamming']:7.1f} {rec['distinct_species_sets']:5d} "
                  f"{rec['mu_mean']:7.3f} {rec['mu_p90']:7.3f} "
                  f"{d0.get('tilt_spread', 0):6.3f}x{d0.get('scale', 0):6.2f}",
                  flush=True)
        if stage == "calib":
            val.close(); print("\nCALIB_DONE", flush=True); return

    print(f"\n== stage 2: {n_battle} teams x {battles} battles per cell ==",
          flush=True)
    opp = A.opponents()
    for mode, lam in cells():
        tag = f"{mode}@{lam:g}"
        if tag in out["cells"]:
            print(f"[{tag}] cached", flush=True); continue
        t0 = time.perf_counter()
        files, teams, pastes, validity, diag = propose_guided(
            p0, C, G, V, look, spreads, val, mode, lam, n_battle,
            ROOT / tag.replace("@", "_"), 7000)
        mu = F.mat(teams) @ wv
        if not smoke:
            wait_for_quiet()
        res = A.pool.score([(f, opp) for f in files], battles=battles, conc=50)
        keep = [i for i, f in enumerate(files)
                if f in res and res[f]["battles"] > 0]
        y = [res[files[i]]["win_rate"] for i in keep]
        st = A.batch_stats(y, [teams[i] for i in keep], corpus_sets, 0.4625)
        st.update(validity=validity, mode=mode, lam=lam,
                  mu_battled=float(np.mean([mu[i] for i in keep])),
                  surrogate_spearman=A.spearman([mu[i] for i in keep], y),
                  proposal=A.memorisation(V, teams, corpus_grid),
                  n_proposed=len(teams), diag=diag,
                  minutes=(time.perf_counter() - t0) / 60)
        out["cells"][tag] = dict(stats=st, pastes=[pastes[i] for i in keep], y=y)
        flush()
        print(A.fmt(tag, st), flush=True)

    val.close()
    print(f"\n{'cell':12s} {'mean':>8s} {'se':>7s} {'p90':>7s} {'max':>7s} "
          f"{'valid':>6s} {'copy':>6s} {'NN':>5s} {'sets':>5s} {'mu':>7s} {'rho':>7s}",
          flush=True)
    for mode, lam in cells():
        tag = f"{mode}@{lam:g}"
        if tag not in out["cells"]:
            continue
        s = out["cells"][tag]["stats"]; p = s.get("proposal") or {}
        r = s.get("surrogate_spearman")
        print(f"{tag:12s} {s['mean']:8.4f} {s['se'] or 0:7.4f} {s['p90']:7.3f} "
              f"{s['max']:7.3f} {s['validity']:6.2f} {p.get('copy_rate', 0):6.2f} "
              f"{p.get('nn_hamming', 0):5.1f} {p.get('distinct_species_sets', 0):5d} "
              f"{s['mu_battled']:7.3f} "
              f"{'  n/a' if r is None else f'{r:+7.3f}'}", flush=True)
    flush()
    print("\nGRADGUIDE_DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "battle")
