"""Inference/decoding-scheme sweep on ONE fixed checkpoint (/tmp/vgc-pilot/wrdiffusion.pt).

THE QUESTION
------------
SubGoal 1.1 asks whether the diffusion generator can drop constrained decoding
altogether: "the goal is to have the diffusion model do not use constraint decoding
at all or at least the generation of non legal pokemon team is small enough to
filter out use the smogon verification feature."  Nothing here retrains anything.
Same weights, same conditioning (win-rate bin 5, style "none", guidance 2 unless
stated), same spread policy — only the DECODER changes.  So the headline number is
the SHOWDOWN ACCEPTANCE RATE of each scheme (valid teams / attempts), and win rate
is the sanity check that a cheaper decoder did not buy legality with strength.

Goal 5.2 also asks for the two use modes: (1) generate a whole legal team,
(2) hand the model a hole — a move slot, a candidate, a pinned species — and let it
fill the rest.  Arms 11-13 are that second mode.

THE ARMS
--------
"Constrained" throughout means an ARC-CONSISTENT projection onto the feasible set: each
slot carries the set of species still compatible with what that slot has already
decoded, a move is feasible iff some candidate species learns it, and the species column
is that candidate set.  This is required for arms 2-6 to be about ORDER: the mask that
ships in diffusion.Constraints is conditioned on the species, so under a random column
order a move decoded before its species is unconstrained and the team is dead on
arrival — measured on this checkpoint, 0/8 rule-legal for random_order, confidence,
entropy_hi, parallel8 and remask8.  Under dependency order the candidate set is the
singleton {species} and the arc-consistent mask is provably (and, in `--selftest`,
assertedly) identical to the shipped one, so arm 1 is still bit-identical to what runs
today.

Full generation, feasibility-masked at every step ("constrained"):
  1  deporder_g2           dependency order (species -> ability/item -> moves -> nature),
                           one column at a time, guidance 2, temp 1.  THE BASELINE:
                           bit-identical to wrdiffusion.sample_constrained.
  2  random_order          a fresh random column permutation per team.
  3  confidence            max-confidence-first: decode the still-masked column whose
                           CONSTRAINED predictive distribution has the LOWEST entropy.
  4  entropy_hi            the ablation: highest-entropy column first.
  5  parallel8             the true MDLM ancestral sampler: 8 reveal steps, each masked
                           column revealed independently with prob (t_now-t_next)/t_now,
                           constraints applied at the moment of reveal (within a step
                           the revealed columns are resolved in dependency order, so a
                           move revealed BEFORE its species gets no learnset mask — that
                           is a property of the sampler, not a bug in the harness).
  6  remask8               parallel8 + ReMDM-style inference-time scaling: after each
                           reveal step, the 20% of already-revealed columns with the
                           lowest CURRENT constrained probability are remasked and redone.
                           A move that became illegal when its species landed has
                           probability ~0, so this doubles as legality repair.
  7  lowtemp               dependency order, temp 0.7, top-p 0.9.

The subgoal arms — no feasibility mask at all, only [MASK] is forbidden:
  8  unconstrained         dependency order, guidance 2, temp 1.
  9  unconstrained_lowtemp the same at temp 0.5, top-p 0.9.  Low temperature piles mass
                           on the modal (usually legal) value; this is the strongest
                           honest shot at "generate freely and filter".
 10  clause_only           A MIDDLE ARM: enforce ONLY the two team-level clauses that
                           need no learnset lookup (Species Clause on the species column,
                           Item Clause on the item column) and leave per-species
                           ability/move legality free.
 14  no_learnset           A SECOND MIDDLE ARM, added after a 512-attempt probe returned
                           0/512 for arms 8, 9 AND 10 alike, with the modal Showdown
                           rejections being "Garchomp can't have Defiant" (ability),
                           "has multiple copies of Protect" (duplicate move) and
                           "Blastoise-Mega transforms in-battle with Blastoisinite"
                           (mega forme) — NOT learnsets.  So arm 10 cannot separate "the
                           clauses are cheap" from "the learnset table is what costs".
                           This arm keeps everything the full mask does EXCEPT the
                           per-species learnset table: clauses, mega rule, ability
                           legality, duplicate-move exclusion.  It is the arm that
                           actually answers SubGoal 1.1, because the learnset table is
                           the only part that needs a per-species lookup.
                           PRE-REGISTERED before its run: if arm 14 lands anywhere near
                           arm 1's acceptance rate, the answer to the subgoal is "drop
                           the learnset mask, keep the four cheap rules, filter the
                           rest"; if it is also near zero, unconstrained decoding is
                           dead at this scale and constrained decoding stays.

Fill-in-the-blank (expectation (2)), all constrained, bin 5, guidance 2:
 11  fill_slot             a real corpus team with ONE whole candidate (8 columns) masked.
 12  fill_move             a real corpus team with ONE single move column masked.
 13  fill_pin              a whole team generated with Garchomp pinned into slot 0 (the GUI case).

PRE-REGISTERED EXPECTATION (written before the run)
---------------------------------------------------
* Every CONSTRAINED full-generation arm (1-7) should land near 0.19 win rate — the
  measured bin-5 numbers are 0.193-0.195 at guidance 4 and 0.168 at guidance 2 — and
  should differ from each other by LESS than the ~3-point run-to-run noise.  Decoding
  order is a reparameterisation of the same joint, so if arms 1-4 separate by more than
  noise the model's conditionals are inconsistent, which is a finding about the model,
  not about the order.  Reference points, do not invent new ones: real corpus team
  0.470-0.475, median real team 0.458, slotcopy 0.420-0.436, unconditioned generator
  0.128, random legal team ~0.011.
* parallel8 is expected to LOSE acceptance rate badly versus arm 1, because moves are
  routinely revealed before the species that licenses them.  remask8 is the fix and is
  expected to recover a large part of that gap.  If it does not, inference-time scaling
  is not worth its extra forward passes here.
* The interesting numbers are the ACCEPTANCE RATES of arms 8-10.  Prior evidence: an
  older checkpoint decoded unconstrained at temp 1 gave 0/256 Showdown-legal.  If
  arm 8 is again ~0 and arm 9 is still under ~1%, "no constrained decoding, just
  filter" is dead at this scale (>100 samples per accepted team) and arm 10 is the
  live compromise.  A workable answer means roughly >= 5% — under 20 samples per
  accepted team, which at ~700 validations/sec is free next to 24 battles.
* Fill arms should be the strongest teams in the sweep (5 or 6 of the 6 candidates are
  real), so fill_move ought to sit close to the real-team band and fill_slot between
  slotcopy and real.  If fill_slot is not clearly above the from-scratch arms, the
  model is not learning the conditional, only the marginal.

HELD CONSTANT ACROSS ARMS (so the comparison is about decoding and nothing else)
--------------------------------------------------------------------------------
Spreads are COPIED, never generated: every rendered slot draws a Stat-Point spread
uniformly from the corpus spreads recorded for that species (measured cost of copying
0.6 points, of generating 2.1 points).  That rule is applied identically to generated
and to carried-over slots, including in the fill arms, so no arm gets a spread
advantage.  Nature is a model column everywhere.  Every emitted team is gated by
Showdown's real TeamValidator before it is battled; win rate is 24 battles against the
same 50 top-placement meta teams; the standard error is CLUSTERED BY TEAM (std over
per-team win rates / sqrt(n_teams)), never binomial over battles.

Arms are capped at --max-attempts (default 20,000) sampling attempts.  A capped arm is
recorded with "capped": true and its real valid count — a truncated arm must never read
as a completed one.

  python schemes.py --selftest
  python schemes.py --smoke
  python schemes.py [--only deporder_g2,unconstrained] [--per 150] [--battles 24]
  python schemes.py --report
"""
import argparse, itertools, json, math, os, random, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/tmp/vgc-pilot/src")
import diffusion as D
import wrdiffusion as W
import pool
from corpus import norm, STATS
from encode import NF, NSLOT
from loop import PMI, opponents
from propose import Validator, slot_to_text

COLS = W.COLS
DEV = W.DEV
OUT = "/tmp/vgc-pilot/schemes_results.json"
GEN = Path("/tmp/vgc-pilot/schemes_gen")
REAL_SURPRISE_REF = 1.844        # PMI surprise of a real corpus team (measured)
REAL_MEDIAN = 0.458              # win rate of the median real team (measured)

# name -> decoding configuration.  Only the decoder varies.
ARMS = [
    ("deporder_g2",           dict(order="dep",  cmode="full")),
    ("random_order",          dict(order="rand", cmode="full")),
    ("confidence",            dict(order="conf", cmode="full")),
    ("entropy_hi",            dict(order="ent",  cmode="full")),
    ("parallel8",             dict(order="par",  cmode="full", steps=8)),
    ("remask8",               dict(order="par",  cmode="full", steps=8, remask=0.20)),
    ("lowtemp",               dict(order="dep",  cmode="full", temp=0.7, top_p=0.9)),
    ("unconstrained",         dict(order="dep",  cmode="none")),
    ("unconstrained_lowtemp", dict(order="dep",  cmode="none", temp=0.5, top_p=0.9)),
    ("clause_only",           dict(order="dep",  cmode="clause")),
    ("no_learnset",           dict(order="dep",  cmode="nolearn")),
    ("fill_slot",             dict(order="dep",  cmode="full", seed="slot")),
    ("fill_move",             dict(order="dep",  cmode="full", seed="move")),
    ("fill_pin",              dict(order="dep",  cmode="full", pin_species="Garchomp")),
    # added 2026-08-27 after lowtemp won by 8.7 points: the GUI's exact call path,
    # pinned generation, re-measured at the winning temperature.
    ("fill_pin_lowtemp",      dict(order="dep",  cmode="full", pin_species="Garchomp", temp=0.7, top_p=0.9)),
]
DEFAULTS = dict(order="dep", cmode="full", guidance=2.0, temp=1.0, top_p=1.0,
                wbin=5, style="none", steps=8, remask=0.0, seed=None, pin_species=None)

# --------------------------------------------------------------------------
# feasibility masks
#
# "Constrained" here means an ARC-CONSISTENT projection onto the feasible set, not
# the one-directional mask that ships in diffusion.Constraints.  That mask is
# conditioned on the species, so it can only bind once the species is known: decode a
# move first and it is unconstrained, and the species that arrives later never learns
# it.  Measured on this checkpoint, the shipped mask under a random column order gives
# 0/8 rule-legal teams — so arms 2-6 would all be dead at 0% acceptance and the sweep
# would be measuring the mask, not the ordering.
#
# So each slot carries a CANDIDATE SPECIES SET: the species still allowed by the
# Species Clause that also learn every move already fixed in the slot, have the ability
# already fixed, and match a mega stone already fixed.  A move is feasible iff some
# candidate species learns it; an ability iff some candidate has it; the species column
# is exactly the candidate set.  Picking from those sets can never empty the set, so
# legality survives any decoding order.
#
# This reduces EXACTLY to diffusion.Constraints.mask_for under dependency order (the
# species is decided first, so the candidate set is the singleton {species}); the
# equality is asserted in `selftest`.  Arm 1 is therefore still bit-identical to what
# runs today.
# --------------------------------------------------------------------------
class Arc:
    """Arc-consistent, order-independent feasibility masks."""
    def __init__(self, C, V):
        from corpus import MEGA_STONE_OF
        self.C, self.V = C, V
        nsp, nmv, nab = V.sizes["species"], V.sizes["move"], V.sizes["ability"]
        z = lambda k: torch.zeros(k, dtype=torch.bool)
        self.MV = torch.stack([C.mv_ok.get(i, z(nmv)) for i in range(nsp)])
        self.AB = torch.stack([C.ab_ok.get(i, z(nab)) for i in range(nsp)])
        bases = sorted({C.base_of[i] for i in range(nsp)})
        bid = {b: i for i, b in enumerate(bases)}
        self.base_id = torch.tensor([bid[C.base_of[i]] for i in range(nsp)])
        self.stone_sp = {}
        for vi, it in enumerate(V.itos["item"]):
            if vi and it in MEGA_STONE_OF:
                b = MEGA_STONE_OF[it]
                self.stone_sp[vi] = torch.tensor(
                    [j > 0 and C.base_of[j] == b and "mega" not in V.itos["species"][j]
                     for j in range(nsp)])
        self.fallbacks = 0

    def cand(self, row, slot):
        """Species still possible for this slot given everything already decided."""
        C = self.C
        si = int(row[slot * NF])
        if si != 0:
            ok = torch.zeros(self.MV.shape[0], dtype=torch.bool); ok[si] = True; return ok
        ok = torch.ones(self.MV.shape[0], dtype=torch.bool); ok[0] = False
        used = {C.base_of[int(row[s * NF])] for s in range(NSLOT)
                if s != slot and int(row[s * NF]) != 0}
        if used:
            ub = torch.tensor(sorted({self.base_id[j].item() for j in range(len(self.base_id))
                                      if C.base_of[j] in used}))
            ok &= ~torch.isin(self.base_id, ub)
        a = int(row[slot * NF + 1])
        if a: ok &= self.AB[:, a]
        it = int(row[slot * NF + 2])
        if it in self.stone_sp: ok &= self.stone_sp[it]
        for j in range(3, 7):
            mv = int(row[slot * NF + j])
            if mv: ok &= self.MV[:, mv]
        return ok

    def mask(self, col, row):
        j, slot = col % NF, col // NF
        if int(row[slot * NF]) != 0 or j == 7:
            return self.C.mask_for(col, row, "cpu")   # fast path: identical by construction
        cd = self.cand(row, slot)
        if not cd.any():
            self.fallbacks += 1
            return self.C.mask_for(col, row, "cpu")   # stranded slot: let Showdown reject it
        if j == 0:
            return cd
        if j == 1:
            ok = self.AB[cd].any(0); ok[0] = False
        elif j == 2:
            ok = torch.ones(self.V.sizes["item"], dtype=torch.bool); ok[0] = False
            for s in range(NSLOT):
                v = int(row[s * NF + 2])
                if s != slot and v != 0: ok[v] = False
            for vi, sp in self.stone_sp.items():
                if not bool((sp & cd).any()): ok[vi] = False
        else:
            ok = self.MV[cd].any(0); ok[0] = False
            for j2 in range(3, 7):
                v = int(row[slot * NF + j2])
                if j2 != j and v != 0: ok[v] = False
        if not ok.any():
            self.fallbacks += 1
            return self.C.mask_for(col, row, "cpu")
        return ok

def nolearn_mask(C, col, row):
    """Everything the full mask does EXCEPT the per-species learnset table: Species
    Clause, Item Clause, the mega-stone/forme rule, ability legality and duplicate-move
    exclusion all stay; move columns are otherwise free.  This isolates the one mask
    that costs a learnset lookup, which is the mask SubGoal 1.1 wants to delete."""
    j, slot = col % NF, col // NF
    if not (3 <= j <= 6):
        return C.mask_for(col, row, "cpu")
    ok = torch.ones(C.V.sizes["move"], dtype=torch.bool); ok[0] = False
    for j2 in range(3, 7):
        v = int(row[slot * NF + j2])
        if j2 != j and v != 0: ok[v] = False
    if not ok.any(): ok[0] = True
    return ok

def clause_mask(C, col, row):
    """Species Clause + Item Clause only - the two team-level rules that need no
    learnset lookup.  Everything else (ability legality, learnsets, duplicate moves)
    is left to the model."""
    k = C.V.key(col); n = C.V.sizes[k]
    ok = torch.ones(n, dtype=torch.bool); ok[0] = False        # never emit [MASK]
    j, slot = col % NF, col // NF
    if j == 0:
        used = {C.base_of[int(row[s * NF])] for s in range(NSLOT)
                if s != slot and int(row[s * NF]) != 0}
        if used:
            for vi in range(1, n):
                if C.base_of[vi] in used: ok[vi] = False
    elif j == 2:
        for s in range(NSLOT):
            v = int(row[s * NF + 2])
            if s != slot and v != 0: ok[v] = False
    if not ok.any(): ok[0] = True
    return ok

_FREE = {}
def feas(A, cmode, col, row):
    """Per-column feasible-value mask (CPU bool) under one of the three regimes."""
    if cmode == "full":
        return A.mask(col, row)
    if cmode == "clause":
        return clause_mask(A.C, col, row)
    if cmode == "nolearn":
        return nolearn_mask(A.C, col, row)
    k = A.V.key(col)                                            # cmode == "none"
    if k not in _FREE:
        m = torch.ones(A.V.sizes[k], dtype=torch.bool); m[0] = False
        _FREE[k] = m
    return _FREE[k]

def stack_masks(A, cmode, col, xc, rows, device):
    return torch.stack([feas(A, cmode, col, xc[b]) for b in rows]).to(device)

# --------------------------------------------------------------------------
# sampling primitives
# --------------------------------------------------------------------------
def top_p_filter(p, top_p):
    if top_p >= 1.0: return p
    sp, idx = torch.sort(p, descending=True, dim=-1)
    keep = (sp.cumsum(-1) - sp) < top_p
    out = torch.zeros_like(p).scatter_(-1, idx, sp * keep)
    return out / out.sum(-1, keepdim=True).clamp_min(1e-12)

def draw(logits, mask, temp, top_p):
    p = torch.softmax(logits.masked_fill(~mask, -1e9) / temp, -1)
    return torch.multinomial(top_p_filter(p, top_p), 1).squeeze(-1)

@torch.no_grad()
def guided(model, x, tt, y, w, guidance):
    """D-CFG log-probabilities for EVERY column at once (the heads are ~6% of a
    forward, so computing all 48 costs nothing over computing one)."""
    hc = model(x, tt, y, w)
    lc = [F.log_softmax(model.logits(hc, c), -1) for c in range(COLS)]
    if guidance == 1.0: return lc
    hu = model(x, tt, torch.full_like(y, D.NULL), torch.full_like(w, W.WNULL))
    return [guidance * lc[c] + (1 - guidance) * F.log_softmax(model.logits(hu, c), -1)
            for c in range(COLS)]

# --------------------------------------------------------------------------
# the decoder: one function, every scheme
# --------------------------------------------------------------------------
@torch.no_grad()
def decode(model, A, n, cfg, x0=None, m0=None, pin=None, rng=None, device=DEV):
    """Returns an (n, COLS) grid.  cfg keys: order, cmode, guidance, temp, top_p,
    wbin, style, steps, remask."""
    g, temp, tp = cfg["guidance"], cfg["temp"], cfg["top_p"]
    cmode, order = cfg["cmode"], cfg["order"]
    y = torch.full((n,), D.STYLES.index(cfg["style"]) if cfg["style"] in D.STYLES else D.NULL,
                   device=device, dtype=torch.long)
    w = torch.full((n,), cfg["wbin"], device=device, dtype=torch.long)
    if x0 is None:
        x = torch.zeros(n, COLS, dtype=torch.long, device=device)
        masked = torch.ones(n, COLS, dtype=torch.bool, device=device)
    else:
        x = x0.clone().to(device); masked = m0.clone().to(device); x[masked] = 0
    if pin:
        for c, v in pin.items(): x[:, c] = v; masked[:, c] = False
    gen_cols = masked.clone()                       # what this call is allowed to touch

    if order == "par":
        return _parallel(model, A, x, masked, gen_cols, y, w, cfg, device)

    # ---- one column at a time -------------------------------------------
    if order in ("dep", "rand"):
        seqs = []
        for b in range(n):
            todo = [c for c in range(COLS) if bool(masked[b, c])]
            if order == "dep":
                seqs.append([c for c in D.ORDER if c in set(todo)])
            else:
                seqs.append([todo[i] for i in rng.permutation(len(todo))])
        nsteps = max((len(s) for s in seqs), default=0)
    else:
        seqs = None
        nsteps = int(masked.sum(1).max()) if masked.any() else 0

    for step in range(nsteps):
        # MDLM time = fraction of the grid still masked (identical to
        # 1 - step/48 for a from-scratch team, honest for a partial fill)
        tt = masked.sum(1).float() / COLS
        lg = guided(model, x, tt, y, w, g)
        xc = x.cpu()
        if seqs is not None:
            bycol = defaultdict(list)
            for b in range(n):
                if step < len(seqs[b]): bycol[seqs[b][step]].append(b)
        else:
            bycol = _pick_by_entropy(A, cmode, lg, xc, masked, temp, order, device)
        for c, rows in bycol.items():
            r = torch.tensor(rows, device=device)
            M = stack_masks(A, cmode, c, xc, rows, device)
            x[r, c] = draw(lg[c][r], M, temp, tp)
            masked[r, c] = False
    return x

def _pick_by_entropy(A, cmode, lg, xc, masked, temp, order, device):
    """confidence / entropy_hi: score every still-masked column by the entropy of its
    CONSTRAINED predictive distribution, then decode one column per row."""
    n = xc.shape[0]
    H = torch.full((n, COLS), float("inf") if order == "conf" else float("-inf"))
    for c in range(COLS):
        rows = torch.nonzero(masked[:, c]).flatten().tolist()
        if not rows: continue
        M = stack_masks(A, cmode, c, xc, rows, device)
        p = torch.softmax(lg[c][torch.tensor(rows, device=device)].masked_fill(~M, -1e9) / temp, -1)
        h = -(p * torch.log(p.clamp_min(1e-12))).sum(-1)
        H[torch.tensor(rows), c] = h.cpu()
    pick = (H.argmin(1) if order == "conf" else H.argmax(1)).tolist()
    bycol = defaultdict(list)
    for b in range(n):
        if bool(masked[b].any()): bycol[pick[b]].append(b)
    return bycol

@torch.no_grad()
def _parallel(model, A, x, masked, gen_cols, y, w, cfg, device):
    """MDLM ancestral sampler, optionally with ReMDM remasking."""
    n = x.shape[0]
    g, temp, tp = cfg["guidance"], cfg["temp"], cfg["top_p"]
    cmode, steps, rf = cfg["cmode"], cfg["steps"], cfg["remask"]
    for i in range(steps):
        if not masked.any(): break
        t_now, t_next = 1.0 - i / steps, 1.0 - (i + 1) / steps
        lg = guided(model, x, torch.full((n,), t_now, device=device), y, w, g)
        p_rev = 1.0 if t_now <= 0 else (t_now - t_next) / t_now
        rev = masked & (torch.rand(n, COLS, device=device) < p_rev)
        if i == steps - 1: rev = masked.clone()
        xc = x.cpu()
        for c in D.ORDER:                       # resolve a step's reveals in dependency
            rows = torch.nonzero(rev[:, c]).flatten().tolist()   # order, so the mask sees
            if not rows: continue                                # what this step already fixed
            r = torch.tensor(rows, device=device)
            M = stack_masks(A, cmode, c, xc, rows, device)
            v = draw(lg[c][r], M, temp, tp)
            x[r, c] = v; masked[r, c] = False; xc[r.cpu(), c] = v.cpu()
        if rf <= 0 or i == steps - 1: continue
        # ---- ReMDM: remask the least-confident revealed columns ----------
        lg2 = guided(model, x, torch.full((n,), max(t_next, 1e-3), device=device), y, w, g)
        shown = gen_cols & ~masked
        P = torch.zeros(n, COLS)
        xc = x.cpu()
        for c in range(COLS):
            rows = torch.nonzero(shown[:, c]).flatten().tolist()
            if not rows: continue
            r = torch.tensor(rows, device=device)
            M = stack_masks(A, cmode, c, xc, rows, device)
            p = torch.softmax(lg2[c][r].masked_fill(~M, -1e9) / temp, -1)
            P[torch.tensor(rows), c] = p.gather(1, x[r, c].view(-1, 1)).squeeze(1).cpu()
        P[~shown.cpu()] = 2.0                    # never remask a fixed / already-masked column
        for b in range(n):
            k = int(round(rf * int(shown[b].sum())))
            if k < 1: continue
            worst = torch.topk(P[b], k, largest=False).indices.to(device)
            x[b, worst] = 0; masked[b, worst] = True
    if masked.any():                             # safety: nothing may leave as [MASK]
        lg = guided(model, x, masked.sum(1).float() / COLS, y, w, g)
        xc = x.cpu()
        for c in D.ORDER:
            rows = torch.nonzero(masked[:, c]).flatten().tolist()
            if not rows: continue
            r = torch.tensor(rows, device=device)
            M = stack_masks(A, cmode, c, xc, rows, device)
            v = draw(lg[c][r], M, temp, tp)
            x[r, c] = v; masked[r, c] = False; xc[r.cpu(), c] = v.cpu()
    return x

# --------------------------------------------------------------------------
# rendering, statistics
# --------------------------------------------------------------------------
def render(V, row, look, spreads, rng):
    """Grid -> parsed slots.  Spreads are COPIED from the corpus for that species,
    identically in every arm (see the module docstring)."""
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
    return slots

def key_of(V, row):
    """Slot-order-invariant identity of a team, for the uniqueness count."""
    sl = [tuple(V.decode_field(i * NF + j, int(row[i * NF + j])) for j in range(NF))
          for i in range(NSLOT)]
    return tuple(sorted(sl))

REASONS = [("can't learn",             "illegal move (learnset)"),
           ("can't have",              "illegal ability"),
           ("multiple copies of",      "duplicate move"),
           ("transforms in-battle",    "mega forme without its stone"),
           ("Species Clause",          "Species Clause"),
           ("Item Clause",             "Item Clause"),
           ("at least one move",       "empty move set"),
           ("does not exist",          "unknown name"),
           ("is banned",               "banned in this format")]

def why(msg):
    """Bucket one Showdown rejection clause, so the sweep says WHICH mask was
    load-bearing instead of only that the team was rejected."""
    m = msg.strip()
    for pat, tag in REASONS:
        if pat in m: return tag
    return "other: " + m[:55]

def wilson(k, n, z=1.96):
    if n == 0: return [0.0, 1.0]
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 6), round(min(1.0, c + h), 6)]

# --------------------------------------------------------------------------
# one arm
# --------------------------------------------------------------------------
def run_arm(name, cfg, ctx, per, battles, max_attempts, batch):
    V, L, A, model = ctx["V"], ctx["L"], ctx["A"], ctx["model"]
    look, spreads, corpus, pmi, opp, val = (ctx["look"], ctx["spreads"], ctx["corpus"],
                                            ctx["pmi"], ctx["opp"], ctx["val"])
    torch.manual_seed(0); random.seed(0); rng = np.random.default_rng(0)
    out = GEN / name; out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.txt"): f.unlink()
    Xc = ctx["Xc"]
    pin = None
    if cfg["pin_species"]:
        si = V.stoi["species"].get(norm(cfg["pin_species"]))
        if si is None: raise SystemExit(f"{cfg['pin_species']} not in the vocabulary")
        pin = {0: si}

    A.fallbacks = 0
    from collections import Counter
    rej = Counter()
    files, slots_of, keys, tries, t0 = [], {}, [], 0, time.perf_counter()
    while len(files) < per and tries < max_attempts:
        n = min(batch, max_attempts - tries)
        x0 = m0 = None
        if cfg["seed"]:
            src = Xc[rng.integers(0, len(Xc), n)]
            x0 = torch.tensor(src, device=DEV)
            m0 = torch.zeros(n, COLS, dtype=torch.bool, device=DEV)
            for b in range(n):
                s = int(rng.integers(0, NSLOT))
                if cfg["seed"] == "slot": m0[b, s * NF:(s + 1) * NF] = True
                else:                     m0[b, s * NF + 3 + int(rng.integers(0, 4))] = True
        X = decode(model, A, n, cfg, x0=x0, m0=m0, pin=pin, rng=rng).cpu().numpy()
        for row in X:
            # an attempt is a team actually rendered and shown to Showdown; rows left
            # over in the last batch are discarded UNCOUNTED, or the final partial batch
            # would inflate the acceptance rate this whole sweep is about
            if len(files) >= per: break
            tries += 1
            sl = render(V, row, look, spreads, rng)
            txt = "\n\n".join(slot_to_text(s) for s in sl) + "\n"
            err = val(txt)
            if err is not None:
                for part in err.split(";"): rej[why(part)] += 1
                continue
            p = out / f"{len(files):04d}.txt"; p.write_text(txt)
            files.append(str(p)); slots_of[str(p)] = sl; keys.append(key_of(V, row))
        print(f"  [{name}] {len(files)}/{per} valid · {tries} attempts "
              f"· {len(files)/max(tries,1):.3f}", flush=True)

    capped = len(files) < per
    acc = len(files) / max(tries, 1)
    r = dict(scheme=name, config={k: cfg[k] for k in DEFAULTS},
             attempts=tries, valid=len(files), n=len(files),
             acceptance=acc, acceptance_ci95=wilson(len(files), max(tries, 1)),
             samples_per_accepted=(tries / len(files)) if files else None,
             unique=(len(set(keys)) / len(keys)) if keys else None,
             capped=capped, cap=max_attempts if capped else None,
             arc_fallbacks=A.fallbacks,
             rejections=dict(rej.most_common(12)),
             gen_minutes=(time.perf_counter() - t0) / 60)
    if capped:
        r["cap_note"] = (f"TRUNCATED: hit the {max_attempts}-attempt cap with only "
                         f"{len(files)} valid teams of the {per} requested — this arm is "
                         f"NOT a completed arm and its win rate rests on {len(files)} teams.")
    if not files:
        r.update(win_rate=None, se=None, p90=None, max=None,
                 above_real_median=None, surprise_mean=None, battles=0)
        return r
    s = np.array([pmi.surprise(slots_of[f]) for f in files])
    res = pool.score([(f, opp) for f in files], battles=battles, conc=50, quiet=True)
    wr = np.array([res[f]["win_rate"] for f in files if f in res])
    r["battles"] = int(sum(res[f]["battles"] for f in res))
    r["surprise_mean"] = float(s.mean())
    r["surprise_ref_real"] = REAL_SURPRISE_REF
    if len(wr) == 0:
        r.update(win_rate=None, se=None, p90=None, max=None, above_real_median=None)
        return r
    r.update(win_rate=float(wr.mean()),
             se=float(wr.std(ddof=1) / math.sqrt(len(wr))) if len(wr) > 1 else None,
             p90=float(np.percentile(wr, 90)), max=float(wr.max()),
             above_real_median=float((wr >= REAL_MEDIAN).mean()), n_scored=int(len(wr)))
    return r

# --------------------------------------------------------------------------
# the visual
# --------------------------------------------------------------------------
def bar(v, hi, width=18):
    if v is None or hi <= 0: return " " * width
    k = int(round(width * min(v / hi, 1.0)))
    return ("█" * k).ljust(width, "·")

def report(res):
    rows = [res[n] for n, _ in ARMS if n in res]
    if not rows: return
    amax = max((r["acceptance"] for r in rows), default=1) or 1
    wmax = max((r["win_rate"] or 0 for r in rows), default=1) or 1
    print("\n" + "=" * 118, flush=True)
    print("DECODING-SCHEME SWEEP  ·  one checkpoint (wrdiffusion.pt), win-rate bin 5, "
          "spreads copied  ·  24 battles vs the top-50 pool", flush=True)
    print("=" * 118)
    print(f"{'scheme':<22}{'acceptance (Wilson 95%)':<26}{'':<19}{'samp/ok':>8}  "
          f"{'win rate (SE clustered)':<18}{'':<18} {'uniq':>6} {'PMI':>6}")
    print("-" * 118)
    for r in rows:
        a, lo, hi = r["acceptance"], *r["acceptance_ci95"]
        wr, se = r["win_rate"], r["se"]
        ws = f"{wr:.3f} +/- {se:.3f}" if wr is not None and se is not None else \
             (f"{wr:.3f}" if wr is not None else "n/a")
        spa = f"{r['samples_per_accepted']:.1f}" if r["samples_per_accepted"] else "inf"
        flag = "  <-- CAPPED" if r["capped"] else ""
        u = f"{r['unique']:>6.2f}" if r["unique"] is not None else "   n/a"
        pm = f"{r['surprise_mean']:>6.2f}" if r.get("surprise_mean") is not None else "   n/a"
        print(f"{r['scheme']:<22} {a:.3g} [{lo:.3g}, {hi:.3g}]".ljust(48)
              + bar(a, amax) + f" {spa:>8}  " + ws.ljust(18) + bar(wr, wmax)
              + f" {u} {pm}  n={r['n']}" + flag, flush=True)
    print("-" * 118)
    print(f"reference win rates: real corpus team 0.470-0.475 · median real 0.458 · "
          f"slotcopy 0.420-0.436 · generator bin5/g4 0.193-0.195 · random legal ~0.011")
    print(f"reference PMI surprise of a real corpus team: {REAL_SURPRISE_REF}")
    weak = [r for r in rows if r["acceptance"] < 0.5 and r.get("rejections")]
    if weak:
        print("why the low-acceptance arms were rejected by Showdown "
              "(clauses per rejected team, top 3):")
        for r in weak:
            top = list(r["rejections"].items())[:3]
            print(f"  {r['scheme']:<22} " + " · ".join(f"{k} x{v}" for k, v in top))
        print("-" * 118)
    caps = [r["scheme"] for r in rows if r["capped"]]
    if caps: print("CAPPED (truncated, NOT completed): " + ", ".join(caps))
    print("=" * 118, flush=True)

# --------------------------------------------------------------------------
def selftest(A, C, V, Xc, n=40):
    """Arm 1 must be bit-identical to what runs today: under dependency order the
    arc-consistent mask has to equal diffusion.Constraints.mask_for at every step."""
    rng = np.random.default_rng(0); checked = 0
    for _ in range(n):
        row = torch.tensor(Xc[rng.integers(0, len(Xc))])
        k = int(rng.integers(0, COLS))                 # a partially decoded dep-order state
        x = row.clone()
        for c in D.ORDER[k:]: x[c] = 0
        for c in D.ORDER[k:]:
            a = A.mask(c, x); b = C.mask_for(c, x, "cpu")
            assert torch.equal(a, b), f"arc mask differs from mask_for at column {c}"
            checked += 1
            x[c] = row[c]
    print(f"selftest OK: arc-consistent mask == Constraints.mask_for on {checked} "
          f"dependency-order states", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma-separated scheme names")
    ap.add_argument("--per", type=int, default=150)
    ap.add_argument("--battles", type=int, default=24)
    ap.add_argument("--max-attempts", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    outfile = "/tmp/vgc-pilot/schemes_smoke.json" if a.smoke else OUT
    if a.report:
        report(json.load(open(outfile)) if os.path.exists(outfile) else {}); return
    per, battles, cap, batch = a.per, a.battles, a.max_attempts, a.batch
    if a.smoke: per, battles, cap, batch = 3, 2, 24, 8
    names = set(a.only.split(",")) if a.only else {n for n, _ in ARMS}

    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    model, V, L, corpus, labelled = W.load_model()
    C = D.Constraints(V, L); A = Arc(C, V)
    look, spreads = {}, defaultdict(list)
    for t in corpus + labelled:
        for s in t:
            look[norm(s["species"])] = s["species"]; look[norm(s["ability"])] = s["ability"]
            if s["item"]: look[norm(s["item"])] = s["item"]
            for mv in s["moves"]: look[norm(mv)] = mv
            if s["nature"]: look[norm(s["nature"])] = s["nature"]
            spreads[norm(s["species"])].append(dict(s["evs"]))
    ctx = dict(V=V, L=L, C=C, A=A, model=model, look=look, spreads=spreads, corpus=corpus,
               pmi=PMI(corpus), opp=opponents(), val=Validator(),
               Xc=np.stack([V.encode(t) for t in corpus]))
    if a.selftest:
        selftest(A, C, V, ctx["Xc"]); ctx["val"].close(); return
    print(f"{len(corpus)} corpus teams · {len(labelled)} labelled · {len(ctx['opp'])} opponents "
          f"· device {DEV} · per={per} battles={battles} cap={cap}", flush=True)

    res = json.load(open(outfile)) if os.path.exists(outfile) else {}
    for name, over in ARMS:
        if name not in names: continue
        if name in res:
            print(f"[{name}] already done, skipping", flush=True); continue
        cfg = dict(DEFAULTS); cfg.update(over)
        t0 = time.perf_counter()
        r = run_arm(name, cfg, ctx, per, battles, cap, batch)
        r["minutes"] = (time.perf_counter() - t0) / 60
        res[name] = r
        json.dump(res, open(outfile, "w"), indent=1)
        wr = f"{r['win_rate']:.4f}" if r["win_rate"] is not None else "n/a"
        se = f" +/- {r['se']:.4f}" if r.get("se") else ""
        print(f"[{name}] DONE  accept {r['acceptance']:.3g} "
              f"{r['acceptance_ci95']}  ({r['valid']}/{r['attempts']})  win rate {wr}{se}  "
              f"uniq {r['unique']}  PMI {r.get('surprise_mean')}  "
              f"{'CAPPED ' if r['capped'] else ''}{r['minutes']:.1f} min", flush=True)
    ctx["val"].close()
    report(res)
    print("SCHEMES_DONE", flush=True)

if __name__ == "__main__":
    main()
