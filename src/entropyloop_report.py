"""Report for entropyloop.py: the three entropy arms beside the cached baseline.

    python entropyloop_report.py            # reads the local staging copy if present

Sections: per-generation batch means; pooled and late-phase differences vs the
baseline (paired by generation, team-level z); diversity trajectories; what the
refit saw (composition, effective sizes, species-set entropy - the baseline's
recomputed from its own labels the same way); the model's unmasking entropy;
the re-battled top teams.  Works on a partial run.
"""
import json, os, re, sys
from collections import defaultdict
import numpy as np

LOCAL = "/tmp/vgc-pilot/local/results"
REPO = "/Users/ramiismael/Documents/code/vgc-team-generator-pilot/results"
def path(n):
    p = os.path.join(LOCAL, n)
    return p if os.path.exists(p) else os.path.join(REPO, n)

R = json.load(open(path("entropyloop.json")))
B = json.load(open(path("gradloop.json")))["gens"]
AS = json.load(open(path("activesearch.json")))
RB = json.load(open(path("rebattle_top.json"))) if os.path.exists(path("rebattle_top.json")) else None
REAL_MEAN = 0.4625
ARMS = [a for a in ["boltz", "niche", "refit"] if any(k.startswith(a + "_g") for k in R["gens"])]

def gens(d, arm):
    return {int(k.rsplit("_g", 1)[1]): v for k, v in d.items() if k.rsplit("_g", 1)[0] == arm}
base = gens(B, "combined")
arm_g = {a: gens(R["gens"], a) for a in ARMS}
G = sorted(base)

def species_set(txt):
    sp = []
    for blk in re.split(r"\n\s*\n", txt.strip()):
        first = blk.strip().splitlines()[0]
        name = first.split("@")[0].strip()
        name = re.sub(r"\s*\((M|F)\)$", "", name)
        m = re.match(r"^.*\((.+)\)$", name)
        if m: name = m.group(1)
        sp.append(name.lower())
    return tuple(sorted(sp))

def hsets(sets, w=None):
    c = defaultdict(float)
    for i, s in enumerate(sets):
        c[s] += (1.0 if w is None else w[i])
    p = np.array(list(c.values())); p = p / p.sum()
    return float(-(p * np.log(p)).sum())

# ---- 1. per-generation means -------------------------------------------------
print("1. batch mean (128 fresh 24-battle labels per generation)")
print(f"{'gen':>3} {'base':>7} " + " ".join(f"{a:>7}" for a in ARMS) + "   T")
for g in range(1, 12):
    row = f"{g:>3} " + (f"{base[g]['stats']['mean']:7.3f} " if g in base else f"{'':7s} ")
    for a in ARMS:
        row += f"{arm_g[a][g]['stats']['mean']:7.3f} " if g in arm_g[a] else f"{'':7s} "
    T = next((arm_g[a][g]['stats'].get('temperature') for a in ARMS if g in arm_g[a]), None)
    print(row + (f"  {T:.2f}" if T else ""))

# ---- 2. pooled + paired differences ------------------------------------------
print("\n2. difference vs baseline (paired by generation over the generations both ran)")
for a in ARMS:
    common = [g for g in G if g in arm_g[a]]
    if not common: continue
    ya = np.concatenate([arm_g[a][g]["y"] for g in common]); yb = np.concatenate([base[g]["y"] for g in common])
    d = np.array([arm_g[a][g]["stats"]["mean"] - base[g]["stats"]["mean"] for g in common])
    se_team = np.sqrt(ya.var(ddof=1) / len(ya) + yb.var(ddof=1) / len(yb))
    ci = 1.96 * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
    late = [g for g in common if g >= 8]
    dl = np.array([arm_g[a][g]["stats"]["mean"] - base[g]["stats"]["mean"] for g in late]) if late else None
    print(f"  {a:6s} gens {common[0]}-{common[-1]}: arm {ya.mean():.4f} vs base {yb.mean():.4f} "
          f"-> diff {d.mean():+.4f} (paired 95% ±{ci:.4f}; team-level z {d.mean()/se_team:+.1f})"
          + (f" · late g{late[0]}-{late[-1]} diff {dl.mean():+.4f}" if late else "")
          + f" · best gen {max(arm_g[a][g]['stats']['mean'] for g in common):.3f} vs base best {max(base[g]['stats']['mean'] for g in common):.3f}")
for a, s in (R.get("stopped") or {}).items():
    print(f"  {a}: STOPPED g{s['gen']} - {s['reason']}")

# ---- 3. diversity -------------------------------------------------------------
print("\n3. proposal diversity per generation: distinct species sets / 512 · novel-set fraction · NN-Hamming to corpus")
print(f"{'gen':>3} " + " ".join(f"{n:>18}" for n in ["base"] + ARMS))
for g in range(1, 12):
    cells = []
    for src in [base] + [arm_g[a] for a in ARMS]:
        if g in src:
            s = src[g]["stats"]; p = s.get("proposal") or {}
            cells.append(f"{p.get('distinct_species_sets',0):4d} {s.get('novel_species_set',0):5.2f} {p.get('nn_hamming',0):6.2f}")
        else:
            cells.append(" " * 18)
    print(f"{g:>3} " + " ".join(f"{c:>18}" for c in cells))

# ---- 4. what the refit saw -----------------------------------------------------
print("\n4. the refit multiset: k · distinct teams · real / proposals · effective size · real weight share · effective species sets (exp H)")
anchor_files = list(AS["anchor"].keys()); anchor_y = [float(v) for v in AS["anchor"].values()]
anchor_sets = []
for f in anchor_files:
    try: anchor_sets.append(species_set(open(f).read()))
    except OSError: anchor_sets.append(None)
g0 = AS["gens"]["gen0"]
lab_y = anchor_y + list(map(float, g0["y"])); lab_sets = anchor_sets + [species_set(p) for p in g0["pastes"]]
n_anchor = len(anchor_y)
base_sel = {}
for g in G:
    y = np.asarray(lab_y); k = max(100, int(0.25 * len(y))); el = np.argsort(-y)[:k]
    sets = [lab_sets[i] for i in el]
    base_sel[g] = dict(k=k, distinct_teams=k, real=int((el < n_anchor).sum()), proposals=int((el >= n_anchor).sum()),
                       ess=float(k), real_weight_share=float((el < n_anchor).sum() / k),
                       effective_species_sets=float(np.exp(hsets([s for s in sets if s is not None]))))
    lab_y += list(map(float, base[g]["y"])); lab_sets += [species_set(p) for p in base[g]["pastes"]]
def selrow(ss):
    return (f"{ss['k']:4d} {ss['distinct_teams']:4d} {ss['real']:4d}/{ss['proposals']:<4d} "
            f"{ss['ess']:5.0f} {ss['real_weight_share']:5.2f} {ss['effective_species_sets']:5.0f}")
print(f"{'gen':>3} " + " ".join(f"{n:>36}" for n in ["base (recomputed)"] + ARMS))
for g in range(1, 12):
    cells = [selrow(base_sel[g]) if g in base_sel else " " * 36]
    for a in ARMS:
        cells.append(selrow(arm_g[a][g]["stats"]["selection"]) if g in arm_g[a] else " " * 36)
    print(f"{g:>3} " + " ".join(f"{c:>36}" for c in cells))

# ---- 5. model entropy ----------------------------------------------------------
p0e = (R.get("config") or {}).get("p0_entropy") or {}
print(f"\n5. unmasking entropy of the refit model on the 200 anchors at 50% masking (species / other, nats); "
      f"p0 = {p0e.get('species', float('nan')):.3f} / {p0e.get('other', float('nan')):.3f}")
print(f"{'gen':>3} " + " ".join(f"{a:>15}" for a in ARMS) + "   beta(refit)")
for g in range(1, 12):
    cells = []
    for a in ARMS:
        if g in arm_g[a]:
            me = arm_g[a][g]["stats"].get("model_entropy") or {}
            cells.append(f"{me.get('species', float('nan')):6.3f} / {me.get('other', float('nan')):6.3f}")
        else:
            cells.append(" " * 15)
    b = arm_g["refit"][g]["stats"].get("beta") if "refit" in arm_g and g in arm_g["refit"] else None
    print(f"{g:>3} " + " ".join(f"{c:>15}" for c in cells) + (f"   {b:.2f}" if b else ""))

# ---- 6. re-battled tops --------------------------------------------------------
print("\n6. re-battled top teams (192 battles, SE ~0.035) - the deliverable")
for a, rb in (R.get("rebattle") or {}).items():
    wr = [t["win_rate"] for t in rb["teams"]]
    if wr:
        srcs = ",".join(sorted({t["source"].rsplit("_g", 1)[1] for t in rb["teams"]}, key=int))
        print(f"  {a:6s}: best {max(wr):.4f} · top-{len(wr)} mean {np.mean(wr):.4f} · ≥ real mean {sum(v >= REAL_MEAN for v in wr)}/{len(wr)} · from gens {srcs}")
        t = rb["teams"][0]
        print(f"          #1 ({t['source']}, label {t['label24']:.3f}): {', '.join(t['species'])}")
if RB:
    rb = [t for t in RB["teams"] if t["source"].startswith("combined_g")]
    wr = [t["win_rate"] for t in rb]
    print(f"  base  : best {max(wr):.4f} · top-{len(wr)} mean {np.mean(wr):.4f} · ≥ real mean {sum(v >= REAL_MEAN for v in wr)}/{len(wr)}  (rebattle_top.json, combined_* rows)")
