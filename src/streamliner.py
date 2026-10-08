"""Constraint acquisition from examples: mine 'at least k of the six slots have
property P' constraints off the Reg M-B meta pool, then score each one battle-free
(meta teams retained x random-legal space pruned) and against the 2,000 battle
labels, which is the only test that asks whether the constraint keeps good teams.

  python src/streamliner.py mine     # single constraints, ranked by pruning
  python src/streamliner.py stack    # conjunctions, with samples-per-hit
  python src/streamliner.py labels   # win rate by Protect count, 48,000 battles
"""
import json, re, sys, math, collections
from pathlib import Path

REPO  = Path(__file__).resolve().parent.parent
META  = REPO/"teams/reg_mb"
HPS   = REPO/"teams/hps_reg_mb_100k.jsonl"
LEGAL = 2.6e118          # size of the legal team space, from the Clause note

def parse(text):
    """Showdown export -> list of slots {species,item,ability,moves,nature}."""
    slots = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = [l.rstrip() for l in block.strip().split("\n") if l.strip()]
        if not lines: continue
        head = lines[0]
        name, item = (head.split(" @ ", 1) + [""])[:2] if " @ " in head else (head, "")
        name = re.sub(r"\s*\((M|F)\)\s*$", "", name.strip())
        m = re.match(r"^(.*?)\s*\((.+)\)\s*$", name)
        s = {"species": (m.group(2) if m else name).strip(), "item": item.strip(),
             "ability": "", "moves": [], "nature": ""}
        for l in lines[1:]:
            ls = l.strip()
            if ls.startswith("Ability:"): s["ability"] = ls.split(":",1)[1].strip()
            elif ls.endswith("Nature"):   s["nature"]  = ls[:-6].strip()
            elif ls.startswith("- "):     s["moves"].append(ls[2:].strip())
        slots.append(s)
    return slots

def load_meta():
    return [t for t in (parse(p.read_text(encoding="utf-8", errors="replace"))
                        for p in sorted(META.glob("*.txt"))) if len(t) == 6]

def load_hps(limit=None):
    out = []
    with open(HPS) as f:
        for i, line in enumerate(f):
            if limit and i >= limit: break
            t = parse(json.loads(line)["team"])
            if len(t) == 6: out.append(t)
    return out

PROT_SELF = {"Protect","Detect","Spiky Shield","Baneful Bunker","King's Shield",
             "Obstruct","Silk Trap","Burning Bulwark"}
PROTECT   = PROT_SELF | {"Wide Guard","Quick Guard"}
SPEEDCTRL = {"Tailwind","Trick Room","Icy Wind","Electroweb","Thunder Wave","Bleakwind Storm",
             "Glaciate","Rock Tomb","Nuzzle","Sticky Web","After You","Quash","Scary Face",
             "String Shot","Cotton Spore","Bulldoze","Low Sweep","Mud Shot","Ally Switch"}
REDIRECT  = {"Follow Me","Rage Powder","Ally Switch","Spotlight"}
FAKEOUT   = {"Fake Out","Upper Hand"}
SUPPORT   = {"Helping Hand","Follow Me","Rage Powder","Light Screen","Reflect","Aurora Veil",
             "Tailwind","Trick Room","Haze","Taunt","Encore","Will-O-Wisp","Fake Out"}

def props(slot):
    mv, p = set(slot["moves"]), set()
    if mv & PROT_SELF:                 p.add("prop:self-protect")
    if mv & PROTECT:                   p.add("prop:protect-family")
    if mv & SPEEDCTRL:                 p.add("prop:speed-control")
    if mv & REDIRECT:                  p.add("prop:redirection")
    if mv & FAKEOUT:                   p.add("prop:fake-out")
    if mv & SUPPORT:                   p.add("prop:any-support")
    if slot["ability"] == "Intimidate": p.add("abil:Intimidate")
    for m in slot["moves"]:            p.add("move:"+m)
    return p

def counts(team):
    c = collections.Counter()
    for s in team: c.update(props(s))
    return c

def pools():
    meta, hps = load_meta(), load_hps()
    print(f"meta teams: {len(meta)}   random-legal teams: {len(hps)}\n")
    return [counts(t) for t in meta], [counts(t) for t in hps]

def mine(floor=0.55):
    mc, hc = pools()
    cand = collections.Counter()
    for c in mc: cand.update(c.keys())
    rows = []
    for prop in cand:
        for k in range(1, 7):
            r = sum(1 for c in mc if c[prop] >= k)/len(mc)
            if r < floor: continue
            q = sum(1 for c in hc if c[prop] >= k)/len(hc)
            rows.append((prop, k, r, q))
    rows.sort(key=lambda x: x[3])
    print(f"{'constraint':42} {'meta kept':>10} {'random kept':>12} {'lift':>9}")
    print("-"*78)
    for prop, k, r, q in rows[:30]:
        print(f"{'>= '+str(k)+'  '+prop:42} {r:>9.1%} {q:>11.4%} "
              f"{(r/q if q else float('inf')):>8.1f}x")

def stack():
    mc, hc = pools()
    P, F, S = "move:Protect", "prop:fake-out", "prop:speed-control"
    conj = [("A  >=3 Protect",                   lambda c: c[P]>=3),
            ("B  A + >=1 Fake Out",              lambda c: c[P]>=3 and c[F]>=1),
            ("C  B + >=1 speed control",         lambda c: c[P]>=3 and c[F]>=1 and c[S]>=1),
            ("D  >=2 Protect + Fake Out + SC",   lambda c: c[P]>=2 and c[F]>=1 and c[S]>=1),
            ("E  >=4 Protect + Fake Out + SC",   lambda c: c[P]>=4 and c[F]>=1 and c[S]>=1)]
    print(f"{'stacked constraint':34} {'meta kept':>10} {'random kept':>12} "
          f"{'lift':>9} {'space left':>12} {'samples/hit':>12}")
    print("-"*96)
    for name, f in conj:
        r = sum(1 for c in mc if f(c))/len(mc)
        q = sum(1 for c in hc if f(c))/len(hc)
        print(f"{name:34} {r:>9.1%} {q:>11.4%} {(r/q if q else float('inf')):>8.1f}x "
              f"{LEGAL*q:>12.1e} {(1/q if q else float('inf')):>11.0f}")

def labels():
    lab = json.load(open(REPO/"results/hps_labels.json"))
    man = json.load(open(REPO/"results/hps_label_manifest.json"))
    want = {man[k]["jsonl_line"]: k for k in man if k in lab}
    rows = []
    with open(HPS) as f:
        for i, line in enumerate(f):
            if i not in want: continue
            t = parse(json.loads(line)["team"])
            if len(t) != 6: continue
            c, v = counts(t), lab[want[i]]
            rows.append((c["move:Protect"], c["prop:speed-control"], v["wins"], v["battles"]))
    print(f"labelled teams: {len(rows)}   battles: {sum(r[3] for r in rows)}")
    for idx, name in ((0, "Protect"), (1, "speed control")):
        g = collections.defaultdict(lambda: [0, 0, 0])
        for r in rows:
            b = g[r[idx]]; b[0] += r[2]; b[1] += r[3]; b[2] += 1
        print(f"\n{name}: win rate by count")
        print(f"{'n':>3} {'teams':>7} {'battles':>9} {'win rate':>10} {'+-1.96se':>10}")
        for n in sorted(g):
            w, b, t = g[n]; p = w/b
            print(f"{n:>3} {t:>7} {b:>9} {p:>9.3%} {1.96*math.sqrt(p*(1-p)/b):>9.3%}")
    for k in (1, 2, 3):
        def agg(sel):
            w = sum(r[2] for r in rows if sel(r[0])); b = sum(r[3] for r in rows if sel(r[0]))
            n = sum(1 for r in rows if sel(r[0])); p = w/b if b else 0
            return n, p, 1.96*math.sqrt(p*(1-p)/b) if b else 0
        (n1,p1,e1), (n0,p0,e0) = agg(lambda x: x >= k), agg(lambda x: x < k)
        print(f"\n>= {k} Protect: pass n={n1:<5} {p1:.3%} +-{e1:.3%}   "
              f"fail n={n0:<5} {p0:.3%} +-{e0:.3%}   ratio {p1/p0 if p0 else float('inf'):.2f}x")

if __name__ == "__main__":
    {"mine": mine, "stack": stack, "labels": labels}[
        sys.argv[1] if len(sys.argv) > 1 else "mine"]()
