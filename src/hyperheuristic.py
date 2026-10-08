"""LLM as hyper-heuristic: the model writes streamliner code, the score costs no battles.

The hyper-heuristic literature searches a space of heuristics instead of a space of
solutions, and every system in the LLM branch (FunSearch, EoH, ReEvo) hands its
generated code to a mandatory evaluator. That evaluator is what this project cannot
afford: an estimator of `f` needs battles, and 73% of a win-rate label's variance is
binomial noise. So the loop here does not evolve a team builder. It evolves a
*streamliner* -- a predicate `keep(team) -> bool` that cuts the search space -- and
scores it against two fixed pools, which needs no battle at all:

  r  meta retention   fraction of the Reg M-B meta teams the rule keeps
  q  space retention  fraction of the hierarchical-product-sampling teams it keeps

A rule is worth having when r stays near 1 while q collapses: it throws away space
without throwing away the teams known to be good. `streamliner.py mine` already does
this over a hand-written property vocabulary ("at least k slots have Protect"); the
control that matters is whether the LLM's arbitrary Python beats those mined rules.

The lossy step is deliberate and is the standing risk: retention is measured against
the *collected meta*, so a rule that keeps every meta team can still cut away the
off-meta counter-team the search exists to find. r is a proxy for "did not destroy
known-good structure", never for "kept the optimum".

Three ways to drive the generation step, in order of how little they assume:

    python src/hyperheuristic.py prompt              # emit the next-generation prompt
    python src/hyperheuristic.py add --file rules.py # score what any LLM wrote back
    python src/hyperheuristic.py run --proposer claude   # automated, needs `claude` auth
    python src/hyperheuristic.py run --proposer api      # automated, needs ANTHROPIC_API_KEY
    python src/hyperheuristic.py run --proposer stub     # seed bank only, no LLM

    python src/hyperheuristic.py report
    python src/hyperheuristic.py selftest            # type chart, sandbox, species coverage

prompt/add is the loop with the model out of process: it needs no credentials, and the
generation it produces is identical to what `run` would have scored.
"""
import argparse, ast, collections, hashlib, json, re, signal, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO/"src"))
from streamliner import (parse, load_meta, load_hps, LEGAL,
                         PROT_SELF, PROTECT, SPEEDCTRL, REDIRECT, FAKEOUT, SUPPORT)

OUT = REPO/"results/hyperheuristic.json"

# ---------------------------------------------------------------- type chart
# Gen 6+ attacking -> defending, non-1.0 entries only. `selftest` checks it.
TYPECHART = {
 "Normal":  {"Rock":.5,"Ghost":0,"Steel":.5},
 "Fire":    {"Fire":.5,"Water":.5,"Grass":2,"Ice":2,"Bug":2,"Rock":.5,"Dragon":.5,"Steel":2},
 "Water":   {"Fire":2,"Water":.5,"Grass":.5,"Ground":2,"Rock":2,"Dragon":.5},
 "Electric":{"Water":2,"Electric":.5,"Grass":.5,"Ground":0,"Flying":2,"Dragon":.5},
 "Grass":   {"Fire":.5,"Water":2,"Grass":.5,"Poison":.5,"Ground":2,"Flying":.5,"Bug":.5,
             "Rock":2,"Dragon":.5,"Steel":.5},
 "Ice":     {"Fire":.5,"Water":.5,"Grass":2,"Ice":.5,"Ground":2,"Flying":2,"Dragon":2,"Steel":.5},
 "Fighting":{"Normal":2,"Ice":2,"Poison":.5,"Flying":.5,"Psychic":.5,"Bug":.5,"Rock":2,
             "Ghost":0,"Dark":2,"Steel":2,"Fairy":.5},
 "Poison":  {"Grass":2,"Poison":.5,"Ground":.5,"Rock":.5,"Ghost":.5,"Steel":0,"Fairy":2},
 "Ground":  {"Fire":2,"Electric":2,"Grass":.5,"Poison":2,"Flying":0,"Bug":.5,"Rock":2,"Steel":2},
 "Flying":  {"Electric":.5,"Grass":2,"Fighting":2,"Bug":2,"Rock":.5,"Steel":.5},
 "Psychic": {"Fighting":2,"Poison":2,"Psychic":.5,"Dark":0,"Steel":.5},
 "Bug":     {"Fire":.5,"Grass":2,"Fighting":.5,"Poison":.5,"Flying":.5,"Psychic":2,"Ghost":.5,
             "Dark":2,"Steel":.5,"Fairy":.5},
 "Rock":    {"Fire":2,"Ice":2,"Fighting":.5,"Ground":.5,"Flying":2,"Bug":2,"Steel":.5},
 "Ghost":   {"Normal":0,"Psychic":2,"Ghost":2,"Dark":.5},
 "Dragon":  {"Dragon":2,"Steel":.5,"Fairy":0},
 "Dark":    {"Fighting":.5,"Psychic":2,"Ghost":2,"Dark":.5,"Fairy":.5},
 "Steel":   {"Fire":.5,"Water":.5,"Electric":.5,"Ice":2,"Rock":2,"Steel":.5,"Fairy":2},
 "Fairy":   {"Fire":.5,"Fighting":2,"Poison":.5,"Dragon":2,"Dark":2,"Steel":.5},
}
TYPES = tuple(TYPECHART)

_DEX = json.load(open(REPO/"data/pokedex.json"))
_BY_ID = {re.sub(r"[^a-z0-9]", "", k.lower()): v for k, v in _DEX.items()}
_BY_ID.update({re.sub(r"[^a-z0-9]", "", v["name"].lower()): v
               for v in _DEX.values() if "name" in v})

def _entry(species):
    return _BY_ID.get(re.sub(r"[^a-z0-9]", "", species.lower()))

# ------------------------------------------------------- the rule-writing API
def types(slot):
    """('Fire','Flying') for the slot's species; () if the species is unknown."""
    e = _entry(slot["species"])
    return tuple(e["types"]) if e else ()

def stats(slot):
    """Base stats dict {'hp','atk','def','spa','spd','spe'}; {} if unknown."""
    e = _entry(slot["species"])
    return dict(e["baseStats"]) if e else {}

def weaknesses(slot):
    """Attacking types that hit this slot for more than 1x."""
    ts = types(slot)
    if not ts: return set()
    return {a for a in TYPES
            if _mult(a, ts) > 1}

def resistances(slot):
    ts = types(slot)
    if not ts: return set()
    return {a for a in TYPES if _mult(a, ts) < 1}

def _mult(atk, defend):
    m = 1.0
    for d in defend: m *= TYPECHART[atk].get(d, 1.0)
    return m

_MOVES = json.load(open(REPO/"data/moves.json"))
_MV = {re.sub(r"[^a-z0-9]", "", k.lower()): v for k, v in _MOVES.items()}
_MV.update({re.sub(r"[^a-z0-9]", "", v["name"].lower()): v
            for v in _MOVES.values() if "name" in v})

def move(name):
    """{'name','type','category','basePower','target','priority'} or {} if unknown."""
    return dict(_MV.get(re.sub(r"[^a-z0-9]", "", name.lower()), {}))

def attacks(slot):
    """The slot's damaging moves, as move dicts."""
    return [m for m in (move(n) for n in slot["moves"])
            if m.get("category") in ("Physical", "Special")]

def spread_moves(slot):
    """Damaging moves that hit more than one target -- the doubles tax."""
    return [m for m in attacks(slot)
            if m.get("target") in ("allAdjacentFoes", "allAdjacent")]

def coverage(team):
    """Defending types the team can hit for more than 1x with some damaging move."""
    out = set()
    for s in team:
        for m in attacks(s):
            t = m.get("type")
            if t in TYPECHART:
                out |= {d for d in TYPES if TYPECHART[t].get(d, 1.0) > 1}
    return out

def has(slot, *names):
    """True if the slot knows any of the named moves."""
    mv = set(slot["moves"])
    return any(n in mv for n in names)

def count(team, fn):
    """How many of the six slots satisfy fn."""
    return sum(1 for s in team if fn(s))

API = {"types": types, "stats": stats, "weaknesses": weaknesses,
       "resistances": resistances, "has": has, "count": count,
       "move": move, "attacks": attacks, "spread_moves": spread_moves,
       "coverage": coverage,
       "PROTECT": PROTECT, "SELF_PROTECT": PROT_SELF, "SPEED_CONTROL": SPEEDCTRL,
       "REDIRECTION": REDIRECT, "FAKE_OUT": FAKEOUT, "SUPPORT": SUPPORT,
       "TYPES": TYPES}

# ------------------------------------------------------------------- sandbox
SAFE_BUILTINS = {b.__name__: b for b in (len, any, all, sum, min, max, abs, sorted,
                                         set, frozenset, list, dict, tuple, str, int,
                                         float, bool, round, range, enumerate, zip,
                                         map, filter, isinstance)}
SAFE_BUILTINS["True"], SAFE_BUILTINS["False"], SAFE_BUILTINS["None"] = True, False, None
BANNED_NAMES = {"open","exec","eval","compile","__import__","globals","locals","vars",
                "getattr","setattr","delattr","input","exit","quit","breakpoint","help"}

class RuleError(Exception): pass
class RuleTimeout(Exception): pass

def _check_ast(tree):
    for n in ast.walk(tree):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            raise RuleError("imports are not allowed")
        if isinstance(n, (ast.Global, ast.Nonlocal)):
            raise RuleError("global/nonlocal are not allowed")
        if isinstance(n, ast.Attribute) and n.attr.startswith("_"):
            raise RuleError(f"private attribute {n.attr}")
        if isinstance(n, ast.Name) and n.id in BANNED_NAMES:
            raise RuleError(f"banned name {n.id}")
        if isinstance(n, ast.Name) and n.id.startswith("__"):
            raise RuleError(f"dunder name {n.id}")

def compile_rule(src):
    """LLM text -> callable keep(team). Raises RuleError on anything unsafe or wrong."""
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        raise RuleError(f"syntax error: {e}")
    _check_ast(tree)
    ns = dict(API); ns["__builtins__"] = SAFE_BUILTINS
    try:
        exec(compile(tree, "<rule>", "exec"), ns)
    except Exception as e:
        raise RuleError(f"failed to define: {type(e).__name__}: {e}")
    fn = ns.get("keep")
    if not callable(fn):
        raise RuleError("no callable named keep")
    return fn

def _alarm(_s, _f): raise RuleTimeout()

def evaluate(fn, meta, hps, timeout=20):
    """Battle-free score. Returns r, q, the killed meta teams, and a behaviour hash."""
    signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        mbits = [bool(fn(t)) for t in meta]
        qbits = [bool(fn(t)) for t in hps]
    except RuleTimeout:
        raise RuleError(f"timed out after {timeout}s")
    except Exception as e:
        raise RuleError(f"raised on a legal team: {type(e).__name__}: {e}")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    r, q = sum(mbits)/len(mbits), sum(qbits)/len(hps)
    sig = hashlib.sha1(bytes(mbits) + bytes(qbits[:4096])).hexdigest()[:12]
    killed = [i for i, b in enumerate(mbits) if not b]
    return {"r": r, "q": q, "killed": killed, "sig": sig}

# ------------------------------------------------------------------ scoring
def summarise(s, floor):
    q, r = s["q"], s["r"]
    lift = r/q if q else float("inf")
    return {"r": r, "q": q, "lift": lift,
            "space_left": LEGAL*q if q else LEGAL/1e5,
            "feasible": r >= floor and q < 1.0}

def pareto(rules):
    """Non-dominated on (r high, q low). The whole point of a portfolio is that it
    spans aggressiveness, so the retention floor marks admissibility for the caller
    and does NOT filter the front -- a floor that hides the trade-off is a floor that
    has already made the choice for you."""
    live = [x for x in rules if not x.get("error") and x["r"] > 0 and x["q"] < 1.0
            and not x.get("duplicate_of")]
    out = []
    for a in live:
        if not any(b is not a and b["r"] >= a["r"] and b["q"] <= a["q"]
                   and (b["r"] > a["r"] or b["q"] < a["q"]) for b in live):
            out.append(a)
    return sorted(out, key=lambda x: -x["q"])

# ------------------------------------------------------------------ proposer
SEEDS = [
 ("three-protect",  "def keep(team):\n    return count(team, lambda s: has(s, 'Protect')) >= 3\n"),
 ("fake-out",       "def keep(team):\n    return count(team, lambda s: bool(set(s['moves']) & FAKE_OUT)) >= 1\n"),
 ("speed-control",  "def keep(team):\n    return count(team, lambda s: bool(set(s['moves']) & SPEED_CONTROL)) >= 1\n"),
 ("protect+fo+sc",  "def keep(team):\n"
                    "    p = count(team, lambda s: has(s, 'Protect')) >= 3\n"
                    "    f = count(team, lambda s: bool(set(s['moves']) & FAKE_OUT)) >= 1\n"
                    "    c = count(team, lambda s: bool(set(s['moves']) & SPEED_CONTROL)) >= 1\n"
                    "    return p and f and c\n"),
 ("no-quad-weak",   "def keep(team):\n"
                    "    seen = {}\n"
                    "    for s in team:\n"
                    "        for w in weaknesses(s):\n"
                    "            seen[w] = seen.get(w, 0) + 1\n"
                    "    return all(v <= 3 for v in seen.values())\n"),
 ("two-types-max",  "def keep(team):\n"
                    "    seen = {}\n"
                    "    for s in team:\n"
                    "        for t in types(s):\n"
                    "            seen[t] = seen.get(t, 0) + 1\n"
                    "    return all(v <= 2 for v in seen.values())\n"),
]

PROMPT = """You are the generation step of a hyper-heuristic. You write STREAMLINERS for a \
Pokemon VGC team search: predicates that cut the search space without cutting away good teams.

Write Python functions `keep(team) -> bool`. `team` is a list of 6 slot dicts with keys
species (str), item (str), ability (str), moves (list of 4 str), nature (str).

Available, no imports allowed:
  types(slot) -> tuple of type names        stats(slot) -> base stat dict
  weaknesses(slot) -> set of types that hit it super-effectively
  resistances(slot) -> set of types it resists
  has(slot, *move_names) -> bool            count(team, fn) -> int
  move(name) -> {type, category, basePower, target, priority}
  attacks(slot) -> damaging move dicts      spread_moves(slot) -> the ones hitting 2+
  coverage(team) -> set of types the team hits super-effectively
  constants: PROTECT, SELF_PROTECT, SPEED_CONTROL, REDIRECTION, FAKE_OUT, SUPPORT, TYPES
  builtins: len any all sum min max abs sorted set list dict tuple str int float bool round \
range enumerate zip map filter isinstance

Each rule is scored on two fixed pools, no battles:
  r = fraction of {nmeta} real Reg M-B meta teams the rule KEEPS   (must stay >= {floor:.2f})
  q = fraction of {nhps} random-legal teams the rule KEEPS          (drive this DOWN)
A good rule keeps almost every real team and almost no random one. A rule that keeps
everything is worthless; a rule that keeps no real team is rejected.

This is Champions VGC 2026 Reg Set M-B: doubles, no Tera type, Stat Points not EVs.

{state}

Write {n} NEW rules, each different in kind from the ones above and from each other.
Prefer structural claims about what a real doubles team must contain or must avoid.
Reply with {n} fenced python blocks, each starting with a comment `# name: short-name`,
nothing else between them."""

def render_state(rules, floor):
    if not rules:
        return "No rules have been scored yet. Start from what a competitive doubles team needs."
    top = sorted([x for x in rules if x["feasible"]], key=lambda x: x["q"])[:6]
    dead = [x for x in rules if not x["feasible"]][:4]
    L = ["Scored so far (r = meta kept, q = space kept; lower q is better):"]
    for x in top:
        L.append(f"  {x['name']:22} r={x['r']:.3f}  q={x['q']:.5f}  lift={x['lift']:.0f}x")
        if x.get("killed_note"): L.append(f"      killed: {x['killed_note']}")
    if dead:
        L.append("Rejected (r below the floor, or kept everything, or crashed):")
        for x in dead:
            L.append(f"  {x['name']:22} " +
                     (x.get("error") or f"r={x['r']:.3f} q={x['q']:.5f}"))
    return "\n".join(L)

def killed_note(idx, meta, cap=3):
    """Which real teams a rule threw away -- the battle-free reflection signal."""
    if not idx: return ""
    sp = collections.Counter()
    for i in idx[:200]:
        for s in meta[i]: sp[s["species"]] += 1
    common = ", ".join(f"{k}" for k, _ in sp.most_common(cap))
    return f"{len(idx)} meta teams, most often featuring {common}"

def ask_api(prompt, model):
    import anthropic                       # optional dependency, only for this backend
    c = anthropic.Anthropic()
    m = c.messages.create(model=model or "claude-opus-4-5", max_tokens=4000,
                          messages=[{"role": "user", "content": prompt}])
    return "".join(b.text for b in m.content if b.type == "text")

def ask_claude(prompt, model, binary, timeout=300):
    cmd = [binary, "-p", prompt] + (["--model", model] if model else [])
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(f"{binary} exited {p.returncode}: {p.stderr[:400]}")
    return p.stdout

def extract(text):
    """Fenced python blocks -> [(name, source)]."""
    out = []
    for block in re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.S):
        m = re.search(r"#\s*name:\s*(\S+)", block)
        name = m.group(1) if m else f"rule{len(out)}"
        if "def keep" in block: out.append((name, block))
    return out

# ---------------------------------------------------------------------- loop
def pools(a):
    meta = load_meta()
    hps  = load_hps(limit=a.pool)
    return meta, hps

def load_state(a, meta, hps, quiet=False):
    """Existing results, or a fresh state seeded with the mined-constraint control."""
    if OUT.exists():
        d = json.load(open(OUT))
        if d["pool"] != len(hps) or d["meta"] != len(meta):
            print(f"note: {OUT.name} was scored on meta={d['meta']} pool={d['pool']}, "
                  f"now meta={len(meta)} pool={len(hps)} -- rescoring from source")
            d = None
        if d: return d
    st = {"floor": a.floor, "pool": len(hps), "meta": len(meta), "rules": []}
    if not quiet:
        print("generation 0 -- seed bank (the control: streamliner.py's mined constraints)")
    for name, src in SEEDS:
        row = admit(st, name, src, 0, meta, hps, a)
        if not quiet: show(row)
    return st

def admit(st, name, src, gen, meta, hps, a):
    seen = {r["sig"]: r["name"] for r in st["rules"] if r.get("sig")}
    row = {"name": name, "gen": gen, "src": src}
    try:
        s = evaluate(compile_rule(src), meta, hps, a.timeout)
    except RuleError as e:
        row.update({"error": str(e), "feasible": False, "r": 0.0, "q": 1.0})
        st["rules"].append(row); return row
    row.update(summarise(s, a.floor))
    row["sig"] = s["sig"]
    row["killed_note"] = killed_note(s["killed"], meta)
    if s["sig"] in seen: row["duplicate_of"] = seen[s["sig"]]
    st["rules"].append(row); return row

def show(row):
    if row.get("error"):
        print(f"  {row['name']:24} rejected: {row['error'][:60]}")
    else:
        dup = f"  == {row['duplicate_of']}" if row.get("duplicate_of") else ""
        flag = " " if row["feasible"] else "x"
        print(f"{flag} {row['name']:24} r={row['r']:.3f}  q={row['q']:.5f}  "
              f"lift={row['lift']:>8.0f}x  space={row['space_left']:.1e}{dup}")

def save(st):
    OUT.parent.mkdir(exist_ok=True)
    st["portfolio"] = [r["name"] for r in pareto(st["rules"])]
    json.dump(st, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT.relative_to(REPO)}")

def build_prompt(st, meta, hps, a):
    return PROMPT.format(nmeta=len(meta), nhps=len(hps), floor=a.floor, n=a.pop,
                         state=render_state(st["rules"], a.floor))

def cmd_prompt(a):
    meta, hps = pools(a)
    st = load_state(a, meta, hps, quiet=True)
    save(st)
    print(build_prompt(st, meta, hps, a))

def cmd_add(a):
    meta, hps = pools(a)
    st = load_state(a, meta, hps)
    text = Path(a.file).read_text()
    got = extract(text) or extract("```python\n" + text + "\n```")
    if not got:
        print(f"no rule blocks in {a.file}: need fenced python blocks defining keep(team)")
        return 1
    gen = max([r["gen"] for r in st["rules"]] or [0]) + 1
    print(f"\ngeneration {gen} -- {len(got)} rules from {a.file}")
    for name, src in got: show(admit(st, f"g{gen}-{name}", src, gen, meta, hps, a))
    save(st)
    report_front(pareto(st["rules"]), len(hps))

def run(a):
    meta, hps = pools(a)
    print(f"meta teams {len(meta)}   random-legal teams {len(hps)}   "
          f"retention floor {a.floor}\n")
    st = load_state(a, meta, hps)

    for g in range(1, a.generations+1):
        print(f"\ngeneration {g} -- {a.proposer} proposes {a.pop}")
        if a.proposer == "stub":
            print("  (stub proposer: no LLM, nothing new to add)"); break
        prompt = build_prompt(st, meta, hps, a)
        try:
            text = (ask_api(prompt, a.model) if a.proposer == "api"
                    else ask_claude(prompt, a.model, a.claude_bin, a.ask_timeout))
        except Exception as e:
            print(f"  proposer failed: {e}")
            print(f"  fall back to: hyperheuristic.py prompt > /tmp/p.txt, then "
                  f"hyperheuristic.py add --file <rules>")
            break
        got = extract(text)
        if not got:
            print("  proposer returned no usable rule block"); break
        for name, src in got: show(admit(st, f"g{g}-{name}", src, g, meta, hps, a))

    save(st)
    report_front(pareto(st["rules"]), len(hps))

def report_front(front, npool):
    print(f"\nportfolio -- non-dominated on (keep the meta, cut the space)")
    print(f"{'':2}{'rule':26} {'meta kept':>10} {'space kept':>12} {'lift':>10} {'space left':>12}")
    print("-"*78)
    for r in front:
        q = f"{r['q']:.5f}" if r["q"] else f"<{1/npool:.5f}"
        print(f"{'  ' if r['feasible'] else 'x '}{r['name']:26} {r['r']:>9.1%} {q:>12} "
              f"{r['lift']:>9.0f}x {r['space_left']:>12.1e}")
    print("x = below the retention floor: it prunes harder by throwing real teams away.")

def report(a):
    d = json.load(open(OUT))
    print(f"meta teams {d['meta']}   random-legal teams {d['pool']}   "
          f"retention floor {d['floor']}")
    for r in sorted(d["rules"], key=lambda x: (x["gen"], x["q"])): show(r)
    report_front(pareto(d["rules"]), d["pool"])

# ------------------------------------------------------------------ selftest
def selftest(a):
    ok = True
    def chk(got, want, what):
        nonlocal ok
        good = abs(got-want) < 1e-9
        ok &= good
        print(f"  {'ok ' if good else 'FAIL'} {what}: {got} (want {want})")
    print("type chart")
    for atk, dfn, want in [("Fire",("Grass",),2), ("Water",("Fire",),2),
                           ("Normal",("Ghost",),0), ("Ground",("Flying",),0),
                           ("Electric",("Ground",),0), ("Dragon",("Fairy",),0),
                           ("Ice",("Dragon","Flying"),4), ("Fighting",("Ghost",),0),
                           ("Grass",("Water","Ground"),4), ("Fire",("Water","Dragon"),0.25)]:
        chk(_mult(atk, dfn), want, f"{atk} -> {'/'.join(dfn)}")
    print("helpers")
    slot = {"species":"Incineroar","item":"","ability":"Intimidate",
            "moves":["Fake Out","Knock Off","Parting Shot","Will-O-Wisp"],"nature":""}
    print(f"  types(Incineroar) = {types(slot)}  (want Fire/Dark)")
    print(f"  weaknesses         = {sorted(weaknesses(slot))}")
    print(f"  has(Fake Out)      = {has(slot,'Fake Out')}")
    print(f"  attacks            = {[m['name'] for m in attacks(slot)]}")
    chk(float(move('Flamethrower')['basePower']), 90.0, "Flamethrower base power")
    chk(float(len(spread_moves({'moves':['Heat Wave','Protect','Snarl','Flamethrower']}))),
        2.0, "spread moves among Heat Wave/Protect/Snarl/Flamethrower")
    print("sandbox")
    for bad in ["import os\ndef keep(t): return True",
                "def keep(t): return open('/etc/passwd')",
                "def keep(t): return t.__class__"]:
        try:
            compile_rule(bad); print(f"  FAIL allowed: {bad[:34]!r}"); ok = False
        except RuleError as e: print(f"  ok  blocked: {str(e)[:50]}")
    print("species coverage")
    meta = load_meta()
    miss = {s["species"] for t in meta for s in t if not _entry(s["species"])}
    print(f"  {'ok ' if not miss else 'FAIL'} unresolved species in the meta pool: "
          f"{len(miss)}{' ' + str(sorted(miss)[:8]) if miss else ''}")
    ok &= not miss
    print("\nSELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run","prompt","add","report","selftest"])
    ap.add_argument("--file", help="add: file of fenced python rule blocks")
    ap.add_argument("--generations", type=int, default=3)
    ap.add_argument("--pop", type=int, default=6)
    ap.add_argument("--pool", type=int, default=20000, help="random-legal teams scored against")
    ap.add_argument("--floor", type=float, default=0.85, help="minimum meta retention")
    ap.add_argument("--timeout", type=int, default=20, help="seconds one rule may take")
    ap.add_argument("--proposer", choices=["claude","api","stub"], default="claude")
    ap.add_argument("--model", default="")
    ap.add_argument("--claude-bin", default="claude")
    ap.add_argument("--ask-timeout", type=int, default=300)
    a = ap.parse_args()
    sys.exit({"run": run, "prompt": cmd_prompt, "add": cmd_add,
              "report": report, "selftest": selftest}[a.cmd](a) or 0)
