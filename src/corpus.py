"""Parse the VGC-Bench Reg M-B team corpus into structured teams, and check legality
against Pokemon Showdown data. A team = 6 slots; a slot = species, ability, item,
4 moves, nature, EV spread ("Stat Points")."""
import json, re, os
from pathlib import Path

DATA = Path("/tmp/vgc-pilot/data")
TEAMS = Path("/tmp/vgc-pilot/vgc-bench/teams/reg_mb")

def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())

_dex = json.load(open(DATA/"pokedex.json"))
_learn = json.load(open(DATA/"learnsets.json"))
_items = json.load(open(DATA/"items.json"))
_abils = json.load(open(DATA/"abilities.json"))
_moves = json.load(open(DATA/"moves.json"))

# pkmn/ps item+ability files are keyed by generation
def _flatten_gen(d):
    if all(k.isdigit() for k in d):
        out = {}
        for g in sorted(d, key=int):
            out.update(d[g])
        return out
    return d
_items = _flatten_gen(_items)
_abils = _flatten_gen(_abils)

ITEM_NAMES = {norm(v.get("name", k)) for k, v in _items.items()} | {norm(k) for k in _items}
ABIL_NAMES = {norm(v.get("name", k)) for k, v in _abils.items()} | {norm(k) for k in _abils}
MOVE_NAMES = {norm(v.get("name", k)) for k, v in _moves.items()} | {norm(k) for k in _moves}
DEX_NAMES  = {norm(v.get("name", k)) for k, v in _dex.items()} | {norm(k) for k in _dex}

STATS = ["HP", "Atk", "Def", "SpA", "SpD", "Spe"]
NATURES = ["Adamant","Bashful","Bold","Brave","Calm","Careful","Docile","Gentle","Hardy",
           "Hasty","Impish","Jolly","Lax","Lonely","Mild","Modest","Naive","Naughty","Quiet",
           "Quirky","Rash","Relaxed","Sassy","Serious","Timid"]

def dex_entry(species):
    k = norm(species)
    if k in _dex: return _dex[k]
    for kk, v in _dex.items():
        if norm(v.get("name", "")) == k: return v
    return None

def _keys_for(species):
    """All learnset keys that contribute to a forme: itself, its base species,
    what it changes from / is a battle-only forme of, and its whole prevo chain.
    Showdown stores a forme's shared moves on the base species entry."""
    keys, stack, seen = [], [species], set()
    while stack:
        cur = stack.pop()
        n = norm(cur)
        if not n or n in seen: continue
        seen.add(n)
        if n in _learn: keys.append(n)
        e = dex_entry(cur)
        if not e: continue
        for fld in ("baseSpecies", "changesFrom", "prevo"):
            v = e.get(fld)
            if isinstance(v, str): stack.append(v)
        bo = e.get("battleOnly")
        if isinstance(bo, str): stack.append(bo)
        elif isinstance(bo, list): stack.extend(bo)
    return keys

def legal_moves(species):
    keys = _keys_for(species)
    if not keys: return None
    ms = set()
    for k in keys:
        ms |= set(_learn[k].get("learnset", {}).keys())
    return ms

def mega_formes(species):
    """Formes reachable by holding a mega stone, e.g. Metagross -> Metagross-Mega."""
    base = norm(species)
    e = dex_entry(species)
    if e: base = norm(e.get("baseSpecies", e.get("name", species)))
    out = []
    for k, v in _dex.items():
        cf = v.get("changesFrom") or v.get("baseSpecies")
        if isinstance(cf, str) and norm(cf) == base and "mega" in norm(v.get("forme", "")):
            out.append(v)
    return out

def legal_abilities(species):
    """A set may list the base ability on a mega forme, or the mega ability on the
    base forme; Showdown accepts both, so union them."""
    e = dex_entry(species)
    if not e: return None
    out = {norm(a) for a in e.get("abilities", {}).values()}
    # A mega forme's ability belongs to the MEGA, not to the base forme holding the
    # stone. Unioning them let "Glimmora with Adaptability" (a Glimmora-Mega ability)
    # pass our oracle while Showdown rejected it.
    if "mega" in norm(e.get("forme", "")):
        for fld in ("baseSpecies", "changesFrom"):
            v = e.get(fld)
            if isinstance(v, str):
                b = dex_entry(v)
                if b: out |= {norm(a) for a in b.get("abilities", {}).values()}
    return out

# stone -> base species, read off each mega forme's requiredItem
MEGA_STONE_OF = {}
for _k, _v in _dex.items():
    if isinstance(_v, dict) and "mega" in norm(_v.get("forme", "")):
        _ri = _v.get("requiredItem")
        _base = _v.get("changesFrom") or _v.get("baseSpecies")
        if _ri and _base: MEGA_STONE_OF[norm(_ri)] = norm(_base)

def parse_team(path):
    return parse_team_text(Path(path).read_text())

def parse_team_text(text):
    slots, cur = [], None
    for line in text.splitlines():
        line = line.rstrip()
        if not line.strip():
            continue
        if " @ " in line or (cur is None and line and not line.startswith("-") and ":" not in line):
            if cur: slots.append(cur)
            if " @ " in line:
                spec, item = line.split(" @ ", 1)
            else:
                spec, item = line, ""
            spec = re.sub(r"\s*\((M|F)\)\s*$", "", spec.strip())
            # strip nickname form: "Nick (Species)"
            m = re.match(r"^(.*?)\s*\((.+)\)$", spec)
            if m and norm(m.group(2)) in DEX_NAMES:
                spec = m.group(2)
            cur = {"species": spec.strip(), "item": item.strip(), "moves": [],
                   "ability": "", "nature": "", "evs": {s: 0 for s in STATS}}
        elif line.startswith("Ability:"):
            cur["ability"] = line.split(":", 1)[1].strip()
        elif line.startswith("EVs:"):
            for val, st in re.findall(r"(\d+)\s+(HP|Atk|Def|SpA|SpD|Spe)", line):
                cur["evs"][st] = int(val)
        elif line.endswith("Nature"):
            cur["nature"] = line.replace("Nature", "").strip()
        elif line.startswith("- "):
            cur["moves"].append(line[2:].strip())
        elif line.startswith("Tera Type:"):
            cur["tera"] = line.split(":", 1)[1].strip()
    if cur: slots.append(cur)
    return slots

def load_corpus():
    teams, names = [], []
    for f in sorted(TEAMS.rglob("*.txt")):   # rglob: include featured/
        t = parse_team(f)
        if len(t) == 6:
            teams.append(t); names.append(f.name)
    return teams, names

def check_slot(slot):
    """Return list of violation strings for one slot."""
    v = []
    e = dex_entry(slot["species"])
    if e is None:
        return ["species:" + slot["species"]]
    if slot["item"] and norm(slot["item"]) not in ITEM_NAMES:
        v.append("item:" + slot["item"])
    la = legal_abilities(slot["species"])
    if la is not None and norm(slot["ability"]) not in la:
        v.append("ability:" + slot["ability"])
    lm = legal_moves(slot["species"])
    ms = [m for m in slot["moves"]]
    if len(set(norm(m) for m in ms)) != len(ms):
        v.append("dupmove")
    if not (1 <= len(ms) <= 4):
        v.append("movecount:%d" % len(ms))
    for m in ms:
        if norm(m) not in MOVE_NAMES:
            v.append("move:" + m)
        elif lm is not None and norm(m) not in lm:
            v.append("learnset:%s/%s" % (slot["species"], m))
    if slot["nature"] and slot["nature"] not in NATURES:
        v.append("nature:" + slot["nature"])
    ev = slot["evs"]
    if any(x < 0 or x > 32 for x in ev.values()):
        v.append("evrange")
    return v

def check_team(team):
    """Violations for a whole team, including Species Clause."""
    v = []
    for i, s in enumerate(team):
        v += ["slot%d/%s" % (i, x) for x in check_slot(s)]
    base = []
    for s in team:
        e = dex_entry(s["species"])
        base.append(norm(e.get("baseSpecies", e.get("name", s["species"]))) if e else norm(s["species"]))
    if len(set(base)) != len(base):
        v.append("speciesclause")
    items = [norm(s["item"]) for s in team if s["item"]]
    return v

if __name__ == "__main__":
    teams, names = load_corpus()
    print("teams parsed:", len(teams))
    bad = 0; kinds = {}
    for t, n in zip(teams, names):
        v = check_team(t)
        if v:
            bad += 1
            for x in v: kinds[x.split(":")[0].split("/")[-1]] = kinds.get(x.split(":")[0].split("/")[-1], 0) + 1
    print("teams with >=1 violation:", bad, "of", len(teams))
    print("violation kinds:", sorted(kinds.items(), key=lambda x: -x[1])[:12])
