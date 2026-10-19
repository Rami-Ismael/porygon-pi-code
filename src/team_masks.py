"""Paths into a six-slot team, hiding them, and the shape a filled team must have.

A team is a list of six slots, each {species, item, ability, nature, moves, evs}.
A path names one field ([slot, "item"]), one move ([slot, "moves", i]) or a whole
slot ([slot]). A masked team has None at every hidden path."""
import copy

STATS = ["HP", "Atk", "Def", "SpA", "SpD", "Spe"]
FIELDS = {"stats": "evs", "item": "item", "ability": "ability", "alignment": "nature"}
# Mask sizes per task; each task is a key paths_for accepts.
GRID = {"stats": [1, 3, 6], "item": [1, 3, 6], "move": [1, 12, 24],
        "pokemon": [1, 3, 6], "ability": [1, 3, 6], "alignment": [1, 3, 6],
        "mixed": [1, 20, 40]}


def paths_for(task):
    if task == "pokemon":
        return [[s] for s in range(6)]
    if task == "move":
        return [[s, "moves", m] for s in range(6) for m in range(4)]
    if task == "mixed":
        return [p for t in FIELDS for p in paths_for(t)] + paths_for("move")
    return [[s, FIELDS[task]] for s in range(6)]


def set_path(team, path, value):
    obj = team
    for key in path[:-1]:
        obj = obj[key]
    obj[path[-1]] = value


def masked(team, paths):
    result = copy.deepcopy(team)
    for path in paths:
        set_path(result, path, None)
    return result


def validate_shape(team):
    if not isinstance(team, list) or len(team) != 6:
        raise ValueError("team must have six slots")
    for slot in team:
        if not isinstance(slot, dict) or set(slot) != {"species", "item", "ability", "nature", "moves", "evs"}:
            raise ValueError("invalid slot fields")
        if any(not isinstance(slot[k], str) for k in ("species", "item", "ability", "nature")):
            raise ValueError("slot names must be strings")
        if (not isinstance(slot["moves"], list) or len(slot["moves"]) != 4
                or any(not isinstance(m, str) for m in slot["moves"])):
            raise ValueError("four move strings required")
        evs = slot["evs"]
        if (not isinstance(evs, dict) or set(evs) != set(STATS)
                or any(type(v) is not int or not 0 <= v <= 32 for v in evs.values())
                or sum(evs.values()) > 66):
            raise ValueError("Stat Points require 0..32 per stat, total <=66")
