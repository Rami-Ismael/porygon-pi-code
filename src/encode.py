"""Encode a team as a fixed grid of categorical fields, and define the legality
oracle used identically by every arm of the pilot.

A team = 6 slots x 8 categorical fields (species, ability, item, move1..4, nature).
Stat Points (this regulation's EV analogue: per-stat 0..32) are carried alongside
as 6 integers per slot and are resampled from the corpus conditional on species.

Slot order is a nuisance variable (a team is a set), so teams are canonicalised by
sorting slots on species before any distance is computed, and slot order is randomly
permuted during training as augmentation.
"""
import json, random
from collections import defaultdict
import numpy as np
from corpus import (load_corpus, norm, dex_entry, legal_moves, legal_abilities,
                    ITEM_NAMES, MOVE_NAMES, STATS, NATURES)

FIELDS = ["species", "ability", "item", "m0", "m1", "m2", "m3", "nature"]
NF = len(FIELDS)
NSLOT = 6

def canon(team):
    return sorted(team, key=lambda s: norm(s["species"]))

def team_fields(team):
    out = []
    for s in canon(team):
        ms = sorted(s["moves"], key=norm) + [""] * (4 - len(s["moves"]))
        out += [norm(s["species"]), norm(s["ability"]), norm(s["item"]),
                norm(ms[0]), norm(ms[1]), norm(ms[2]), norm(ms[3]), norm(s["nature"])]
    return out

class Vocab:
    """One vocabulary per field type; species/ability/item/nature share a column each,
    the four move columns share a single move vocabulary."""
    def __init__(self, teams):
        cols = defaultdict(set)
        for t in teams:
            f = team_fields(t)
            for i, v in enumerate(f):
                k = FIELDS[i % NF]
                cols["move" if k.startswith("m") and k != "nature" else k].add(v)
        self.itos, self.stoi = {}, {}
        for k, vals in cols.items():
            vs = ["[MASK]"] + sorted(vals)
            self.itos[k] = vs
            self.stoi[k] = {v: i for i, v in enumerate(vs)}
        self.sizes = {k: len(v) for k, v in self.itos.items()}
    def key(self, col):
        k = FIELDS[col % NF]
        return "move" if k.startswith("m") and k != "nature" else k
    def encode(self, team):
        f = team_fields(team)
        return np.array([self.stoi[self.key(i)].get(v, 0) for i, v in enumerate(f)], dtype=np.int64)
    def decode_field(self, col, idx):
        return self.itos[self.key(col)][idx]

# ---- legality oracle -------------------------------------------------------
class Legality:
    """Legality against an AUTHORITATIVE per-species move table built by querying
    Showdown's own TeamValidator (see build_learnset.py). The earlier hand-rolled
    merge rules were wrong in both directions: the merged rule allowed ~94 moves
    per species where the truth is ~48, which is why generated teams passed this
    oracle at 96.9% and Showdown at 32.8%."""
    def __init__(self, teams):
        import json as _json, os as _os
        self._true = None
        _p = "/tmp/vgc-pilot/learnset_true.json"
        if _os.path.exists(_p):
            self._true = {k: set(v) for k, v in _json.load(open(_p)).items()}
        self.corpus_moves = defaultdict(set)
        self.corpus_abils = defaultdict(set)
        self.corpus_items = set()
        self.max_sp = 0
        for t in teams:
            for s in t:
                sp = norm(s["species"])
                self.corpus_moves[sp] |= {norm(m) for m in s["moves"]}
                self.corpus_abils[sp].add(norm(s["ability"]))
                if s["item"]: self.corpus_items.add(norm(s["item"]))
                self.max_sp = max(self.max_sp, sum(s["evs"].values()))
        self._lm, self._la = {}, {}
    def moves_for(self, sp):
        if sp not in self._lm:
            if self._true is not None and sp in self._true:
                self._lm[sp] = self._true[sp] | self.corpus_moves.get(sp, set())
            else:
                self._lm[sp] = (legal_moves(sp) or set()) | self.corpus_moves.get(sp, set())
        return self._lm[sp]
    def abils_for(self, sp):
        if sp not in self._la:
            base = legal_abilities(sp) or set()
            self._la[sp] = base | self.corpus_abils.get(sp, set())
        return self._la[sp]
    def violations(self, fields):
        """fields = flat list of 48 normalised strings."""
        v = []
        bases = []
        for i in range(NSLOT):
            sl = fields[i*NF:(i+1)*NF]
            sp, ab, it = sl[0], sl[1], sl[2]
            ms = [m for m in sl[3:7] if m]
            nat = sl[7]
            e = dex_entry(sp)
            if e is None:
                v.append("species"); bases.append(sp); continue
            bases.append(norm(e.get("baseSpecies", e.get("name", sp))))
            if it and it not in ITEM_NAMES and it not in self.corpus_items:
                v.append("item")
            if ab not in self.abils_for(sp):
                v.append("ability")
            if not (1 <= len(ms) <= 4):
                v.append("movecount")  # Showdown allows 1-3 move sets
            if len(set(ms)) != len(ms):
                v.append("dupmove")
            lm = self.moves_for(sp)
            for m in ms:
                if m not in lm: v.append("learnset")
            if nat and nat not in {norm(n) for n in NATURES}:
                v.append("nature")  # empty nature defaults to Serious
        if len(set(bases)) != NSLOT:
            v.append("speciesclause")
        # Item Clause is in force in this regulation: 0/692 corpus teams repeat an
        # item. Omitting this check flatters any operator that pastes corpus slots.
        items = [f for f in (fields[i*NF+2] for i in range(NSLOT)) if f]
        if len(set(items)) != len(items):
            v.append("itemclause")
        return v
    def legal(self, fields):
        return len(self.violations(fields)) == 0

if __name__ == "__main__":
    teams, names = load_corpus()
    teams = [t for t in teams if len(t) == 6]
    V = Vocab(teams); L = Legality(teams)
    print("teams:", len(teams), "field-grid:", NSLOT*NF)
    print("vocab sizes:", V.sizes)
    bad = sum(0 if L.legal(team_fields(t)) else 1 for t in teams)
    print("corpus teams failing the widened oracle:", bad, "of", len(teams))
    X = np.stack([V.encode(t) for t in teams])
    print("encoded:", X.shape, "unique rows:", len(np.unique(X, axis=0)))
    print("max stat points on a slot:", L.max_sp)
