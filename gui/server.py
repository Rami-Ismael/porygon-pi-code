"""Team sheet generator - local web service.

Pin a candidate, build teams around it, and MEASURE each one against the top-50
meta teams instead of asserting it is good. The measurement is the point: this
project's central finding was that legality, novelty and style adherence were all
green while win rate was red, so the interface refuses to show a team without
offering to battle it.

Two build methods, named for what they do:
  graft  - start from a real tournament team and swap the pinned candidate in.
           Measured baseline for this move: 0.420 (slotcopy).
  invent - generate all six slots from the diffusion model with the pin held fixed.
           Measured baseline: 0.093. Offered because it is what the research goal
           asked for, and labelled honestly.
"""
import asyncio, json, os, random, sys, threading, time
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, "/tmp/vgc-pilot/src")
sys.path.insert(0, "/tmp/vgc-pilot/vgc-bench")
os.chdir("/tmp/vgc-pilot/vgc-bench")

import numpy as np, torch
from corpus import load_corpus, norm, STATS
from encode import Vocab, Legality, canon, NF, NSLOT
import diffusion as D
import wrdiffusion as W
from propose import Validator, slot_to_text

HERE = Path("/tmp/vgc-pilot/gui")
# Every value here is MEASURED against the top-50 meta pool with the behaviour-cloning
# policy on both sides, errors clustered by team. `invent` was 0.0928 (unconditioned,
# temp 1.0) and then 0.179 (win-rate bin 5, temp 1.0, Garchomp pinned); the sampler now
# runs at temp 0.7 / top-p 0.9, measured 2026-08-27 at 0.2686 +/- 0.0098 over 150 teams
# with a pinned candidate, at 100% Showdown validity. See results/schemes_results.json.
BASELINES = [
    {"key": "invent",  "label": "invented from scratch", "value": 0.2686},
    {"key": "graft",   "label": "one candidate grafted", "value": 0.4199},
    {"key": "asis",    "label": "a real team, untouched", "value": 0.4751},
]

print("loading corpus and model ...", flush=True)
TEAMS, V, L, X, Y = D.load_all()
MODEL = D._load(V)
CONS = D.Constraints(V, L)
# The `invent` path uses the win-rate-conditioned checkpoint asked for at its top bin,
# not the unconditioned one: 0.269 vs 0.093 measured. Its vocabulary is a superset of the
# corpus vocabulary, so the display-name lookups below still resolve.
WMODEL, WV, WL, _wcorpus, _wlab = W.load_model()
WCONS = D.Constraints(WV, WL)
LOOK, SPREADS, BY_SPECIES = {}, defaultdict(list), defaultdict(list)
for _t in TEAMS:
    for _s in _t:
        LOOK[norm(_s["species"])] = _s["species"]; LOOK[norm(_s["ability"])] = _s["ability"]
        if _s["item"]: LOOK[norm(_s["item"])] = _s["item"]
        for _m in _s["moves"]: LOOK[norm(_m)] = _m
        if _s["nature"]: LOOK[norm(_s["nature"])] = _s["nature"]
        SPREADS[norm(_s["species"])].append(dict(_s["evs"]))
        BY_SPECIES[norm(_s["species"])].append(_s)
SPECIES = sorted({norm(s["species"]) for t in TEAMS for s in t},
                 key=lambda k: (-len(BY_SPECIES[k]), k))
VAL = Validator(); VAL_LOCK = threading.Lock()

OPP = []
_root = Path("/tmp/vgc-pilot/vgc-bench/teams/reg_mb")
for _i in [t["id"] for t in json.load(open("/tmp/vgc-pilot/top50_evs.json"))]:
    for _c in (_root/f"{_i}.txt", _root/"featured"/f"{_i}.txt"):
        if _c.exists(): OPP.append(_c); break
print(f"ready: {len(TEAMS)} teams, {len(SPECIES)} candidates, {len(OPP)} opponents", flush=True)

_POLICY = None
def policy():
    global _POLICY
    if _POLICY is None:
        from stable_baselines3 import PPO
        _POLICY = PPO.load("/tmp/bc_100.zip", device=torch.device("cpu")).policy
    return _POLICY

def validate(text):
    with VAL_LOCK:
        return VAL(text)

def slots_to_text(slots):
    return "\n\n".join(slot_to_text(s) for s in slots) + "\n"

def row_to_slots(row, spread_from=None, vocab=None):
    vocab = vocab or V
    out = []
    for i in range(NSLOT):
        b = i * NF
        g = lambda j: vocab.decode_field(b + j, int(row[b + j]))
        mv = [LOOK.get(g(3+j), g(3+j)) for j in range(4) if g(3+j) and g(3+j) != "[MASK]"]
        sp = g(0)
        evs = (spread_from[i]["evs"] if spread_from
               else random.choice(SPREADS.get(sp) or [{s: 0 for s in STATS}]))
        out.append(dict(species=LOOK.get(sp, sp), item=LOOK.get(g(2), g(2)),
                        ability=LOOK.get(g(1), g(1)), nature=LOOK.get(g(7), g(7)),
                        moves=mv, evs=evs))
    return out

def build(pin, style, method, n):
    """Return up to n Showdown-valid teams containing `pin`."""
    rng = np.random.default_rng()
    si = V.stoi["species"].get(pin)
    if si is None: return [], "That candidate is not in the Reg M-B corpus."
    made, tries = [], 0
    while len(made) < n and tries < 220:
        tries += 1
        if method == "graft":
            base = canon(TEAMS[int(rng.integers(0, len(TEAMS)))])
            if any(norm(s["species"]) == pin for s in base):
                slot = next(i for i, s in enumerate(base) if norm(s["species"]) == pin)
                donor = dict(random.choice(BY_SPECIES[pin]))
            else:
                slot = int(rng.integers(0, NSLOT))
                donor = dict(random.choice(BY_SPECIES[pin]))
            cand = [dict(s) for s in base]; cand[slot] = donor
            slots = cand
        else:
            g = 2.0 if style != "none" else 1.0
            wsi = WV.stoi["species"].get(pin)
            if wsi is None: return [], "That candidate is not in the model's vocabulary."
            row = W.sample_constrained(WMODEL, WCONS, 1, 5, style=style, guidance=g,
                                       temp=0.7, pin={0: wsi}, top_p=0.9).cpu().numpy()[0]
            slots = row_to_slots(row, vocab=WV)
        txt = slots_to_text(slots)
        if not any(norm(s["species"]) == pin for s in slots): continue
        if validate(txt) is not None: continue
        if any(m["paste"] == txt for m in made): continue
        made.append({"paste": txt, "slots": [
            {"species": s["species"], "item": s["item"], "ability": s["ability"],
             "nature": s["nature"], "moves": s["moves"],
             "evs": " / ".join(f"{s['evs'].get(k,0)} {k}" for k in STATS if s["evs"].get(k)),
             "pinned": norm(s["species"]) == pin} for s in slots]})
    return made, None

def measure(paste, n_opp, n_bat):
    import winrate as W
    tmp = Path("/tmp/vgc-pilot/gui/_measure.txt"); tmp.write_text(paste)
    pol = policy(); wins = fin = 0
    for of in OPP[:n_opp]:
        w, f = asyncio.run(W.play(pol, str(tmp), str(of), n_bat, 25, os.urandom(3).hex()))
        wins += w; fin += f
    wr = wins / max(fin, 1)
    return {"win_rate": wr, "se": (wr*(1-wr)/max(fin,1))**.5, "battles": fin}

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, code, body, ctype="application/json"):
        b = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        p = urlparse(self.path).path
        if p in ("/", "/index.html"):
            return self._send(200, (HERE/"static"/"index.html").read_bytes(), "text/html; charset=utf-8")
        if p == "/hist":
            return self._send(200, (HERE/"static"/"hist.html").read_bytes(), "text/html; charset=utf-8")
        if p == "/masks":
            return self._send(200, (HERE/"static"/"masks.html").read_bytes(), "text/html; charset=utf-8")
        if p == "/schemes":
            return self._send(200, (HERE/"static"/"schemes.html").read_bytes(), "text/html; charset=utf-8")
        if p == "/api/meta":
            return self._send(200, {
                "species": [{"key": k, "name": LOOK.get(k, k), "count": len(BY_SPECIES[k])}
                            for k in SPECIES],
                "styles": D.STYLES, "baselines": BASELINES,
                "corpus": len(TEAMS), "opponents": len(OPP)})
        self._send(404, {"error": "no such path"})
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        p = urlparse(self.path).path
        try:
            if p == "/api/build":
                teams, err = build(req.get("pin", ""), req.get("style", "none"),
                                   req.get("method", "graft"), int(req.get("n", 4)))
                if err: return self._send(200, {"error": err})
                return self._send(200, {"teams": teams})
            if p == "/api/measure":
                return self._send(200, measure(req["paste"], int(req.get("opponents", 12)),
                                               int(req.get("battles", 8))))
        except Exception as e:
            return self._send(200, {"error": f"{type(e).__name__}: {e}"})
        self._send(404, {"error": "no such path"})

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8770
    print(f"team sheet generator on http://localhost:{port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
