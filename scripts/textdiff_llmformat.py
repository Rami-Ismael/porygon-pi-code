"""Text diffusion writers whose prompts copy the LLM baseline's prompt versions: one fine-tune per version.

The masked-completion baseline sent Ling five prompt versions: the Mask Lab's `baseline` and `regmb`
(team_completion_pipeline/pipeline.py) and the registry's `mb-original-v1`, `mb-grounded-v2` and `mb-regulation-v3`
(scripts/prompt_versions.py, docs/prompt-changelog.md). Each version here keeps that system text and that user
message's structure, and changes what the text model cannot take (docs/textdiff.md, "LLM-format prompts"):
- the meta teams are one line each, the team's six species (1,284 tokens for the 49 instead of 17,543 as pastes);
- Stat Points are not text: unmasked spreads are left out and masked ones are filled from a same-species corpus
  spread, as for the first text writer; the sentences that ask for spreads in the answer are adapted;
- a target_win_rate field is added: the model's steering input (guidance compares it with "unknown").
The answer keeps each version's shape: the Mask Lab versions return the completed team (as a paste, not JSON
objects), the registry versions return {"fills": [...]} patches.

Training data, the same for every version (teams, masks, opponent lists and labels; only the formatting differs):
- 4,000 rows like the first text writer's: every real-corpus row of F1_final_medium_noheldout_regmb_v3's data, then
  family and synthetic rows, each listing all 49 top-50 teams with the team's win rate against them (or unknown);
- 4,000 rows with a battle against each of the 49 (the matchup-matrix teams the battle-records model trained on),
  each listing a random 8-49 of the 49 with the team's win rate against those.
Each row gets one mask drawn from the Mask Lab grid's 48 non-Stat-Point cells (items, abilities, natures and whole
Pokémon k = 1-6, moves k = 1-24). Ranks 51-100 are not in it (heldout.row_mask, as in textdiff.py prepare).

    /tmp/vgc-pilot/.venv/bin/python scripts/textdiff_llmformat.py prepare
    ~/.local/share/vgc-pilot-runtime/venv-dlm/bin/python scripts/textdiff_llmformat.py stats
"""
import argparse, copy, json, random, statistics, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(1, str(REPO / "scripts"))
PIPELINE = Path.home() / ("Documents/yakumsi-vault/Personal Project/Using Advance method search in the whole search of "
                          "possible vgc pokemon format to find the right counter to a pokemon team/team_completion_pipeline")
sys.path.insert(2, str(PIPELINE))
RUNTIME = Path.home() / ".local/share/vgc-pilot-runtime"
REG = RUNTIME / "regmb-v3"
OUT = RUNTIME / "textdiff/llmformat"
FORMAT = "gen9championsvgc2026regmb"
SPREAD_SYSTEM = "Champions Stat Points; use the allocation rules for the specified format."     # benchmark.ling.json
VERSIONS = ["regmb", "baseline", "mb-original-v1", "mb-regulation-v3", "mb-grounded-v2"]
MASK_LAB = ("regmb", "baseline")
SEED, P_UNCOND, N_EACH, N_VAL = 20261021, 0.15, 4000, 512
CELLS = [(t, k) for t, top in (("items", 6), ("moves", 24), ("pokemon", 6), ("abilities", 6), ("natures", 6))
         for k in range(1, top + 1)]                  # the Mask Lab grid without its Stat Point family


# ---------------------------------------------------------------- the prompt versions
def system_text(version):
    """The version's system text with only the answer-format sentences adapted (Stat Points are not text here)."""
    if version in MASK_LAB:
        import pipeline as P
        text = P.PROMPTS[version]
        old = 'Return JSON only: {"completed_team": [six complete Pokémon objects], "reasoning": []}.'
        new = ("Return only the completed team as a Showdown paste: the six Pokémon in slot order, each with its "
               "item, ability, nature and four moves, without Stat Points.")
    else:
        text = (REPO / "prompts" / f"{version}.system.txt").read_text()
        if version == "mb-grounded-v2":
            old, new = ", and evs containing all six Stat Point keys", ""
        else:
            old = "For a whole slot return {species,item,ability,nature,moves,evs}; evs keys are HP,Atk,Def,SpA,SpD,Spe."
            new = "For a whole slot return {species,item,ability,nature,moves}."
    if old not in text: raise ValueError(f"{version}: the answer-format sentence moved; check the adaptation")
    return text.replace(old, new)

def mask_paths(mask, registry):
    """pipeline.masked_paths strings for the Mask Lab versions, path lists for the registry versions."""
    import pipeline as P
    task, slots = mask["task"], mask["slots"]
    if not registry: return P.masked_paths(task, slots)
    if task == "moves": return [[i // 4, "moves", i % 4] for i in slots]
    if task == "pokemon": return [[i] for i in slots]
    return [[i, P.FIELDS[task].replace("statPoints", "evs")] for i in slots]

def masked_team(team, mask, registry):
    out = [dict(species=m["species"], item=m["item"], ability=m["ability"], nature=m["nature"], moves=list(m["moves"])) for m in team]
    if not registry: out = [dict(slot=i + 1, **m) for i, m in enumerate(out)]
    for i in mask["slots"]:
        if mask["task"] == "moves": out[i // 4]["moves"][i % 4] = None
        elif mask["task"] == "pokemon": out[i] = None
        else: out[i][{"items": "item", "abilities": "ability", "natures": "nature"}[mask["task"]]] = None
    return out

def user_text(version, team, mask, opponents, target):
    """The user message: the version's request with the meta as species lines and a target win rate added."""
    registry = version not in MASK_LAB
    target = "unknown" if target is None else f"{target:.2f}"
    if not registry:                                    # pipeline.generate sends canonical(request)
        import pipeline as P
        request = dict(format=FORMAT, spread_system=SPREAD_SYSTEM,
                       task=dict(task=mask["task"], k=mask["k"], slots=mask["slots"], masked_paths=mask_paths(mask, False)),
                       masked_team=masked_team(team, mask, False), meta_teams=opponents, target_win_rate=target, feedback=None)
        return P.canonical(request)
    content = {"format": FORMAT, "opponents": opponents, "masked_team": masked_team(team, mask, True),
               "requested_paths": mask_paths(mask, True)}           # llm_completion.prompt's request, in its key order
    if version == "mb-grounded-v2":                     # mb_prompt.strengthen adds the per-slot legal choices
        from mb_prompt import strengthen
        msgs = strengthen([{"role": "system", "content": ""}, {"role": "user", "content": json.dumps(content)}],
                          {"format": FORMAT}, {"paths": content["requested_paths"]})
        content = json.loads(msgs[1]["content"])
    content["target_win_rate"] = target
    return json.dumps(content, ensure_ascii=False)

def answer_text(version, team, mask):
    if version in MASK_LAB:
        from textdiff import slot_text
        return "\n\n".join(slot_text(m["species"], m["item"], m["ability"], m["nature"], [x for x in m["moves"] if x]) for m in team) + "\n"
    fills = []
    for p in mask_paths(mask, True):
        m = team[p[0]]
        value = (dict(species=m["species"], item=m["item"], ability=m["ability"], nature=m["nature"], moves=list(m["moves"]))
                 if len(p) == 1 else m["moves"][p[2]] if p[1] == "moves" else m[p[1]])
        fills.append({"path": p, "value": value})
    return json.dumps({"fills": fills}, ensure_ascii=False)


# ---------------------------------------------------------------- prepare (/tmp/vgc-pilot/.venv)
def meta_lines():
    """The 49 distinct top-50 teams as species lines in placement order, with their column in the matchup rows."""
    import numpy as np, sqlite3, lossdown as L, matchup_db as MDB, top50 as T50
    top = [int(t) for t in np.load(L.DATA)["top"]]
    con = sqlite3.connect(f"file:{Path.home() / 'vgc-data/matchup_regmb.sqlite'}?mode=ro", uri=True)
    source = dict(con.execute("SELECT source_id, team_id FROM team_source WHERE source='vgcpastes'").fetchall())
    order, seen = [], set()
    for t in json.load(open(T50.TOP50)):
        tid = source[t["id"]]
        if tid in seen: continue                        # MB493 = MB494: one team, listed once
        seen.add(tid); order.append(tid)
    if sorted(order) != sorted(top): raise ValueError("top-50 placement order does not cover the 49 matchup columns")
    line = lambda tid: ", ".join(sorted(s["species"] for s in MDB.canonical_slots(
        con.execute("SELECT paste FROM team WHERE team_id=?", (tid,)).fetchone()[0])))
    return [dict(team_id=tid, column=top.index(tid), line=line(tid)) for tid in order]

def prepare(a):
    import numpy as np, lossdown as L, asked_vs_got as X, heldout as HO, encode
    reg = json.loads((REG / "training-regulation.json").read_text())
    V = encode.Vocab.from_regulation(reg); names = reg["names"]
    _, old, *_ = X.setup()
    d = L.load_data(include_heldout=True); mapped = np.load(REG / "remapped.npz"); synthetic = np.load(REG / "synthetic.npy")
    keep = HO.row_mask(d["X"], old)
    tr, val = L.splits(d); tr = tr[keep[tr]]; val = val[keep[val]]; tag = L.tag_of(d["S"])
    valid = mapped["valid"]

    def decode(r):
        nm = lambda v: names.get(v, v) if v else ""
        team = []
        for s in range(6):
            g = lambda j: V.decode_field(s * 8 + j, int(r[s * 8 + j]))
            moves = [nm(g(3 + j)) if g(3 + j) not in ("", "[MASK]") else "" for j in range(4)]
            team.append(dict(species=nm(g(0)), item=nm(g(2)), ability=nm(g(1)), nature=nm(g(7)), moves=moves))
        return team

    rng = random.Random(SEED)
    def mask_for(team):
        while True:                                     # a cell the team can take (moves only at written move slots)
            task, k = rng.choice(CELLS)
            pos = ([s * 4 + j for s, m in enumerate(team) for j, mv in enumerate(m["moves"]) if mv] if task == "moves"
                   else list(range(6)))
            if k <= len(pos): return dict(task=task, k=k, slots=sorted(rng.sample(pos, k)))

    meta = meta_lines()
    W, M = d["W"], d["M"]
    f1 = tr[((tag[tr] == L.TEST_TAG) | (tag[tr] == 0)) & valid[tr]]
    real = [int(i) for i in f1[tag[f1] == 0]]; fam = [int(i) for i in f1[tag[f1] != 0]]
    rest = [("row", i) for i in fam] + [("synthetic", j) for j in range(len(synthetic))]
    rng.shuffle(rest)
    pool_a = [("row", i) for i in real] + rest[:N_EACH - len(real)]
    full = np.where((M[:, :, 1] > 0).all(1) & np.isfinite(W))[0]
    pool_b = [int(i) for i in np.intersect1d(full, tr) if valid[i]]
    pool_b = rng.sample(pool_b, N_EACH)

    plan = []
    for kind, i in pool_a:
        team = decode(synthetic[i] if kind == "synthetic" else mapped["X"][i])
        w = None if kind == "synthetic" or not np.isfinite(W[i]) else float(W[i])
        if w is not None and rng.random() < P_UNCOND: w = None
        source = "synthetic" if kind == "synthetic" else ("real" if tag[i] == 0 else "family")
        plan.append(dict(row=i if kind == "row" else None, synthetic=i if kind == "synthetic" else None, source=source,
                         team=team, mask=mask_for(team), opponents=list(range(len(meta))), label=w))
    for i in pool_b:
        team = decode(mapped["X"][i])
        chosen = sorted(rng.sample(range(len(meta)), rng.randint(8, len(meta))))
        cols = [meta[j]["column"] for j in chosen]
        w = float(M[i, cols, 0].sum() / M[i, cols, 1].sum())
        if rng.random() < P_UNCOND: w = None
        plan.append(dict(row=i, synthetic=None, source="battle-records", team=team, mask=mask_for(team), opponents=chosen, label=w))
    rng.shuffle(plan)
    vplan = []
    for i in [int(v) for v in val[valid[val]]][:N_VAL]:
        team = decode(mapped["X"][i])
        w = None if not np.isfinite(W[i]) else float(W[i])
        vplan.append(dict(row=i, synthetic=None, source="val", team=team, mask=mask_for(team), opponents=list(range(len(meta))), label=w))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "plan.json").write_text(json.dumps(dict(seed=SEED, meta=meta, train=plan, val=vplan)) + "\n")
    for version in VERSIONS:
        folder = OUT / version; folder.mkdir(exist_ok=True)
        system = system_text(version)
        for name, rows in (("train", plan), ("val", vplan)):
            with open(folder / f"{name}.jsonl", "w") as f:
                for e in rows:
                    f.write(json.dumps(dict(system=system, prompt=user_text(version, e["team"], e["mask"], [meta[j]["line"] for j in e["opponents"]], e["label"]),
                                            answer=answer_text(version, e["team"], e["mask"]), label=e["label"], source=e["source"],
                                            task=e["mask"]["task"], k=e["mask"]["k"], opponents=len(e["opponents"])), ensure_ascii=False) + "\n")
    count = lambda rows, key: dict(sorted(__import__("collections").Counter(r[key] for r in rows).items()))
    summary = dict(train=len(plan), val=len(vplan), sources=count(plan, "source"), tasks=count([e["mask"] for e in plan], "task"),
                   labelled=sum(e["label"] is not None for e in plan), opponents_mean=statistics.mean(len(e["opponents"]) for e in plan),
                   heldout_rows=int((~keep[[e["row"] for e in plan + vplan if e["row"] is not None]]).sum()))
    (OUT / "data.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


# ---------------------------------------------------------------- stats (venv-dlm: the model's tokenizer)
def stats(a):
    sys.path.insert(0, str(REPO / "src"))
    import textdiff_port as P, textdiff as T
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(P.MODEL_ID, trust_remote_code=True)       # as textdiff_port.load
    out = {}
    for version in VERSIONS:
        rows = [json.loads(l) for l in open(OUT / version / "train.jsonl")][:a.sample]
        lens = [len(T.encode_example(tok, r["prompt"], r["answer"], 32, "full", r["system"])[0]) for r in rows]
        ans = [len(tok(r["answer"], add_special_tokens=False).input_ids) for r in rows]
        q = sorted(lens)
        out[version] = dict(system=len(tok(rows[0]["system"], add_special_tokens=False).input_ids), example_mean=round(statistics.mean(lens)),
                            example_p95=q[int(0.95 * len(q))], example_max=q[-1], answer_mean=round(statistics.mean(ans)))
        print(version, out[version], flush=True)
    (OUT / "tokens.json").write_text(json.dumps(out, indent=1) + "\n")


# ---------------------------------------------------------------- fill the Mask Lab grids
GRIDS = {"random": PIPELINE / "runs/prompt-ab", "heldout": PIPELINE / "runs/heldout-starters"}
WORK = Path.home() / "vgc-data/textdiff_llmformat"
GEN_SEED = 20261008                                   # mask_lab_textdiff.py's seed, + trial index

def generate(a):                                      # venv-dlm
    """One answer per mask: the version's prompt with all 49 meta teams listed and target 0.50, guidance 2 against the
    same prompt with target unknown, the released block sampler (threshold 0.9, temperature 1, top-p 1). Stat Point
    masks are not asked: collect fills them from the corpus, as for the first text writer."""
    import time, torch, textdiff_port as P
    from peft import PeftModel
    adapter = Path(a.adapter) if a.adapter else RUNTIME / "textdiff" / f"llm_{a.version}" / "final"
    tok, m = P.load(); m = PeftModel.from_pretrained(m, str(adapter)).merge_and_unload().eval()
    lines = [x["line"] for x in json.loads((OUT / "plan.json").read_text())["meta"]]
    system = system_text(a.version)
    chat = lambda u: tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": u}],
                                             add_generation_prompt=True, return_tensors="pt").to(P.DEV)
    tasks = json.loads((GRIDS[a.grid] / "eval2.tasks.json").read_text())["tasks"]
    dest = WORK / a.grid / (a.version + a.tag) / "raw"; dest.mkdir(parents=True, exist_ok=True)
    (dest.parent / "generation-config.json").write_text(json.dumps(dict(adapter=str(adapter), version=a.version, ask=a.ask,
        guidance=a.guidance, threshold=0.9, temperature=1.0, top_p=1.0, block_size=32, seed=GEN_SEED, opponents=len(lines)), indent=1))
    done = 0
    for i, t in enumerate(tasks):
        f = dest / f"{t['id']}.json"
        if f.exists() or t["task"] == "stats": continue
        if a.limit and done >= a.limit: break
        torch.manual_seed(GEN_SEED + i)
        team = [dict(species=x["species"], item=x["item"], ability=x["ability"], nature=x["nature"], moves=list(x["moves"])) for x in t["original"]]
        mask = dict(task=t["task"], k=t["k"], slots=t["slots"])
        c, u = chat(user_text(a.version, team, mask, lines, a.ask)), chat(user_text(a.version, team, mask, lines, None))
        t0 = time.perf_counter()
        out = P.generate_cfg(m, c, u, a.guidance, max_new_tokens=320 if a.version in MASK_LAB else 448,
                             threshold=0.9, temperature=1.0, top_p=1.0)
        new = out[0, c.shape[1]:]
        f.write_text(json.dumps(dict(task=t["id"], family=t["task"], k=t["k"], text=tok.decode(new, skip_special_tokens=True),
                                     stopped=bool((new == 151645).any()), prompt_tokens=int(c.shape[1]), seconds=time.perf_counter() - t0)))
        done += 1
        if done % 10 == 1: print(f"{done} {t['task']} k={t['k']} {time.perf_counter() - t0:.1f}s", flush=True)

def parse_answer(version, text, team, mask):
    """The completed team (species, item, ability, nature, moves per slot), or ValueError. Unmasked fields must come
    back unchanged, as pipeline.check_completion and llm_completion.apply_completion require of Ling."""
    norm = lambda s: "".join(ch for ch in (s or "").lower() if ch.isalnum())
    out = copy.deepcopy(team)
    if version in MASK_LAB:
        blocks = [b for b in text.strip().split("\n\n") if b.strip()]
        if len(blocks) != 6: raise ValueError(f"{len(blocks)} Pokémon in the answer")
        got = []
        for b in blocks:
            lines = b.strip().split("\n"); sp, _, it = lines[0].partition(" @ ")
            ab = next((l[9:].strip() for l in lines if l.startswith("Ability: ")), None)
            na = next((l[:-7].strip() for l in lines if l.endswith(" Nature")), "")   # slot_text omits an empty nature
            mv = [l[2:].strip() for l in lines if l.startswith("- ")]
            if ab is None or len(mv) != 4: raise ValueError("a Pokémon block is incomplete")
            got.append(dict(species=sp.strip(), item=it.strip(), ability=ab, nature=na, moves=mv))
        for s, g in enumerate(got):
            if mask["task"] == "pokemon" and s in mask["slots"]: out[s] = g; continue
            if norm(g["species"]) != norm(team[s]["species"]): raise ValueError("an unmasked Pokémon changed")
            field = {"items": "item", "abilities": "ability", "natures": "nature"}.get(mask["task"])
            for k in ("item", "ability", "nature"):
                if field == k and s in mask["slots"]: out[s][k] = g[k]
                elif norm(g[k]) != norm(team[s][k]): raise ValueError(f"an unmasked {k} changed")
            if mask["task"] == "moves":
                masked = [i % 4 for i in mask["slots"] if i // 4 == s]
                fixed = [team[s]["moves"][j] for j in range(4) if j not in masked]
                left = [m_ for m_ in g["moves"]]
                for mv in fixed:
                    hit = next((x for x in left if norm(x) == norm(mv)), None)
                    if hit is None: raise ValueError("an unmasked move changed")
                    left.remove(hit)
                for j, mv in zip(masked, left): out[s]["moves"][j] = mv
            elif sorted(map(norm, g["moves"])) != sorted(map(norm, team[s]["moves"])): raise ValueError("an unmasked move changed")
        return out
    fills = json.loads(text)
    if not isinstance(fills, dict) or set(fills) != {"fills"} or not isinstance(fills["fills"], list): raise ValueError("expected {fills: [...]}")
    want = mask_paths(mask, True)
    if len(fills["fills"]) != len(want): raise ValueError("wrong number of fills")
    for fill in fills["fills"]:
        if not isinstance(fill, dict) or set(fill) != {"path", "value"} or fill["path"] not in want: raise ValueError("a fill has an unrequested path")
        want.remove(fill["path"]); p, v = fill["path"], fill["value"]
        if len(p) == 1:
            if not isinstance(v, dict) or set(v) != {"species", "item", "ability", "nature", "moves"} or len(v["moves"]) != 4:
                raise ValueError("a whole-slot fill is incomplete")
            out[p[0]] = dict(v, moves=list(v["moves"]))
        elif p[1] == "moves": out[p[0]]["moves"][p[2]] = v
        else: out[p[0]][p[1]] = v
        if not all(isinstance(x, str) and x for x in [out[p[0]]["species"], *out[p[0]]["moves"]]): raise ValueError("an empty value")
    return out

NATURES = set("hardy lonely brave adamant naughty bold docile relaxed impish lax timid hasty serious jolly naive modest mild "
              "quiet bashful rash calm gentle sassy careful quirky".split())
# The validator below accepted "Very Brave" (2026-10-24), but the battle runtime rejects any nature outside these 25 when
# the challenge is sent, and every battle of that team hung. A trailing " Nature" is dropped by the battle client, as
# "Bold Nature" showed, so it is allowed.

def collect(a):                                       # /tmp/vgc-pilot/.venv
    import numpy as np, asked_vs_got as X, hps_generate, corpus   # project modules before the vault code
    from mask_lab_heldout_starters import llm_baseline_2026_10_15
    B = llm_baseline_2026_10_15()                      # the helpers mask_lab_textdiff.collect uses, as they were
    import mask_lab_prepare as MP
    _, _, _, _, spreads = X.setup()
    h, _ = B.runtime(); validator = h.Validator()
    run = GRIDS[a.grid]; tasks = json.loads((run / "eval2.tasks.json").read_text())["tasks"]
    raw = WORK / a.grid / (a.version + a.tag) / "raw"; kind = f"llm_{a.version}{a.tag}"
    rng = np.random.default_rng(GEN_SEED)
    rows, new = {}, []
    try:
        for t in tasks:
            team = [dict(species=x["species"], item=x["item"], ability=x["ability"], nature=x["nature"], moves=list(x["moves"]),
                         evs=dict(x["statPoints"])) for x in t["original"]]
            mask = dict(task=t["task"], k=t["k"], slots=t["slots"]); seconds = 0.0; error = None
            try:
                if t["task"] != "stats":
                    f = raw / f"{t['id']}.json"
                    if not f.exists(): rows[t["id"]] = dict(valid=False, error="missing", seconds=None); continue
                    g = json.loads(f.read_text()); seconds = g["seconds"]
                    done = parse_answer(a.version, g["text"], [{k: v for k, v in m.items() if k != "evs"} for m in team], mask)
                    team = [dict(d_, evs=m["evs"]) for d_, m in zip(done, team)]
                    bad = [m["nature"] for m in team if isinstance(m["nature"], str) and m["nature"].strip()
                           and m["nature"].lower().removesuffix(" nature").replace(" ", "") not in NATURES]
                    if bad: raise ValueError(f"invalid nature: {bad[0]}")
                for s in sorted({i // 4 if t["task"] == "moves" else i for i in t["slots"]}):
                    if t["task"] in ("stats", "pokemon"):      # Stat Points: a same-species corpus spread, else a random legal one
                        pl = spreads.get(corpus.norm(team[s]["species"]))
                        team[s]["evs"] = dict(pl[int(rng.integers(len(pl)))]) if pl else hps_generate.sample_spread(rng)
                B.validate_shape(team)
                error = validator(B.paste(team, h))
            except (ValueError, KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
                error = f"{type(exc).__name__}: {exc}"
            rows[t["id"]] = dict(valid=error is None, error=(error or "")[:160], seconds=seconds)
            if error is None:
                t2 = [dict(slot=i + 1, species=s["species"], item=s["item"], ability=s["ability"], nature=s["nature"],
                           moves=list(s["moves"]), statPoints=dict(s["evs"])) for i, s in enumerate(team)]
                new.append(dict(label=f"{kind}:{t['id']}", kind=kind, task_id=t["id"], source=t["source"], team=t2, paste=MP.to_paste(t2)))
    finally:
        validator.close()
    dest = WORK / "smoke" if a.tag else run if a.grid == "heldout" else WORK / "random-run"   # a tagged run never touches real candidates
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / "candidates.json"
    cands = json.loads(path.read_text())["candidates"] if path.exists() else []
    cands = [c for c in cands if c["kind"] != kind] + new
    path.write_text(json.dumps({"candidates": cands}, indent=1) + "\n")
    (dest / f"{kind}_rows.json").write_text(json.dumps(rows, indent=1) + "\n")
    if a.grid == "random" and not (dest / "eval2.tasks.json").exists():
        (dest / "eval2.tasks.json").write_text((run / "eval2.tasks.json").read_text())
    legal = sum(r["valid"] for r in rows.values())
    print(f"{kind} on the {a.grid} grid: {legal}/{len(rows)} legal; candidates now {len(cands)} in {path}")


# ---------------------------------------------------------------- upload to the Hugging Face Hub (venv-dlm)
HUB = "Rami/fast-dllm-v2-1.5b-vgc-mask-fill-ling-{}-prompt-lora"

def upload(a):
    """Model card from this version's prompt, training log and battle reports; public repo like the first text writer's
    (owner's choice, 2026-10-21); stops unless the hub copy's SHA-256 equals the local adapter's."""
    import hashlib, shutil
    from huggingface_hub import HfApi
    v = a.version; repo = HUB.format(v)
    run = RUNTIME / "textdiff" / f"llm_{v}"
    log = json.loads((run / "log.json").read_text())
    vals = [e for e in log["log"] if "val" in e]
    rep = {g: json.loads((WORK / f"report_{g}.json").read_text()).get(v) for g in ("heldout", "random")}
    if not all(rep.values()): raise SystemExit(f"{v}: battle reports missing; run report --grid heldout/random first")
    pp = lambda c: "—" if not c else f"{100*c['mean']:+.1f} [{100*c['lo']:+.1f}, {100*c['hi']:+.1f}]"
    lines = [x["line"] for x in json.loads((OUT / "plan.json").read_text())["meta"]]
    ex = next(e for e in json.loads((OUT / "plan.json").read_text())["train"] if e["mask"]["task"] == "items" and e["mask"]["k"] == 1)
    user = user_text(v, ex["team"], ex["mask"], lines[:2] + ["…"], 0.50)
    ans = answer_text(v, ex["team"], ex["mask"])
    shape = ("the whole completed team, six blocks of species @ item, ability, nature and four moves, without Stat Points "
             "or level; keep only the blanks' values and reject the answer if an unmasked field comes back changed"
             if v in MASK_LAB else "a JSON list of fills, one per requested path, in the form shown")
    ling = {"regmb": ("regmb@16384", 149), "baseline": ("baseline@16384", 113)}.get(v)
    h, r = rep["heldout"]["all"], rep["random"]["all"]
    row = lambda g, o, base: (f"| {g} | {o['legal']}/{o['n']} | {pp(o['vs_original'])} | {pp(o.get('vs_text'))} | "
                              + (pp(o.get(f'vs_ling_{v}')) if ling and g.startswith('five random') else "not run") + " |")
    card = f"""---
license: apache-2.0
base_model: Efficient-Large-Model/Fast_dLLM_v2_1.5B
base_model_relation: adapter
library_name: peft
pipeline_tag: text-generation
language:
- en
tags:
- lora
- peft
- diffusion-language-model
- block-diffusion
- pokemon
- vgc
- pokemon-showdown
---

# Fast-dLLM v2 1.5B LoRA: Pokémon VGC team blank-filler with Ling's `{v}` prompt (Champions Regulation M-B)

A LoRA adapter for [Fast-dLLM v2 1.5B](https://huggingface.co/Efficient-Large-Model/Fast_dLLM_v2_1.5B), a block-diffusion
language model adapted from Qwen2.5-1.5B-Instruct. Given a team with blanks (items, abilities, natures, moves or whole
Pokémon), it fills them for `gen9championsvgc2026regmb` (Champions VGC 2026, Regulation Set M-B). Its prompt copies the
`{v}` prompt version the [Battle Factory of Porygon-Pi](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi) project
sent to an LLM (`inclusionai/ling-3.0-flash-vl`) in its masked-completion baseline. Sibling adapters: the short-prompt text
writer [Rami/fast-dllm-v2-1.5b-vgc-regmb-lora](https://huggingface.co/Rami/fast-dllm-v2-1.5b-vgc-regmb-lora) and one writer per
LLM prompt version (`{HUB.format("<version>")}`).

## Prompt and output

System message: `system.txt` in this repo, the LLM's `{v}` text with only the answer-format sentence changed.

User message: the LLM's request with each meta team as one line of its six species (`meta_teams.txt` holds the 49 lines)
and a `target_win_rate` (two decimals or `unknown`). One training row, its meta list shortened:

```
{user}
```

Answer: {shape}:

```
{ans}
```

Add Stat Points yourself and validate the team with Showdown before use.

## How to use

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base, rev = "Efficient-Large-Model/Fast_dLLM_v2_1.5B", "25093b6f63300adfd57f72145083c8a528fe4f16"
tok = AutoTokenizer.from_pretrained(base, revision=rev, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base, revision=rev, trust_remote_code=True, dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, "{repo}").merge_and_unload().eval()
msgs = [{{"role": "system", "content": open("system.txt").read()}}, {{"role": "user", "content": user_message}}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
out = model.generate(ids, max_new_tokens={320 if v in MASK_LAB else 448}, threshold=0.9, block_size=32, small_block_size=8, temperature=1.0, top_p=1.0)
print(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))
```

The project fills with classifier-free guidance 2 (the request with `"0.50"` and with `"unknown"` in one batch, logits
combined as 2·cond − 1·uncond at every step); plain `generate` is guidance 1.

## Training

8,000 rows from the project's Regulation M-B teams, none of them a team ranked 51 to 100 (its held-out set): 4,000 list
all 49 distinct top-50 meta teams with the team's win rate against them, 4,000 come from teams with a recorded battle
against every top-50 team and list a random 8 to 49 of them with the win rate against those. Each row blanks one cell of
the project's Mask Lab grid (items, abilities, natures or whole Pokémon k = 1 to 6; moves k = 1 to 24). LoRA rank 32,
alpha 64, every attention and MLP projection; 2,000 steps of 4 examples; learning rate 2e-4, 3% warm-up, linear decay;
bf16; Fast-dLLM v2's block-diffusion loss; full gradient checkpointing; one Apple M4 Pro, {log.get('minutes', 0) / 60:.1f} hours.
Validation loss at steps {', '.join(str(e['step']) for e in vals)}: {', '.join(f"{e['val']:.3f}" for e in vals)} (on the whole answer).

## Evaluation

Win rate is wins against the 49 distinct top-50 meta teams, one battle each, with a behaviour-cloning battle policy on both
sides. The Mask Lab grid blanks 270 cells across five starting teams; a fill is legal if Showdown accepts the team and no
unmasked field changed. Percentage points, paired on the blank, 95% bootstrap over the five starting teams.

| starting teams | legal fills | fill − original | vs the short-prompt writer, both legal | vs the LLM on this prompt, both legal |
|---|---:|---|---|---|
{row('five held-out real teams, ranks 51–100 (originals win 0.48)', h, None)}
{row('five random legal teams (originals win 0.03)', r, None)}

The short-prompt writer had 173 and 149 legal fills on the same blanks{f", the LLM on `{ling[0]}` {ling[1]} on the random teams" if ling else ""}.
A real top-50 team wins 0.531 against the other 49.

## Limitations

Regulation M-B only. No learnset or clause constraints at decoding, so many fills are illegal and need the validator. One
training seed, five starting teams per grid, one battle per opponent, one battle policy. The training rows differ from the
short-prompt writer's, so a comparison between them measures prompt and data together.
"""
    stage = WORK / "hf" / v
    if stage.exists(): shutil.rmtree(stage)
    stage.mkdir(parents=True)
    for f in ("adapter_model.safetensors", "adapter_config.json"): shutil.copy2(run / "final" / f, stage / f)
    (stage / "README.md").write_text(card)
    (stage / "system.txt").write_text(system_text(v))
    (stage / "meta_teams.txt").write_text("\n".join(lines) + "\n")
    api = HfApi(); api.create_repo(repo, private=False, exist_ok=True)
    commit = api.upload_folder(repo_id=repo, folder_path=str(stage), commit_message=f"Mask-fill LoRA trained on Ling's {v} prompt format")
    local = hashlib.sha256((stage / "adapter_model.safetensors").read_bytes()).hexdigest()
    hub = next(x for x in api.model_info(repo, files_metadata=True).siblings if x.rfilename == "adapter_model.safetensors")
    if hub.lfs.sha256 != local: raise SystemExit(f"{repo}: hub SHA-256 differs from the local adapter")
    print(f"uploaded {repo} ({getattr(commit, 'oid', '')[:7]}), sha256 {local[:12]} matches")


LING = {"regmb": "ling:regmb@16384:", "baseline": "ling:baseline@16384:"}   # Ling's arm with the same prompt, random grid

def report(a):
    """Each LLM-format model against the original, random fill, scratch decoder, the first text writer and, on the random
    grid, Ling with the same prompt. Percentage points of win rate vs the 49 top-50 teams, paired on the mask, five
    starting teams weighted equally, 95% bootstrap over them (mask_lab_heldout.cluster)."""
    from mask_lab_heldout import cluster, fmt
    tasks = json.loads((GRIDS[a.grid] / "eval2.tasks.json").read_text())["tasks"]
    panels = dict(json.loads((GRIDS[a.grid] / "panels.json").read_text())["panels"])
    if a.grid == "random":                             # the new arms' panels: same opponents, same battle identity
        extra = WORK / "random-run" / "panels.json"
        if extra.exists(): panels.update(json.loads(extra.read_text())["panels"])
    score = lambda lab: (panels.get(lab) or {}).get("score")
    versions = [v for v in VERSIONS if any(l.startswith(f"llm_{v}:") for l in panels)]
    fam = {}
    for t in tasks:
        r = dict(start=t["source"], original=score(f"orig:{t['source']}"), random=score(f"random:{t['id']}"),
                 diffusion=score(f"diffusion:{t['id']}"), text=score(f"textdiff:{t['id']}"),
                 **{f"llm_{v}": score(f"llm_{v}:{t['id']}") for v in versions},
                 **{f"ling_{v}": score(LING[v] + t["id"]) for v in versions if v in LING and a.grid == "random"})
        fam.setdefault(t["task"], []).append(r)
    groups = list(fam.items()) + [("all", [r for v in fam.values() for r in v])]
    net = lambda r, k: (r[k] if r[k] is not None else r["original"]) - r["original"]
    both = lambda rs, x, y: cluster([(r["start"], r[x] - r[y]) for r in rs if r[x] is not None and r[y] is not None])
    out = {}
    for v in versions:
        k = f"llm_{v}"; refs = ["random", "diffusion", "text"] + ([f"ling_{v}"] if f"ling_{v}" in groups[0][1][0] else [])
        print(f"\n### {v} ({a.grid} starting teams)\n\n| family | legal | − original | " + " | ".join(f"− {x}, both legal" for x in refs)
              + " | net per mask | net − text writer |" + (" net − Ling |" if refs[-1].startswith("ling") else ""))
        print("|---|---:|---|" + "---|" * len(refs) + "---|---|" + ("---|" if refs[-1].startswith("ling") else ""))
        out[v] = {}
        for family, rs in groups:
            o = dict(n=len(rs), legal=sum(r[k] is not None for r in rs),
                     vs_original=cluster([(r["start"], r[k] - r["original"]) for r in rs if r[k] is not None]),
                     **{f"vs_{x}": both(rs, k, x) for x in refs}, net=cluster([(r["start"], net(r, k)) for r in rs]),
                     net_minus_text=cluster([(r["start"], net(r, k) - net(r, "text")) for r in rs]))
            if refs[-1].startswith("ling"): o["net_minus_ling"] = cluster([(r["start"], net(r, k) - net(r, refs[-1])) for r in rs])
            out[v][family] = o
            print(f"| {family} | {o['legal']}/{o['n']} | {fmt(o['vs_original'])} | " + " | ".join(fmt(o[f'vs_{x}']) for x in refs)
                  + f" | {fmt(o['net'])} | {fmt(o['net_minus_text'])} |" + (f" {fmt(o['net_minus_ling'])} |" if 'net_minus_ling' in o else ""))
    (WORK / f"report_{a.grid}.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"\nwrote {WORK / f'report_{a.grid}.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    s = sub.add_parser("stats"); s.add_argument("--sample", type=int, default=400)
    for name in ("generate", "collect"):
        g = sub.add_parser(name); g.add_argument("--version", choices=VERSIONS, required=True)
        g.add_argument("--grid", choices=list(GRIDS), required=True); g.add_argument("--limit", type=int, default=0)
        g.add_argument("--ask", type=float, default=0.5); g.add_argument("--guidance", type=float, default=2.0)
        g.add_argument("--adapter", default="", help="generate: another adapter folder (smoke tests)")
        g.add_argument("--tag", default="", help="suffix for the output folder and arm name; a tagged collect writes to WORK/smoke")
    r = sub.add_parser("report"); r.add_argument("--grid", choices=list(GRIDS), required=True)
    u = sub.add_parser("upload"); u.add_argument("--version", choices=VERSIONS, required=True)
    a = ap.parse_args()
    {"prepare": prepare, "stats": stats, "generate": generate, "collect": collect, "report": report, "upload": upload}[a.cmd](a)
