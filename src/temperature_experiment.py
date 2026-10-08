"""Matched selection-temperature experiment; see docs/temperature-experiment.md.

Run through entropyloop.py compare, or directly. Offline diagnostics and preflight
do not import Torch, the simulator, or the generator's data-dependent modules.
"""
import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import fcntl
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import socket
import sys
import time

import numpy as np
from boltzmann_selection import ARMS, selection_temperature, select_boltz
from diversity_metrics import Reference, measure

REPO = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Config:
    seeds: tuple = (101, 202, 303)
    ports: tuple = (8130, 8131, 8132, 8133)
    generations: int = 11
    anneal_generations: int = 11
    propose: int = 512
    battle: int = 128
    battles: int = 24
    final_sample: int = 128
    final_battles: int = 192
    finalists: int = 16
    epochs: int = 40
    batch: int = 64
    adaptive_ess: float = .50
    checkpoint: str = "/tmp/vgc-pilot/activesearch_p0.pt"
    labels: str = str(REPO / "results/activesearch.json")
    # The existing guided sampler applies softmax(logits) (decode T=1).
    decoding_temperature: float = 1.0
    loss_entropy_bonus: float = 0.0
    guidance: str = "gloss"
    guidance_strength: float = 108.0

    def validate(self):
        for key in ("generations", "anneal_generations", "propose", "battle", "battles", "final_sample",
                    "final_battles", "finalists", "epochs", "batch"):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive")
        if self.battle > self.propose or self.finalists > self.generations * self.battle:
            raise ValueError("battle/proposal or finalist budget is inconsistent")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds) or min(self.seeds) < 0:
            raise ValueError("provide distinct nonnegative seeds")
        if not self.ports or len(set(self.ports)) != len(self.ports) or any(not 1024 <= p <= 65535 for p in self.ports):
            raise ValueError("provide distinct server ports in [1024, 65535]")
        if not 0 < self.adaptive_ess <= 1:
            raise ValueError("adaptive ESS target must be in (0, 1]")
        if self.decoding_temperature != 1 or self.loss_entropy_bonus != 0:
            raise ValueError("this study fixes decoding T=1 and loss entropy bonus=0")

    def budget(self):
        per_arm = self.generations * self.battle * self.battles
        per_arm += (self.final_sample + self.finalists) * self.final_battles
        return dict(arms=list(ARMS), repetitions=len(self.seeds),
                    total_battles=per_arm * len(ARMS) * len(self.seeds),
                    generations=len(ARMS) * len(self.seeds) * self.generations)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def phase_seed(seed, generation, phase):
    # Stable and independent of arm, invocation order, or Python hash salt.
    raw = f"temperature-v1/{seed}/{generation}/{phase}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "little")


def opponent_schedule(opponents, n, seed):
    """Balanced shuffled cycles: same multiset/order for every arm in a block."""
    if not opponents or n < 1:
        raise ValueError("nonempty opponents and positive battle budget required")
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        out.extend(str(opponents[i]) for i in rng.permutation(len(opponents)))
    return out[:n]


def canonical_paste(paste):
    # Ignore slot and move order, retaining item/ability/nature/stat spread.
    slots = []
    for block in re.split(r"\n\s*\n", paste.strip()):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if lines:
            slots.append("\n".join([lines[0]] + sorted(lines[1:])))
    return "\n\n".join(sorted(slots))


def paste_species(paste):
    species = []
    for block in re.split(r"\n\s*\n", paste.strip()):
        name = block.strip().splitlines()[0].split("@")[0].strip()
        name = re.sub(r"\s*\((M|F)\)$", "", name)
        match = re.match(r"^.*\((.+)\)$", name)
        species.append(re.sub(r"[^a-z0-9]", "", (match[1] if match else name).lower()))
    return tuple(sorted(species))


def diversity(pastes):
    """Equal-size proposal stream, including repeats, before score ranking."""
    if not pastes:
        raise ValueError("cannot measure an empty proposal stream")
    sets = [paste_species(p) for p in pastes]
    counts = np.array(list(Counter(sets).values()), dtype=float)
    probabilities = counts / counts.sum()
    entropy = float(-np.dot(probabilities, np.log(probabilities)))
    distances = [1 - len(set(a) & set(b)) / len(set(a) | set(b))
                 for i, a in enumerate(sets) for b in sets[i + 1:]]
    return dict(n=len(pastes), distinct_teams=len({canonical_paste(p) for p in pastes}),
                distinct_species_sets=len(counts), species_set_entropy=entropy,
                effective_species_sets=float(np.exp(entropy)),
                mean_pairwise_species_jaccard_distance=float(np.mean(distances)) if distances else 0.)


def require_scores(results, files, battles):
    """Incomplete simulator jobs must never silently reduce an arm's budget."""
    rows = []
    for file in files:
        r = results.get(file)
        if not r or r.get("battles") != battles:
            raise RuntimeError(f"incomplete battle budget for {file}: {r}")
        wins = r.get("wins")
        if not isinstance(wins, int) or not 0 <= wins <= battles:
            raise RuntimeError(f"invalid win count for {file}: {r}")
        rows.append(dict(wins=wins, battles=battles, win_rate=wins / battles))
    return rows


def preflight(config):
    required = [Path(config.checkpoint), Path(config.labels), Path("/tmp/bc_100.zip"),
                Path("/tmp/vgc-pilot/learnset_true.json"), Path("/tmp/vgc-pilot/top50_evs.json"),
                Path("/tmp/vgc-pilot/vgc-bench/vgc_bench/src/policy_player.py"),
                Path("/tmp/vgc-pilot/vgc-bench/pokemon-showdown/validate-teams-batch.js"),
                Path("/tmp/vgc-pilot/src/shard.py")]
    required += [Path("/tmp/vgc-pilot/data") / f"{name}.json"
                 for name in ("pokedex", "learnsets", "items", "abilities", "moves")]
    missing = [str(p) for p in required if not p.is_file()]
    if not list(Path("/tmp/vgc-pilot/vgc-bench/teams/reg_mb").rglob("*.txt")):
        missing.append("/tmp/vgc-pilot/vgc-bench/teams/reg_mb/*.txt")
    ports = []
    for port in config.ports:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=.1):
                ports.append(port)
        except OSError:
            pass
    return dict(ready=not missing and len(ports) == len(config.ports), missing=missing, live_ports=ports,
                required_ports=list(config.ports),
                budget=config.budget(), note="Imports and checkpoint vocabulary are checked on run.")


def fingerprints(config, opponents):
    files = [Path(config.checkpoint), Path(config.labels), Path("/tmp/bc_100.zip"),
             Path("/tmp/vgc-pilot/learnset_true.json"), Path("/tmp/vgc-pilot/top50_evs.json")]
    files += sorted(Path("/tmp/vgc-pilot/vgc-bench/teams/reg_mb").rglob("*.txt"))
    files += sorted((REPO / "src").glob("*.py"))
    files += [Path("/tmp/vgc-pilot/src/shard.py")]
    bench = Path("/tmp/vgc-pilot/vgc-bench")
    files += sorted((bench / "vgc_bench").rglob("*.py"))
    files += sorted((bench / "data").glob("*.json"))
    files += sorted((bench / "pokemon-showdown/dist").rglob("*.js"))
    files += [Path("/tmp/vgc-pilot/data") / f"{name}.json"
              for name in ("pokedex", "learnsets", "items", "abilities", "moves")]
    return {str(p): digest(p) for p in dict.fromkeys(files + list(map(Path, opponents)))}


class BattleEngine:
    def __init__(self, config):
        import entropyloop as E
        self.E, self.c = E, config
        A, D = E.A, E.D
        self.cf = A.load_corpus_files()
        self.corpus = [team for _, team in self.cf]
        self.diversity_reference = Reference(self.corpus, scope="corpus", format_id="reg_mb",
            provenance={"files": {str(file): digest(file) for file, _ in self.cf}})
        self.V = A.Vocab(self.corpus)
        self.C = D.Constraints(self.V, A.Legality(self.corpus))
        self.grid = np.stack([self.V.encode(t) for t in self.corpus])
        self.look, self.spreads = A.decode_tables(self.corpus)
        self.opponents = A.opponents()
        if len(self.opponents) != 50:
            raise RuntimeError(f"expected 50 meta opponents, found {len(self.opponents)}")
        saved = json.loads(Path(config.labels).read_text())
        self.anchor = [E.parse_team(f) for f in saved["anchor"]]
        self.initial_teams = self.anchor + [E.parse_team_text(p) for p in saved["gens"]["gen0"]["pastes"]]
        self.initial_y = list(saved["anchor"].values()) + saved["gens"]["gen0"]["y"]
        if len(self.initial_teams) != len(self.initial_y) or any(len(t) != 6 for t in self.initial_teams):
            raise RuntimeError("initial labels do not match six-slot teams")
        # Vocabulary compatibility is mandatory; a same-shaped unrelated checkpoint is not substituted.
        check = E.H.TeamDiffusionHPS(self.V).to(D.DEV)
        check.load_state_dict(E.torch.load(config.checkpoint, map_location=D.DEV)["sd"])
        del check
        self.validator = E.Validator()

    def close(self):
        self.validator.close()

    def parse(self, pastes):
        return [self.E.parse_team_text(p) for p in pastes]

    def fit(self, teams, y, seed, generation, arm):
        E, c = self.E, self.c
        temp, tc = selection_temperature(arm, generation, y, c.anneal_generations, c.adaptive_ess)
        k = max(E.ELITE_MIN, int(E.RHO * len(y)))
        idx, weights = select_boltz(y, k, temp, np.random.default_rng(phase_seed(seed, generation, "select")))
        model, steps = E.resteer(self.V, [teams[i] for i in idx], beta=0.,
                                seed=phase_seed(seed, generation, "refit"), batch=c.batch,
                                epochs=c.epochs, checkpoint=c.checkpoint)
        expected = c.epochs * math.ceil(k / c.batch)
        if steps != expected:
            raise RuntimeError(f"training budget mismatch: {steps} != {expected}")
        feats, coef = E.GG.fit_ridge(teams, y, self.corpus)
        guide = E.GG.Guide(feats, coef, self.V, E.D.DEV)
        stats = dict(temperature=temp, concentration=tc, refit_steps=steps, beta=0.,
                     selection=E.selection_stats(idx, weights, teams, y, len(self.anchor)))
        return (model, feats, coef, guide), stats

    def propose(self, fitted, n, directory, seed):
        E, c = self.E, self.c
        model, feats, coef, guide = fitted
        sampling = {}
        files, teams, pastes, validity, diag = E.GG.propose_guided(
            model, self.C, guide, self.V, self.look, self.spreads, self.validator,
            c.guidance, c.guidance_strength, n, directory, seed, sampling_stats=sampling)
        stats = dict(**(diversity(pastes) if pastes else {"n": 0}), validity=validity,
                     memorisation=E.A.memorisation(self.V, teams, self.grid),
                     team_diversity=measure(teams, self.diversity_reference,
                                            expected_n=n, attempts=sampling["attempts"]))
        stats["team_diversity"]["sampling"]["attempt_limit"] = n * 8
        if len(pastes) != n:
            atomic_json(directory / "sampling_failure.json", dict(pastes=pastes, diversity=stats,
                        seed=seed, checkpoint=c.checkpoint, guidance=c.guidance,
                        guidance_strength=c.guidance_strength))
            raise RuntimeError(f"proposal budget incomplete: {len(pastes)} != {n}; sampling_failure.json saved")
        return dict(pastes=pastes, mu=(feats.mat(teams) @ coef).tolist(), diversity=stats)

    def score(self, pastes, directory, n, schedule, seed):
        directory.mkdir(parents=True, exist_ok=True)
        files = []
        for i, paste in enumerate(pastes):
            file = directory / f"{i:04d}.txt"
            file.write_text(paste)
            files.append(str(file))
        res = self.E.A.pool.score([(file, self.opponents) for file in files],
                                 battles=n, conc=50, ports=list(self.c.ports),
                                 opp_schedule=schedule, seed=seed)
        return require_scores(res, files, n)


def load_or_create(path, manifest):
    if path.exists():
        out = json.loads(path.read_text())
        if out["manifest"] != manifest:
            raise ValueError("refusing resume: configuration or input/source fingerprints changed")
        return out
    return dict(manifest=manifest, runs={}, status="running")


def run(config, output, engine, input_hashes):
    """Resume only completed phases; hold-out scores never feed the training pool."""
    manifest = json.loads(json.dumps(dict(schema=1, config=asdict(config), inputs=input_hashes,
                                         arms=ARMS, opponents=engine.opponents)))
    if hasattr(engine, "diversity_reference"):
        ref = engine.diversity_reference
        manifest["diversity_reference"] = json.loads(json.dumps(
            dict(metadata=ref.metadata, normalized_records=ref.snapshot)))
    out = load_or_create(output, manifest)
    work = output.parent / (output.stem + "_teams")
    for seed in config.seeds:
        # Rotate execution order to distribute machine/server time effects.
        order = np.random.default_rng(phase_seed(seed, 0, "arm_order")).permutation(ARMS)
        for arm in order.tolist():
            key = f"seed{seed}/{arm}"
            rec = out["runs"].setdefault(key, dict(seed=seed, arm=arm, generations={}))
            if "final" in rec:
                continue
            teams, y = list(engine.initial_teams), list(engine.initial_y)
            candidates = []
            for generation in range(1, config.generations + 1):
                name = str(generation)
                cell = rec["generations"].setdefault(name, {})
                directory = work / key / f"g{generation}"
                fitted = None
                t0 = time.monotonic()
                if "proposal" not in cell:
                    fitted, cell["training"] = engine.fit(teams, y, seed, generation, arm)
                    cell["proposal"] = engine.propose(fitted, config.propose, directory / "proposal",
                                                     phase_seed(seed, generation, "proposal"))
                    atomic_json(output, out)
                if "scores" not in cell:
                    proposal = cell["proposal"]
                    idx = np.argsort(-np.asarray(proposal["mu"]), kind="stable")[:config.battle]
                    cell["selected"] = [proposal["pastes"][i] for i in idx]
                    bs = phase_seed(seed, generation, "search_battles")
                    cell["schedule"] = opponent_schedule(engine.opponents, config.battles, bs)
                    cell["battle_seed"] = bs
                    cell["scores"] = engine.score(cell["selected"], directory / "search",
                                                  config.battles, cell["schedule"], bs)
                    cell["seconds_this_attempt"] = time.monotonic() - t0
                    atomic_json(output, out)
                # Final generator is p_G: the model trained BEFORE generation G's labels.
                # Restore it from exactly that pool on resume, without an extra refit step.
                if generation == config.generations:
                    if fitted is None:
                        fitted, training = engine.fit(teams, y, seed, generation, arm)
                        if training != cell["training"]:
                            raise RuntimeError("resume refit differs from recorded selection/training budget")
                    final_fitted = fitted
                labels = [r["win_rate"] for r in cell["scores"]]
                candidates.extend(zip(labels, cell["selected"], [generation] * len(labels)))
                teams.extend(engine.parse(cell["selected"]))
                y.extend(labels)
                print(f"{key} g{generation}: mean={np.mean(labels):.4f}, "
                      f"T={cell['training']['temperature']:.4f}, "
                      f"sets={cell['proposal']['diversity']['distinct_species_sets']}", flush=True)
            # Random final-generator samples measure generator win rate without acquisition bias.
            holdout = rec.setdefault("holdout", {})
            if "proposal" not in holdout:
                holdout["proposal"] = engine.propose(final_fitted, config.final_sample,
                    work / key / "holdout_proposals", phase_seed(seed, 0, "holdout_proposals"))
                atomic_json(output, out)
            if "scores" not in holdout:
                bs = phase_seed(seed, 0, "holdout_battles")
                holdout["schedule"] = opponent_schedule(engine.opponents, config.final_battles, bs)
                holdout["battle_seed"] = bs
                holdout["scores"] = engine.score(holdout["proposal"]["pastes"], work / key / "holdout",
                                                  config.final_battles, holdout["schedule"], bs)
                atomic_json(output, out)
            # Preselect finalists on search labels only; never re-rank by hold-out results.
            top, seen = [], set()
            for label, paste, generation in sorted(candidates, key=lambda row: -row[0]):
                canonical = canonical_paste(paste)
                if canonical not in seen:
                    seen.add(canonical)
                    top.append(dict(label=label, paste=paste, generation=generation))
                if len(top) == config.finalists:
                    break
            if len(top) != config.finalists:
                raise RuntimeError("too few unique finalists; cannot silently change battle budget")
            bs = phase_seed(seed, 0, "finalist_battles")
            schedule = opponent_schedule(engine.opponents, config.final_battles, bs)
            rec["final"] = dict(teams=top, schedule=schedule, battle_seed=bs,
                scores=engine.score([t["paste"] for t in top], work / key / "finalists",
                                    config.final_battles, schedule, bs))
            atomic_json(output, out)
    out["status"] = "complete"
    atomic_json(output, out)
    return out


def report(out):
    lines = ["# Selection-temperature comparison", "", f"Status: {out['status']}.", "",
             "Seed-level results; win rates below use fresh battles. Finalists were selected on search labels.", "",
             "| Seed | Arm | Final generator mean | Finalist mean | Original #1 fresh rate | Species sets (holdout) |",
             "|---|---|---:|---:|---:|---:|"]
    means = {}
    for rec in out["runs"].values():
        if "final" not in rec:
            continue
        holdout = rec["holdout"]
        mean = float(np.mean([r["win_rate"] for r in holdout["scores"]]))
        means[(rec["seed"], rec["arm"])] = mean
        final = rec["final"]["scores"]
        lines.append(f"| {rec['seed']} | {rec['arm']} | {mean:.4f} | "
                     f"{np.mean([r['win_rate'] for r in final]):.4f} | {final[0]['win_rate']:.4f} | "
                     f"{holdout['proposal']['diversity']['distinct_species_sets']} |")
    lines += ["", "Paired final-generator differences (annealed minus comparator):", ""]
    for arm in ARMS[1:]:
        differences = [means[(s, "annealed")] - means[(s, arm)]
                       for s in out["manifest"]["config"]["seeds"]
                       if (s, "annealed") in means and (s, arm) in means]
        if differences:
            lines.append(f"- {arm}: mean {np.mean(differences):+.4f}; seed differences "
                         + ", ".join(f"{v:+.4f}" for v in differences))
    lines += ["", "Seeds, not battles or generations, are the independent training replicates. "
              "Three seeds provide an exploratory comparison; a positive mean alone is not evidence of improvement. "
              "Assess diversity jointly using equal-size proposal and holdout samples, effective species sets, "
              "pairwise species distance, and corpus copy rate. Shared policy RNG seeds do not seed Showdown's battle RNG."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "run", "report"), nargs="?", default="preflight")
    parser.add_argument("--output", type=Path, default=REPO / "results/temperature_comparison.json")
    parser.add_argument("--seeds", default="101,202,303")
    parser.add_argument("--ports", default="8130,8131,8132,8133")
    for key in ("generations", "anneal-generations", "propose", "battle", "battles", "final-sample", "final-battles", "finalists", "epochs", "batch"):
        parser.add_argument("--" + key, type=int, default=getattr(Config(), key.replace("-", "_")))
    parser.add_argument("--adaptive-ess", type=float, default=.50)
    parser.add_argument("--checkpoint", default=Config().checkpoint)
    parser.add_argument("--labels", default=Config().labels)
    args = parser.parse_args(argv)
    if args.action == "report":
        result = report(json.loads(args.output.read_text()))
        args.output.with_suffix(".md").write_text(result)
        print(result)
        return
    fields = {k: v for k, v in vars(args).items() if k not in ("action", "output", "seeds", "ports")}
    config = Config(seeds=tuple(int(s) for s in args.seeds.split(",")),
                    ports=tuple(int(p) for p in args.ports.split(",")), **fields)
    config.validate()
    ready = preflight(config)
    if args.action == "preflight" or not ready["ready"]:
        print(json.dumps(ready, indent=2))
        atomic_json(args.output.with_suffix(".preflight.json"), ready)
        if not ready["ready"]:
            raise SystemExit(2)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("another process is already running this output")
        engine = BattleEngine(config)
        try:
            hashes = fingerprints(config, engine.opponents)
            hashes["runtime_versions"] = {name: importlib.metadata.version(name)
                                           for name in ("torch", "numpy", "stable-baselines3", "poke-env")}
            out = run(config, args.output.resolve(), engine, hashes)
            args.output.with_suffix(".md").write_text(report(out))
        finally:
            engine.close()


if __name__ == "__main__":
    main()
