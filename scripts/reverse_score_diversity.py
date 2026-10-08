"""Fixed-checkpoint comparison of normal, absent, and reversed score guidance.

No retraining, acquisition filtering, or battles. Every cell receives the same
raw generation budget. Completed cells resume only with an identical manifest.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from signed_score_guidance import sample_direction, SignedGuide

ARMS = ("normal", "none", "reverse")
PRIMARY = "distinct_species_sets"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def jaccard(a, b):
    return 1 - len(a & b) / len(a | b)


def rarefied_distinct(counts, sample_size=128):
    n = sum(counts)
    if n < sample_size:
        return None
    denominator = math.comb(n, sample_size)
    return sum(1 - (math.comb(n - c, sample_size) if n - c >= sample_size else 0)
               / denominator for c in counts)


def species_metrics(combinations, corpus_combinations):
    counts = Counter(tuple(sorted(s)) for s in combinations)
    corpus = {tuple(sorted(s)) for s in corpus_combinations}
    n = sum(counts.values())
    if not n:
        return dict(n=0, distinct_species_sets=0, effective_species_sets=0.,
                    species_set_entropy=None, novel_distinct_species_sets=0,
                    corpus_combination_rate=None, mean_pairwise_jaccard=None,
                    mean_nearest_corpus_jaccard=None, rarefied_distinct_128=None)
    entropy = -sum((c / n) * math.log(c / n) for c in counts.values())
    items = [(frozenset(s), c) for s, c in counts.items()]
    corpus_sets = [frozenset(s) for s in corpus]
    pair_sum = sum(ca * cb * jaccard(a, b)
                   for i, (a, ca) in enumerate(items) for b, cb in items[i + 1:])
    nearest = sum(c * min(jaccard(a, b) for b in corpus_sets) for a, c in items)
    return dict(n=n, distinct_species_sets=len(counts),
                effective_species_sets=math.exp(entropy), species_set_entropy=entropy,
                novel_distinct_species_sets=sum(s not in corpus for s in counts),
                corpus_combination_rate=sum(c for s, c in counts.items() if s in corpus) / n,
                mean_pairwise_jaccard=pair_sum / math.comb(n, 2) if n > 1 else None,
                mean_nearest_corpus_jaccard=nearest / n,
                rarefied_distinct_128=rarefied_distinct(list(counts.values())))


def paired(values):
    result = dict(n=len(values), differences=values, mean=statistics.mean(values))
    if len(values) == 3:
        se = statistics.stdev(values) / math.sqrt(3)
        q = 2 * (1 - .05 / 4) - 1  # Two two-sided primary comparisons, df=2.
        critical = math.sqrt(2) * q / math.sqrt(1 - q * q)
        result["ci95_two_comparisons"] = [result["mean"] - critical * se,
                                           result["mean"] + critical * se]
    return result


class Engine:
    def __init__(self, args):
        import numpy as np
        import torch
        import gradguide as GG
        self.np, self.torch, self.GG = np, torch, GG
        A, D = GG.A, GG.D
        self.cf = A.load_corpus_files()
        self.corpus = [t for _, t in self.cf]
        metadata = json.loads(args.checkpoint_metadata.read_text())
        if digest(args.checkpoint) != metadata["checkpoint_sha256"]:
            raise ValueError("checkpoint does not match its training metadata")
        if len(self.corpus) != metadata["corpus_size"]:
            raise ValueError("checkpoint corpus size changed")
        current_paths = {p for p, _ in self.cf}
        valid_training_paths = {p for p in metadata["corpus"] if len(GG.parse_team(p)) == 6}
        if current_paths != valid_training_paths:
            raise ValueError("checkpoint corpus membership changed")
        for p, expected in metadata["corpus"].items():
            if digest(p) != expected:
                raise ValueError(f"checkpoint corpus changed: {p}")
        # Checkpoint compatibility depends on the architecture and vocabulary,
        # not later edits to independent experiment/reporting drivers.
        for name in ("hpsdiffusion.py", "diffusion.py", "encode.py", "corpus.py"):
            p = ROOT / "src" / name
            if digest(p) != metadata["source"][str(p)]:
                raise ValueError(f"checkpoint model/vocabulary source changed: {p}")
        self.V = A.Vocab(self.corpus)
        self.C = D.Constraints(self.V, A.Legality(self.corpus))
        self.look, self.spreads = A.decode_tables(self.corpus)
        labels = json.loads(args.labels.read_text())
        self.anchor_paths = list(labels["anchor"])
        teams = [GG.parse_team(p) for p in self.anchor_paths]
        teams += [GG.parse_team_text(p) for p in labels["gens"]["gen0"]["pastes"]]
        y = list(labels["anchor"].values()) + labels["gens"]["gen0"]["y"]
        if len(teams) != 328 or len(y) != 328 or any(len(t) != 6 for t in teams):
            raise ValueError("expected the shared 200-anchor plus 128-gen0 label pool")
        self.F, self.coef = GG.fit_ridge(teams, y, self.corpus)
        self.G = GG.Guide(self.F, self.coef, self.V, D.DEV)
        self.model = GG.H.TeamDiffusionHPS(self.V).to(D.DEV)
        self.model.load_state_dict(torch.load(args.checkpoint, map_location=D.DEV)["sd"])
        self.model.eval()
        self.corpus_sets = [A.species_set(t) for t in self.corpus]
        os.environ["PATH"] = str(args.node.parent) + os.pathsep + os.environ.get("PATH", "")
        self.validator = GG.Validator()

    def close(self):
        self.validator.close()

    def manifest(self, args):
        import importlib.metadata
        from hps_generate import SHOWDOWN
        paths = set(ROOT.joinpath("src").glob("*.py"))
        paths.update(Path(p) for p, _ in self.cf)
        paths.update(map(Path, self.anchor_paths))
        paths.update(SHOWDOWN.joinpath("dist").rglob("*.js"))
        paths.update([Path(__file__), args.checkpoint, args.checkpoint_metadata,
                      args.labels, SHOWDOWN / "validate-teams-batch.js"])
        return dict(protocol="reverse-score-diversity-v1", attempts=args.attempts,
                    seeds=args.seeds, magnitude=args.magnitude, chunk=args.chunk,
                    arms=list(ARMS), guidance="gloss", decoding_temperature=1.,
                    checkpoint=str(args.checkpoint), labels=str(args.labels),
                    surrogate_labels=328, corpus_teams=len(self.corpus),
                    surrogate_coef_sha256=hashlib.sha256(self.coef.tobytes()).hexdigest(),
                    device=self.GG.D.DEV, torch=importlib.metadata.version("torch"),
                    numpy=importlib.metadata.version("numpy"),
                    node=subprocess.check_output([str(args.node), "--version"], text=True).strip(),
                    training_steps=0, measured_battles=0,
                    inputs={str(p): digest(p) for p in sorted(paths)})

    def controls(self, magnitude):
        torch, GG = self.torch, self.GG
        seed, n = 940001, 4
        with torch.no_grad():
            torch.manual_seed(seed)
            original, _ = GG.sample_guided(self.model, self.C, self.G, n, "gloss", magnitude)
            torch.manual_seed(seed)
            positive, _ = sample_direction(GG.sample_guided, self.model, self.C, self.G, n, magnitude)
            torch.manual_seed(seed)
            original_none, _ = GG.sample_guided(self.model, self.C, self.G, n, "none", 0.)
            torch.manual_seed(seed)
            zero, _ = sample_direction(GG.sample_guided, self.model, self.C, self.G, n, 0.)
            assert torch.equal(original, positive), "normal control changed the frozen sampler"
            assert torch.equal(original_none, zero), "zero control differs from no guidance"
            x = torch.zeros(n, GG.D.COLS, dtype=torch.long, device=GG.D.DEV)
            tt = torch.ones(n, device=GG.D.DEV)
            w = torch.full((n,), GG.H.WNULL, dtype=torch.long, device=GG.D.DEV)
            h = self.model(x, tt, w)
            c = GG.D.ORDER[0]
            t = self.G.tilt(self.model, h, x, c, "gloss")
            gap = self.G.gap(self.model, h, x).unsqueeze(1)
            opposite = SignedGuide(self.G, -1)
            delta = magnitude * gap * t
            reversed_delta = magnitude * opposite.gap(self.model, h, x).unsqueeze(1) * opposite.tilt(self.model, h, x, c, "gloss")
            assert torch.equal(delta, -reversed_delta), "guidance is not exactly antisymmetric"
            assert float(delta.abs().max()) > 0, "sign check was vacuous"
        return dict(status="passed", control_seed=seed, control_samples=n,
                    normal_matches_original=True, zero_matches_original=True,
                    reversal_is_exact=True, maximum_initial_logit_tilt=float(delta.abs().max()))

    def cell(self, seed, arm, args):
        GG, np, torch = self.GG, self.np, self.torch
        strength = {"normal": args.magnitude, "none": 0., "reverse": -args.magnitude}[arm]
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        records, diagnostics = [], []
        start = time.perf_counter()
        for offset in range(0, args.attempts, args.chunk):
            size = min(args.chunk, args.attempts - offset)
            with torch.no_grad():
                x, diag = sample_direction(GG.sample_guided, self.model, self.C, self.G, size, strength)
            diagnostics.append(dict(offset=offset, entries=diag))
            for row in x.cpu().numpy():
                paste = GG.A.row_to_paste(self.V, row, self.look, self.spreads, rng)
                team = GG.parse_team_text(paste)
                assert len(team) == 6
                error = self.validator(paste)
                records.append(dict(index=len(records), paste=paste,
                                    species=list(GG.A.species_set(team)), valid=error is None,
                                    validation_error=error,
                                    surrogate_score=float((self.F.mat([team]) @ self.coef)[0])))
            print(f"seed{seed}/{arm}: {len(records)}/{args.attempts} attempts", flush=True)
        valid = [r for r in records if r["valid"]]
        return dict(seed=seed, arm=arm, strength=strength, records=records,
                    raw=species_metrics([r["species"] for r in records], self.corpus_sets),
                    legal=species_metrics([r["species"] for r in valid], self.corpus_sets),
                    legality_rate=len(valid) / len(records),
                    mean_surrogate_score=statistics.mean(r["surrogate_score"] for r in records),
                    mean_legal_surrogate_score=statistics.mean(r["surrogate_score"] for r in valid) if valid else None,
                    diagnostics=diagnostics, seconds=time.perf_counter() - start)


def summarize(data):
    if data["status"] != "complete":
        raise ValueError("full configured experiment must complete before analysis")
    m, runs = data["manifest"], data["runs"]
    expected = {f"seed{s}/{a}" for s in m["seeds"] for a in ARMS}
    assert set(runs) == expected
    for key, row in runs.items():
        assert len(row["records"]) == m["attempts"], key
        assert row["legal"]["n"] == sum(r["valid"] for r in row["records"]), key
        assert row["strength"] == {"normal": m["magnitude"], "none": 0., "reverse": -m["magnitude"]}[row["arm"]]
    means = {}
    for arm in ARMS:
        rows = [runs[f"seed{s}/{arm}"] for s in m["seeds"]]
        means[arm] = {metric: statistics.mean(r["legal"][metric] for r in rows)
                      for metric in ("distinct_species_sets", "effective_species_sets", "novel_distinct_species_sets")}
        means[arm]["legality_rate"] = statistics.mean(r["legality_rate"] for r in rows)
        for metric in ("mean_nearest_corpus_jaccard", "rarefied_distinct_128"):
            values = [r["legal"][metric] for r in rows]
            means[arm][metric] = statistics.mean(values) if all(v is not None for v in values) else None
        values = [r["mean_legal_surrogate_score"] for r in rows]
        means[arm]["mean_legal_surrogate_score"] = statistics.mean(values) if all(v is not None for v in values) else None
    comparisons = {}
    for other in ("normal", "none"):
        differences = [runs[f"seed{s}/reverse"]["legal"][PRIMARY] - runs[f"seed{s}/{other}"]["legal"][PRIMARY] for s in m["seeds"]]
        comparisons[other] = paired(differences)
    return dict(means=means, reverse_minus_comparator=comparisons,
                primary_endpoint="distinct legal six-species combinations per fixed raw attempt budget",
                inference_unit="generation seed, conditional on one fixed checkpoint and one fixed surrogate",
                measured_battles=0)


def write_report(path, data):
    summary = summarize(data)
    summary["raw_result_sha256"] = digest(path)
    summary["analysis_source_sha256"] = digest(__file__)
    atomic_json(path.with_suffix(".analysis.json"), summary)
    m = data["manifest"]
    lines = ["# Reversed score guidance: diversity experiment", "",
             f"Completed {len(m['seeds'])} seeds × 3 directions × {m['attempts']} raw generation attempts. No retraining or battles.", "",
             "The reversed arm negates the existing gloss logit tilt while retaining its original nonnegative gap multiplier. The no-guidance arm distinguishes reversal from simply removing positive guidance.", "",
             "## Means across seeds", "",
             "| Direction | Legal attempts | Distinct legal combinations | Effective legal combinations | Novel distinct legal combinations | Distinct in 128 legal draws |",
             "|---|---:|---:|---:|---:|---:|"]
    for arm, row in summary["means"].items():
        rare = row["rarefied_distinct_128"]
        lines.append(f"| {arm} | {row['legality_rate']:.1%} | {row['distinct_species_sets']:.1f} | {row['effective_species_sets']:.1f} | {row['novel_distinct_species_sets']:.1f} | {rare:.1f} |" if rare is not None else
                     f"| {arm} | {row['legality_rate']:.1%} | {row['distinct_species_sets']:.1f} | {row['effective_species_sets']:.1f} | {row['novel_distinct_species_sets']:.1f} | unavailable |")
    lines += ["", "Novel combinations are absent from the checkpoint's training corpus. Species combinations ignore slot order, moves, items, and stat spreads. The primary endpoint counts unique legal combinations per fixed attempt budget; invalid draws are retained and never replaced. Rarefaction reports the exact expected distinct combinations in 128 valid draws without replacement, and is unavailable when a cell has fewer than 128 valid draws.", "",
              "## Individual seeds", "", "| Seed | Direction | Legal / attempted | Distinct legal combinations | Novel distinct legal combinations |", "|---|---|---:|---:|---:|"]
    for seed in m["seeds"]:
        for arm in ARMS:
            r = data["runs"][f"seed{seed}/{arm}"]
            lines.append(f"| {seed} | {arm} | {r['legal']['n']} / {m['attempts']} | {r['legal'][PRIMARY]} | {r['legal']['novel_distinct_species_sets']} |")
    lines += ["", "## Reversed minus comparator", "", "| Comparator | Mean difference in distinct legal combinations | Seed differences | Adjusted 95% paired t interval |", "|---|---:|---|---|"]
    for arm, row in summary["reverse_minus_comparator"].items():
        ci = row.get("ci95_two_comparisons")
        interval = f"[{ci[0]:+.1f}, {ci[1]:+.1f}]" if ci else "not computed"
        lines.append(f"| {arm} | {row['mean']:+.1f} | {', '.join(f'{v:+d}' for v in row['differences'])} | {interval} |")
    lines += ["", "Three-seed t intervals are exploratory and require a normal-effects assumption; a Bonferroni correction covers the two primary comparisons. Other metrics are descriptive. The measured effect applies to this checkpoint, this frozen surrogate, the chosen guidance magnitude, and these generation seeds. Surrogate scores in the JSON are predictions, not measured battle win rates.", "",
              f"![Diversity and legality comparison]({path.with_suffix('.png')})", "",
              f"[Raw proposals and validation outcomes]({path}) · [Analysis JSON]({path.with_suffix('.analysis.json')}) · [Protocol]({ROOT / 'docs/reverse-score-diversity.md'})"]
    path.with_suffix(".md").write_text("\n".join(lines) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    colors = ("#21618c", "#777777", "#b94735")
    for ax, (metric, title) in zip(axes.flat, [(PRIMARY, "Distinct legal species combinations"),
            ("effective_species_sets", "Effective legal species combinations"),
            ("novel_distinct_species_sets", "Novel distinct legal combinations"),
            ("legality_rate", "Legal generation attempts (%)")]):
        for i, (arm, color) in enumerate(zip(ARMS, colors)):
            values = [(100 * data['runs'][f'seed{s}/{arm}'][metric] if metric == 'legality_rate' else data['runs'][f'seed{s}/{arm}']['legal'][metric]) for s in m['seeds']]
            ax.scatter([i] * len(values), values, color=color)
            ax.plot([i-.2, i+.2], [statistics.mean(values)] * 2, color=color, linewidth=2)
        ax.set_xticks(range(3), ["Normal", "No guidance", "Reversed"])
        ax.set_title(title)
        ax.set_ylim(bottom=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
    fig.suptitle(f"Score direction and generated diversity\n{m['attempts']} fixed attempts per cell; dots are seeds, bars are means")
    fig.savefig(path.with_suffix(".png"), dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "results/temperature_p0.pt")
    parser.add_argument("--checkpoint-metadata", type=Path, default=ROOT / "results/temperature_p0.training.json")
    parser.add_argument("--labels", type=Path, default=ROOT / "results/activesearch.json")
    parser.add_argument("--node", type=Path, default=Path('/Users/ramiismael/.nvm/versions/node/v22.22.0/bin/node'))
    parser.add_argument("--output", type=Path, default=ROOT / "results/reverse_score_diversity.json")
    parser.add_argument("--attempts", type=int, default=512)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1101, 1202, 1303])
    parser.add_argument("--magnitude", type=float, default=108.)
    parser.add_argument("--chunk", type=int, default=48)
    args = parser.parse_args()
    for name in ("checkpoint", "checkpoint_metadata", "labels", "node", "output"):
        setattr(args, name, getattr(args, name).resolve())
    if args.attempts < 2 or args.chunk < 1 or args.magnitude <= 0 or not math.isfinite(args.magnitude) or len(set(args.seeds)) != len(args.seeds):
        parser.error("positive finite magnitude, unique seeds, at least 2 attempts, and positive chunk required")
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        engine = Engine(args)
        try:
            manifest = engine.manifest(args)
            if args.output.exists():
                data = json.loads(args.output.read_text())
                if data["manifest"] != manifest:
                    raise ValueError("existing output has different configuration or input fingerprints")
            else:
                data = dict(status="running", manifest=manifest, created_at=datetime.now(timezone.utc).isoformat(), runs={})
            if "controls" not in data:
                data["controls"] = engine.controls(args.magnitude)
                print("Real-model sign and sampler controls passed", flush=True)
                atomic_json(args.output, data)
            for i, seed in enumerate(args.seeds):
                order = ARMS[i % 3:] + ARMS[:i % 3]
                for arm in order:
                    key = f"seed{seed}/{arm}"
                    if key in data["runs"]:
                        continue
                    data["active_case"] = key
                    atomic_json(args.output, data)
                    data["runs"][key] = engine.cell(seed, arm, args)
                    data.pop("active_case", None)
                    atomic_json(args.output, data)
            data["status"] = "complete"
            data.setdefault("completed_at", datetime.now(timezone.utc).isoformat())
            atomic_json(args.output, data)
            write_report(args.output, data)
            print(f"COMPLETE: {args.output}", flush=True)
        finally:
            engine.close()


if __name__ == "__main__":
    main()
