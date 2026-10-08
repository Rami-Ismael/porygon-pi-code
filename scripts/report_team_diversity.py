"""Recompute team diversity on a completed temperature campaign, without battles.

Requires its initial checkpoint training manifest and unchanged corpus files.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from diversity_metrics import Reference, measure


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_reference(manifest_path, config, inputs):
    from corpus import parse_team_text
    manifest = json.loads(manifest_path.read_text())
    checkpoint = config["checkpoint"]
    expected = manifest["checkpoint_sha256"]
    if inputs.get(checkpoint) != expected or digest(checkpoint) != expected:
        raise ValueError("training reference does not match the campaign checkpoint")
    parser_path = str(REPO / "src/corpus.py")
    if manifest["source"].get(parser_path) != digest(parser_path):
        raise ValueError("corpus parser differs from the checkpoint training parser")
    for name in ("pokedex", "learnsets", "items", "abilities", "moves"):
        file = f"/tmp/vgc-pilot/data/{name}.json"
        if inputs.get(file) != digest(file):
            raise ValueError(f"parser data differs from campaign inputs: {file}")
    teams = []
    for file, expected_hash in manifest["corpus"].items():
        if digest(file) != expected_hash:
            raise ValueError(f"training corpus content changed: {file}")
        teams.append(parse_team_text(Path(file).read_text()))
    if len(teams) != manifest["corpus_size"]:
        raise ValueError("reference corpus count mismatch")
    return Reference(teams, scope="initial_training", format_id="reg_mb", provenance=dict(
        training_manifest=str(manifest_path.resolve()), training_manifest_sha256=digest(manifest_path),
        checkpoint_sha256=manifest["checkpoint_sha256"], source_files=manifest["corpus"])), parse_team_text


def analyze(data, reference, parse):
    if data.get("status") != "complete":
        raise ValueError("expected a completed campaign")
    config, manifest = data["manifest"]["config"], data["manifest"]
    expected_runs = {f"seed{seed}/{arm}" for seed in config["seeds"] for arm in manifest["arms"]}
    if set(data["runs"]) != expected_runs:
        raise ValueError("missing or unexpected campaign runs")
    rows = []
    for name, run in sorted(data["runs"].items()):
        if name != f"seed{run['seed']}/{run['arm']}":
            raise ValueError("run identity mismatch")
        if set(run["generations"]) != {str(i) for i in range(1, config["generations"] + 1)}:
            raise ValueError("incomplete generations")
        phases = [(int(g), "generation", cell, config["propose"], config["battle"], config["battles"])
                  for g, cell in run["generations"].items()]
        phases.append((config["generations"], "holdout", run["holdout"],
                       config["final_sample"], config["final_sample"], config["final_battles"]))
        for generation, phase, cell, expected_n, expected_scores, battles in phases:
            pastes = cell["proposal"]["pastes"]
            if len(pastes) != expected_n:
                raise ValueError(f"{name}/{generation}/{phase}: sample-size mismatch")
            scores = cell["scores"]
            if len(scores) != expected_scores or any(
                type(s["wins"]) is not int or not 0 <= s["wins"] <= battles or s["battles"] != battles
                for s in scores):
                raise ValueError(f"{name}/{generation}/{phase}: incomplete battle scores")
            if phase == "generation" and len(cell["selected"]) != expected_scores:
                raise ValueError("selected team count mismatch")
            # Old records preserve an acceptance fraction, not an integer counter.
            # Keep attempts unknown rather than infer exact provenance from rounding.
            metrics = measure([parse(p) for p in pastes], reference, expected_n=expected_n)
            metrics["sampling"]["historical_acceptance_fraction"] = cell["proposal"]["diversity"]["validity"]
            rows.append(dict(run=name, seed=run["seed"], arm=run["arm"], generation=generation,
                phase=phase, metrics=metrics,
                win_rate=dict(population="score_selected" if phase == "generation" else "unranked_generator_sample",
                              teams=len(scores), battles_per_team=battles,
                              mean=sum(s["wins"] for s in scores) / (len(scores) * battles))))
    return rows


def markdown(rows, source):
    arms = list(dict.fromkeys(r["arm"] for r in rows))
    last = max(r["generation"] for r in rows)
    lines = ["# Team diversity on the saved temperature experiment", "",
        f"Source: `{source.name}`. The initial-training reference was verified against its checkpoint and all corpus-file hashes.", "",
        "Every percentage below is a mean across run seeds. Draws retain duplicates; proposal streams precede ranking. "
        "These are representation-level measurements, not tests of strategic generalization.", "",
        "## First and last saved generations", "",
        "Generation 1 is already after fine-tuning. This campaign contains no comparable pre-fine-tuning proposal sample. "
        "Win rate here belongs to score-selected teams, whereas diversity uses all 512 proposals.", "",
        "| Arm | Generation | Unique 48-field | Unique with spreads | Novel with spreads | Compositions | Selected win rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    def mean(group, path):
        vals = []
        for row in group:
            value = row
            for key in path:
                value = value[key]
            vals.append(value)
        return statistics.mean(vals)
    for arm in arms:
        for g in sorted({1, last}):
            group = [r for r in rows if r["arm"] == arm and r["generation"] == g and r["phase"] == "generation"]
            a = mean(group, ("metrics", "categorical_48", "unique_fraction"))
            b = mean(group, ("metrics", "with_spreads", "unique_fraction"))
            c = mean(group, ("metrics", "with_spreads", "novel_draw_fraction"))
            d = mean(group, ("metrics", "composition", "unique_count"))
            w = mean(group, ("win_rate", "mean"))
            lines.append(f"| {arm} | {g} | {a:.1%} | {b:.1%} | {c:.1%} | {d:.1f} | {w:.1%} |")
    lines += ["", "## Independent final generator sample", "",
        "These 128-draw samples have a different sample size from the generation streams; compare arms within this table. "
        "Win rate and diversity describe the same unranked draws.", "",
        "| Arm | Unique with spreads | Novel with spreads | Compositions | Generator win rate |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for arm in arms:
        group = [r for r in rows if r["arm"] == arm and r["phase"] == "holdout"]
        b = mean(group, ("metrics", "with_spreads", "unique_fraction"))
        c = mean(group, ("metrics", "with_spreads", "novel_draw_fraction"))
        d = mean(group, ("metrics", "composition", "unique_count"))
        w = mean(group, ("win_rate", "mean"))
        lines.append(f"| {arm} | {b:.1%} | {c:.1%} | {d:.1f} | {w:.1%} |")
    lines += ["", "The JSON companion includes every generation, run seed, repeat fraction, "
        "novel-unique fraction, composition novelty, complete composition frequency table, and frozen normalized reference. "
        "Historical attempt counters are unavailable; their recorded acceptance fractions are retained separately. "
        "This analysis runs no new battles and changes no historical experiment results.", ""]
    return "\n".join(lines)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("source", type=Path)
    cli.add_argument("--training-manifest", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    if args.output.resolve() in (args.source.resolve(), args.training_manifest.resolve()):
        raise ValueError("analysis must not overwrite input artifacts")
    data = json.loads(args.source.read_text())
    reference, parse = load_reference(args.training_manifest, data["manifest"]["config"], data["manifest"]["inputs"])
    rows = analyze(data, reference, parse)
    report = dict(source=str(args.source.resolve()), source_sha256=digest(args.source),
                  reference=reference.metadata, reference_snapshot=reference.snapshot,
                  metric_source_sha256=digest(REPO / "src/diversity_metrics.py"),
                  analysis_source_sha256=digest(__file__), rows=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    args.output.with_suffix(".md").write_text(markdown(rows, args.source))
    print(f"Verified reference and measured {len(rows)} streams: {args.output}")


if __name__ == "__main__":
    main()
