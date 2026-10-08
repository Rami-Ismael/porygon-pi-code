"""Post-hoc Pokemon-ID-only diversity analysis of saved direction experiments.

Run: python scripts/pokemon_id_diversity.py results/reverse_score_diversity.json
No generation, retraining, battles, or changes to the frozen experiment inputs.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

TEAM_SIZE = 6
SCORE = "pokemon_id_diversity_percent"
MEASURES = (SCORE, "mean_id_replacements", "mean_shared_ids", "distinct_ids", "effective_ids")


def canonical_ids(ids):
    if not isinstance(ids, (list, tuple)) or len(ids) != TEAM_SIZE:
        raise ValueError("each roster must be a list or tuple of six Pokemon IDs")
    if any(not isinstance(value, str) for value in ids):
        raise ValueError("Pokemon IDs must be strings")
    normalized = [re.sub(r"[^a-z0-9]", "", value.lower()) for value in ids]
    if any(not value or value == "mask" for value in normalized):
        raise ValueError("empty or masked Pokemon IDs are not rosters")
    return tuple(sorted(normalized))


def pokemon_id_metrics(rosters):
    """Average ID replacements between distinct draw indices, preserving repeats.

    For two six-slot rosters, shared = sum_id min(count_A(id), count_B(id)).
    Distance = (6 - shared) / 6. Count thresholds also handle repeated IDs in
    invalid raw proposals without silently deleting slots or replacing attempts.
    """
    slots, incidence = Counter(), Counter()
    n = 0
    for roster in rosters:
        counts = Counter(canonical_ids(roster))
        slots.update(counts)
        for pokemon_id, count in counts.items():
            incidence.update((pokemon_id, level) for level in range(1, count + 1))
        n += 1
    shared = replacements = score = None
    if n >= 2:
        # Each threshold contributes one shared slot to every pair containing it.
        shared = sum(math.comb(count, 2) for count in incidence.values()) / math.comb(n, 2)
        replacements = TEAM_SIZE - shared
        score = 100 * replacements / TEAM_SIZE
    entropy = -sum(count / (TEAM_SIZE * n) * math.log(count / (TEAM_SIZE * n))
                   for count in slots.values()) if n else None
    return dict(n=n, pokemon_id_diversity_percent=score,
                mean_id_replacements=replacements, mean_shared_ids=shared,
                distinct_ids=len(slots), effective_ids=math.exp(entropy) if n else 0.,
                id_slot_counts=dict(sorted(slots.items())))


def analyze(data):
    if data.get("status") != "complete":
        raise ValueError("only completed experiments can be analyzed")
    manifest, runs = data["manifest"], data["runs"]
    arms, seeds, attempts = manifest["arms"], manifest["seeds"], manifest["attempts"]
    if not arms or not seeds or len(set(arms)) != len(arms) or len(set(seeds)) != len(seeds):
        raise ValueError("arms and seeds must be nonempty and unique")
    if not isinstance(attempts, int) or attempts < 1:
        raise ValueError("attempt budget must be positive")
    expected = {f"seed{seed}/{arm}" for seed in seeds for arm in arms}
    if set(runs) != expected:
        raise ValueError("missing or unexpected experiment cells")
    cells = {}
    for seed in seeds:
        for arm in arms:
            key = f"seed{seed}/{arm}"
            run = runs[key]
            rows = run["records"]
            if run["seed"] != seed or run["arm"] != arm:
                raise ValueError("cell identity does not match its key")
            if len(rows) != attempts or [r["index"] for r in rows] != list(range(attempts)):
                raise ValueError("record count or draw indices do not match the budget")
            if any(not isinstance(row["valid"], bool) for row in rows):
                raise ValueError("legality flags must be booleans")
            # Only IDs enter the metric. Stored legality defines the legal subset.
            cells[key] = dict(
                raw=pokemon_id_metrics([r["species"] for r in rows]),
                legal=pokemon_id_metrics([r["species"] for r in rows if r["valid"]]))
    means = {}
    for subset in ("raw", "legal"):
        means[subset] = {}
        for arm in arms:
            values = [cells[f"seed{seed}/{arm}"][subset] for seed in seeds]
            means[subset][arm] = {"mean_teams": statistics.mean(v["n"] for v in values)}
            for metric in MEASURES:
                numbers = [v[metric] for v in values]
                means[subset][arm][metric] = statistics.mean(numbers) if all(v is not None for v in numbers) else None
    contrasts = {}
    if "reverse" in arms:
        for subset in ("raw", "legal"):
            contrasts[subset] = {}
            for arm in arms:
                if arm == "reverse":
                    continue
                differences = []
                for seed in seeds:
                    a, b = cells[f"seed{seed}/reverse"][subset][SCORE], cells[f"seed{seed}/{arm}"][subset][SCORE]
                    differences.append(a - b if a is not None and b is not None else None)
                contrasts[subset][arm] = dict(
                    differences_percentage_points=differences,
                    mean_percentage_points=statistics.mean(differences) if all(v is not None for v in differences) else None)
    return dict(metric_version="pokemon-id-diversity-v1", post_hoc=True,
                arms=arms, seeds=seeds, raw_attempts_per_cell=attempts,
                identity="normalized model species/form ID; different form IDs remain distinct",
                definition="100 * mean_pair(6 - multiset_ID_overlap) / 6",
                ignored_fields=["ev", "iv", "nature", "ability", "moves", "item", "nickname", "slot_order", "surrogate_score"],
                cells=cells, means=means, reverse_minus_comparator=contrasts,
                inference="descriptive paired seed differences; metric added after the original run")


def display(value, decimals=2):
    return "unavailable" if value is None else f"{value:.{decimals}f}"


def render_report(result, source, output):
    lines = ["# Pokemon-ID-only diversity", "",
             "This score uses only the six Pokemon species/form IDs. EVs, IVs, nature, ability, moves, items, nicknames, and slot order do not affect the distance.", "",
             "For each pair of generated teams, count how many Pokemon IDs must be replaced to make their rosters match, then divide by six. The reported score is the average across all pairs, multiplied by 100.", "",
             "| Pair of rosters | ID diversity |", "|---|---:|",
             "| Same six IDs, including a different slot order | 0% |",
             "| Five IDs shared; one replacement | 16.67% |", "| No IDs shared | 100% |", "",
             "Different form IDs count separately, following the model vocabulary. Repeated outputs are retained. Repeated IDs within invalid proposals use multiset overlap. A pairwise score needs at least two teams.", "",
             f"Analysis uses {len(result['seeds'])} seeds and {result['raw_attempts_per_cell']} raw attempts per direction and seed. These are additional, descriptive measurements of the saved experiment; no new teams or battles were generated.", ""]
    names = {"normal": "Normal", "none": "No guidance", "reverse": "Reversed"}
    for subset, title in (("legal", "Legal outputs"), ("raw", "All raw outputs")):
        lines += [f"## {title}", "", "Means across seeds, calculated within each seed's batch.", "",
                  "| Direction | Teams | ID diversity (%) | IDs replaced per pair | Distinct IDs | Effective IDs |",
                  "|---|---:|---:|---:|---:|---:|"]
        for arm in result["arms"]:
            r = result["means"][subset][arm]
            lines.append(f"| {names.get(arm, arm)} | {display(r['mean_teams'], 1)} | {display(r[SCORE])} | {display(r['mean_id_replacements'], 3)} | {display(r['distinct_ids'], 1)} | {display(r['effective_ids'])} |")
        lines += [""]
    lines += ["Effective IDs is the exponential entropy of Pokemon-ID usage across slots: the equivalent number of equally common IDs. Repeating one six-ID team gives six effective IDs but zero roster diversity. Distinct-ID coverage and effective-ID estimates can depend on the number of outputs; the raw view uses the same draw count in every cell.", "",
              "The legal view uses the stored legality flags to select teams; attributes can affect whether a team is legal, but the distance calculation itself reads only Pokemon IDs. The raw view includes every attempt.", "",
              "## Paired seed differences in ID diversity", "",
              "Values are reversed minus comparator, in percentage points. They are descriptive, without a new significance claim.", "",
              "| Subset | Comparator | Mean difference (pp) | Seed differences (pp) |",
              "|---|---|---:|---|"]
    for subset, contrasts in result["reverse_minus_comparator"].items():
        for arm, r in contrasts.items():
            lines.append(f"| {subset} | {names.get(arm, arm)} | {display(r['mean_percentage_points'])} | {', '.join(display(v) for v in r['differences_percentage_points'])} |")
    lines += ["", "## Individual seeds: legal outputs", "",
              "| Seed | Direction | Legal teams | ID diversity (%) | IDs replaced per pair |",
              "|---|---|---:|---:|---:|"]
    for seed in result["seeds"]:
        for arm in result["arms"]:
            r = result["cells"][f"seed{seed}/{arm}"]["legal"]
            lines.append(f"| {seed} | {names.get(arm, arm)} | {r['n']} | {display(r[SCORE])} | {display(r['mean_id_replacements'], 3)} |")
    lines += ["", "This measures overlap of Pokemon IDs, not battle quality or strategic diversity. The average overlap largely reflects which IDs are used and how frequently. It complements the original unique-roster count rather than measuring the fraction of all possible teams explored.", "",
              f"[Metric JSON]({output}) · [Original proposals]({source})", "",
              "## Recompute", "", "From the repository root:", "", "```sh",
              "python scripts/pokemon_id_diversity.py results/reverse_score_diversity.json", "```", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    source = args.results.resolve()
    snapshot = source.read_bytes()
    result = analyze(json.loads(snapshot))
    result["source_sha256"] = hashlib.sha256(snapshot).hexdigest()
    result["analysis_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["generated_at"] = datetime.now(timezone.utc).isoformat()
    receipt = source.with_suffix(".verification.json")
    result["source_legality_audit"] = None
    if receipt.exists():
        verification = json.loads(receipt.read_bytes())
        if verification["status"] != "passed" or verification["result_sha256"] != result["source_sha256"]:
            raise ValueError("source verification receipt does not match the saved experiment")
        result["source_legality_audit"] = dict(path=str(receipt), sha256=hashlib.sha256(receipt.read_bytes()).hexdigest())
    output = source.with_suffix(".pokemon-id-diversity.json")
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    output.with_suffix(".md").write_text(render_report(result, source, output))
    print(f"Analyzed {len(result['cells'])} cells; wrote {output.name} and {output.with_suffix('.md').name}")


if __name__ == "__main__":
    main()
