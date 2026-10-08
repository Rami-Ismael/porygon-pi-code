"""Independently audit the full temperature-study protocol from saved records.

This does not import or modify the frozen training implementation. Partial audits
are explicitly incomplete; only a complete 15-run campaign can pass the full gate.
"""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re


ARMS = ["annealed", "fixed_010", "fixed_020", "fixed_030", "adaptive"]
EXPECTED = dict(seeds=[101, 202, 303], generations=11, anneal_generations=11,
                propose=512, battle=128, battles=24, final_sample=128,
                final_battles=192, finalists=16, epochs=40, batch=64,
                adaptive_ess=.5, decoding_temperature=1., loss_entropy_bonus=0.,
                guidance="gloss", guidance_strength=108.)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, message):
    require(math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-10), message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def team_key(paste):
    return tuple(sorted(tuple([lines[0], *sorted(lines[1:])])
                        for block in re.split(r"\n\s*\n", paste.strip())
                        if (lines := [line.strip() for line in block.splitlines() if line.strip()])))


def species_key(paste):
    names = []
    for block in re.split(r"\n\s*\n", paste.strip()):
        name = block.splitlines()[0].split("@")[0].strip()
        name = re.sub(r"\s*\((M|F)\)$", "", name)
        nickname = re.fullmatch(r".*\((.+)\)", name)
        names.append(re.sub(r"[^a-z0-9]", "", (nickname[1] if nickname else name).lower()))
    require(len(names) == 6 and len(set(names)) == 6, "proposal is not a six-species team")
    return tuple(sorted(names))


def audit_diversity(proposal, n, label):
    pastes, recorded = proposal["pastes"], proposal["diversity"]
    require(len(pastes) == len(proposal["mu"]) == n, f"{label}: proposal count")
    require(all(math.isfinite(v) for v in proposal["mu"]), f"{label}: surrogate score")
    counts = Counter(species_key(p) for p in pastes)
    entropy = -sum((count / n) * math.log(count / n) for count in counts.values())
    require(recorded["n"] == n, f"{label}: diversity sample size")
    require(recorded["distinct_teams"] == len({team_key(p) for p in pastes}), f"{label}: distinct teams")
    require(recorded["distinct_species_sets"] == len(counts), f"{label}: species-set count")
    close(recorded["species_set_entropy"], entropy, f"{label}: entropy")
    close(recorded["effective_species_sets"], math.exp(entropy), f"{label}: effective species sets")
    groups = [(set(names), count) for names, count in counts.items()]
    distance_sum = sum(ca * cb * (1 - len(a & b) / len(a | b))
                       for i, (a, ca) in enumerate(groups) for b, cb in groups[i + 1:])
    close(recorded["mean_pairwise_species_jaccard_distance"],
          distance_sum / (n * (n - 1) / 2), f"{label}: species distance")
    require(0 < recorded["validity"] <= 1, f"{label}: validity")
    memory = recorded["memorisation"]
    require(0 <= memory["copy_rate"] <= 1 and 0 <= memory["nn_hamming"] <= 48,
            f"{label}: memorisation metric bounds")
    require(memory["distinct_species_sets"] == len(counts), f"{label}: memorisation species count")


def weights(labels, temperature):
    peak = max(labels)
    raw = [math.exp((y - peak) / temperature) for y in labels]
    total = sum(raw)
    return [value / total for value in raw]


def ess_fraction(labels, temperature):
    return 1 / sum(w * w for w in weights(labels, temperature)) / len(labels)


def audit_training(training, labels, arm, generation, label):
    temperature = training["temperature"]
    require(.1 <= temperature <= .3, f"{label}: temperature bounds")
    concentration = training["concentration"]
    if arm == "adaptive":
        lo, hi = ess_fraction(labels, .1), ess_fraction(labels, .3)
        expected_status = "lower_bound" if lo >= .5 else "upper_bound" if hi <= .5 else "target"
        require(concentration["status"] == expected_status, f"{label}: adaptive status")
        close(concentration["target_ess_fraction"], .5, f"{label}: adaptive target")
        if expected_status == "target":
            close(ess_fraction(labels, temperature), .5, f"{label}: adaptive concentration target")
        else:
            close(temperature, .1 if expected_status == "lower_bound" else .3, f"{label}: adaptive boundary")
    else:
        expected_t = (.3 - .02 * (generation - 1)) if arm == "annealed" else int(arm[-3:]) / 100
        close(temperature, expected_t, f"{label}: temperature schedule")
        require(concentration["status"] == ("scheduled" if arm == "annealed" else "fixed"), f"{label}: status")
    w = weights(labels, temperature)
    ess = 1 / sum(value * value for value in w)
    for field, expected in dict(ess=ess, ess_fraction=ess / len(w), max_weight=max(w),
                                entropy=-sum(value * math.log(value) for value in w if value)).items():
        close(concentration[field], expected, f"{label}: concentration {field}")
    k = max(100, len(labels) // 4)
    require(training["beta"] == 0, f"{label}: loss entropy bonus")
    require(training["refit_steps"] == 40 * math.ceil(k / 64), f"{label}: training steps")
    require(training["selection"]["k"] == k, f"{label}: resample size")
    close(training["selection"]["ess"], ess, f"{label}: selection ESS")
    close(training["selection"]["real_weight_share"], sum(w[:200]), f"{label}: anchor weight share")


def audit(path, allow_incomplete=False, verify_inputs=True):
    snapshot = Path(path).read_bytes()
    data = json.loads(snapshot)
    manifest, runs = data["manifest"], data["runs"]
    config = manifest["config"]
    for key, value in EXPECTED.items():
        require(config[key] == value, f"protocol config: {key}")
    require(manifest["arms"] == ARMS, "all five temperature arms are required")
    expected_runs = {f"seed{s}/{a}" for s in EXPECTED["seeds"] for a in ARMS}
    require(set(runs) <= expected_runs, "unexpected arm or seed")
    complete = data["status"] == "complete"
    require(allow_incomplete or complete, "campaign is not complete")
    require(not complete or set(runs) == expected_runs, "missing full-study runs")
    opponents = manifest["opponents"]
    require(len(opponents) == len(set(opponents)) == 50, "expected 50 unique opponents")
    initial = json.loads(Path(config["labels"]).read_text())
    require(len(initial["anchor"]) == 200 and len(initial["gens"]["gen0"]["y"]) == 128, "initial pool size")
    initial_y = list(initial["anchor"].values()) + initial["gens"]["gen0"]["y"]
    require(all(0 <= value <= 1 for value in initial_y), "initial pool scores")
    phase_controls, phase_seeds = {}, {}
    generations = final_runs = total_battles = 0

    def audit_phase(cell, n, battles, seed, phase, label):
        nonlocal total_battles
        require(len(cell["scores"]) == n and len(cell["schedule"]) == battles, f"{label}: battle count")
        counts = Counter(cell["schedule"])
        require(set(counts) <= set(opponents), f"{label}: unknown opponent")
        coverage = [counts[o] for o in opponents]
        require(max(coverage) - min(coverage) <= 1, f"{label}: unbalanced opponent schedule")
        pair = (cell["schedule"], cell["battle_seed"])
        control_key = (seed, phase)
        require(phase_controls.setdefault(control_key, pair) == pair, f"{label}: unmatched paired phase")
        require(phase_seeds.setdefault(cell["battle_seed"], control_key) == control_key, f"{label}: reused phase seed")
        for score in cell["scores"]:
            require(type(score["wins"]) is int and 0 <= score["wins"] <= battles, f"{label}: wins")
            require(score["battles"] == battles and score["win_rate"] == score["wins"] / battles, f"{label}: score denominator")
        total_battles += n * battles

    for key, record in runs.items():
        seed, arm = record["seed"], record["arm"]
        require(key == f"seed{seed}/{arm}", f"{key}: run identity")
        cells = record["generations"]
        require(set(cells) <= set(map(str, range(1, 12))), f"{key}: generation range")
        labels, candidates = list(initial_y), []
        for generation in range(1, 12):
            cell = cells.get(str(generation), {})
            if "proposal" not in cell:
                require(not complete, f"{key}: missing generation {generation}")
                break
            label = f"{key}/g{generation}"
            audit_training(cell["training"], labels, arm, generation, label)
            audit_diversity(cell["proposal"], 512, label)
            if "scores" not in cell:
                require(not complete, f"{label}: missing scores")
                break
            proposal = cell["proposal"]
            selected = [proposal["pastes"][i] for i in sorted(range(512), key=lambda i: -proposal["mu"][i])[:128]]
            require(cell["selected"] == selected, f"{label}: surrogate top-128 selection")
            audit_phase(cell, 128, 24, seed, f"g{generation}", label)
            new_labels = [s["win_rate"] for s in cell["scores"]]
            candidates.extend(zip(new_labels, selected, [generation] * 128))
            labels.extend(new_labels)
            generations += 1
        holdout = record.get("holdout", {})
        if "proposal" in holdout:
            require(len(candidates) == 1408, f"{key}: holdout before full search")
            audit_diversity(holdout["proposal"], 128, f"{key}/holdout")
        if "scores" in holdout:
            audit_phase(holdout, 128, 192, seed, "holdout", f"{key}/holdout")
        final = record.get("final", {})
        if "scores" in final:
            require("scores" in holdout, f"{key}: incomplete fresh generator evaluation")
            expected, seen = [], set()
            for win_rate, paste, generation in sorted(candidates, key=lambda row: -row[0]):
                canonical = team_key(paste)
                if canonical not in seen:
                    seen.add(canonical)
                    expected.append(dict(label=win_rate, paste=paste, generation=generation))
                if len(expected) == 16:
                    break
            require(final["teams"] == expected and len(expected) == 16, f"{key}: finalists must be the top unique search-scored teams")
            audit_phase(final, 16, 192, seed, "finalists", f"{key}/finalists")
            final_runs += 1
        else:
            require(not complete, f"{key}: missing fresh finalist evaluation")

    controls = []
    for seed in EXPECTED["seeds"]:
        left = runs.get(f"seed{seed}/annealed", {}).get("generations", {}).get("1", {})
        right = runs.get(f"seed{seed}/fixed_030", {}).get("generations", {}).get("1", {})
        if "proposal" in left and "proposal" in right:
            require(left["proposal"] == right["proposal"], f"seed{seed}: equal-temperature proposal control")
            require(left["training"]["selection"] == right["training"]["selection"], f"seed{seed}: equal-temperature resampling control")
            controls.append(seed)
    if complete:
        require((generations, final_runs, total_battles) == (165, 15, 921600), "full study budget")
        require(controls == EXPECTED["seeds"], "missing equal-temperature controls")
        execution = manifest["execution"]
        require(execution["initial_controller_stopped_after_seed101"], "original controller was not stopped after seed101")
        for seed in EXPECTED["seeds"]:
            source = execution["sources"][str(seed)]
            require(sha(source["path"]) == source["raw_result_sha256"], f"seed{seed}: raw source digest")
            raw = json.loads(Path(source["path"]).read_text())
            require(hashlib.sha256(json.dumps(raw["manifest"], sort_keys=True).encode()).hexdigest() == source["manifest_sha256"], f"seed{seed}: source manifest digest")
            for arm in ARMS:
                key = f"seed{seed}/{arm}"
                require(raw["runs"][key] == runs[key], f"{key}: merged record differs from raw source")
            for field in ("inputs", "arms", "opponents"):
                require(raw["manifest"][field] == manifest[field], f"seed{seed}: source {field}")
            require({k: v for k, v in raw["manifest"]["config"].items() if k not in ("seeds", "ports")} ==
                    {k: v for k, v in config.items() if k != "seeds"}, f"seed{seed}: source protocol")
    input_count = 0
    runtime_checked = []
    if verify_inputs:
        for filename, expected in manifest["inputs"].items():
            if filename == "runtime_versions":
                continue
            require(sha(filename) == expected, f"changed input: {filename}")
            input_count += 1
        versions = manifest["inputs"]["runtime_versions"]
        require(set(versions) == {"torch", "numpy", "stable-baselines3", "poke-env"}, "runtime version coverage")
        for package, expected in versions.items():
            require(importlib.metadata.version(package) == expected, f"changed runtime package: {package}")
            runtime_checked.append(package)
    return dict(status="passed" if complete else "partial_checks_passed", study_complete=complete,
                audited_generations=generations, completed_runs=final_runs, measured_battles=total_battles,
                equal_temperature_controls=controls, input_file_hashes_checked=input_count,
                runtime_versions_checked=runtime_checked,
                source_records_checked=complete, audit_script_sha256=sha(__file__),
                result_sha256=hashlib.sha256(snapshot).hexdigest(),
                diversity_recomputed=["distinct_teams", "distinct_species_sets", "species_set_entropy",
                                      "effective_species_sets", "mean_pairwise_species_jaccard_distance"],
                memorisation_check="bounds and species count; copy/NN values rely on the fingerprinted producer")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--skip-input-hashes", action="store_true")
    args = parser.parse_args()
    receipt = audit(args.results, args.allow_incomplete, not args.skip_input_hashes)
    args.results.with_suffix(".protocol-verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
