"""Rebuild one shared p0 after loss of the historical /tmp checkpoint.

Never overwrites an existing checkpoint. All arms/seeds use this same artifact;
the study is conditional on it and does not reproduce the historical p0 exactly.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from temperature_experiment import atomic_json, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO / "results/temperature_p0.pt")
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"checkpoint already exists: {args.output}")
    import entropyloop as E
    corpus = E.A.load_corpus_files()
    teams = [team for _, team in corpus]
    manifest = dict(reason="Historical /tmp/vgc-pilot/activesearch_p0.pt was missing",
                    shared_across_all_arms_and_seeds=True, seed=args.seed, epochs=args.epochs,
                    torch=E.torch.__version__, device=E.D.DEV, corpus_size=len(teams),
                    corpus={file: digest(file) for file, _ in corpus},
                    source={str(file): digest(file) for file in (REPO / "src").glob("*.py")})
    atomic_json(args.output.with_suffix(".training.json"), manifest)
    temp = args.output.with_suffix(".building.pt")
    E.A.P0_CKPT = str(temp)
    t0 = time.monotonic()
    E.A.train_p0(E.A.Vocab(teams), teams, epochs=args.epochs, seed=args.seed)
    os.replace(temp, args.output)
    manifest.update(checkpoint_sha256=digest(args.output), seconds=time.monotonic() - t0)
    atomic_json(args.output.with_suffix(".training.json"), manifest)
    print(f"SHARED_CHECKPOINT_READY {args.output}", flush=True)


if __name__ == "__main__":
    main()
