"""Frozen, resumable Ling-only masked-completion pilot. See docs/llm-baseline.md.

prepare   freeze the study manifest                (src/completion_study.py)
complete  ask the language model, once per trial   (src/llm_completion.py)
battle    score every candidate's 50-battle panel  (src/battles.py)
report    pair completions with their controls     (src/completion_study.py)"""
import argparse
import fcntl
from pathlib import Path

from repo_paths import ROOT  # also puts src/ first on sys.path
import battles
import completion_study
import llm_completion

COMMANDS = {"prepare": completion_study.prepare, "complete": llm_completion.complete,
            "battle": battles.battle, "report": completion_study.report}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=list(COMMANDS))
    parser.add_argument("--output", type=Path, default=ROOT / "results/llm-baseline-instant-pilot")
    parser.add_argument("--teams", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--shard', type=int, default=0)
    # Kept so an old command asking for the removed legacy engine gets battles.battle's reason.
    parser.add_argument('--battle-backend', help=argparse.SUPPRESS)
    parser.add_argument('--fast-panels', type=int, default=battles.FAST_PANELS, help='battle: panels per fast evaluator call')
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.teams < 1: parser.error("--teams must be positive")
    if not 0 <= args.shard < args.shards: parser.error('invalid shard')
    args.output.mkdir(parents=True, exist_ok=True)
    lock_name = '.' + args.command + (f'-{args.shard}-of-{args.shards}' if args.command=='complete' else '') + '.lock'
    with (args.output / lock_name).open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        COMMANDS[args.command](args)


if __name__ == "__main__":
    main()
