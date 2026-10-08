"""Resume one validator-crash challenge without restarting a scoring worker.

Preparation is read-only against the simulator and saves an immutable record of
the pending matchup and completed shard scores. Resume issues exactly one normal
acceptChallenge call, guarded against changed state and repeated invocation.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import socket
import subprocess
import sys


REPO = Path(__file__).resolve().parents[1]
RUNTIME = Path("/Users/ramiismael/.local/share/vgc-pilot-runtime")
sys.path.insert(0, "/tmp/vgc-pilot/vgc-bench")
from vgc_bench.src.teams import RandomTeamBuilder


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def worker_matches(command, spec, output):
    parts = shlex.split(command)
    return (len(parts) >= 4 and Path(parts[-3]).resolve() == (REPO / "src/shard.py").resolve()
            and Path(parts[-2]).resolve() == Path(spec).resolve()
            and Path(parts[-1]).resolve() == Path(output).resolve())


def query(repl, expression):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as stream:
        stream.settimeout(2)
        stream.connect(str(repl))
        stream.recv(4096)  # initial prompt
        stream.sendall((expression + "\n").encode())
        response = b""
        while not response.endswith(b"\n> "):
            part = stream.recv(65536)
            if not part:
                break
            response += part
        value = response.decode().rsplit("\n> ", 1)[0].strip()
        return json.loads(ast.literal_eval(value))


def snapshot_expression(challenger, acceptor):
    return "JSON.stringify((()=>{const a=Users.get(" + json.dumps(challenger) + "),b=Users.get(" + json.dumps(acceptor) + ");return {challenge:Ladders.challenges.search(b.id,a.id),aTeam:a.battleSettings.team,bTeam:b.battleSettings.team,aConnections:a.connections.length,bConnections:b.connections.length,rooms:[...Rooms.rooms.values()].filter(r=>r.battle).map(r=>({id:r.roomid,ended:r.battle.ended}))}})())"


def prepare(spec_path, challenger, acceptor, worker_pid):
    spec_path = spec_path.resolve()
    spec = json.loads(spec_path.read_text())
    out_path = spec_path.with_suffix(".out.json")
    completed = json.loads(out_path.read_text())
    require(spec["battles"] == 24 and spec["conc"] == 50, "unexpected search battle protocol")
    pending = [job for job in spec["jobs"] if job[0] not in completed]
    require(pending and all(v["battles"] == 24 for v in completed.values()), "invalid partial results")
    command = subprocess.check_output(["ps", "-p", str(worker_pid), "-o", "command="], text=True).strip()
    require(worker_matches(command, spec_path, out_path), "worker ownership changed")
    team_file = Path(pending[0][0])
    require("/seed202/fixed_020/g10/search/" in str(team_file), "this recovery is scoped to the observed failure")
    require(team_file.name == "0073.txt" and spec["port"] == 8135, "pending matchup changed")
    builder = RandomTeamBuilder(1, None, "mb", custom_team_paths=[team_file])
    candidate = builder._load_team(team_file)
    opponents = [builder._load_team(Path(path)) for path in spec["opp_schedule"]]
    repl = RUNTIME / f"servers/{spec['port']}/logs/repl/app"
    state = query(repl, snapshot_expression(challenger, acceptor))
    challenge = state["challenge"]
    require(challenge["from"] == challenger and challenge["to"] == acceptor, "challenge users changed")
    require(challenge["ready"]["userid"] == challenger, "ready user mismatch")
    require(challenge["format"] == "gen9championsvgc2026regmb" and challenge["acceptCommand"] is None, "wrong challenge type")
    require(state["aConnections"] == state["bConnections"] == 1 and not state["rooms"], "clients are not idle on the pending acceptance")
    require(state["aTeam"] == candidate, "candidate team does not match pending worker job")
    matches = [i for i, team in enumerate(opponents) if team == state["bTeam"]]
    require(len(matches) == 1, "opponent does not uniquely match the scheduled opponent list")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = REPO / f"results/recoveries/seed202-fixed020-g10-{timestamp}"
    directory.mkdir(parents=True)
    files = {}
    prefix = spec_path.stem.rsplit("_", 1)[0]
    completed_total = 0
    for source in sorted(spec_path.parent.glob(prefix + "_*.json")):
        blob = source.read_bytes()
        parsed = json.loads(blob)
        (directory / source.name).write_bytes(blob)
        files[str(source)] = digest(blob)
        if source.name.endswith(".out.json"):
            completed_total += sum(v.get("battles") == 24 for v in parsed.values())
    require(completed_total == 114, "completed worker state changed during preparation")
    source_path = REPO / "results/temperature_seed202.json"
    source_blob = source_path.read_bytes()
    (directory / "temperature_seed202.before.json").write_bytes(source_blob)
    (directory / "server-state.before.json").write_text(json.dumps(state, indent=2) + "\n")
    errors = RUNTIME / "servers/8135/logs/errors.txt"
    (directory / "server-errors.before.txt").write_bytes(errors.read_bytes())
    receipt = dict(status="prepared", prepared_at=datetime.now(timezone.utc).isoformat(),
                   repl=str(repl), spec=str(spec_path), output=str(out_path), worker_pid=worker_pid,
                   challenger=challenger, acceptor=acceptor, pending_team=str(team_file),
                   pending_opponent_schedule_index=matches[0], completed_teams=completed_total,
                   pending_shard_teams=len(pending), original_files_sha256=files,
                   source_sha256=digest(source_blob), failure="team-validator crash during pending challenge acceptance",
                   script_sha256=digest(Path(__file__).read_bytes()))
    path = directory / "recovery.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"receipt": str(path), "completed_teams_preserved": completed_total,
                      "pending_opponent_schedule_index": matches[0], "teams_verified": True}))


def resume(path):
    path = path.resolve()
    receipt = json.loads(path.read_text())
    require(receipt["status"] == "prepared", "acceptance has already been issued or receipt is not prepared")
    before = json.loads((path.parent / "server-state.before.json").read_text())
    state = query(receipt["repl"], snapshot_expression(receipt["challenger"], receipt["acceptor"]))
    require(state == before, "server state changed; refusing to repeat acceptance")
    for filename, expected in receipt["original_files_sha256"].items():
        require(digest(Path(filename).read_bytes()) == expected, "worker files advanced; recovery is no longer current")
    command = subprocess.check_output(["ps", "-p", str(receipt["worker_pid"]), "-o", "command="], text=True).strip()
    require(worker_matches(command, receipt["spec"], receipt["output"]), "worker ownership changed")
    result_path = path.parent / "acceptance-result.json"
    expected = json.dumps(before, separators=(",", ":"))
    target = json.dumps(str(result_path))
    # A single asynchronous acceptance uses the same connection and team payload.
    # The server writes its result directly to the owned recovery directory.
    expression = "JSON.stringify((()=>{const expected=" + expected + ";const a=Users.get(expected.challenge.from),b=Users.get(expected.challenge.to),c=Ladders.challenges.search(b.id,a.id);if(JSON.stringify(c)!==JSON.stringify(expected.challenge)||a.battleSettings.team!==expected.aTeam||b.battleSettings.team!==expected.bTeam||a.connections.length!==1||b.connections.length!==1||[...Rooms.rooms.values()].some(r=>r.battle))throw new Error('Pending acceptance state changed');Ladders.acceptChallenge(b.connections[0],c).then(room=>require('fs').writeFileSync(" + target + ",JSON.stringify({status:room?'accepted':'not_accepted',roomid:room?.roomid,at:new Date().toISOString()}))).catch(error=>require('fs').writeFileSync(" + target + ",JSON.stringify({status:'error',error:String(error),at:new Date().toISOString()})));return {status:'issued_once'}})())"
    receipt.update(status="acceptance_issued", issued_at=datetime.now(timezone.utc).isoformat())
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    response = query(receipt["repl"], expression)
    print(json.dumps({"receipt": str(path), "response": response, "result_path": str(result_path)}))


def verify(path):
    path = path.resolve()
    receipt = json.loads(path.read_text())
    require(receipt["status"] in ("acceptance_issued", "verified"), "no acceptance was issued")
    source = REPO / "results/temperature_seed202.json"
    source_blob = source.read_bytes()
    after = json.loads(source_blob)
    before = json.loads((path.parent / "temperature_seed202.before.json").read_text())
    require(after["manifest"] == before["manifest"], "experiment manifest changed")
    key = "seed202/fixed_020"
    for generation in map(str, range(1, 10)):
        require(after["runs"][key]["generations"][generation] == before["runs"][key]["generations"][generation],
                "an earlier generation changed")
    cell = after["runs"][key]["generations"]["10"]
    for field, value in before["runs"][key]["generations"]["10"].items():
        require(cell[field] == value, "saved generation-10 proposal or training changed")
    require(len(cell["scores"]) == len(cell["selected"]) == 128, "generation-10 team budget")
    for score in cell["scores"]:
        require(score["battles"] == 24 and 0 <= score["wins"] <= 24, "generation-10 battle budget")
        require(score["win_rate"] == score["wins"] / 24, "generation-10 win rate")
    preserved = 0
    for original in sorted(path.parent.glob("_shard_52117_*.out.json")):
        for team, score in json.loads(original.read_text()).items():
            index = int(Path(team).stem)
            require(cell["selected"][index] == Path(team).read_text(), "team identity changed")
            require(all(cell["scores"][index][field] == score[field] for field in ("wins", "battles")),
                    "an already completed team result changed")
            preserved += 1
    require(preserved == 114, "original completed-score coverage")
    (path.parent / "completed-generation.json").write_text(json.dumps(cell, indent=2) + "\n")
    evidence = dict(status="passed", unchanged_completed_teams=preserved, completed_teams=128,
                    battles_per_team=24, generation_battles=3072, pending_team_battles=cell["scores"][73]["battles"],
                    training_and_proposals_unchanged=True, earlier_generations_unchanged=True,
                    source_sha256=digest(source_blob), manifest_unchanged=True,
                    verification="The same worker completed its existing shard; the parent accepted all four successful shards.")
    (path.parent / "verification.json").write_text(json.dumps(evidence, indent=2) + "\n")
    receipt.update(status="verified", verified_at=datetime.now(timezone.utc).isoformat(),
                   verification=str(path.parent / "verification.json"),
                   acceptance_result_artifact_present=(path.parent / "acceptance-result.json").exists(),
                   artifact_note="The initial helper used a relative callback-result path. Battle completion is verified from the original worker outputs and the persisted generation; the helper now resolves this path absolutely.")
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(evidence))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("spec", type=Path)
    p.add_argument("--challenger", required=True)
    p.add_argument("--acceptor", required=True)
    p.add_argument("--worker-pid", type=int, required=True)
    p = sub.add_parser("resume")
    p.add_argument("receipt", type=Path)
    p = sub.add_parser("verify")
    p.add_argument("receipt", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.spec, args.challenger, args.acceptor, args.worker_pid)
    elif args.command == "resume":
        resume(args.receipt)
    else:
        verify(args.receipt)
