"""Persist reusable smoothness experiments alongside the sparse matchup matrix.

Pool-level panels never become pairwise cells: they include self matchups and do
not contain per-opponent attribution. Exact battled pastes retain slot/move order.
Import is incremental and idempotent; export needs only SQLite and this module.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import sys

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = Path.home() / "vgc-data/matchup_regmb.sqlite"
SCHEMA = """
CREATE TABLE IF NOT EXISTS smoothness_run (
 run_id TEXT PRIMARY KEY, format_id TEXT NOT NULL, parent_run TEXT,
 source_directory TEXT NOT NULL, manifest_json TEXT NOT NULL,
 analysis_json TEXT, benchmark_json TEXT, status TEXT NOT NULL,
 expected_panels INTEGER NOT NULL, imported_at TEXT NOT NULL,
 FOREIGN KEY(parent_run) REFERENCES smoothness_run(run_id)
);
CREATE TABLE IF NOT EXISTS smoothness_anchor (
 run_id TEXT NOT NULL REFERENCES smoothness_run, anchor_id TEXT NOT NULL,
 group_name TEXT NOT NULL, source_id TEXT NOT NULL, cluster_id TEXT NOT NULL,
 record_json TEXT NOT NULL, PRIMARY KEY(run_id,anchor_id)
);
CREATE TABLE IF NOT EXISTS smoothness_candidate (
 run_id TEXT NOT NULL, candidate_id TEXT NOT NULL, anchor_id TEXT NOT NULL,
 matrix_team_id INTEGER REFERENCES team(team_id), kind TEXT NOT NULL,
 exact_paste TEXT NOT NULL, exact_sha256 TEXT NOT NULL, edit_json TEXT,
 PRIMARY KEY(run_id,candidate_id),
 FOREIGN KEY(run_id,anchor_id) REFERENCES smoothness_anchor(run_id,anchor_id)
);
CREATE TABLE IF NOT EXISTS smoothness_opponent (
 run_id TEXT NOT NULL REFERENCES smoothness_run, ordinal INTEGER NOT NULL,
 source_id TEXT NOT NULL, matrix_team_id INTEGER REFERENCES team(team_id),
 exact_paste TEXT NOT NULL, exact_sha256 TEXT NOT NULL, record_json TEXT NOT NULL,
 PRIMARY KEY(run_id,ordinal)
);
CREATE TABLE IF NOT EXISTS smoothness_panel (
 run_id TEXT NOT NULL, candidate_id TEXT NOT NULL, replicate INTEGER NOT NULL,
 seed INTEGER NOT NULL, wins INTEGER NOT NULL, battles INTEGER NOT NULL,
 record_json TEXT NOT NULL, record_sha256 TEXT NOT NULL,
 PRIMARY KEY(run_id,candidate_id,replicate),
 FOREIGN KEY(run_id,candidate_id) REFERENCES smoothness_candidate(run_id,candidate_id),
 CHECK(battles>0 AND wins>=0 AND wins<=battles)
);
CREATE TABLE IF NOT EXISTS smoothness_edge (
 run_id TEXT NOT NULL, candidate_id TEXT NOT NULL,
 baseline_run TEXT NOT NULL, baseline_candidate TEXT NOT NULL,
 delta REAL NOT NULL, ci_low REAL NOT NULL, ci_high REAL NOT NULL,
 corrected_mse REAL NOT NULL, record_json TEXT NOT NULL,
 PRIMARY KEY(run_id,candidate_id),
 FOREIGN KEY(run_id,candidate_id) REFERENCES smoothness_candidate(run_id,candidate_id),
 FOREIGN KEY(baseline_run,baseline_candidate) REFERENCES smoothness_candidate(run_id,candidate_id),
 CHECK(ci_low<=ci_high)
);
CREATE TABLE IF NOT EXISTS smoothness_benchmark (
 run_id TEXT NOT NULL REFERENCES smoothness_run, domain TEXT NOT NULL,
 step REAL NOT NULL, fixed_scale REAL NOT NULL, record_json TEXT NOT NULL,
 PRIMARY KEY(run_id,domain,step)
);
CREATE TABLE IF NOT EXISTS smoothness_artifact (
 run_id TEXT NOT NULL REFERENCES smoothness_run, name TEXT NOT NULL,
 sha256 TEXT NOT NULL, content BLOB NOT NULL,
 PRIMARY KEY(run_id,name,sha256)
);
CREATE TABLE IF NOT EXISTS smoothness_visual (
 name TEXT NOT NULL, run_id TEXT NOT NULL REFERENCES smoothness_run,
 sha256 TEXT NOT NULL, document_json TEXT NOT NULL,
 PRIMARY KEY(name,run_id,sha256)
);
CREATE VIEW IF NOT EXISTS smoothness_coverage AS
 SELECT c.run_id,c.anchor_id,json_extract(c.edit_json,'$.slot') AS pokemon_slot,
 c.candidate_id,c.kind,COUNT(p.replicate) AS completed_panels,
 COALESCE(SUM(p.battles),0) AS battles,COALESCE(SUM(p.wins),0) AS wins,
 e.delta,e.ci_low,e.ci_high
 FROM smoothness_candidate c
 LEFT JOIN smoothness_panel p USING(run_id,candidate_id)
 LEFT JOIN smoothness_edge e USING(run_id,candidate_id)
 WHERE c.edit_json IS NOT NULL
 GROUP BY c.run_id,c.candidate_id;
"""


def packed(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def open_db(path=DEFAULT_DB, readonly=False):
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"existing matchup database required: {path}")
    con = sqlite3.connect(f"file:{path}?mode={'ro' if readonly else 'rw'}", uri=True, timeout=60)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
    row = con.execute("SELECT value FROM meta WHERE key='format_id'").fetchone()
    if not row:
        raise ValueError("not a matchup-matrix database")
    return con


def backup(path):
    path = Path(path).expanduser().resolve()
    destination = path.parent / "backups" / f"{path.stem}.before-smoothness-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}.sqlite"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open_db(path, readonly=True) as source, sqlite3.connect(destination) as target:
        source.backup(target)
    return destination


def matrix_team(con, text, origin):
    # The logical matrix identity is useful for joins; exact_paste is authoritative
    # for this experiment, because matrix canonicalization sorts slots and moves.
    import matchup_db as M
    canon = M.canonical_paste(text)
    digest = M.canon_hash(canon)
    row = con.execute("SELECT team_id FROM team WHERE format_id=? AND canon_hash=?", (M.FORMAT, digest)).fetchone()
    if row:
        return row[0]
    full, base = M.species_keys(canon)
    return con.execute("INSERT INTO team(format_id,canon_hash,paste,species_key,base_key,origin) VALUES(?,?,?,?,?,?)",
                       (M.FORMAT, digest, canon, full, base, origin)).lastrowid


def immutable(con, table, key, values):
    columns = list(key) + list(values)
    where = " AND ".join(f"{column}=?" for column in key)
    old = con.execute(f"SELECT * FROM {table} WHERE {where}", tuple(key.values())).fetchone()
    if old is not None:
        if any(old[column] != value for column, value in values.items()):
            raise ValueError(f"conflicting saved {table} record: {dict(key)}")
        return False
    con.execute(f"INSERT INTO {table}({','.join(columns)}) VALUES({','.join('?' for _ in columns)})", tuple(key.values()) + tuple(values.values()))
    return True


def import_run(con, directory, visual=None):
    sys.path.insert(0, str(ROOT / "scripts"))
    import smoothness_experiment as protocol
    directory = Path(directory).resolve()
    manifest_path = directory / "manifest.json"
    manifest = protocol.verify_manifest(directory)
    run_id = sha(manifest_path.read_bytes())
    if manifest["format"] != con.execute("SELECT value FROM meta WHERE key='format_id'").fetchone()[0]:
        raise ValueError("experiment format differs from this matchup database")
    parent = manifest.get("parent_manifest_sha256")
    if parent and not con.execute("SELECT 1 FROM smoothness_run WHERE run_id=?", (parent,)).fetchone():
        raise ValueError("import the parent experiment before its coverage extension")
    jobs = protocol.job_list(manifest)
    panels = []
    for job in jobs:
        path = directory / "labels" / (job["id"] + ".json")
        if path.exists():
            panels.append(protocol.checked_result(json.loads(path.read_text()), job, run_id))
    analysis_path, benchmark_path = directory / "analysis.json", directory / "benchmarks.json"
    analysis = json.loads(analysis_path.read_text()) if analysis_path.exists() else None
    benchmarks = json.loads(benchmark_path.read_text()) if benchmark_path.exists() else None
    for result in [analysis, benchmarks]:
        if result and result["manifest_sha256"] != run_id:
            raise ValueError("result manifest does not match")
    if analysis and analysis.get("complete") and len(panels) != len(jobs):
        raise ValueError("complete analysis has missing battle panels")
    # Read each immutable source before the transaction; the runner writes panels atomically.
    candidate_text = {c["id"]: Path(c["file"]).read_text() for c in manifest["candidates"]}
    opponent_text = [Path(o["file"]).read_text() for o in manifest["opponents"]]
    artifacts = [("manifest.json", manifest_path.read_bytes())]
    for filename in ["analysis.json", "benchmarks.json", "verification.json", "report.md", "edits.csv"]:
        path = directory / filename
        if path.exists():
            artifacts.append((filename, path.read_bytes()))
    for candidate in manifest["candidates"]:
        artifacts.append(("teams/" + candidate["id"] + ".txt", candidate_text[candidate["id"]].encode()))
    for i, opponent in enumerate(manifest["opponents"]):
        artifacts.append((f"opponents/{i:03d}-{opponent['id']}.txt", opponent_text[i].encode()))
    for panel in panels:
        artifacts.append(("labels/" + panel["job"]["id"] + ".json", packed(panel).encode()))
    if visual:
        text = Path(visual).read_text()
        document = json.loads(text.split("window.SMOOTHNESS_DATA = ", 1)[1].strip().removesuffix(";"))
        if document["manifestSha256"] != run_id:
            raise ValueError("visual snapshot belongs to another experiment")
        artifacts.append(("visual/source.json", packed(document).encode()))
    status = "complete" if analysis and analysis.get("complete") else "panels-complete" if len(panels) == len(jobs) else "partial" if panels else "prepared"
    with con:
        old = con.execute("SELECT manifest_json FROM smoothness_run WHERE run_id=?", (run_id,)).fetchone()
        if old and old[0] != packed(manifest):
            raise ValueError("immutable manifest changed")
        con.execute("""INSERT INTO smoothness_run VALUES(?,?,?,?,?,?,?,?,?,?)
          ON CONFLICT(run_id) DO UPDATE SET
          analysis_json=COALESCE(excluded.analysis_json,smoothness_run.analysis_json),
          benchmark_json=COALESCE(excluded.benchmark_json,smoothness_run.benchmark_json),
          status=excluded.status,imported_at=excluded.imported_at""",
                    (run_id, manifest["format"], parent, str(directory), packed(manifest),
                     packed(analysis) if analysis else None, packed(benchmarks) if benchmarks else None,
                     status, len(jobs), datetime.now(timezone.utc).isoformat()))
        for anchor in manifest["anchors"]:
            immutable(con, "smoothness_anchor", dict(run_id=run_id, anchor_id=anchor["id"]),
                      dict(group_name=anchor["group"], source_id=anchor["source_id"], cluster_id=anchor["cluster"], record_json=packed(anchor)))
        for candidate in manifest["candidates"]:
            text = candidate_text[candidate["id"]]
            immutable(con, "smoothness_candidate", dict(run_id=run_id, candidate_id=candidate["id"]),
                      dict(anchor_id=candidate["anchor"], matrix_team_id=matrix_team(con,text,"smoothness:"+run_id),
                           kind=candidate["kind"], exact_paste=text, exact_sha256=candidate["sha256"],
                           edit_json=packed(candidate["edit"]) if candidate.get("edit") else None))
        for i, opponent in enumerate(manifest["opponents"]):
            immutable(con, "smoothness_opponent", dict(run_id=run_id, ordinal=i),
                      dict(source_id=opponent["id"], matrix_team_id=matrix_team(con,opponent_text[i],"smoothness:"+run_id),
                           exact_paste=opponent_text[i], exact_sha256=opponent["sha256"], record_json=packed(opponent)))
        added = 0
        for panel in panels:
            job = panel["job"]
            added += immutable(con,"smoothness_panel",dict(run_id=run_id,candidate_id=job["candidate"],replicate=job["replicate"]),
                               dict(seed=job["seed"],wins=panel["wins"],battles=panel["battles"],record_json=packed(panel),record_sha256=sha(packed(panel).encode())))
        if analysis:
            for edge in analysis["edges"]:
                baseline_run = edge.get("baseline_experiment", run_id)
                baseline_candidate = edge.get("baseline_candidate", edge["anchor"]+"-base")
                immutable(con,"smoothness_edge",dict(run_id=run_id,candidate_id=edge["candidate"]),
                          dict(baseline_run=baseline_run,baseline_candidate=baseline_candidate,delta=edge["delta"],
                               ci_low=edge["ci95"][0],ci_high=edge["ci95"][1],corrected_mse=edge["corrected_mse"],record_json=packed(edge)))
        if benchmarks:
            for condition in benchmarks["conditions"]:
                immutable(con,"smoothness_benchmark",dict(run_id=run_id,domain=condition["domain"],step=condition["step"]),
                          dict(fixed_scale=condition["fixed_scale"],record_json=packed(condition)))
        for name, content in artifacts:
            con.execute("INSERT OR IGNORE INTO smoothness_artifact VALUES(?,?,?,?)", (run_id,name,sha(content),content))
        if visual:
            con.execute("INSERT OR IGNORE INTO smoothness_visual VALUES(?,?,?,?)", ("smoothness-explorer",run_id,sha(packed(document).encode()),packed(document)))
        # A later import may contain only a subset of the already archived files.
        # Status follows durable records, never regresses to that subset's state.
        saved_count = con.execute('SELECT COUNT(*) FROM smoothness_panel WHERE run_id=?',(run_id,)).fetchone()[0]
        saved_analysis = con.execute('SELECT analysis_json FROM smoothness_run WHERE run_id=?',(run_id,)).fetchone()[0]
        status = 'complete' if saved_analysis and json.loads(saved_analysis).get('complete') else 'panels-complete' if saved_count==len(jobs) else 'partial' if saved_count else 'prepared'
        con.execute('UPDATE smoothness_run SET status=? WHERE run_id=?',(status,run_id))
    return dict(run_id=run_id,status=status,new_panels=added,saved_panels=con.execute("SELECT COUNT(*) FROM smoothness_panel WHERE run_id=?",(run_id,)).fetchone()[0],expected_panels=len(jobs))


def save_reference(con, run_id, name, content):
    with con:
        con.execute("INSERT OR IGNORE INTO smoothness_artifact VALUES(?,?,?,?)",(run_id,name,sha(content),content))


def held_items_from_paste(text):
    """Read battle-start items in exact slot order, including itemless members."""
    return [block.splitlines()[0].partition(' @ ')[2].strip() or None
            for block in re.split(r'\n\s*\n', text.strip()) if block.strip()]


def snapshot(con, run_id=None):
    query = "SELECT * FROM smoothness_visual WHERE name='smoothness-explorer'"
    rows = con.execute(query + (" AND run_id=?" if run_id else "") + " ORDER BY rowid", (run_id,) if run_id else ()).fetchall()
    if not rows or len({row['run_id'] for row in rows}) != 1:
        raise ValueError("specify one visualization run; ambiguous or missing saved visual")
    document = json.loads(rows[-1]["document_json"])
    run_id = rows[-1]["run_id"]
    primary_run = con.execute("SELECT * FROM smoothness_run WHERE run_id=?",(run_id,)).fetchone()
    primary_analysis = json.loads(primary_run['analysis_json'])
    document['summary'] = primary_analysis['summary']
    document['battles'] = primary_analysis['battles']
    document['benchmarks'] = [{k:v for k,v in json.loads(row[0]).items() if k!='edges'} for row in con.execute('SELECT record_json FROM smoothness_benchmark WHERE run_id=? ORDER BY domain,step',(run_id,))]
    reference = con.execute("SELECT content FROM smoothness_artifact WHERE run_id=? AND name='reference/move-names.json'",(run_id,)).fetchone()
    moves = json.loads(reference[0]) if reference else {}
    reference = con.execute("SELECT content FROM smoothness_artifact WHERE run_id=? AND name='reference/item-sprites.json' ORDER BY rowid DESC LIMIT 1",(run_id,)).fetchone()
    item_sprites = json.loads(reference[0]) if reference else {}
    def panels_for(experiment,candidate):
        return [dict(replicate=p['replicate'],wins=p['wins'],battles=p['battles']) for p in con.execute('SELECT * FROM smoothness_panel WHERE run_id=? AND candidate_id=? ORDER BY replicate',(experiment,candidate))]
    for team in document['teams']:
        paste = con.execute('SELECT exact_paste FROM smoothness_candidate WHERE run_id=? AND candidate_id=?',(run_id,team['id']+'-base')).fetchone()[0]
        items = held_items_from_paste(paste)
        if len(items) != len(team['roster']):
            raise ValueError('saved roster and battled paste have different slot counts')
        for pokemon, item in zip(team['roster'], items):
            item_id = re.sub(r'[^a-z0-9]', '', item.lower()) if item else None
            pokemon.update(heldItem=item, itemSprite=item_sprites.get(item_id, {}).get('sprite'))
        saved_names = {e['edit']['new_move']:e['edit'].get('new_name',e['edit']['new_move']) for e in team['edges'] if e.get('edit')}
        team['edges'] = []
        anchor = next(a for a in primary_analysis['anchors'] if a['id']==team['id'])
        team.update(original=anchor['original_win_rate'],corrected=anchor['corrected_mse'],absolute=anchor['observed_mean_absolute_delta'],baselinePanels=panels_for(run_id,team['id']+'-base'))
        for row in con.execute('SELECT e.record_json FROM smoothness_edge e JOIN smoothness_candidate c USING(run_id,candidate_id) WHERE e.run_id=? AND c.anchor_id=? ORDER BY e.candidate_id',(run_id,team['id'])):
            edge=json.loads(row[0])
            if edge.get('edit'):
                edge['edit']['new_name']=moves.get(edge['edit']['new_move'],saved_names.get(edge['edit']['new_move'],edge['edit']['new_move']))
            edge.update(panels=panels_for(run_id,edge['candidate']),experimentId=run_id,primary_sample=True)
            team['edges'].append(edge)
    document["storage"] = {"kind":"matchup-matrix-sqlite","primaryRun":run_id,"sourceOfTruth":"smoothness_run / candidate / panel / edge","exportedAt":datetime.now(timezone.utc).isoformat()}
    document["coverageRunIds"] = []
    for row in con.execute("SELECT run_id FROM smoothness_run WHERE parent_run=? AND status='complete' ORDER BY run_id",(run_id,)):
        coverage_id = row[0]
        document["coverageRunIds"].append(coverage_id)
        for row in con.execute("SELECT * FROM smoothness_edge WHERE run_id=? ORDER BY candidate_id",(coverage_id,)):
            edge = json.loads(row["record_json"])
            edge["edit"]["new_name"] = moves.get(edge["edit"]["new_move"], edge["edit"]["new_move"])
            edge["panels"] = panels_for(coverage_id,edge['candidate'])
            edge["experimentId"] = coverage_id
            next(t for t in document["teams"] if t["id"]==edge["anchor"])["edges"].append(edge)
    ids = [run_id] + document["coverageRunIds"]
    document["primaryBattles"] = document["battles"]
    document["battles"] = con.execute(f"SELECT SUM(battles) FROM smoothness_panel WHERE run_id IN ({','.join('?' for _ in ids)})",ids).fetchone()[0]
    document["measuredMembers"] = sum(len({e["edit"]["slot"] for e in t["edges"] if e.get("edit")}) for t in document["teams"])
    document["displayedMembers"] = sum(len(t["roster"]) for t in document["teams"])
    document["schema"] = 3
    large = []
    for row in con.execute("SELECT run_id,status,expected_panels,manifest_json FROM smoothness_run WHERE parent_run IS NULL AND run_id!=?",(run_id,)):
        manifest = json.loads(row['manifest_json'])
        if len(manifest['anchors']) == 100:
            count = con.execute('SELECT COUNT(*) FROM smoothness_panel WHERE run_id=?',(row['run_id'],)).fetchone()[0]
            large.append(dict(runId=row['run_id'],status=row['status'],savedPanels=count,expectedPanels=row['expected_panels']))
    document['largerStudies'] = large
    return document


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db",type=Path,default=DEFAULT_DB)
    sub = p.add_subparsers(dest="command",required=True)
    imp = sub.add_parser("import-run")
    imp.add_argument("directory",type=Path)
    imp.add_argument("--visual",type=Path)
    imp.add_argument("--backup",action="store_true")
    exp = sub.add_parser("export")
    exp.add_argument("--run")
    exp.add_argument("--output",type=Path,required=True)
    exp.add_argument("--javascript",action="store_true")
    sub.add_parser("list")
    args = p.parse_args()
    if args.command=="import-run":
        if args.backup:
            print(json.dumps({"backup":str(backup(args.db))}),flush=True)
        with open_db(args.db) as con:
            con.executescript(SCHEMA)
            print(json.dumps(import_run(con,args.directory,args.visual)),flush=True)
    elif args.command=="export":
        with open_db(args.db,readonly=True) as con:
            document = snapshot(con,args.run)
        body = json.dumps(document,ensure_ascii=False,indent=2,allow_nan=False)
        args.output.write_text(("// Exported from the matchup matrix database. Do not edit measured values.\nwindow.SMOOTHNESS_DATA = "+body+";\n") if args.javascript else body+"\n")
        print(json.dumps({"output":str(args.output),"teams":len(document["teams"]),"measured_members":document["measuredMembers"],"battles":document["battles"]}))
    else:
        with open_db(args.db,readonly=True) as con:
            print(json.dumps([dict(r) for r in con.execute("SELECT r.run_id,r.status,r.parent_run,r.expected_panels,COUNT(p.replicate) AS saved_panels,SUM(p.battles) AS saved_battles FROM smoothness_run r LEFT JOIN smoothness_panel p USING(run_id) GROUP BY r.run_id")],indent=2))


if __name__=="__main__":
    main()
