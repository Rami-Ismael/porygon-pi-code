"""Archive held-item sprites for a saved experiment; never starts battles."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import sys
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
import smoothness_db as store


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', type=Path, default=store.DEFAULT_DB)
    parser.add_argument('--run', required=True)
    parser.add_argument('--visual', type=Path, required=True)
    args = parser.parse_args()
    with store.open_db(args.db, readonly=True) as con:
        document = store.snapshot(con, args.run)
        names = {p['heldItem'] for t in document['teams'] for p in t['roster'] if p['heldItem']}
        saved = con.execute("SELECT content FROM smoothness_artifact WHERE run_id=? AND name='reference/item-sprites.json' ORDER BY rowid DESC LIMIT 1", (args.run,)).fetchone()
        reference = json.loads(saved[0]) if saved else {}
        cached = {}
        for item_id, entry in reference.items():
            row = con.execute('SELECT content FROM smoothness_artifact WHERE run_id=? AND name=? AND sha256=?', (args.run, 'visual/assets/' + Path(entry['sprite']).name, entry['sha256'])).fetchone()
            if row:
                cached[item_id] = bytes(row[0])

    ids = {re.sub(r'[^a-z0-9]', '', name.lower()): name for name in names}
    missing = set(ids) - set(cached)
    if missing:
        # Individual source images from the public Pokémon Showdown sprite repository.
        with urlopen('https://api.github.com/repos/smogon/sprites/contents/src/minisprites/items', timeout=30) as response:
            listing = json.load(response)
        assets = {re.sub(r'[^a-z0-9]', '', Path(entry['name']).stem[1:]): entry
                  for entry in listing if entry['name'].startswith('i') and entry['name'].endswith('.png')}
        if missing - assets.keys():
            raise ValueError('No source sprite for: ' + ', '.join(sorted(missing - assets.keys())))

        def fetch(item_id):
            entry = assets[item_id]
            with urlopen(entry['download_url'], timeout=30) as response:
                content = response.read()
            if not content.startswith(b'\x89PNG\r\n\x1a\n'):
                raise ValueError('Expected PNG for ' + item_id)
            return item_id, content, dict(name=ids[item_id], sprite='items/item-' + item_id + '.png',
                source=entry['download_url'], sourceBlobSha=entry['sha'], sha256=store.sha(content))

        with ThreadPoolExecutor(max_workers=6) as pool:
            for item_id, content, entry in pool.map(fetch, sorted(missing)):
                cached[item_id] = content
                reference[item_id] = entry

    # Asset copies support file:// use; the server reads their archived bytes instead.
    (args.visual / 'items').mkdir(exist_ok=True)
    artifacts = []
    for item_id in sorted(ids):
        entry, content = reference[item_id], cached[item_id]
        (args.visual / entry['sprite']).write_bytes(content)
        artifacts.append((args.run, 'visual/assets/' + Path(entry['sprite']).name, store.sha(content), content))
    encoded = (json.dumps(reference, indent=2, ensure_ascii=False) + '\n').encode()
    (args.visual / 'items/sources.json').write_bytes(encoded)
    artifacts.append((args.run, 'reference/item-sprites.json', store.sha(encoded), encoded))
    with store.open_db(args.db) as con:
        con.executemany('INSERT OR IGNORE INTO smoothness_artifact VALUES(?,?,?,?)', artifacts)
    print(json.dumps(dict(items=len(ids), downloaded=len(missing), database=str(args.db))))


if __name__ == '__main__':
    main()
