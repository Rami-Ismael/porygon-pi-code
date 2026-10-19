"""Frozen copies of the result files some tests read.

results/ is gitignored, so a fresh checkout (or the monorepo mirror) has no
results; tests that check behaviour against a finished study read these copies
instead. Each one is the original file gzipped, byte-identical once
decompressed; SHA256SUMS records the original's hash and test_frozen checks it
(and compares with results/ whenever the original is present).
"""
import gzip
import hashlib
import json
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent / 'fixtures'
RESULTS = Path(__file__).resolve().parent.parent / 'results'


def recorded():
    """{results-relative path: sha256 of the original file}"""
    rows = (FIXTURES / 'SHA256SUMS').read_text().splitlines()
    return {line.split('  ', 1)[1]: line.split('  ', 1)[0] for line in rows if line}


def read_bytes(rel):
    """The original bytes of results/<rel>, refusing a copy that no longer matches its hash."""
    data = gzip.decompress((FIXTURES / (rel + '.gz')).read_bytes())
    if hashlib.sha256(data).hexdigest() != recorded()[rel]:
        raise ValueError(f'frozen copy of {rel} does not match SHA256SUMS')
    return data


def load(rel):
    return json.loads(read_bytes(rel))
