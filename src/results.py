"""Write a result file so a crash never leaves it half-written, and fingerprint a JSON value."""
import hashlib
import json
import os
from pathlib import Path


def digest(value):
    """sha256 of the value as JSON with sorted keys: the same for any key order."""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def save(path, value):
    """Write value as indented JSON to path, creating its folder.

    Writes a .tmp sibling, fsyncs it, renames it over path, then fsyncs the
    folder, so path holds either the old file or the whole new one."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open('w') as handle:
        handle.write(json.dumps(value, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)
    directory = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
