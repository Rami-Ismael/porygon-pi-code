"""Start isolated Showdown instances for independent temperature-study seeds."""
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time

RUNTIME = Path("/Users/ramiismael/.local/share/vgc-pilot-runtime")
SOURCE = Path("/tmp/vgc-pilot/vgc-bench/pokemon-showdown")
NODE = "/Users/ramiismael/.nvm/versions/node/v22.22.0/bin/node"


def link_or_copy(source, target):
    try:
        return os.link(source, target)
    except OSError:
        return shutil.copy2(source, target)


def listening(port):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=.2):
            return True
    except OSError:
        return False


def main():
    records = {}
    for port in range(8134, 8142):
        if listening(port):
            raise RuntimeError(f"port already occupied: {port}")
        root = RUNTIME / "servers" / str(port)
        if root.exists():
            raise RuntimeError(f"server directory already exists: {root}")
        # Real paths for Showdown code keep its FS root isolated; dependencies
        # can be shared. Mutable configuration/database files get separate inodes.
        excluded = {".git", "node_modules", "config", "databases", "logs"}
        shutil.copytree(SOURCE, root, copy_function=link_or_copy,
                        ignore=lambda directory, names: [name for name in names
                            if Path(directory) == SOURCE and name in excluded])
        (root / "node_modules").symlink_to(SOURCE / "node_modules", target_is_directory=True)
        for directory in ("config", "databases"):
            if (SOURCE / directory).exists():
                shutil.copytree(SOURCE / directory, root / directory)
        (root / "logs").mkdir(exist_ok=True)
        (root / "logs/repl").mkdir(exist_ok=True)
        with (root / "config/config.js").open("a") as f:
            f.write('\nexports.bindaddress = "127.0.0.1";\n')
        log = (RUNTIME / f"showdown-{port}.log").open("ab")
        process = subprocess.Popen([NODE, "pokemon-showdown", "start", str(port), "--no-security"],
                                   cwd=root, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        for _ in range(100):
            if listening(port):
                break
            if process.poll() is not None:
                raise RuntimeError(f"server {port} exited: see its log")
            time.sleep(.1)
        else:
            raise RuntimeError(f"server {port} did not become ready")
        records[str(port)] = dict(pid=process.pid, root=str(root), node=NODE)
        (RUNTIME / "temperature-additional-servers.json").write_text(json.dumps(records, indent=2) + "\n")
        print(f"SERVER_READY {port} pid={process.pid}", flush=True)


if __name__ == "__main__":
    main()
