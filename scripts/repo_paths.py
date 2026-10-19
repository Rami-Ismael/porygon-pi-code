"""The repository root, and src/ first on sys.path.

A script that imports a module from src/ imports this first; nothing else
changes sys.path for it as a side effect."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
