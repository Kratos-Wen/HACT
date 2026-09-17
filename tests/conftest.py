"""Pytest configuration: the scripts import their siblings as top-level modules, so the source
directories are put on ``sys.path`` the same way the scripts do when run standalone."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for name in ("pipeline", "utils", "integrations"):
    path = os.path.join(ROOT, name)
    if path not in sys.path:
        sys.path.insert(0, path)
