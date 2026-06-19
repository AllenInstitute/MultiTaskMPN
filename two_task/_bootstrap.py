"""Path bootstrap: make core/ importable via flat imports after the dir move.

Importing this module (``import _bootstrap``) prepends the repository's
``core/`` directory to ``sys.path``, so existing flat imports such as
``import helper`` / ``import mpn`` continue to resolve. Scripts are still run
from the repository root (e.g. ``python two_task/two_task.py``).
"""
import sys as _sys
import pathlib as _pathlib

_CORE = _pathlib.Path(__file__).resolve().parent.parent / "core"
if str(_CORE) not in _sys.path:
    _sys.path.insert(0, str(_CORE))
