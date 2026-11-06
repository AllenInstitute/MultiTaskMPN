"""Make shared code importable when a multiple_task script is run directly.

Importing this module (``import _bootstrap``) prepends the repository's
root and ``core/`` directory to ``sys.path``. This supports package imports
such as ``from core.task_variance import ...`` as well as existing flat
imports such as ``import helper`` / ``import mpn``. Scripts are still run
from the repository root (e.g. ``python multiple_task/run_pipeline.py``).
"""
import sys as _sys
import pathlib as _pathlib

_ROOT = _pathlib.Path(__file__).resolve().parent.parent
for _directory in (_ROOT, _ROOT / "core"):
    if str(_directory) not in _sys.path:
        _sys.path.insert(0, str(_directory))
