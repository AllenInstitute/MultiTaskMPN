"""Path bootstrap: make the repository's packages importable from figures/.

The figure scripts import shared code as packages (``from core.clustering
import ...``, ``from pretrain.pretraining_utils import ...``), so they need the
repository root on ``sys.path``. Running ``python figures/paper_plot.py``
puts only ``figures/`` there; importing this module (``import _bootstrap``)
prepends the root as well. Scripts are still run from the repository root,
which is where their output directories (``paper_plot/``, ``cartoon_plot/``)
are written.
"""
import sys as _sys
import pathlib as _pathlib

_ROOT = _pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
