"""Make repository modules importable when tests are run from ``test/``."""

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

for module_dir in (
    REPO_ROOT,
    REPO_ROOT / "core",
    REPO_ROOT / "multiple_task",
    REPO_ROOT / "pretrain",
):
    module_path = str(module_dir)
    if module_path not in sys.path:
        sys.path.insert(0, module_path)
