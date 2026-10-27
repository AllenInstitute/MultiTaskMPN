"""Tests for reading the job's cgroup memory limit."""

import tempfile
import unittest
from pathlib import Path

import _bootstrap  # noqa: F401

import memory_limit

GIB = memory_limit.GIB


def _cgroup_tree(root, limits):
    """A fake /sys/fs/cgroup holding job/step/task, with memory.max and
    memory.current per level from `limits` {relative path: (max, current)},
    and the /proc/self/cgroup line pointing at the task."""
    for relative, (limit, current) in limits.items():
        directory = root / "cgroup" / relative
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "memory.max").write_text(f"{limit}\n")
        (directory / "memory.current").write_text(f"{current}\n")
    proc = root / "proc_cgroup"
    proc.write_text("0::/job/step/task\n")
    return {"proc_cgroup": proc, "cgroup_root": root / "cgroup"}


class CgroupMemoryTests(unittest.TestCase):
    def test_tightest_limit_over_the_process_wins(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = _cgroup_tree(Path(directory), {
                "job": (100 * GIB, 40 * GIB),
                "job/step": ("max", 30 * GIB),
                "job/step/task": (200 * GIB, 10 * GIB),
            })
            self.assertEqual(memory_limit.cgroup_memory(**paths), (100 * GIB, 40 * GIB))
            report = memory_limit.memory_report(**paths)
        self.assertIn("100.0 GiB (cgroup)", report)
        self.assertIn("in use 40.0 GiB", report)
        self.assertIn("free 60.0 GiB", report)

    def test_no_limit_falls_back_to_the_node(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = _cgroup_tree(Path(directory), {"job": ("max", 1), "job/step/task": ("max", 1)})
            self.assertIsNone(memory_limit.cgroup_memory(**paths))
            self.assertIn("No cgroup memory limit", memory_limit.memory_report(**paths))

    def test_without_cgroup_v2_there_is_no_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            proc = Path(directory) / "proc_cgroup"
            proc.write_text("12:memory:/job\n")
            self.assertIsNone(memory_limit.cgroup_memory(proc_cgroup=proc,
                                                         cgroup_root=Path(directory)))
            self.assertIsNone(memory_limit.cgroup_memory(proc_cgroup=Path(directory) / "missing"))


if __name__ == "__main__":
    unittest.main()
