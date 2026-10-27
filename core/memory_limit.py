"""
The memory a process may use: its Slurm job's cgroup limit, not the node's.

On the cluster every job runs in a cgroup v2 whose ``memory.max`` is the
memory the job requested (e.g. 100 GiB); a job that goes over it is killed
(the shell prints "Killed"). psutil reports the whole node (about 540 GB), so
a script that prints psutil's numbers looks far from its limit when it is
not. `memory_report` gives the job's limit and use instead, and falls back to
the node's memory where no cgroup limit applies.
"""
from pathlib import Path

import psutil

GIB = 1024 ** 3


def cgroup_memory(proc_cgroup="/proc/self/cgroup", cgroup_root="/sys/fs/cgroup"):
    """(limit, used) in bytes of the tightest cgroup v2 memory limit over this
    process, read from ``memory.max`` and ``memory.current`` of the cgroup
    that sets it, or None when there is no cgroup v2 or no limit."""
    try:
        lines = Path(proc_cgroup).read_text().splitlines()
    except OSError:
        return None
    relative = next((line[3:] for line in lines if line.startswith("0::")), None)
    if relative is None:
        return None
    root = Path(cgroup_root)
    directory = root / relative.strip().lstrip("/")
    tightest = None
    while True:
        try:
            text = (directory / "memory.max").read_text().strip()
            if text != "max":
                limit = int(text)
                if tightest is None or limit < tightest[0]:
                    tightest = (limit, int((directory / "memory.current").read_text()))
        except (OSError, ValueError):
            pass
        if directory == root or directory.parent == directory:
            return tightest
        directory = directory.parent


def memory_report(**paths):
    """One line on the memory this process may use: the job's cgroup limit and
    its use, or the node's total and available memory without a limit.
    `paths` pass through to `cgroup_memory`."""
    job = cgroup_memory(**paths)
    if job is not None:
        limit, used = job
        return (f"Job memory limit {limit / GIB:.1f} GiB (cgroup), in use {used / GIB:.1f} GiB, "
                f"free {(limit - used) / GIB:.1f} GiB")
    node = psutil.virtual_memory()
    return (f"No cgroup memory limit; node total {node.total / GIB:.1f} GiB, "
            f"available {node.available / GIB:.1f} GiB")
