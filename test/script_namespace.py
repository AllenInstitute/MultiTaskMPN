"""Execute selected functions and constants of a training script without
importing it: the scripts start CUDA, plotting and the training stack at
import, which the naming and seed tests do not need."""

import ast
import random
import re
from pathlib import Path

import numpy as np


def load_script_namespace(source, functions, constants, extra=None):
    """Namespace holding the named top-level functions and constant
    assignments of `source`, executed in file order with numpy, re, random
    and Path available (plus `extra`)."""
    source = Path(source)
    tree = ast.parse(source.read_text())
    nodes = [node for node in tree.body
             if (isinstance(node, ast.FunctionDef) and node.name in functions)
             or (isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id in constants
                         for target in node.targets))]
    found = {node.name for node in nodes if isinstance(node, ast.FunctionDef)}
    if found != set(functions):
        raise AssertionError(f"{source}: missing {sorted(set(functions) - found)}")
    namespace = {"np": np, "re": re, "random": random, "Path": Path, **(extra or {})}
    exec(  # noqa: S102 -- selected definitions from trusted local source
        compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    return namespace
