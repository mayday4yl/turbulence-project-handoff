"""Load unchanged pinned official class bodies without BasicSR's eager plugins.

Only import isolation is performed. AST class nodes are compiled verbatim; no
operators, initialization, forward, or backward bodies are rewritten. The full
upstream sources and their license remain in vendor/nafnet for auditing.
"""
import ast
import hashlib
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
COMMIT = '2b4af71ebe098a92a75910c233a3965a3e93ede4'


def verify_upstream():
    identity = json.loads((ROOT/'protocol/UPSTREAM_IDENTITY.json').read_text())
    if identity['commit'] != COMMIT:
        raise ValueError('NAFNet official source commit changed')
    for rel, expected in identity['files'].items():
        if hashlib.sha256((ROOT/'vendor/nafnet'/rel).read_bytes()).hexdigest() != expected:
            raise ValueError('Pinned NAFNet source changed: '+rel)
    return identity


def _load_classes(rel, names):
    path = ROOT/'vendor/nafnet'/rel
    tree = ast.parse(path.read_text(), filename=str(path))
    selected = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
    if [node.name for node in selected] != list(names):
        raise RuntimeError('Unexpected upstream class structure')
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), globals())


verify_upstream()
_load_classes('basicsr/models/archs/arch_util.py', ('LayerNormFunction', 'LayerNorm2d'))
_load_classes('basicsr/models/archs/NAFNet_arch.py', ('SimpleGate', 'NAFBlock', 'NAFNet'))
