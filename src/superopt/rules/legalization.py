"""Legalization rewrite rules.

These rules lower unsupported ops into supported equivalents.
They mirror the decompositions in onnx_rewrite/passes/ but are
encoded as e-graph equality rules.

Key difference from onnx_rewrite passes: in the e-graph, both
the original and decomposed forms coexist as equivalents.
The extraction phase decides which to pick based on the cost model
and legality constraints.
"""

from __future__ import annotations

from src.common.rules import get_legalization_specs

from .base import RewriteRule
from .wrapper import rulespecs_to_rewrites


def get_legalization_rules() -> list[RewriteRule]:
    """Return legalization rewrite rules."""
    return rulespecs_to_rewrites(get_legalization_specs())
