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

from .base import RewriteRule
from ..egraph.pattern import PatternNode, PatternVar


def get_legalization_rules() -> list[RewriteRule]:
    """Return legalization rewrite rules."""
    x = PatternVar("?x")

    rules: list[RewriteRule] = []

    # TODO: implement legalization rules that lower unsupported ops
    # into supported equivalents. Examples from todo.md:
    #   Pow(x, 2) → Mul(x, x)
    #   Neg(x) → Mul(x, -1)           (requires constant synthesis)
    #   LayerNorm → ReduceMean + Sub + Mul + ...
    #   GELU_exact → GELU_tanh approx
    #
    # Current stubs (Sub→Add+Neg, Div→Mul+Reciprocal) were wrong:
    # they introduced NEW unsupported ops (Neg, Reciprocal).
    # Real legalization must only produce ops in supported_ops.

    # TODO: implement actual legalization rules.
    return rules
