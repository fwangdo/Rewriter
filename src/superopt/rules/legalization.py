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

    # These initial rules intentionally avoid constant synthesis.
    # They are legality-oriented canonicalizations that can coexist with
    # baseline ONNX rewrites.
    rules.append(RewriteRule(
        name="sub_to_add_neg",
        source=PatternNode("Sub", (x, PatternVar("?y"))),
        target=PatternNode("Add", (x, PatternNode("Neg", (PatternVar("?y"),)))),
    ))

    rules.append(RewriteRule(
        name="div_to_mul_recip",
        source=PatternNode("Div", (x, PatternVar("?y"))),
        target=PatternNode("Mul", (x, PatternNode("Reciprocal", (PatternVar("?y"),)))),
    ))

    return rules
