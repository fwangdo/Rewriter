"""Layout rewrite rules.

Transpose fusion, reshape chain simplification, and other
layout/shape transformation rules.
"""

from __future__ import annotations

from .base import RewriteRule
from ..egraph.pattern import PatternNode, PatternVar


def get_layout_rules() -> list[RewriteRule]:
    """Return layout-related rewrite rules."""
    x = PatternVar("?x")
    y = PatternVar("?y")

    rules: list[RewriteRule] = []

    rules.append(RewriteRule(
        name="reshape_reshape",
        source=PatternNode("Reshape", (PatternNode("Reshape", (x, y)), PatternVar("?z"))),
        target=PatternNode("Reshape", (x, PatternVar("?z"))),
    ))

    rules.append(RewriteRule(
        name="transpose_transpose_identity",
        source=PatternNode("Transpose", (PatternNode("Transpose", (x,), attrs=(("perm", (0, 1)),)),), attrs=(("perm", (0, 1)),)),
        target=x,
    ))

    return rules
