"""Arithmetic rewrite rules.

Identity/zero elimination and simple algebraic simplifications.
These are basic equality rules that any e-graph optimizer should have.
"""

from __future__ import annotations

from .base import RewriteRule
from ..egraph.pattern import PatternNode, PatternVar


def get_arithmetic_rules() -> list[RewriteRule]:
    """Return the standard arithmetic rewrite rules."""
    x = PatternVar("?x")
    y = PatternVar("?y")

    rules: list[RewriteRule] = []

    # --- commutativity ---
    # Add(x, y) = Add(y, x)
    rules.append(RewriteRule(
        name="add_comm",
        source=PatternNode("Add", (x, y)),
        target=PatternNode("Add", (y, x)),
    ))

    # Mul(x, y) = Mul(y, x)
    rules.append(RewriteRule(
        name="mul_comm",
        source=PatternNode("Mul", (x, y)),
        target=PatternNode("Mul", (y, x)),
    ))

    # Low-risk skeleton rules that do not require constant synthesis.
    rules.append(RewriteRule(
        name="add_assoc_right",
        source=PatternNode("Add", (PatternNode("Add", (x, y)), PatternVar("?z"))),
        target=PatternNode("Add", (x, PatternNode("Add", (y, PatternVar("?z"))))),
    ))

    rules.append(RewriteRule(
        name="mul_assoc_right",
        source=PatternNode("Mul", (PatternNode("Mul", (x, y)), PatternVar("?z"))),
        target=PatternNode("Mul", (x, PatternNode("Mul", (y, PatternVar("?z"))))),
    ))

    return rules
