"""Fusion rewrite rules (multi-pattern).

These rules merge multiple operations into fewer, more efficient ones.
This is where equality saturation shines: fusion and fission rules
can coexist without phase-ordering conflicts.

Reference: Tensat Figure 8, 9, 10, 11.
"""

from __future__ import annotations

from .base import RewriteRule
from ..egraph.pattern import PatternNode, PatternVar


def get_fusion_rules() -> list[RewriteRule]:
    """Return fusion rewrite rules.

    Note: multi-pattern rules (multiple matched outputs) require
    special handling in the exploration phase.  The rules below
    are single-pattern approximations.
    """
    x = PatternVar("?x")

    rules: list[RewriteRule] = []

    # Conservative skeleton rules only. Real multi-root fusion should be
    # introduced after extractor and constant synthesis are stable.
    rules.append(RewriteRule(
        name="bias_add_commute",
        source=PatternNode("Add", (PatternNode("MatMul", (x, PatternVar("?w"))), PatternVar("?b"))),
        target=PatternNode("Add", (PatternVar("?b"), PatternNode("MatMul", (x, PatternVar("?w"))))),
    ))

    return rules
