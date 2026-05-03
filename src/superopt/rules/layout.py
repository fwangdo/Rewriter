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

    # Transpose(Transpose(x, p), p) → x  for common permutations.
    # Each fires only when both transposes have the exact same perm,
    # which is its own inverse (involution).
    # (0,1) — 2D swap, (1,0) — 2D swap
    for perm in [(0, 1), (1, 0)]:
        rules.append(RewriteRule(
            name=f"transpose_cancel_perm_{'_'.join(str(p) for p in perm)}",
            source=PatternNode(
                "Transpose",
                (PatternNode("Transpose", (x,), attrs=(("perm", perm),)),),
                attrs=(("perm", perm),),
            ),
            target=x,
        ))

    # General transpose cancel: Transpose(Transpose(x, p1), inv(p1)) → x
    # Requires extending pattern matching to expose matched attrs.
    # TODO: implement when attr-aware substitutions are supported.

    return rules
