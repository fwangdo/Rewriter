"""Common fusion-oriented rule specifications."""

from __future__ import annotations

from .spec import PatternSpec as P
from .spec import RuleSpec


def get_fusion_specs() -> list[RuleSpec]:
    return [
        RuleSpec(
            name="bias_add_commute",
            source=P("Add", (P("MatMul", ("?x", "?w")), "?b")),
            target=P("Add", ("?b", P("MatMul", ("?x", "?w")))),

        ),
    ]
