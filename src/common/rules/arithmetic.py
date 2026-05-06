"""Common arithmetic rule specifications."""

from __future__ import annotations

from .spec import PatternSpec as P
from .spec import RuleSpec


def get_arithmetic_specs() -> list[RuleSpec]:
    return [
        RuleSpec(
            name="add_comm",
            source=P("Add", ("?x", "?y")),
            target=P("Add", ("?y", "?x")),
            family="arithmetic",
        ),
        RuleSpec(
            name="mul_comm",
            source=P("Mul", ("?x", "?y")),
            target=P("Mul", ("?y", "?x")),
            family="arithmetic",
        ),
        RuleSpec(
            name="add_assoc_right",
            source=P("Add", (P("Add", ("?x", "?y")), "?z")),
            target=P("Add", ("?x", P("Add", ("?y", "?z")))),
            family="arithmetic",
        ),
        RuleSpec(
            name="mul_assoc_right",
            source=P("Mul", (P("Mul", ("?x", "?y")), "?z")),
            target=P("Mul", ("?x", P("Mul", ("?y", "?z")))),
            family="arithmetic",
        ),
    ]
