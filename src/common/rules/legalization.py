"""Common legalization rule specifications."""

from __future__ import annotations

from .spec import PatternSpec as P
from .spec import RuleSpec


def get_pure_legalization_specs() -> list[RuleSpec]:
    """Rules that are pure source/target pattern rewrites."""
    return [
        RuleSpec(
            name="eliminate_identity",
            source=P("Identity", ("?x",)),
            target="?x",
            family="legalization",
        ),
        RuleSpec(
            name="greater_to_less",
            source=P("Greater", ("?a", "?b")),
            target=P("Less", ("?b", "?a")),
            family="legalization",
        ),
        RuleSpec(
            name="sub_to_add_neg",
            source=P("Sub", ("?x", "?y")),
            target=P("Add", ("?x", P("Neg", ("?y",)))),
            family="legalization",
        ),
        RuleSpec(
            name="clip_decompose",
            source=P("Clip", ("?x", "?min", "?max")),
            target=P("Min", (P("Max", ("?x", "?min")), "?max")),
            family="legalization",
        ),
    ]


def get_legalization_specs() -> list[RuleSpec]:
    """Return common legalization specs currently shared across backends."""
    return get_pure_legalization_specs()
