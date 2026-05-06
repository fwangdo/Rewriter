"""Common layout rule specifications."""

from __future__ import annotations

from .spec import PatternSpec as P
from .spec import RuleSpec


def get_layout_specs() -> list[RuleSpec]:
    rules = [
        RuleSpec(
            name="reshape_reshape",
            source=P("Reshape", (P("Reshape", ("?x", "?y")), "?z")),
            target=P("Reshape", ("?x", "?z")),
            family="layout",
        ),
    ]
    for perm in ((0, 1), (1, 0)):
        suffix = "_".join(str(p) for p in perm)
        attrs = (("perm", perm),)
        rules.append(
            RuleSpec(
                name=f"transpose_cancel_perm_{suffix}",
                source=P("Transpose", (P("Transpose", ("?x",), attrs=attrs),), attrs=attrs),
                target="?x",
                family="layout",
            )
        )
    return rules
