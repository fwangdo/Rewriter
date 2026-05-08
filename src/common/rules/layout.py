"""Common layout rule specifications."""

from __future__ import annotations

from .legalization import RuleSpec
from src.superopt.egraph.pattern import PatternNode as PN, PatternVar as PV


def get_layout_specs() -> list[RuleSpec]:
    rules = [
        RuleSpec(
            name="reshape_reshape",
            source=PN("Reshape", (PN("Reshape", (PV("?x"), PV("?y"))), PV("?z"))),
            build_fn=lambda b, v: b.add_op("Reshape", [v["?x"], v["?z"]]),
        ),
    ]
    for perm in ((0, 1), (1, 0)):
        suffix = "_".join(str(p) for p in perm)
        attrs = (("perm", perm),)
        rules.append(
            RuleSpec(
                name=f"transpose_cancel_perm_{suffix}",
                source=PN("Transpose", (PN("Transpose", (PV("?x"),), attrs=attrs),), attrs=attrs),
                build_fn=lambda b, v: v["?x"],
            )
        )
    return rules
