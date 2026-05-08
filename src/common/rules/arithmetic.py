"""Common arithmetic rule specifications."""

from __future__ import annotations

from .legalization import RuleSpec
from src.superopt.egraph.pattern import PatternNode as PN, PatternVar as PV


def get_arithmetic_specs() -> list[RuleSpec]:
    return [
        RuleSpec(
            name="add_comm",
            source=PN("Add", (PV("?x"), PV("?y"))),
            build_fn=lambda b, v: b.add_op("Add", [v["?y"], v["?x"]]),
        ),
        RuleSpec(
            name="mul_comm",
            source=PN("Mul", (PV("?x"), PV("?y"))),
            build_fn=lambda b, v: b.add_op("Mul", [v["?y"], v["?x"]]),
        ),
        RuleSpec(
            name="add_assoc_right",
            source=PN("Add", (PN("Add", (PV("?x"), PV("?y"))), PV("?z"))),
            build_fn=lambda b, v: b.add_op("Add", [v["?x"], b.add_op("Add", [v["?y"], v["?z"]])]),
        ),
        RuleSpec(
            name="mul_assoc_right",
            source=PN("Mul", (PN("Mul", (PV("?x"), PV("?y"))), PV("?z"))),
            build_fn=lambda b, v: b.add_op("Mul", [v["?x"], b.add_op("Mul", [v["?y"], v["?z"]])]),
        ),
    ]
