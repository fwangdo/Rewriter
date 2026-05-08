"""Common fusion-oriented rule specifications."""

from __future__ import annotations

from .legalization import RuleSpec
from src.superopt.egraph.pattern import PatternNode as PN, PatternVar as PV


def get_fusion_specs() -> list[RuleSpec]:
    return [
        RuleSpec(
            name="bias_add_commute",
            source=PN("Add", (PN("MatMul", (PV("?x"), PV("?w"))), PV("?b"))),
            build_fn=lambda b, v: b.add_op("Add", [v["?b"], b.add_op("MatMul", [v["?x"], v["?w"]])]),
        ),
    ]
