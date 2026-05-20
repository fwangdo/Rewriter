"""Common fusion-oriented rule specifications."""

from __future__ import annotations

from .legalization import RuleSpec
from src.common.egraph.pattern import PatternNode as PN, PatternVar as PV


def get_fusion_specs() -> list[RuleSpec]:
    return [
        # Add(MatMul(x, w), b) → Add(b, MatMul(x, w)).
        # Commutes bias addition so that the bias comes first. This can
        # help ORT's pattern matcher recognize MatMul+Add as a fused op
        # when it expects a specific operand order.
        RuleSpec(
            name="bias_add_commute",
            source=PN("Add", (PN("MatMul", (PV("?x"), PV("?w"))), PV("?b"))),
            build_fn=lambda b, v: b.add_op("Add", [v["?b"], b.add_op("MatMul", [v["?x"], v["?w"]], b.get_matched_shape(), b.get_dtype("?x"))], b.get_matched_shape(), b.get_dtype("?x")),
        ),
    ]
