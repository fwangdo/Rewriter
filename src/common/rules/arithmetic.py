"""Common arithmetic rule specifications."""

from __future__ import annotations

from .legalization import RuleSpec
from src.common.egraph.pattern import PatternNode as PN, PatternVar as PV


def get_arithmetic_specs() -> list[RuleSpec]:
    return [
        # Add(x, y) → Add(y, x). Addition is commutative; lets the
        # e-graph explore equivalent operand orderings.
        RuleSpec(
            name="add_comm",
            source=PN("Add", (PV("?x"), PV("?y"))),
            build_fn=lambda b, v: b.add_op("Add", [v["?y"], v["?x"]], b.get_matched_shape(), b.get_dtype("?x")),
        ),
        # Mul(x, y) → Mul(y, x). Multiplication is commutative.
        RuleSpec(
            name="mul_comm",
            source=PN("Mul", (PV("?x"), PV("?y"))),
            build_fn=lambda b, v: b.add_op("Mul", [v["?y"], v["?x"]], b.get_matched_shape(), b.get_dtype("?x")),
        ),
        # Add(Add(x, y), z) → Add(x, Add(y, z)). Addition is associative;
        # re-associates to expose constant-folding or fusion opportunities.
        RuleSpec(
            name="add_assoc_right",
            source=PN("Add", (PN("Add", (PV("?x"), PV("?y"))), PV("?z"))),
            build_fn=lambda b, v: b.add_op("Add", [v["?x"], b.add_op("Add", [v["?y"], v["?z"]], b.get_matched_shape(), b.get_dtype("?x"))], b.get_matched_shape(), b.get_dtype("?x")),
        ),
        # Mul(Mul(x, y), z) → Mul(x, Mul(y, z)). Multiplication is associative.
        RuleSpec(
            name="mul_assoc_right",
            source=PN("Mul", (PN("Mul", (PV("?x"), PV("?y"))), PV("?z"))),
            build_fn=lambda b, v: b.add_op("Mul", [v["?x"], b.add_op("Mul", [v["?y"], v["?z"]], b.get_matched_shape(), b.get_dtype("?x"))], b.get_matched_shape(), b.get_dtype("?x")),
        ),
    ]
