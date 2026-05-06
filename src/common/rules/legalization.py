"""Common legalization rule specifications."""

from __future__ import annotations

import numpy as np

from .spec import PatternSpec as P
from .spec import GraphBuilder, RuleSpec, VarCheck


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


def get_simple_build_legalization_specs() -> list[RuleSpec]:
    """Rules with common build functions but no backend-specific semantics."""
    return [
        RuleSpec(
            name="neg_to_mul",
            source=P("Neg", ("?x",)),
            build_fn=_build_neg_to_mul,
            family="legalization",
        ),
        RuleSpec(
            name="squeeze_to_reshape",
            source=P("Squeeze", ("?x", "?axes")),
            checks=(VarCheck("?x", has_shape=True),),
            build_fn=_build_shape_to_reshape,
            family="legalization",
        ),
        RuleSpec(
            name="unsqueeze_to_reshape",
            source=P("Unsqueeze", ("?x", "?axes")),
            checks=(VarCheck("?x", has_shape=True),),
            build_fn=_build_shape_to_reshape,
            family="legalization",
        ),
        RuleSpec(
            name="pow_to_identity",
            source=P("Pow", ("?x", "?e")),
            checks=(VarCheck("?e", scalar_close=1.0),),
            build_fn=lambda builder, vars: vars["?x"],
            family="legalization",
        ),
        RuleSpec(
            name="pow_to_sqrt",
            source=P("Pow", ("?x", "?e")),
            checks=(VarCheck("?e", scalar_close=0.5),),
            build_fn=lambda builder, vars: builder.add_op("Sqrt", [vars["?x"]]),
            family="legalization",
        ),
        RuleSpec(
            name="pow_to_mul",
            source=P("Pow", ("?x", "?e")),
            checks=(VarCheck("?e", scalar_close=2.0),),
            build_fn=lambda builder, vars: builder.add_op("Mul", [vars["?x"], vars["?x"]]),
            family="legalization",
        ),
        RuleSpec(
            name="pow_to_cube",
            source=P("Pow", ("?x", "?e")),
            checks=(VarCheck("?e", scalar_close=3.0),),
            build_fn=_build_pow_to_cube,
            family="legalization",
        ),
        RuleSpec(
            name="pow_to_reciprocal",
            source=P("Pow", ("?x", "?e")),
            checks=(VarCheck("?e", scalar_close=-1.0),),
            build_fn=_build_pow_to_reciprocal,
            family="legalization",
        ),
        RuleSpec(
            name="pow_to_rsqrt",
            source=P("Pow", ("?x", "?e")),
            checks=(VarCheck("?e", scalar_close=-0.5),),
            build_fn=_build_pow_to_rsqrt,
            family="legalization",
        ),
    ]


def get_legalization_specs() -> list[RuleSpec]:
    """Return common legalization specs currently shared across backends."""
    return get_pure_legalization_specs() + get_simple_build_legalization_specs()


def _build_neg_to_mul(builder: GraphBuilder, vars: dict[str, object]) -> object:
    return builder.add_op("Mul", [vars["?x"], builder.add_scalar(-1.0)])


def _build_shape_to_reshape(builder: GraphBuilder, vars: dict[str, object]) -> object:
    shape = builder.get_matched_shape()
    if shape is None or sum(1 for dim in shape if dim == -1) > 1:
        return vars["?x"]
    shape_value = builder.add_array(
        np.array(shape, dtype=np.int64),
        name=f"__shape_{shape}",
        dtype_code=7,
    )
    return builder.add_op("Reshape", [vars["?x"], shape_value])


def _build_pow_to_cube(builder: GraphBuilder, vars: dict[str, object]) -> object:
    squared = builder.add_op("Mul", [vars["?x"], vars["?x"]])
    return builder.add_op("Mul", [squared, vars["?x"]])


def _build_pow_to_reciprocal(builder: GraphBuilder, vars: dict[str, object]) -> object:
    return builder.add_op("Div", [builder.add_scalar(1.0), vars["?x"]])


def _build_pow_to_rsqrt(builder: GraphBuilder, vars: dict[str, object]) -> object:
    sqrt = builder.add_op("Sqrt", [vars["?x"]])
    return builder.add_op("Div", [builder.add_scalar(1.0), sqrt])
