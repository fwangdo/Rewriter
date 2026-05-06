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
        RuleSpec(
            name="layernorm_decompose",
            source=P("LayerNormalization", ("?x", "?scale", "?bias")),
            build_fn=_build_layernorm_decompose,
            family="legalization",
        ),
        RuleSpec(
            name="where_mask_decompose",
            source=P("Where", ("?cond", "?true", "?false")),
            checks=(
                VarCheck("?true", scalar_abs_lt=1e-8),
                VarCheck("?false", scalar_lte=-1.0e30),
            ),
            build_fn=_build_where_mask_decompose,
            family="legalization",
        ),
        RuleSpec(
            name="range_decompose",
            source=P("Range", ("?start", "?limit", "?step")),
            checks=(
                VarCheck("?start", scalar_close=0.0),
                VarCheck("?step", scalar_close=1.0),
            ),
            build_fn=_build_range_decompose,
            family="legalization",
        ),
        RuleSpec(
            name="erf_to_tanh",
            source=P("Erf", ("?x",)),
            build_fn=_build_erf_to_tanh,
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
        return builder.get_match()
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


def _build_layernorm_decompose(builder: GraphBuilder, vars: dict[str, object]) -> object:
    axis = builder.get_matched_attr("axis")
    epsilon = builder.get_matched_attr("epsilon")
    axis = -1 if axis is None else int(axis)
    epsilon = 1e-5 if epsilon is None else float(epsilon)

    axes = builder.add_array(np.array([axis], dtype=np.int64), f"__ln_axes_{axis}", dtype_code=7)
    eps = builder.add_scalar(epsilon, name=f"__ln_eps_{epsilon}")
    mean = builder.add_op("ReduceMean", [vars["?x"], axes], attrs={"keepdims": 1})
    centered = builder.add_op("Sub", [vars["?x"], mean])
    squared = builder.add_op("Mul", [centered, centered])
    var = builder.add_op("ReduceMean", [squared, axes], attrs={"keepdims": 1})
    var_eps = builder.add_op("Add", [var, eps])
    std = builder.add_op("Sqrt", [var_eps])
    normalized = builder.add_op("Div", [centered, std])
    scaled = builder.add_op("Mul", [normalized, vars["?scale"]])
    return builder.add_op("Add", [scaled, vars["?bias"]])


def _build_where_mask_decompose(builder: GraphBuilder, vars: dict[str, object]) -> object:
    cast = builder.add_op("Cast", [vars["?cond"]], attrs={"to": 1})
    inverse = builder.add_op("Sub", [builder.add_scalar(1.0), cast])
    return builder.add_op("Mul", [inverse, vars["?false"]])


def _build_range_decompose(builder: GraphBuilder, vars: dict[str, object]) -> object:
    table = builder.add_array(np.arange(4096, dtype=np.int64), "__arange_table_4096", dtype_code=7)
    starts = builder.add_array(np.array([0], dtype=np.int64), "__slice_starts_0", dtype_code=7)
    axes = builder.add_array(np.array([0], dtype=np.int64), "__slice_axes_0", dtype_code=7)
    steps = builder.add_array(np.array([1], dtype=np.int64), "__slice_steps_1", dtype_code=7)
    ends_shape = builder.add_array(np.array([1], dtype=np.int64), "__shape_1", dtype_code=7)
    ends = builder.add_op("Reshape", [vars["?limit"], ends_shape])
    return builder.add_op("Slice", [table, starts, ends, axes, steps])


def _build_erf_to_tanh(builder: GraphBuilder, vars: dict[str, object]) -> object:
    c1 = builder.add_scalar(0.044715, "__erf_c1")
    c2 = builder.add_scalar(0.7978845608, "__erf_c2")
    one = builder.add_scalar(1.0, "__erf_one")
    x2 = builder.add_op("Mul", [vars["?x"], vars["?x"]])
    cx2 = builder.add_op("Mul", [c1, x2])
    inner = builder.add_op("Add", [one, cx2])
    xc = builder.add_op("Mul", [vars["?x"], c2])
    arg = builder.add_op("Mul", [xc, inner])
    return builder.add_op("Tanh", [arg])
