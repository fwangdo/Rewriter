"""Common legalization rule specifications."""

from __future__ import annotations
from typing import Callable, Any

import numpy as np
from dataclasses import dataclass

from .spec import GraphBuilder, VarCheck
from src.superopt.egraph.pattern import Pattern, PatternNode as PN, PatternVar as PV

BuildFn = Callable[[GraphBuilder, dict[str, Any]], Any]


@dataclass(frozen=True)
class RuleSpec:
    """Backend-independent rewrite rule specification."""
    name: str
    source: Pattern
    build_fn: BuildFn
    checks: tuple[VarCheck, ...] = ()


def get_legalization_specs() -> list[RuleSpec]:
    """All legalization rules as build rules."""
    return [
        RuleSpec(
            name="eliminate_identity",
            source=PN("Identity", (PV("?x"),)),
            build_fn=_build_eliminate_identity,
        ),
        RuleSpec(
            name="greater_to_less",
            source=PN("Greater", (PV("?a"), PV("?b"))),
            build_fn=_build_greater_to_less,
        ),
        RuleSpec(
            name="sub_to_add_neg",
            source=PN("Sub", (PV("?x"), PV("?y"))),
            build_fn=_build_sub_to_add_neg,
        ),
        RuleSpec(
            name="neg_to_mul",
            source=PN("Neg", (PV("?x"),)),
            build_fn=_build_neg_to_mul,
        ),
        RuleSpec(
            name="squeeze_to_reshape",
            source=PN("Squeeze", (PV("?x"), PV("?axes"))),
            checks=(VarCheck("?x", has_shape=True),),
            build_fn=_build_shape_to_reshape,
        ),
        RuleSpec(
            name="unsqueeze_to_reshape",
            source=PN("Unsqueeze", (PV("?x"), PV("?axes"))),
            checks=(VarCheck("?x", has_shape=True),),
            build_fn=_build_shape_to_reshape,
        ),
        RuleSpec(
            name="pow_to_identity",
            source=PN("Pow", (PV("?x"), PV("?e"))),
            checks=(VarCheck("?e", scalar_close=1.0),),
            build_fn=lambda builder, vars: vars["?x"],
        ),
        RuleSpec(
            name="pow_to_sqrt",
            source=PN("Pow", (PV("?x"), PV("?e"))),
            checks=(VarCheck("?e", scalar_close=0.5),),
            build_fn=lambda builder, vars: builder.add_op("Sqrt", [vars["?x"]]),
        ),
        RuleSpec(
            name="pow_to_mul",
            source=PN("Pow", (PV("?x"), PV("?e"))),
            checks=(VarCheck("?e", scalar_close=2.0),),
            build_fn=lambda builder, vars: builder.add_op("Mul", [vars["?x"], vars["?x"]]),
        ),
        RuleSpec(
            name="pow_to_cube",
            source=PN("Pow", (PV("?x"), PV("?e"))),
            checks=(VarCheck("?e", scalar_close=3.0),),
            build_fn=_build_pow_to_cube,
        ),
        RuleSpec(
            name="pow_to_reciprocal",
            source=PN("Pow", (PV("?x"), PV("?e"))),
            checks=(VarCheck("?e", scalar_close=-1.0),),
            build_fn=_build_pow_to_reciprocal,
        ),
        RuleSpec(
            name="pow_to_rsqrt",
            source=PN("Pow", (PV("?x"), PV("?e"))),
            checks=(VarCheck("?e", scalar_close=-0.5),),
            build_fn=_build_pow_to_rsqrt,
        ),
        RuleSpec(
            name="layernorm_decompose",
            source=PN("LayerNormalization", (PV("?x"), PV("?scale"), PV("?bias"))),
            build_fn=_build_layernorm_decompose,
        ),
        RuleSpec(
            name="where_mask_decompose",
            source=PN("Where", (PV("?cond"), PV("?true"), PV("?false"))),
            checks=(
                VarCheck("?true", scalar_abs_lt=1e-8),
                VarCheck("?false", scalar_lte=-1.0e30),
            ),
            build_fn=_build_where_mask_decompose,
        ),
        RuleSpec(
            name="where_to_arithmetic",
            source=PN("Where", (PV("?cond"), PV("?true"), PV("?false"))),
            build_fn=_build_where_to_arithmetic,
        ),
        RuleSpec(
            name="range_decompose",
            source=PN("Range", (PV("?start"), PV("?limit"), PV("?step"))),
            checks=(
                VarCheck("?start", scalar_close=0.0),
                VarCheck("?step", scalar_close=1.0),
            ),
            build_fn=_build_range_decompose,
        ),
        RuleSpec(
            name="bn_decompose",
            source=PN("BatchNormalization", (PV("?x"), PV("?s"), PV("?bn_b"), PV("?bn_m"), PV("?bn_v"))),
            build_fn=_build_bn_decompose,
        ),
        RuleSpec(
            name="gemm_decompose",
            source=PN("Gemm", (PV("?a"), PV("?w"), PV("?b"))),
            build_fn=_build_gemm_decompose,
        ),
        RuleSpec(
            name="gemm_decompose_no_bias",
            source=PN("Gemm", (PV("?a"), PV("?w"))),
            build_fn=_build_gemm_decompose_no_bias,
        ),
        RuleSpec(
            name="matmul_to_conv",
            source=PN("MatMul", (PV("?a"), PV("?w"))),
            checks=(VarCheck("?w", is_constant=True),),
            build_fn=_build_matmul_to_conv,
        ),
        RuleSpec(
            name="shape_fold",
            source=PN("Shape", (PV("?x"),)),
            checks=(VarCheck("?x", has_shape=True),),
            build_fn=_build_shape_fold,
        ),
        RuleSpec(
            name="constantofshape_fold",
            source=PN("ConstantOfShape", (PV("?shape"),)),
            checks=(VarCheck("?shape", is_constant=True),),
            build_fn=_build_constantofshape_fold,
        ),
        RuleSpec(
            name="flatten_to_reshape",
            source=PN("Flatten", (PV("?x"),)),
            checks=(VarCheck("?x", has_shape=True),),
            build_fn=_build_flatten_to_reshape,
        ),
        RuleSpec(
            name="expand_to_mul_ones",
            source=PN("Expand", (PV("?x"), PV("?shape"))),
            checks=(VarCheck("?shape", is_constant=True),),
            build_fn=_build_expand_to_mul_ones,
        ),
        RuleSpec(
            name="cos_fold",
            source=PN("Cos", (PV("?x"),)),
            checks=(VarCheck("?x", is_constant=True),),
            build_fn=_build_cos_fold,
        ),
        RuleSpec(
            name="sin_fold",
            source=PN("Sin", (PV("?x"),)),
            checks=(VarCheck("?x", is_constant=True),),
            build_fn=_build_sin_fold,
        ),
        RuleSpec(
            name="pad_eliminate_zero",
            source=PN("Pad", (PV("?x"), PV("?pads"))),
            checks=(VarCheck("?pads", is_constant=True),),
            build_fn=_build_pad_eliminate_zero,
        ),
        RuleSpec(
            name="equal_fold",
            source=PN("Equal", (PV("?a"), PV("?b"))),
            checks=(
                VarCheck("?a", is_constant=True),
                VarCheck("?b", is_constant=True),
            ),
            build_fn=_build_equal_fold,
        ),
        RuleSpec(
            name="less_fold",
            source=PN("Less", (PV("?a"), PV("?b"))),
            checks=(
                VarCheck("?a", is_constant=True),
                VarCheck("?b", is_constant=True),
            ),
            build_fn=_build_less_fold,
        ),
        RuleSpec(
            name="not_to_sub",
            source=PN("Not", (PV("?x"),)),
            build_fn=_build_not_to_sub,
        ),
        RuleSpec(
            name="abs_decompose",
            source=PN("Abs", (PV("?x"),)),
            build_fn=_build_abs_decompose,
        ),
        RuleSpec(
            name="reciprocal_to_div",
            source=PN("Reciprocal", (PV("?x"),)),
            build_fn=lambda builder, vars: builder.add_op(
                "Div", [builder.add_scalar(1.0), vars["?x"]]
            ),
        ),
        RuleSpec(
            name="ceil_fold",
            source=PN("Ceil", (PV("?x"),)),
            checks=(VarCheck("?x", is_constant=True),),
            build_fn=_build_ceil_fold,
        ),
        RuleSpec(
            name="floor_fold",
            source=PN("Floor", (PV("?x"),)),
            checks=(VarCheck("?x", is_constant=True),),
            build_fn=_build_floor_fold,
        ),
    ]


def _build_eliminate_identity(builder: GraphBuilder, vars: dict[str, object]) -> object:
    return vars["?x"]


def _build_greater_to_less(builder: GraphBuilder, vars: dict[str, object]) -> object:
    return builder.add_op("Less", [vars["?b"], vars["?a"]])


def _build_sub_to_add_neg(builder: GraphBuilder, vars: dict[str, object]) -> object:
    neg = builder.add_op("Neg", [vars["?y"]])
    return builder.add_op("Add", [vars["?x"], neg])


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


def _build_bn_decompose(builder: GraphBuilder, vars: dict[str, object]) -> object:
    epsilon = builder.get_matched_attr("epsilon")
    epsilon = 1e-5 if epsilon is None else float(epsilon)

    s_data = builder.get_weight_data("?s")
    b_data = builder.get_weight_data("?bn_b")
    m_data = builder.get_weight_data("?bn_m")
    v_data = builder.get_weight_data("?bn_v")
    if any(data is None for data in (s_data, b_data, m_data, v_data)):
        return builder.get_match()

    scale_factor = (s_data / np.sqrt(v_data + epsilon)).astype(np.float32)
    bias_factor = (b_data - m_data * scale_factor).astype(np.float32)

    x_shape = builder.get_shape("?x")
    if x_shape is not None and len(x_shape) == 4:
        channels = scale_factor.shape[0]
        scale_factor = scale_factor.reshape(1, channels, 1, 1)
        bias_factor = bias_factor.reshape(1, channels, 1, 1)

    sf = builder.add_array(scale_factor, f"__bn_scale_{id(scale_factor)}")
    bf = builder.add_array(bias_factor, f"__bn_bias_{id(bias_factor)}")
    mul = builder.add_op("Mul", [vars["?x"], sf])
    return builder.add_op("Add", [mul, bf])


def _build_gemm_decompose(builder: GraphBuilder, vars: dict[str, object]) -> object:
    a_value, _, b_value = vars["?a"], vars["?w"], vars["?b"]
    trans_a, trans_b, alpha, beta = _get_gemm_attrs(builder)

    w_data = builder.get_weight_data("?w")
    if w_data is None:
        return builder.get_match()
    if trans_b:
        w_data = w_data.T
    w_data = (alpha * w_data).astype(np.float32)

    if trans_a:
        a_value = builder.add_op("Transpose", [a_value], attrs={"perm": (1, 0)})

    w_new = builder.add_array(w_data, f"__gemm_w_{id(w_data)}")
    matmul = builder.add_op("MatMul", [a_value, w_new])

    b_data = builder.get_weight_data("?b")
    if b_data is not None and not _is_close(beta, 1.0):
        b_data = (beta * b_data).astype(np.float32)
        b_value = builder.add_array(b_data, f"__gemm_bias_{id(b_data)}")

    return builder.add_op("Add", [matmul, b_value])


def _build_gemm_decompose_no_bias(builder: GraphBuilder, vars: dict[str, object]) -> object:
    a_value = vars["?a"]
    trans_a, trans_b, alpha, _beta = _get_gemm_attrs(builder)

    w_data = builder.get_weight_data("?w")
    if w_data is None:
        return builder.get_match()
    if trans_b:
        w_data = w_data.T
    w_data = (alpha * w_data).astype(np.float32)

    if trans_a:
        a_value = builder.add_op("Transpose", [a_value], attrs={"perm": (1, 0)})

    w_new = builder.add_array(w_data, f"__gemm_w_{id(w_data)}")
    return builder.add_op("MatMul", [a_value, w_new])


def _build_matmul_to_conv(builder: GraphBuilder, vars: dict[str, object]) -> object:
    w_data = builder.get_weight_data("?w")
    if w_data is None or w_data.ndim != 2:
        return builder.get_match()

    a_shape = builder.get_shape("?a")
    if a_shape is None or len(a_shape) != 2:
        return builder.get_match()

    k_size, n_size = w_data.shape
    conv_weight = w_data.T.reshape(n_size, k_size, 1, 1).astype(np.float32)
    conv_w = builder.add_array(conv_weight, f"__matmul_conv_w_{id(conv_weight)}")

    reshape_in_shape = builder.add_array(np.array([1, 0, -1, 1], dtype=np.int64), "__reshape_10n11", dtype_code=7)
    reshape_in = builder.add_op("Reshape", [vars["?a"], reshape_in_shape])
    conv = builder.add_op("Conv", [reshape_in, conv_w], attrs={"kernel_shape": (1, 1)})
    t2 = builder.add_op("Transpose", [conv], attrs={"perm": (0, 2, 1, 3)})
    reshape_out_shape = builder.add_array(np.array([-1, n_size], dtype=np.int64), f"__reshape_n1_{n_size}", dtype_code=7)
    return builder.add_op("Reshape", [t2, reshape_out_shape])


def _build_shape_fold(builder: GraphBuilder, vars: dict[str, object]) -> object:
    """Shape(x) → constant int64 tensor when shape is fully static."""
    shape = builder.get_shape("?x")
    if shape is None or any(d < 0 for d in shape):
        return builder.get_match()
    return builder.add_array(
        np.array(shape, dtype=np.int64), f"__shape_const_{shape}", dtype_code=7
    )


def _build_constantofshape_fold(
    builder: GraphBuilder, vars: dict[str, object]
) -> object:
    """ConstantOfShape(shape) → constant tensor filled with the default value."""
    shape_data = builder.get_weight_data("?shape")
    if shape_data is None:
        return builder.get_match()
    target_shape = tuple(int(d) for d in shape_data.flat)
    if any(d < 0 for d in target_shape):
        return builder.get_match()
    # ConstantOfShape default value comes from the "value" attribute (scalar tensor).
    # In the e-graph, numpy arrays are stored as (dtype_str, shape, bytes) tuples.
    value_attr = builder.get_matched_attr("value")
    if value_attr is not None:
        if isinstance(value_attr, np.ndarray):
            fill_val = float(value_attr.flat[0])
        elif isinstance(value_attr, tuple) and len(value_attr) == 3:
            dtype_str, _shape, data = value_attr
            fill_val = float(np.frombuffer(data, dtype=np.dtype(dtype_str)).flat[0])
        else:
            try:
                fill_val = float(value_attr)
            except (TypeError, ValueError):
                return builder.get_match()
    else:
        fill_val = 0.0
    arr = np.full(target_shape, fill_val, dtype=np.float32)
    return builder.add_array(arr, f"__constshape_{target_shape}_{fill_val}")


def _build_flatten_to_reshape(
    builder: GraphBuilder, vars: dict[str, object]
) -> object:
    """Flatten(x, axis) → Reshape(x, [pre, post])."""
    shape = builder.get_shape("?x")
    if shape is None or any(d < 0 for d in shape):
        return builder.get_match()
    axis = builder.get_matched_attr("axis")
    axis = 1 if axis is None else int(axis)
    if axis < 0:
        axis += len(shape)
    pre = 1
    for d in shape[:axis]:
        pre *= d
    post = 1
    for d in shape[axis:]:
        post *= d
    flat_shape = np.array([pre, post], dtype=np.int64)
    shape_w = builder.add_array(flat_shape, f"__flat_{pre}_{post}", dtype_code=7)
    return builder.add_op("Reshape", [vars["?x"], shape_w])


def _build_expand_to_mul_ones(
    builder: GraphBuilder, vars: dict[str, object]
) -> object:
    """Expand(x, shape) → Mul(x, ones_of_shape).

    Mul broadcasts x to the target shape automatically.
    """
    shape_data = builder.get_weight_data("?shape")
    if shape_data is None:
        return builder.get_match()
    target_shape = tuple(int(d) for d in shape_data.flat)
    if any(d < 0 for d in target_shape):
        return builder.get_match()
    ones = np.ones(target_shape, dtype=np.float32)
    ones_w = builder.add_array(ones, f"__expand_ones_{target_shape}")
    return builder.add_op("Mul", [vars["?x"], ones_w])


def _build_cos_fold(builder: GraphBuilder, vars: dict[str, object]) -> object:
    """Cos(x) → constant when x is a constant tensor."""
    data = builder.get_weight_data("?x")
    if data is None:
        return builder.get_match()
    result = np.cos(data).astype(np.float32)
    return builder.add_array(result, f"__cos_folded_{id(result)}")


def _build_sin_fold(builder: GraphBuilder, vars: dict[str, object]) -> object:
    """Sin(x) → constant when x is a constant tensor."""
    data = builder.get_weight_data("?x")
    if data is None:
        return builder.get_match()
    result = np.sin(data).astype(np.float32)
    return builder.add_array(result, f"__sin_folded_{id(result)}")


def _build_pad_eliminate_zero(
    builder: GraphBuilder, vars: dict[str, object]
) -> object:
    """Pad(x, pads) → x when all pad values are zero."""
    pads_data = builder.get_weight_data("?pads")
    if pads_data is None:
        return builder.get_match()
    if np.all(pads_data == 0):
        return vars["?x"]
    return builder.get_match()


def _build_where_to_arithmetic(
    builder: GraphBuilder, vars: dict[str, object]
) -> object:
    """Where(cond, A, B) → Cast(cond)*A + (1-Cast(cond))*B."""
    cast = builder.add_op("Cast", [vars["?cond"]], attrs={"to": 1})
    one = builder.add_scalar(1.0, "__where_one")
    inv = builder.add_op("Sub", [one, cast])
    true_branch = builder.add_op("Mul", [cast, vars["?true"]])
    false_branch = builder.add_op("Mul", [inv, vars["?false"]])
    return builder.add_op("Add", [true_branch, false_branch])


def _build_equal_fold(builder: GraphBuilder, vars: dict[str, object]) -> object:
    """Equal(a, b) → constant bool tensor when both inputs are constant."""
    a_data = builder.get_weight_data("?a")
    b_data = builder.get_weight_data("?b")
    if a_data is None or b_data is None:
        return builder.get_match()
    try:
        result = (a_data == b_data).astype(np.float32)
    except (ValueError, TypeError):
        return builder.get_match()
    return builder.add_array(result, f"__equal_folded_{id(result)}")


def _build_less_fold(builder: GraphBuilder, vars: dict[str, object]) -> object:
    """Less(a, b) → constant bool tensor when both inputs are constant."""
    a_data = builder.get_weight_data("?a")
    b_data = builder.get_weight_data("?b")
    if a_data is None or b_data is None:
        return builder.get_match()
    try:
        result = (a_data < b_data).astype(np.float32)
    except (ValueError, TypeError):
        return builder.get_match()
    return builder.add_array(result, f"__less_folded_{id(result)}")


def _build_not_to_sub(builder: GraphBuilder, vars: dict[str, object]) -> object:
    """Not(x) → Sub(1, Cast(x, float))."""
    cast = builder.add_op("Cast", [vars["?x"]], attrs={"to": 1})
    one = builder.add_scalar(1.0, "__not_one")
    return builder.add_op("Sub", [one, cast])


def _build_abs_decompose(builder: GraphBuilder, vars: dict[str, object]) -> object:
    """Abs(x) → Relu(x) + Relu(Neg(x)) = Relu(x) + Relu(Mul(x, -1))."""
    neg_one = builder.add_scalar(-1.0, "__abs_neg")
    relu_pos = builder.add_op("Relu", [vars["?x"]])
    neg_x = builder.add_op("Mul", [vars["?x"], neg_one])
    relu_neg = builder.add_op("Relu", [neg_x])
    return builder.add_op("Add", [relu_pos, relu_neg])


def _build_ceil_fold(builder: GraphBuilder, vars: dict[str, object]) -> object:
    """Ceil(x) → constant when x is constant."""
    data = builder.get_weight_data("?x")
    if data is None:
        return builder.get_match()
    result = np.ceil(data).astype(data.dtype)
    return builder.add_array(result, f"__ceil_folded_{id(result)}")


def _build_floor_fold(builder: GraphBuilder, vars: dict[str, object]) -> object:
    """Floor(x) → constant when x is constant."""
    data = builder.get_weight_data("?x")
    if data is None:
        return builder.get_match()
    result = np.floor(data).astype(data.dtype)
    return builder.add_array(result, f"__floor_folded_{id(result)}")


def _get_gemm_attrs(builder: GraphBuilder) -> tuple[int, int, float, float]:
    trans_a = builder.get_matched_attr("transA")
    trans_b = builder.get_matched_attr("transB")
    alpha = builder.get_matched_attr("alpha")
    beta = builder.get_matched_attr("beta")
    return (
        0 if trans_a is None else int(trans_a),
        0 if trans_b is None else int(trans_b),
        1.0 if alpha is None else float(alpha),
        1.0 if beta is None else float(beta),
    )


def _is_close(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-6
