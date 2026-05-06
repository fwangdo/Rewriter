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
        RuleSpec(
            name="bn_decompose",
            source=P("BatchNormalization", ("?x", "?s", "?bn_b", "?bn_m", "?bn_v")),
            build_fn=_build_bn_decompose,
            family="legalization",
        ),
        RuleSpec(
            name="gemm_decompose",
            source=P("Gemm", ("?a", "?w", "?b")),
            build_fn=_build_gemm_decompose,
            family="legalization",
        ),
        RuleSpec(
            name="gemm_decompose_no_bias",
            source=P("Gemm", ("?a", "?w")),
            build_fn=_build_gemm_decompose_no_bias,
            family="legalization",
        ),
        RuleSpec(
            name="matmul_to_conv",
            source=P("MatMul", ("?a", "?w")),
            checks=(VarCheck("?w", is_constant=True),),
            build_fn=_build_matmul_to_conv,
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
    a_value, w_value, b_value = vars["?a"], vars["?w"], vars["?b"]
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
    if a_shape is None or len(a_shape) not in (2, 3):
        return builder.get_match()

    k_size, n_size = w_data.shape
    conv_weight = w_data.T.reshape(n_size, k_size, 1, 1).astype(np.float32)
    conv_w = builder.add_array(conv_weight, f"__matmul_conv_w_{id(conv_weight)}")

    if len(a_shape) == 3:
        t1 = builder.add_op("Transpose", [vars["?a"]], attrs={"perm": (0, 2, 1)})
        axes = builder.add_array(np.array([3], dtype=np.int64), "__unsq_axes_3", dtype_code=7)
        unsqueezed = builder.add_op("Unsqueeze", [t1, axes])
        conv = builder.add_op("Conv", [unsqueezed, conv_w], attrs={"kernel_shape": (1, 1)})
        t2 = builder.add_op("Transpose", [conv], attrs={"perm": (0, 2, 1, 3)})
        reshape_shape = builder.add_array(np.array([0, 0, -1], dtype=np.int64), "__reshape_00n1", dtype_code=7)
        return builder.add_op("Reshape", [t2, reshape_shape])

    reshape_in_shape = builder.add_array(np.array([1, 0, -1, 1], dtype=np.int64), "__reshape_10n11", dtype_code=7)
    reshape_in = builder.add_op("Reshape", [vars["?a"], reshape_in_shape])
    conv = builder.add_op("Conv", [reshape_in, conv_w], attrs={"kernel_shape": (1, 1)})
    t2 = builder.add_op("Transpose", [conv], attrs={"perm": (0, 2, 1, 3)})
    reshape_out_shape = builder.add_array(np.array([-1, n_size], dtype=np.int64), f"__reshape_n1_{n_size}", dtype_code=7)
    return builder.add_op("Reshape", [t2, reshape_out_shape])


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
