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
        # General Where: Where(c,A,B) → Cast(c)*A + (1-Cast(c))*B
        RuleSpec(
            name="where_to_arithmetic",
            source=P("Where", ("?cond", "?true", "?false")),
            build_fn=_build_where_to_arithmetic,
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
        # Clip(x, 0, max) → Relu(x) - Relu(x - max)
        RuleSpec(
            name="clip_to_relu",
            source=P("Clip", ("?x", "?min", "?max")),
            checks=(
                VarCheck("?min", scalar_close=0.0),
                VarCheck("?max", is_constant=True),
            ),
            build_fn=_build_clip_to_relu,
            family="legalization",
        ),
        # Shape(x) → constant fold when shape is fully static
        RuleSpec(
            name="shape_fold",
            source=P("Shape", ("?x",)),
            checks=(VarCheck("?x", has_shape=True),),
            build_fn=_build_shape_fold,
            family="legalization",
        ),
        # ConstantOfShape(shape) → constant tensor when shape input is constant
        RuleSpec(
            name="constantofshape_fold",
            source=P("ConstantOfShape", ("?shape",)),
            checks=(VarCheck("?shape", is_constant=True),),
            build_fn=_build_constantofshape_fold,
            family="legalization",
        ),
        # Flatten(x) → Reshape(x, [d0*...*d_{axis-1}, d_axis*...*d_n])
        RuleSpec(
            name="flatten_to_reshape",
            source=P("Flatten", ("?x",)),
            checks=(VarCheck("?x", has_shape=True),),
            build_fn=_build_flatten_to_reshape,
            family="legalization",
        ),
        # Gelu(x) → x * Sigmoid(1.702 * x)  (SiLU approximation)
        RuleSpec(
            name="gelu_decompose",
            source=P("Gelu", ("?x",)),
            build_fn=_build_gelu_decompose,
            family="legalization",
        ),
        # LeakyRelu(x, alpha) → Relu(x) + alpha*(x - Relu(x))
        RuleSpec(
            name="leakyrelu_decompose",
            source=P("LeakyRelu", ("?x",)),
            build_fn=_build_leakyrelu_decompose,
            family="legalization",
        ),
        # Expand(x, shape) → Mul(x, ones) when target shape is constant
        RuleSpec(
            name="expand_to_mul_ones",
            source=P("Expand", ("?x", "?shape")),
            checks=(VarCheck("?shape", is_constant=True),),
            build_fn=_build_expand_to_mul_ones,
            family="legalization",
        ),
        # Cos(x) → constant fold when x is constant
        RuleSpec(
            name="cos_fold",
            source=P("Cos", ("?x",)),
            checks=(VarCheck("?x", is_constant=True),),
            build_fn=_build_cos_fold,
            family="legalization",
        ),
        # Sin(x) → constant fold when x is constant
        RuleSpec(
            name="sin_fold",
            source=P("Sin", ("?x",)),
            checks=(VarCheck("?x", is_constant=True),),
            build_fn=_build_sin_fold,
            family="legalization",
        ),
        # HardSigmoid(x) → Relu(x*a + b) - Relu(x*a + b - 1)
        RuleSpec(
            name="hardsigmoid_decompose",
            source=P("HardSigmoid", ("?x",)),
            build_fn=_build_hardsigmoid_decompose,
            family="legalization",
        ),
        # HardSwish(x) → x * HardSigmoid(x)  (decomposed further)
        RuleSpec(
            name="hardswish_decompose",
            source=P("HardSwish", ("?x",)),
            build_fn=_build_hardswish_decompose,
            family="legalization",
        ),
        # Pad(x, pads, val) → identity when all pads are zero
        RuleSpec(
            name="pad_eliminate_zero",
            source=P("Pad", ("?x", "?pads")),
            checks=(VarCheck("?pads", is_constant=True),),
            build_fn=_build_pad_eliminate_zero,
            family="legalization",
        ),
        # Equal/Less/Greater with constant → Cast(comparison) for known patterns
        RuleSpec(
            name="equal_fold",
            source=P("Equal", ("?a", "?b")),
            checks=(
                VarCheck("?a", is_constant=True),
                VarCheck("?b", is_constant=True),
            ),
            build_fn=_build_equal_fold,
            family="legalization",
        ),
        RuleSpec(
            name="less_fold",
            source=P("Less", ("?a", "?b")),
            checks=(
                VarCheck("?a", is_constant=True),
                VarCheck("?b", is_constant=True),
            ),
            build_fn=_build_less_fold,
            family="legalization",
        ),
        # Not(x) → 1 - Cast(x, float) for boolean tensors (via arithmetic)
        RuleSpec(
            name="not_to_sub",
            source=P("Not", ("?x",)),
            build_fn=_build_not_to_sub,
            family="legalization",
        ),
        # Abs(x) → Relu(x) + Relu(-x)  = Relu(x) + Relu(Mul(x, -1))
        RuleSpec(
            name="abs_decompose",
            source=P("Abs", ("?x",)),
            build_fn=_build_abs_decompose,
            family="legalization",
        ),
        # Reciprocal(x) → Div(1, x)
        RuleSpec(
            name="reciprocal_to_div",
            source=P("Reciprocal", ("?x",)),
            build_fn=lambda builder, vars: builder.add_op(
                "Div", [builder.add_scalar(1.0), vars["?x"]]
            ),
            family="legalization",
        ),
        # Ceil/Floor with constant input → fold
        RuleSpec(
            name="ceil_fold",
            source=P("Ceil", ("?x",)),
            checks=(VarCheck("?x", is_constant=True),),
            build_fn=_build_ceil_fold,
            family="legalization",
        ),
        RuleSpec(
            name="floor_fold",
            source=P("Floor", ("?x",)),
            checks=(VarCheck("?x", is_constant=True),),
            build_fn=_build_floor_fold,
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


def _build_clip_to_relu(builder: GraphBuilder, vars: dict[str, object]) -> object:
    """Clip(x, 0, max) → Relu(x) - Relu(x - max).

    Works for ReLU6 and any Clip with min=0 and constant max.
    Identity: x<0 → 0-0=0; 0<=x<=max → x-0=x; x>max → x-(x-max)=max.
    """
    max_data = builder.get_weight_data("?max")
    if max_data is None:
        return builder.get_match()
    max_val = float(max_data.flat[0])
    relu_x = builder.add_op("Relu", [vars["?x"]])
    max_const = builder.add_scalar(max_val, f"__clip_max_{max_val}")
    shifted = builder.add_op("Sub", [vars["?x"], max_const])
    relu_shifted = builder.add_op("Relu", [shifted])
    return builder.add_op("Sub", [relu_x, relu_shifted])


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


def _build_gelu_decompose(
    builder: GraphBuilder, vars: dict[str, object]
) -> object:
    """Gelu(x) → x * Sigmoid(1.702 * x).

    Uses the SiLU-based GELU approximation (fast, accurate for inference).
    """
    c = builder.add_scalar(1.702, "__gelu_c")
    cx = builder.add_op("Mul", [c, vars["?x"]])
    sig = builder.add_op("Sigmoid", [cx])
    return builder.add_op("Mul", [vars["?x"], sig])


def _build_leakyrelu_decompose(
    builder: GraphBuilder, vars: dict[str, object]
) -> object:
    """LeakyRelu(x, alpha) → Relu(x) + alpha * (x - Relu(x))."""
    alpha = builder.get_matched_attr("alpha")
    alpha = 0.01 if alpha is None else float(alpha)
    alpha_w = builder.add_scalar(alpha, f"__lrelu_alpha_{alpha}")
    relu_x = builder.add_op("Relu", [vars["?x"]])
    neg_part = builder.add_op("Sub", [vars["?x"], relu_x])
    scaled_neg = builder.add_op("Mul", [alpha_w, neg_part])
    return builder.add_op("Add", [relu_x, scaled_neg])


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


def _build_hardsigmoid_decompose(
    builder: GraphBuilder, vars: dict[str, object]
) -> object:
    """HardSigmoid(x) → Relu(x*alpha + beta) - Relu(x*alpha + beta - 1).

    Default: alpha=1/6, beta=0.5.
    clip(x*a+b, 0, 1) = Relu(y) - Relu(y-1) where y = x*a + b.
    """
    alpha = builder.get_matched_attr("alpha")
    beta = builder.get_matched_attr("beta")
    alpha = 1.0 / 6.0 if alpha is None else float(alpha)
    beta = 0.5 if beta is None else float(beta)
    a = builder.add_scalar(alpha, f"__hsig_a_{alpha}")
    b = builder.add_scalar(beta, f"__hsig_b_{beta}")
    one = builder.add_scalar(1.0, "__hsig_one")
    y = builder.add_op("Mul", [vars["?x"], a])
    y = builder.add_op("Add", [y, b])
    relu_y = builder.add_op("Relu", [y])
    y_minus_1 = builder.add_op("Sub", [y, one])
    relu_y1 = builder.add_op("Relu", [y_minus_1])
    return builder.add_op("Sub", [relu_y, relu_y1])


def _build_hardswish_decompose(
    builder: GraphBuilder, vars: dict[str, object]
) -> object:
    """HardSwish(x) → x * HardSigmoid(x), decomposed inline.

    = x * clip(x/6 + 0.5, 0, 1)
    = x * (Relu(x/6+0.5) - Relu(x/6-0.5))
    """
    a = builder.add_scalar(1.0 / 6.0, "__hswish_a")
    b = builder.add_scalar(0.5, "__hswish_b")
    one = builder.add_scalar(1.0, "__hswish_one")
    y = builder.add_op("Mul", [vars["?x"], a])
    y = builder.add_op("Add", [y, b])
    relu_y = builder.add_op("Relu", [y])
    y_minus_1 = builder.add_op("Sub", [y, one])
    relu_y1 = builder.add_op("Relu", [y_minus_1])
    hsig = builder.add_op("Sub", [relu_y, relu_y1])
    return builder.add_op("Mul", [vars["?x"], hsig])


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
    except Exception:
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
    except Exception:
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
