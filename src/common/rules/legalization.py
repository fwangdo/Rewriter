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
        # Pow(x, 1) → x. Raising to the power 1 is identity.
        RuleSpec(
            name="pow_to_identity",
            source=PN("Pow", (PV("?x"), PV("?e"))),
            checks=(VarCheck("?e", scalar_close=1.0),),
            build_fn=lambda builder, vars: vars["?x"],
        ),
        # Pow(x, 0.5) → Sqrt(x). Square root is more widely supported than Pow.
        RuleSpec(
            name="pow_to_sqrt",
            source=PN("Pow", (PV("?x"), PV("?e"))),
            checks=(VarCheck("?e", scalar_close=0.5),),
            build_fn=lambda builder, vars: builder.add_op("Sqrt", [vars["?x"]], builder.get_matched_shape(), builder.get_dtype("?x")),
        ),
        # Pow(x, 2) → Mul(x, x). Squaring via Mul avoids Pow.
        RuleSpec(
            name="pow_to_mul",
            source=PN("Pow", (PV("?x"), PV("?e"))),
            checks=(VarCheck("?e", scalar_close=2.0),),
            build_fn=lambda builder, vars: builder.add_op("Mul", [vars["?x"], vars["?x"]], builder.get_matched_shape(), builder.get_dtype("?x")),
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
        # RuleSpec(
        #     name="where_mask_decompose",
        #     source=PN("Where", (PV("?cond"), PV("?true"), PV("?false"))),
        #     checks=(
        #         VarCheck("?true", scalar_abs_lt=1e-8),
        #         VarCheck("?false", scalar_lte=-1.0e30),
        #     ),
        #     build_fn=_build_where_mask_decompose,
        # ),
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
        # Reciprocal(x) → Div(1, x). Reciprocal computes 1/x element-wise.
        # Replaced with Div which is more widely supported.
        RuleSpec(
            name="reciprocal_to_div",
            source=PN("Reciprocal", (PV("?x"),)),
            build_fn=lambda builder, vars: builder.add_op(
                "Div", [builder.add_scalar(1.0, "?x"), vars["?x"]],
                builder.get_matched_shape(), builder.get_dtype("?x"),
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
    """Identity(x) → x.
    Identity is a no-op that passes its input through unchanged. Removed
    because it adds a node without any computation."""
    return vars["?x"]


def _build_greater_to_less(builder: GraphBuilder, vars: dict[str, object]) -> object:
    """Greater(a, b) → Less(b, a).
    Greater returns element-wise a > b. Equivalent to Less with swapped
    operands. Canonicalizes comparison ops to reduce the op vocabulary."""
    shape = builder.get_shape("?b")
    dtype = builder.get_dtype("?b")
    return builder.add_op("Less", [vars["?b"], vars["?a"]], shape, dtype)


def _build_sub_to_add_neg(builder: GraphBuilder, vars: dict[str, object]) -> object:
    """Sub(x, y) → Add(x, Neg(y)).
    Sub computes element-wise x - y. Decomposed into Add + Neg so that
    the backend only needs to support Add, and additive rewrites
    (commutativity, associativity) apply uniformly."""
    shape = builder.get_matched_shape()
    dtype = builder.get_dtype("?y")
    neg = builder.add_op("Neg", [vars["?y"]], builder.get_shape("?y"), dtype)
    return builder.add_op("Add", [vars["?x"], neg], shape, dtype)


def _build_neg_to_mul(builder: GraphBuilder, vars: dict[str, object]) -> object:
    """Neg(x) → Mul(x, -1).
    Neg returns element-wise -x. Expressed as Mul so the backend only
    needs Mul, and Neg is removed from the op vocabulary."""
    shape = builder.get_shape("?x")
    dtype = builder.get_dtype("?x")
    return builder.add_op("Mul", [vars["?x"], builder.add_scalar(-1, "?x")], shape, dtype)


def _build_shape_to_reshape(builder: GraphBuilder, vars: dict[str, object]) -> object:
    """Squeeze(x, axes) or Unsqueeze(x, axes) → Reshape(x, target_shape).
    Squeeze removes size-1 dimensions; Unsqueeze inserts them. Both are
    special cases of Reshape when the full shape is statically known.
    Canonicalized to Reshape to reduce the op vocabulary."""
    shape = builder.get_matched_shape()
    if shape is None or sum(1 for dim in shape if dim == -1) > 1:
        return builder.get_match("shape_to_reshape", "matched shape unknown or ambiguous")
    shape_value = builder.add_array(
        np.array(shape, dtype=np.int64),
        name=f"__shape_{shape}",
        dtype_code=7,
    )

    dtype = builder.get_dtype("?x")
    return builder.add_op("Reshape", [vars["?x"], shape_value], shape, dtype)


def _build_pow_to_cube(builder: GraphBuilder, vars: dict[str, object]) -> object:
    """Pow(x, 3) → Mul(Mul(x, x), x).
    Pow computes element-wise x^e. When e=3, replaced with two Muls to
    avoid Pow which many backends don't support."""
    shape = builder.get_shape("?x")
    dtype = builder.get_dtype("?x")
    squared = builder.add_op("Mul", [vars["?x"], vars["?x"]], shape, dtype)
    return builder.add_op("Mul", [squared, vars["?x"]], shape, dtype)


def _build_pow_to_reciprocal(builder: GraphBuilder, vars: dict[str, object]) -> object:
    """Pow(x, -1) → Div(1, x).
    Computes 1/x. Replaces Pow with Div which is more widely supported."""
    shape = builder.get_shape("?x")
    dtype = builder.get_dtype("?x")
    return builder.add_op("Div", [builder.add_scalar(1.0, "?x"), vars["?x"]], shape, dtype)


def _build_pow_to_rsqrt(builder: GraphBuilder, vars: dict[str, object]) -> object:
    """Pow(x, -0.5) → Div(1, Sqrt(x)).
    Computes 1/sqrt(x). Replaces Pow with Sqrt + Div which are more
    widely supported."""
    shape = builder.get_shape("?x")
    dtype = builder.get_dtype("?x")
    sqrt = builder.add_op("Sqrt", [vars["?x"]], shape, dtype)
    return builder.add_op("Div", [builder.add_scalar(1.0, "?x"), sqrt], shape, dtype)


def _build_layernorm_decompose(builder: GraphBuilder, vars: dict[str, object]) -> object:
    """Decompose LayerNorm(x, scale, bias) into primitive arithmetic ops.

    LayerNorm normalizes the input along ``axis`` so that the slice has
    zero mean and unit variance, then applies an affine transform::

        mean      = ReduceMean(x, axis, keepdims=1)   # shape: [..., 1], dtype: same as x
        centered  = x - mean                          # shape: same as x (broadcast)
        var       = ReduceMean(centered², axis, kd=1)  # shape: [..., 1]
        std       = sqrt(var + epsilon)                # shape: [..., 1]
        out       = (centered / std) * scale + bias    # shape: same as x

    Why decompose:
    - LayerNorm is a single fused op in ONNX (opset 17+), but many
      target backends (e.g. QNN EP) do not support it natively.
    - The decomposed form uses only ReduceMean, Sub, Mul, Add, Sqrt, Div
      which are universally supported.
    - epsilon (default 1e-5) prevents division by zero when variance is
      near zero. It is always float regardless of input dtype, because
      the variance computation is inherently floating-point.
    """
    from src.common.analysis.shape import infer_reduce_mean

    axis = builder.get_matched_attr("axis")
    epsilon = builder.get_matched_attr("epsilon")
    axis = -1 if axis is None else int(axis)
    epsilon = 1e-5 if epsilon is None else float(epsilon)

    x_shape = builder.get_shape("?x")
    x_dtype = builder.get_dtype("?x")
    if x_shape is not None and x_dtype is not None:
        reduced_shape, _ = infer_reduce_mean(
            x_shape, x_dtype, [axis],
            keepdims=True, opset_version=builder.get_opset_version(),
        )
    else:
        reduced_shape = None

    axes = builder.add_array(np.array([axis], dtype=np.int64), f"__ln_axes_{axis}", dtype_code=7)
    # epsilon is always float32: variance is a floating-point quantity.
    eps = builder.add_scalar_float(epsilon, name=f"__ln_eps_{epsilon}")
    mean = builder.add_op("ReduceMean", [vars["?x"], axes],
                          shape=reduced_shape, dtype=x_dtype,
                          attrs={"keepdims": 1})
    centered = builder.add_op("Sub", [vars["?x"], mean],
                              shape=x_shape, dtype=x_dtype)
    squared = builder.add_op("Mul", [centered, centered],
                             shape=x_shape, dtype=x_dtype)
    var = builder.add_op("ReduceMean", [squared, axes],
                         shape=reduced_shape, dtype=x_dtype,
                         attrs={"keepdims": 1})
    var_eps = builder.add_op("Add", [var, eps],
                             shape=reduced_shape, dtype=x_dtype)
    std = builder.add_op("Sqrt", [var_eps],
                         shape=reduced_shape, dtype=x_dtype)
    normalized = builder.add_op("Div", [centered, std],
                                shape=x_shape, dtype=x_dtype)
    scaled = builder.add_op("Mul", [normalized, vars["?scale"]],
                            shape=x_shape, dtype=x_dtype)
    return builder.add_op("Add", [scaled, vars["?bias"]],
                          shape=x_shape, dtype=x_dtype)


# def _build_where_mask_decompose(builder: GraphBuilder, vars: dict[str, object]) -> object:
#     cast = builder.add_op("Cast", [vars["?cond"]], attrs={"to": 1})
#     inverse = builder.add_op("Sub", [builder.add_scalar_float(1.0), cast]) # To check. 
#     return builder.add_op("Mul", [inverse, vars["?false"]])


def _build_range_decompose(builder: GraphBuilder, vars: dict[str, object]) -> object:
    """Range(start=0, limit, step=1) → Slice(arange_table, 0, limit).
    Range generates a 1-D sequence [start, start+step, ..., limit). When
    start=0 and step=1 (checked by VarCheck), this is equivalent to
    slicing a pre-built [0..4095] lookup table up to ``limit``.
    Eliminates Range which many backends don't support."""
    # TODO: we need to check 4096. 
    table = builder.add_array(np.arange(4096, dtype=np.int64), "__arange_table_4096", dtype_code=7)
    starts = builder.add_array(np.array([0], dtype=np.int64), "__slice_starts_0", dtype_code=7)
    axes = builder.add_array(np.array([0], dtype=np.int64), "__slice_axes_0", dtype_code=7)
    steps = builder.add_array(np.array([1], dtype=np.int64), "__slice_steps_1", dtype_code=7)
    ends_shape = builder.add_array(np.array([1], dtype=np.int64), "__shape_1", dtype_code=7)
    ends = builder.add_op("Reshape", [vars["?limit"], ends_shape], (1,), 7)
    return builder.add_op("Slice", [table, starts, ends, axes, steps], builder.get_matched_shape(), 7)


def _build_bn_decompose(builder: GraphBuilder, vars: dict[str, object]) -> object:
    """BatchNormalization(x, s, b, mean, var) → Mul(x, scale_factor) + Add(bias_factor).
    BN normalizes x per-channel using running mean/variance, then applies
    affine transform: out = s * (x - mean) / sqrt(var + eps) + b.
    Since s, b, mean, var are all constant weights, the entire normalization
    is folded into a single Mul + Add with precomputed scale/bias tensors.
    Eliminates BN which is inference-only and easily foldable."""
    epsilon = builder.get_matched_attr("epsilon")
    epsilon = 1e-5 if epsilon is None else float(epsilon)

    s_data = builder.get_weight_data("?s")
    b_data = builder.get_weight_data("?bn_b")
    m_data = builder.get_weight_data("?bn_m")
    v_data = builder.get_weight_data("?bn_v")
    if any(data is None for data in (s_data, b_data, m_data, v_data)):
        return builder.get_match("bn_decompose", "missing BN weight data")

    scale_factor = (s_data / np.sqrt(v_data + epsilon)).astype(s_data.dtype) # type: ignore
    bias_factor = (b_data - m_data * scale_factor).astype(b_data.dtype) # type: ignore

    x_shape = builder.get_shape("?x")
    if x_shape is not None and len(x_shape) == 4:
        channels = scale_factor.shape[0]
        scale_factor = scale_factor.reshape(1, channels, 1, 1)
        bias_factor = bias_factor.reshape(1, channels, 1, 1)

    sf = builder.add_array(scale_factor, f"__bn_scale_{id(scale_factor)}")
    bf = builder.add_array(bias_factor, f"__bn_bias_{id(bias_factor)}")
    shape = builder.get_matched_shape()
    dtype = builder.get_dtype("?x")
    mul = builder.add_op("Mul", [vars["?x"], sf], shape, dtype)
    return builder.add_op("Add", [mul, bf], shape, dtype)


def _build_gemm_decompose(builder: GraphBuilder, vars: dict[str, object]) -> object:
    """Gemm(a, w, b) → Add(MatMul(a', w'), b').
    Gemm computes alpha * A @ B + beta * C with optional transposes.
    Decomposed into MatMul + Add because Gemm is not widely supported
    on edge backends. Transpose and alpha/beta scaling are folded into
    the constant weight and bias at rewrite time."""
    a_value, _, b_value = vars["?a"], vars["?w"], vars["?b"]
    trans_a, trans_b, alpha, beta = _get_gemm_attrs(builder)

    w_data = builder.get_weight_data("?w")
    if w_data is None:
        return builder.get_match("gemm_decompose", "w is not constant")
    if trans_b:
        w_data = w_data.T
    w_data = (alpha * w_data).astype(w_data.dtype)

    a_dtype = builder.get_dtype("?a")
    if trans_a:
        a_shape = builder.get_shape("?a")
        t_shape = (a_shape[1], a_shape[0]) if a_shape is not None and len(a_shape) == 2 else None
        a_value = builder.add_op("Transpose", [a_value], t_shape, a_dtype, attrs={"perm": (1, 0)})

    w_new = builder.add_array(w_data, f"__gemm_w_{id(w_data)}")
    matched_shape = builder.get_matched_shape()
    matmul = builder.add_op("MatMul", [a_value, w_new], matched_shape, a_dtype)

    b_data = builder.get_weight_data("?b")
    # we dont have to consider beta when it is 1.0
    if b_data is not None and not _is_close(beta, 1.0):
        b_data = (beta * b_data).astype(b_data.dtype)
        b_value = builder.add_array(b_data, f"__gemm_bias_{id(b_data)}")

    return builder.add_op("Add", [matmul, b_value], matched_shape, a_dtype)


def _build_gemm_decompose_no_bias(builder: GraphBuilder, vars: dict[str, object]) -> object:
    """Gemm(a, w) → MatMul(a', w').
    Same as gemm_decompose but for the 2-input variant without bias.
    Transpose and alpha scaling are folded into the weight."""
    a_value = vars["?a"]
    trans_a, trans_b, alpha, _beta = _get_gemm_attrs(builder)

    w_data = builder.get_weight_data("?w")
    if w_data is None:
        return builder.get_match("gemm_decompose_no_bias", "w is not constant")
    if trans_b:
        w_data = w_data.T
    w_data = (alpha * w_data).astype(w_data.dtype)

    a_dtype = builder.get_dtype("?a")
    if trans_a:
        a_shape = builder.get_shape("?a")
        t_shape = (a_shape[1], a_shape[0]) if a_shape is not None and len(a_shape) == 2 else None
        a_value = builder.add_op("Transpose", [a_value], t_shape, a_dtype, attrs={"perm": (1, 0)})

    w_new = builder.add_array(w_data, f"__gemm_w_{id(w_data)}")
    matched_shape = builder.get_matched_shape()
    return builder.add_op("MatMul", [a_value, w_new], matched_shape, a_dtype)


def _build_matmul_to_conv(builder: GraphBuilder, vars: dict[str, object]) -> object:
    """MatMul(a, w) → Conv(Reshape(a), Reshape(w)) when both are 2-D.
    MatMul performs matrix multiplication. Some edge backends (e.g. TIDL)
    support Conv but not MatMul. The 2-D matmul [M,K]@[K,N] is equivalent
    to a 1x1 Conv with weight reshaped to [N,K,1,1] after reshaping the
    input to [1,K,M,1] and transposing the output back."""
    w_data = builder.get_weight_data("?w")
    if w_data is None or w_data.ndim != 2:
        return builder.get_match("matmul_to_conv", "w is not 2-D constant")

    a_shape = builder.get_shape("?a")
    if a_shape is None or len(a_shape) != 2:
        return builder.get_match("matmul_to_conv", "a is not 2-D")

    k_size, n_size = w_data.shape
    conv_weight = w_data.T.reshape(n_size, k_size, 1, 1).astype(w_data.dtype)
    conv_w = builder.add_array(conv_weight, f"__matmul_conv_w_{id(conv_weight)}")

    a_dtype = builder.get_dtype("?a")
    m_size = a_shape[0]
    reshape_in_shape = builder.add_array(np.array([1, 0, -1, 1], dtype=np.int64), "__reshape_10n11", dtype_code=7)
    reshape_in = builder.add_op("Reshape", [vars["?a"], reshape_in_shape], (1, k_size, m_size, 1), a_dtype)
    conv = builder.add_op("Conv", [reshape_in, conv_w], (1, n_size, m_size, 1), a_dtype, attrs={"kernel_shape": (1, 1)})
    t2 = builder.add_op("Transpose", [conv], (1, m_size, n_size, 1), a_dtype, attrs={"perm": (0, 2, 1, 3)})
    reshape_out_shape = builder.add_array(np.array([-1, n_size], dtype=np.int64), f"__reshape_n1_{n_size}", dtype_code=7)
    matched_shape = builder.get_matched_shape()
    return builder.add_op("Reshape", [t2, reshape_out_shape], matched_shape, a_dtype)


# TODO: add matmul -> conv for 3d, 4d version. 

def _build_shape_fold(builder: GraphBuilder, vars: dict[str, object]) -> object:
    """Shape(x) → constant int64 tensor when shape is fully static."""
    shape = builder.get_shape("?x")
    if shape is None or any(d < 0 for d in shape):
        return builder.get_match("shape_fold", "shape unknown or dynamic")
    return builder.add_array(
        np.array(shape, dtype=np.int64), f"__shape_const_{shape}", dtype_code=7
    )


def _build_constantofshape_fold(
    builder: GraphBuilder, vars: dict[str, object]
) -> object:
    """ConstantOfShape(shape) → constant tensor filled with the default value."""
    shape_data = builder.get_weight_data("?shape")
    if shape_data is None:
        return builder.get_match("constantofshape_fold", "shape is not constant")
    target_shape = tuple(int(d) for d in shape_data.flat) # flat == flatten.
    if any(d < 0 for d in target_shape):
        return builder.get_match("constantofshape_fold", "shape has dynamic dims")
    # ConstantOfShape default value comes from the "value" attribute (scalar tensor).
    # In the e-graph, numpy arrays are stored as (dtype_str, shape, bytes) tuples.
    value_attr = builder.get_matched_attr("value")
    fill_dtype = np.float32
    if value_attr is not None:
        if isinstance(value_attr, np.ndarray):
            fill_array = np.asarray(value_attr)
            fill_val = fill_array.reshape(-1)[0].item()
            fill_dtype = fill_array.dtype
        elif isinstance(value_attr, tuple) and len(value_attr) == 3:
            dtype_str, _shape, data = value_attr
            fill_array = np.frombuffer(data, dtype=np.dtype(dtype_str))
            fill_val = fill_array.reshape(-1)[0].item()
            fill_dtype = fill_array.dtype
        else:
            try:
                fill_val = float(value_attr) # type: ignore 
            except (TypeError, ValueError):
                return builder.get_match("constantofshape_fold", "unparseable value attr")
    else:
        fill_val = 0.0
    arr = np.full(target_shape, fill_val, dtype=fill_dtype)
    return builder.add_array(
        arr,
        f"__constshape_{target_shape}_{fill_val}",
        dtype_code=_onnx_dtype_code(arr.dtype),
    )


def _build_flatten_to_reshape(
    builder: GraphBuilder, vars: dict[str, object]
) -> object:
    """Flatten(x, axis) → Reshape(x, [pre, post])."""
    shape = builder.get_shape("?x")
    if shape is None or any(d < 0 for d in shape):
        return builder.get_match("flatten_to_reshape", "shape unknown or dynamic")
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
    dtype = builder.get_dtype("?x")
    return builder.add_op("Reshape", [vars["?x"], shape_w], (pre, post), dtype)


def _build_expand_to_mul_ones(
    builder: GraphBuilder, vars: dict[str, object]
) -> object:
    """Expand(x, shape) → Mul(x, ones_of_shape).

    Mul broadcasts x to the target shape automatically.
    """
    shape_data = builder.get_weight_data("?shape")
    if shape_data is None:
        return builder.get_match("expand_to_mul_ones", "shape is not constant")
    target_shape = tuple(int(d) for d in shape_data.flat)
    if any(d < 0 for d in target_shape):
        return builder.get_match("expand_to_mul_ones", "shape has dynamic dims")
    ones = np.ones(target_shape, dtype=np.float32)
    ones_w = builder.add_array(ones, f"__expand_ones_{target_shape}")
    dtype = builder.get_dtype("?x")
    return builder.add_op("Mul", [vars["?x"], ones_w], target_shape, dtype)


def _build_cos_fold(builder: GraphBuilder, vars: dict[str, object]) -> object:
    """Cos(x) → constant when x is a constant tensor."""
    data = builder.get_weight_data("?x")
    if data is None:
        return builder.get_match("cos_fold", "x is not constant")
    result = np.cos(data).astype(data.dtype)
    return builder.add_array(result, f"__cos_folded_{id(result)}")


def _build_sin_fold(builder: GraphBuilder, vars: dict[str, object]) -> object:
    """Sin(x) → constant when x is a constant tensor."""
    data = builder.get_weight_data("?x")
    if data is None:
        return builder.get_match("sin_fold", "x is not constant")
    result = np.sin(data).astype(data.dtype)
    return builder.add_array(result, f"__sin_folded_{id(result)}")


def _build_pad_eliminate_zero(
    builder: GraphBuilder, vars: dict[str, object]
) -> object:
    """Pad(x, pads) → x when all pad values are zero."""
    pads_data = builder.get_weight_data("?pads")
    if pads_data is None:
        return builder.get_match("pad_eliminate_zero", "pads is not constant")
    if np.all(pads_data == 0):
        return vars["?x"]
    return builder.get_match("pad_eliminate_zero", "pads are non-zero")


def _build_where_to_arithmetic(
    builder: GraphBuilder, vars: dict[str, object]
) -> object:
    """Where(cond, A, B) → Cast(cond)*A + (1-Cast(cond))*B.
    Where selects elements from A where cond is True, B where False.
    Decomposed into arithmetic: Cast bool→float as a mask, multiply
    each branch by mask/inverse-mask, then add. Eliminates Where which
    some backends don't support."""
    cond_shape = builder.get_shape("?cond")
    out_dtype = builder.get_dtype("?true")
    cast_to = out_dtype if out_dtype is not None else 1
    cast = builder.add_op("Cast", [vars["?cond"]], cond_shape, cast_to, attrs={"to": cast_to})
    one = builder.add_scalar(1.0, "?true")
    inv = builder.add_op("Sub", [one, cast], cond_shape, cast_to)
    matched_shape = builder.get_matched_shape()
    true_branch = builder.add_op("Mul", [cast, vars["?true"]], matched_shape, out_dtype)
    false_branch = builder.add_op("Mul", [inv, vars["?false"]], matched_shape, out_dtype)
    return builder.add_op("Add", [true_branch, false_branch], matched_shape, out_dtype)


def _build_equal_fold(builder: GraphBuilder, vars: dict[str, object]) -> object:
    """Equal(a, b) → constant bool tensor when both inputs are constant."""
    a_data = builder.get_weight_data("?a")
    b_data = builder.get_weight_data("?b")
    if a_data is None or b_data is None:
        return builder.get_match("equal_fold", "inputs are not both constant")
    try:
        result = a_data == b_data
    except (ValueError, TypeError):
        return builder.get_match("equal_fold", "comparison failed")
    return builder.add_array(result, f"__equal_folded_{id(result)}", dtype_code=9)


def _build_less_fold(builder: GraphBuilder, vars: dict[str, object]) -> object:
    """Less(a, b) → constant bool tensor when both inputs are constant."""
    a_data = builder.get_weight_data("?a")
    b_data = builder.get_weight_data("?b")
    if a_data is None or b_data is None:
        return builder.get_match("less_fold", "inputs are not both constant")
    try:
        result = a_data < b_data
    except (ValueError, TypeError):
        return builder.get_match("less_fold", "comparison failed")
    return builder.add_array(result, f"__less_folded_{id(result)}", dtype_code=9)


def _build_not_to_sub(builder: GraphBuilder, vars: dict[str, object]) -> object:
    """Not(x) → Sub(1, Cast(x, float)).
    Not returns element-wise logical negation of a bool tensor.
    Cast bool→float maps True→1, False→0, then 1-x flips it.
    Eliminates Not by decomposing into Cast + Sub."""
    x_shape = builder.get_shape("?x")
    cast = builder.add_op("Cast", [vars["?x"]], x_shape, 1, attrs={"to": 1})
    one = builder.add_scalar(1.0, "?x")
    matched_shape = builder.get_matched_shape()
    return builder.add_op("Sub", [one, cast], matched_shape, 1)


def _build_abs_decompose(builder: GraphBuilder, vars: dict[str, object]) -> object:
    """Abs(x) → Relu(x) + Relu(-x).
    Abs returns element-wise |x|. Decomposed using Relu(x) for the
    positive part and Relu(-x) for the negative part. Their sum equals
    |x|. Eliminates Abs by using Relu + Mul + Add."""
    shape = builder.get_shape("?x")
    dtype = builder.get_dtype("?x")
    neg_one = builder.add_scalar(-1.0, "?x")
    relu_pos = builder.add_op("Relu", [vars["?x"]], shape, dtype)
    neg_x = builder.add_op("Mul", [vars["?x"], neg_one], shape, dtype)
    relu_neg = builder.add_op("Relu", [neg_x], shape, dtype)
    return builder.add_op("Add", [relu_pos, relu_neg], shape, dtype)


def _build_ceil_fold(builder: GraphBuilder, vars: dict[str, object]) -> object:
    """Ceil(x) → constant when x is constant."""
    data = builder.get_weight_data("?x")
    if data is None:
        return builder.get_match("ceil_fold", "x is not constant")
    result = np.ceil(data).astype(data.dtype)
    return builder.add_array(result, f"__ceil_folded_{id(result)}")


def _build_floor_fold(builder: GraphBuilder, vars: dict[str, object]) -> object:
    """Floor(x) → constant when x is constant."""
    data = builder.get_weight_data("?x")
    if data is None:
        return builder.get_match("floor_fold", "x is not constant")
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


def _onnx_dtype_code(dtype: np.dtype) -> int:
    dtype = np.dtype(dtype)
    if dtype == np.dtype("float32"):
        return 1
    if dtype == np.dtype("uint8"):
        return 2
    if dtype == np.dtype("int8"):
        return 3
    if dtype == np.dtype("uint16"):
        return 4
    if dtype == np.dtype("int16"):
        return 5
    if dtype == np.dtype("int32"):
        return 6
    if dtype == np.dtype("int64"):
        return 7
    if dtype == np.dtype("bool"):
        return 9
    if dtype == np.dtype("float64"):
        return 11
    return 1
