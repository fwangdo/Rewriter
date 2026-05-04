"""Legalization rewrite rules.

These rules lower unsupported ops into supported equivalents.
They mirror the decompositions in onnx_rewrite/passes/ but are
encoded as e-graph equality rules.

Key difference from onnx_rewrite passes: in the e-graph, both
the original and decomposed forms coexist as equivalents.
The extraction phase decides which to pick based on the cost model
and legality constraints.
"""

from __future__ import annotations

from .base import RewriteRule
from ..egraph.enode import EClassId, ENode
from ..egraph.egraph import EGraph
from ..egraph.pattern import PatternNode, PatternVar, Subst


def get_legalization_rules() -> list[RewriteRule]:
    """Return legalization rewrite rules."""
    x = PatternVar("?x")
    a = PatternVar("?a")
    b = PatternVar("?b")
    y = PatternVar("?y")
    e = PatternVar("?e")
    scale = PatternVar("?scale")
    bias = PatternVar("?bias")
    clip_min = PatternVar("?min")
    clip_max = PatternVar("?max")
    cond = PatternVar("?cond")
    true_val = PatternVar("?true")
    false_val = PatternVar("?false")
    start = PatternVar("?start")
    limit = PatternVar("?limit")
    step = PatternVar("?step")
    s = PatternVar("?s")
    bn_b = PatternVar("?bn_b")
    bn_m = PatternVar("?bn_m")
    bn_v = PatternVar("?bn_v")
    w = PatternVar("?w")

    rules: list[RewriteRule] = []

    # --- F2: eliminate_identity: Identity(x) → x ---
    rules.append(RewriteRule(
        name="eliminate_identity",
        source=PatternNode("Identity", (x,)),
        target=x,
    ))

    # --- neg_to_mul: Neg(x) → Mul(x, Constant(-1)) ---
    rules.append(RewriteRule(
        name="neg_to_mul",
        source=PatternNode("Neg", (x,)),
        target=PatternNode("Neg", (x,)),  # placeholder, apply_fn overrides
        apply_fn=_apply_neg_to_mul,
    ))

    # --- greater_to_less: Greater(a, b) → Less(b, a) ---
    rules.append(RewriteRule(
        name="greater_to_less",
        source=PatternNode("Greater", (a, b)),
        target=PatternNode("Less", (b, a)),
    ))

    # --- sub_to_add_neg: Sub(x, y) → Add(x, Neg(y)) ---
    rules.append(RewriteRule(
        name="sub_to_add_neg",
        source=PatternNode("Sub", (x, y)),
        target=PatternNode("Add", (x, PatternNode("Neg", (y,)))),
    ))

    # --- squeeze_to_reshape: Squeeze(x, axes) → Reshape(x, shape) ---
    rules.append(RewriteRule(
        name="squeeze_to_reshape",
        source=PatternNode("Squeeze", (x, PatternVar("?axes"))),
        target=PatternNode("Squeeze", (x, PatternVar("?axes"))),  # placeholder
        check=_check_has_shape,
        apply_fn=_apply_squeeze_to_reshape,
    ))

    # --- unsqueeze_to_reshape: Unsqueeze(x, axes) → Reshape(x, shape) ---
    rules.append(RewriteRule(
        name="unsqueeze_to_reshape",
        source=PatternNode("Unsqueeze", (x, PatternVar("?axes"))),
        target=PatternNode("Unsqueeze", (x, PatternVar("?axes"))),  # placeholder
        check=_check_has_shape,
        apply_fn=_apply_unsqueeze_to_reshape,
    ))

    # --- F3: Pow decomposition (6 variants) ---
    # Pow(x, e) where e is a scalar constant exponent
    rules.append(RewriteRule(
        name="pow_to_identity",
        source=PatternNode("Pow", (x, e)),
        target=PatternNode("Pow", (x, e)),  # placeholder
        check=_check_pow_exp(1.0),
        apply_fn=_apply_pow_to_identity,
    ))
    rules.append(RewriteRule(
        name="pow_to_sqrt",
        source=PatternNode("Pow", (x, e)),
        target=PatternNode("Pow", (x, e)),  # placeholder
        check=_check_pow_exp(0.5),
        apply_fn=_apply_pow_to_sqrt,
    ))
    rules.append(RewriteRule(
        name="pow_to_mul",
        source=PatternNode("Pow", (x, e)),
        target=PatternNode("Pow", (x, e)),  # placeholder
        check=_check_pow_exp(2.0),
        apply_fn=_apply_pow_to_mul,
    ))
    rules.append(RewriteRule(
        name="pow_to_cube",
        source=PatternNode("Pow", (x, e)),
        target=PatternNode("Pow", (x, e)),  # placeholder
        check=_check_pow_exp(3.0),
        apply_fn=_apply_pow_to_cube,
    ))
    rules.append(RewriteRule(
        name="pow_to_reciprocal",
        source=PatternNode("Pow", (x, e)),
        target=PatternNode("Pow", (x, e)),  # placeholder
        check=_check_pow_exp(-1.0),
        apply_fn=_apply_pow_to_reciprocal,
    ))
    rules.append(RewriteRule(
        name="pow_to_rsqrt",
        source=PatternNode("Pow", (x, e)),
        target=PatternNode("Pow", (x, e)),  # placeholder
        check=_check_pow_exp(-0.5),
        apply_fn=_apply_pow_to_rsqrt,
    ))

    # --- F4: LayerNorm decomposition ---
    rules.append(RewriteRule(
        name="layernorm_decompose",
        source=PatternNode("LayerNormalization", (x, scale, bias)),
        target=PatternNode("LayerNormalization", (x, scale, bias)),  # placeholder
        apply_fn=_apply_layernorm_decompose,
    ))

    # --- F5: Clip decomposition ---
    # Clip(x, min, max) → Min(Max(x, min), max)
    rules.append(RewriteRule(
        name="clip_decompose",
        source=PatternNode("Clip", (x, clip_min, clip_max)),
        target=PatternNode("Min", (PatternNode("Max", (x, clip_min)), clip_max)),
    ))

    # --- F6: WhereMask decomposition ---
    # Where(cond, 0, -inf) → Mul(Sub(1, Cast(cond)), -inf)
    rules.append(RewriteRule(
        name="where_mask_decompose",
        source=PatternNode("Where", (cond, true_val, false_val)),
        target=PatternNode("Where", (cond, true_val, false_val)),  # placeholder
        check=_check_where_mask,
        apply_fn=_apply_where_mask_decompose,
    ))

    # --- F7: Range decomposition ---
    # Range(0, limit, 1) → Slice(arange_table, 0, limit)
    rules.append(RewriteRule(
        name="range_decompose",
        source=PatternNode("Range", (start, limit, step)),
        target=PatternNode("Range", (start, limit, step)),  # placeholder
        check=_check_range,
        apply_fn=_apply_range_decompose,
    ))

    # --- F8: BN standalone decomposition ---
    rules.append(RewriteRule(
        name="bn_decompose",
        source=PatternNode("BatchNormalization", (x, s, bn_b, bn_m, bn_v)),
        target=PatternNode("BatchNormalization", (x, s, bn_b, bn_m, bn_v)),  # placeholder
        apply_fn=_apply_bn_decompose,
    ))

    # --- F9: Gemm decomposition ---
    rules.append(RewriteRule(
        name="gemm_decompose",
        source=PatternNode("Gemm", (a, w, b)),
        target=PatternNode("Gemm", (a, w, b)),  # placeholder
        apply_fn=_apply_gemm_decompose,
    ))
    rules.append(RewriteRule(
        name="gemm_decompose_no_bias",
        source=PatternNode("Gemm", (a, w)),
        target=PatternNode("Gemm", (a, w)),  # placeholder
        apply_fn=_apply_gemm_decompose_no_bias,
    ))

    # --- F10: MatMul→Conv (static weight) ---
    rules.append(RewriteRule(
        name="matmul_to_conv",
        source=PatternNode("MatMul", (a, w)),
        target=PatternNode("MatMul", (a, w)),  # placeholder
        check=_check_matmul_static_weight,
        apply_fn=_apply_matmul_to_conv,
    ))

    return rules


# --- apply_fn implementations ---

import numpy as np
from ..egraph.eclass import AnalysisData


def _add_scalar_constant(egraph: EGraph, value: float, dtype: int = 1) -> EClassId:
    """Add a scalar constant enode to the egraph, return its eclass id."""
    arr = np.array(value, dtype=np.float32)
    hashable = (str(arr.dtype), arr.shape, arr.tobytes())
    enode = ENode("weight", (), attrs=(
        ("__name__", f"__const_{value}"),
        ("__synth__", hashable),
    ))
    cid = egraph.add(enode)
    egraph.update_analysis(cid, AnalysisData(
        shape=tuple(arr.shape),
        dtype=dtype,
        is_constant=True,
        scalar_value=value,
    ))
    return cid


def _add_shape_constant(egraph: EGraph, shape: tuple[int, ...]) -> EClassId:
    """Add a shape constant (weight node) to the egraph."""
    arr = np.array(shape, dtype=np.int64)
    hashable = (str(arr.dtype), arr.shape, arr.tobytes())
    enode = ENode("weight", (), attrs=(
        ("__name__", f"__shape_{shape}"),
        ("__synth__", hashable),
    ))
    cid = egraph.add(enode)
    egraph.update_analysis(cid, AnalysisData(
        shape=tuple(arr.shape),
        dtype=7,  # TensorProto.INT64
        is_constant=True,
    ))
    return cid


def _add_ndarray_constant(egraph: EGraph, arr: np.ndarray, name: str, dtype: int = 1) -> EClassId:
    """Add an arbitrary ndarray constant to the egraph."""
    arr = np.ascontiguousarray(arr)
    hashable = (str(arr.dtype), arr.shape, arr.tobytes())
    enode = ENode("weight", (), attrs=(
        ("__name__", name),
        ("__synth__", hashable),
    ))
    cid = egraph.add(enode)
    scalar_value = float(arr.reshape(-1)[0]) if arr.size == 1 else None
    egraph.update_analysis(cid, AnalysisData(
        shape=tuple(arr.shape),
        dtype=dtype,
        is_constant=True,
        scalar_value=scalar_value,
    ))
    return cid


# --- Neg ---

def _apply_neg_to_mul(
    egraph: EGraph, match_cid: EClassId, subst: Subst,
) -> EClassId:
    """Neg(x) → Mul(x, -1_constant)."""
    x_cid = subst["?x"]
    neg1_cid = _add_scalar_constant(egraph, -1.0)
    mul_enode = ENode("Mul", (x_cid, neg1_cid))
    return egraph.add(mul_enode)


# --- Squeeze/Unsqueeze ---

def _check_has_shape(egraph: EGraph, subst: Subst) -> bool:
    """Check that the matched source eclass has known shape."""
    x_cid = subst["?x"]
    return egraph.eclass(x_cid).data.shape is not None


def _apply_squeeze_to_reshape(
    egraph: EGraph, match_cid: EClassId, subst: Subst,
) -> EClassId:
    """Squeeze(x, axes) → Reshape(x, target_shape)."""
    x_cid = subst["?x"]
    target_shape = egraph.eclass(match_cid).data.shape
    if target_shape is None:
        return match_cid
    shape_cid = _add_shape_constant(egraph, target_shape)
    reshape_enode = ENode("Reshape", (x_cid, shape_cid))
    return egraph.add(reshape_enode)


def _apply_unsqueeze_to_reshape(
    egraph: EGraph, match_cid: EClassId, subst: Subst,
) -> EClassId:
    """Unsqueeze(x, axes) → Reshape(x, target_shape)."""
    x_cid = subst["?x"]
    target_shape = egraph.eclass(match_cid).data.shape
    if target_shape is None:
        return match_cid
    shape_cid = _add_shape_constant(egraph, target_shape)
    reshape_enode = ENode("Reshape", (x_cid, shape_cid))
    return egraph.add(reshape_enode)


# --- F3: Pow decomposition ---

def _is_close(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-6


def _check_pow_exp(target_exp: float):
    """Return a check function for Pow with a specific exponent."""
    def _check(egraph: EGraph, subst: Subst) -> bool:
        e_cid = subst["?e"]
        sv = egraph.eclass(e_cid).data.scalar_value
        return sv is not None and _is_close(sv, target_exp)
    return _check


def _apply_pow_to_identity(
    egraph: EGraph, match_cid: EClassId, subst: Subst,
) -> EClassId:
    """Pow(x, 1) → x."""
    return subst["?x"]


def _apply_pow_to_sqrt(
    egraph: EGraph, match_cid: EClassId, subst: Subst,
) -> EClassId:
    """Pow(x, 0.5) → Sqrt(x)."""
    x_cid = subst["?x"]
    return egraph.add(ENode("Sqrt", (x_cid,)))


def _apply_pow_to_mul(
    egraph: EGraph, match_cid: EClassId, subst: Subst,
) -> EClassId:
    """Pow(x, 2) → Mul(x, x)."""
    x_cid = subst["?x"]
    return egraph.add(ENode("Mul", (x_cid, x_cid)))


def _apply_pow_to_cube(
    egraph: EGraph, match_cid: EClassId, subst: Subst,
) -> EClassId:
    """Pow(x, 3) → Mul(Mul(x, x), x)."""
    x_cid = subst["?x"]
    sq_cid = egraph.add(ENode("Mul", (x_cid, x_cid)))
    return egraph.add(ENode("Mul", (sq_cid, x_cid)))


def _apply_pow_to_reciprocal(
    egraph: EGraph, match_cid: EClassId, subst: Subst,
) -> EClassId:
    """Pow(x, -1) → Div(1, x)."""
    x_cid = subst["?x"]
    one_cid = _add_scalar_constant(egraph, 1.0)
    return egraph.add(ENode("Div", (one_cid, x_cid)))


def _apply_pow_to_rsqrt(
    egraph: EGraph, match_cid: EClassId, subst: Subst,
) -> EClassId:
    """Pow(x, -0.5) → Div(1, Sqrt(x))."""
    x_cid = subst["?x"]
    sqrt_cid = egraph.add(ENode("Sqrt", (x_cid,)))
    one_cid = _add_scalar_constant(egraph, 1.0)
    return egraph.add(ENode("Div", (one_cid, sqrt_cid)))


# --- F4: LayerNorm decomposition ---

def _apply_layernorm_decompose(
    egraph: EGraph, match_cid: EClassId, subst: Subst,
) -> EClassId:
    """LayerNormalization(x, scale, bias) → ReduceMean+Sub+Mul+ReduceMean+Add+Sqrt+Div+Mul+Add.

    Reads axis/epsilon from the matched enode's attrs.
    """
    x_cid = subst["?x"]
    scale_cid = subst["?scale"]
    bias_cid = subst["?bias"]

    # Get axis and epsilon from the matched LayerNorm enode attrs
    axis = -1
    epsilon = 1e-5
    ec = egraph.eclass(match_cid)
    for nid in ec.nodes:
        enode = egraph.enode(nid)
        if enode.op == "LayerNormalization":
            for k, v in enode.attrs:
                if k == "axis":
                    axis = int(v)
                elif k == "epsilon":
                    epsilon = float(v)
            break

    # Create constants
    axes_arr = np.array([axis], dtype=np.int64)
    axes_cid = _add_ndarray_constant(egraph, axes_arr, f"__ln_axes_{axis}", dtype=7)
    eps_cid = _add_scalar_constant(egraph, epsilon)

    # mean = ReduceMean(x, axes)
    mean_cid = egraph.add(ENode("ReduceMean", (x_cid, axes_cid), attrs=(("keepdims", 1),)))
    # centered = Sub(x, mean)
    centered_cid = egraph.add(ENode("Sub", (x_cid, mean_cid)))
    # squared = Mul(centered, centered)
    squared_cid = egraph.add(ENode("Mul", (centered_cid, centered_cid)))
    # var = ReduceMean(squared, axes)
    var_cid = egraph.add(ENode("ReduceMean", (squared_cid, axes_cid), attrs=(("keepdims", 1),)))
    # var_eps = Add(var, epsilon)
    var_eps_cid = egraph.add(ENode("Add", (var_cid, eps_cid)))
    # std = Sqrt(var_eps)
    std_cid = egraph.add(ENode("Sqrt", (var_eps_cid,)))
    # normalized = Div(centered, std)
    normalized_cid = egraph.add(ENode("Div", (centered_cid, std_cid)))
    # scaled = Mul(normalized, scale)
    scaled_cid = egraph.add(ENode("Mul", (normalized_cid, scale_cid)))
    # output = Add(scaled, bias)
    output_cid = egraph.add(ENode("Add", (scaled_cid, bias_cid)))
    return output_cid


# --- F6: WhereMask decomposition ---

def _check_where_mask(egraph: EGraph, subst: Subst) -> bool:
    """Check Where(cond, 0, -inf) pattern."""
    true_cid = subst["?true"]
    false_cid = subst["?false"]
    true_sv = egraph.eclass(true_cid).data.scalar_value
    false_sv = egraph.eclass(false_cid).data.scalar_value
    if true_sv is None or false_sv is None:
        return False
    return abs(true_sv) < 1e-8 and false_sv <= -1.0e30


def _apply_where_mask_decompose(
    egraph: EGraph, match_cid: EClassId, subst: Subst,
) -> EClassId:
    """Where(cond, 0, -inf) → Mul(Sub(1, Cast(cond)), -inf)."""
    cond_cid = subst["?cond"]
    false_cid = subst["?false"]  # the -inf value

    # Cast(cond, to=FLOAT)
    cast_cid = egraph.add(ENode("Cast", (cond_cid,), attrs=(("to", 1),)))
    # Sub(1, cast)
    one_cid = _add_scalar_constant(egraph, 1.0)
    inverse_cid = egraph.add(ENode("Sub", (one_cid, cast_cid)))
    # Mul(inverse, -inf)
    return egraph.add(ENode("Mul", (inverse_cid, false_cid)))


# --- F7: Range decomposition ---

def _check_range(egraph: EGraph, subst: Subst) -> bool:
    """Check Range(0, limit, 1)."""
    start_sv = egraph.eclass(subst["?start"]).data.scalar_value
    step_sv = egraph.eclass(subst["?step"]).data.scalar_value
    if start_sv is None or step_sv is None:
        return False
    return _is_close(start_sv, 0.0) and _is_close(step_sv, 1.0)


def _apply_range_decompose(
    egraph: EGraph, match_cid: EClassId, subst: Subst,
) -> EClassId:
    """Range(0, limit, 1) → Slice(arange_table, 0, Reshape(limit,[1]), [0], [1])."""
    limit_cid = subst["?limit"]

    MAX_TABLE = 4096
    table_arr = np.arange(MAX_TABLE, dtype=np.int64)
    table_cid = _add_ndarray_constant(egraph, table_arr, "__arange_table_4096", dtype=7)

    starts_cid = _add_ndarray_constant(egraph, np.array([0], dtype=np.int64), "__slice_starts_0", dtype=7)
    axes_cid = _add_ndarray_constant(egraph, np.array([0], dtype=np.int64), "__slice_axes_0", dtype=7)
    steps_cid = _add_ndarray_constant(egraph, np.array([1], dtype=np.int64), "__slice_steps_1", dtype=7)

    # Reshape limit to [1] for Slice ends input
    ends_shape_cid = _add_ndarray_constant(egraph, np.array([1], dtype=np.int64), "__shape_1", dtype=7)
    ends_cid = egraph.add(ENode("Reshape", (limit_cid, ends_shape_cid)))

    # Slice(table, starts, ends, axes, steps)
    return egraph.add(ENode("Slice", (table_cid, starts_cid, ends_cid, axes_cid, steps_cid)))


# --- F8: BN standalone decomposition ---

def _apply_bn_decompose(
    egraph: EGraph, match_cid: EClassId, subst: Subst,
) -> EClassId:
    """BatchNormalization(x, s, b, m, v) → Add(Mul(x, scale_factor), bias_factor).

    Computes scale_factor = s / sqrt(v + eps) and bias_factor = b - m * scale_factor
    from initializer data.
    """
    x_cid = subst["?x"]

    # Get epsilon from matched enode attrs
    epsilon = 1e-5
    ec = egraph.eclass(match_cid)
    for nid in ec.nodes:
        enode = egraph.enode(nid)
        if enode.op == "BatchNormalization":
            for k, v in enode.attrs:
                if k == "epsilon":
                    epsilon = float(v)
            break

    # Try to extract weight data from the eclasses
    # We need the actual numpy arrays for s, b, m, v
    s_data = _get_synth_data(egraph, subst["?s"])
    b_data = _get_synth_data(egraph, subst["?bn_b"])
    m_data = _get_synth_data(egraph, subst["?bn_m"])
    v_data = _get_synth_data(egraph, subst["?bn_v"])

    if any(d is None for d in (s_data, b_data, m_data, v_data)):
        # Can't decompose without weight data; return match_cid to avoid change
        return match_cid

    scale_factor = s_data / np.sqrt(v_data + epsilon)
    bias_factor = b_data - m_data * scale_factor
    scale_factor = scale_factor.astype(np.float32)
    bias_factor = bias_factor.astype(np.float32)

    # Reshape to [1, C, 1, 1] for 4D broadcasting if needed
    x_shape = egraph.eclass(x_cid).data.shape
    if x_shape is not None and len(x_shape) == 4:
        C = scale_factor.shape[0]
        scale_factor = scale_factor.reshape(1, C, 1, 1)
        bias_factor = bias_factor.reshape(1, C, 1, 1)

    sf_cid = _add_ndarray_constant(egraph, scale_factor, f"__bn_scale_{id(scale_factor)}")
    bf_cid = _add_ndarray_constant(egraph, bias_factor, f"__bn_bias_{id(bias_factor)}")

    # Mul(x, scale_factor)
    mul_cid = egraph.add(ENode("Mul", (x_cid, sf_cid)))
    # Add(mul, bias_factor)
    return egraph.add(ENode("Add", (mul_cid, bf_cid)))


# --- F9: Gemm decomposition ---

def _apply_gemm_decompose(
    egraph: EGraph, match_cid: EClassId, subst: Subst,
) -> EClassId:
    """Gemm(a, w, b) → Reshape(Transpose(Conv(Unsqueeze(Transpose(a)), w'), ...), ...) + bias.

    Simplification: we lower Gemm to MatMul + Add which are both supported,
    letting the cost model and other rules handle further optimization.
    """
    a_cid = subst["?a"]
    w_cid = subst["?w"]
    b_cid = subst["?b"]

    # Get Gemm attributes
    trans_a = 0
    trans_b = 0
    alpha = 1.0
    beta = 1.0
    ec = egraph.eclass(match_cid)
    for nid in ec.nodes:
        enode = egraph.enode(nid)
        if enode.op == "Gemm":
            for k, v in enode.attrs:
                if k == "transA":
                    trans_a = int(v)
                elif k == "transB":
                    trans_b = int(v)
                elif k == "alpha":
                    alpha = float(v)
                elif k == "beta":
                    beta = float(v)
            break

    # Get weight data to compute transposed weight
    w_data = _get_synth_data(egraph, w_cid)
    if w_data is None:
        return match_cid

    # Prepare weight: Gemm computes alpha * (A @ B) + beta * C
    # where B = W.T if transB else W
    if trans_b:
        w_data = w_data.T
    w_data = (alpha * w_data).astype(np.float32)

    # Gemm(a, w, b) → Add(MatMul(a', w'), b * beta)
    # Handle transA
    if trans_a:
        a_cid = egraph.add(ENode("Transpose", (a_cid,), attrs=(("perm", (1, 0)),)))

    # MatMul(a, w)
    w_new_cid = _add_ndarray_constant(egraph, w_data, f"__gemm_w_{id(w_data)}")
    matmul_cid = egraph.add(ENode("MatMul", (a_cid, w_new_cid)))

    # Handle bias
    b_data = _get_synth_data(egraph, b_cid)
    if b_data is not None and not _is_close(beta, 1.0):
        b_data = (beta * b_data).astype(np.float32)
        b_cid = _add_ndarray_constant(egraph, b_data, f"__gemm_bias_{id(b_data)}")

    # Add(matmul, bias)
    return egraph.add(ENode("Add", (matmul_cid, b_cid)))


def _apply_gemm_decompose_no_bias(
    egraph: EGraph, match_cid: EClassId, subst: Subst,
) -> EClassId:
    """Gemm(a, w) → MatMul(a', w') for bias-less Gemm."""
    a_cid = subst["?a"]
    w_cid = subst["?w"]

    trans_a = 0
    trans_b = 0
    alpha = 1.0
    ec = egraph.eclass(match_cid)
    for nid in ec.nodes:
        enode = egraph.enode(nid)
        if enode.op == "Gemm":
            for k, v in enode.attrs:
                if k == "transA":
                    trans_a = int(v)
                elif k == "transB":
                    trans_b = int(v)
                elif k == "alpha":
                    alpha = float(v)
            break

    w_data = _get_synth_data(egraph, w_cid)
    if w_data is None:
        return match_cid

    if trans_b:
        w_data = w_data.T
    w_data = (alpha * w_data).astype(np.float32)

    if trans_a:
        a_cid = egraph.add(ENode("Transpose", (a_cid,), attrs=(("perm", (1, 0)),)))

    w_new_cid = _add_ndarray_constant(egraph, w_data, f"__gemm_w_{id(w_data)}")
    return egraph.add(ENode("MatMul", (a_cid, w_new_cid)))


# --- F10: MatMul → Conv ---

def _check_matmul_static_weight(egraph: EGraph, subst: Subst) -> bool:
    """Check that the right operand of MatMul is a constant."""
    w_cid = subst["?w"]
    return egraph.eclass(w_cid).data.is_constant


def _apply_matmul_to_conv(
    egraph: EGraph, match_cid: EClassId, subst: Subst,
) -> EClassId:
    """MatMul(a, w) → Reshape(Transpose(Conv(Reshape(a), w_conv))) when w is static.

    Converts the weight to [N, K, 1, 1] conv format.
    """
    a_cid = subst["?a"]
    w_cid = subst["?w"]

    w_data = _get_synth_data(egraph, w_cid)
    if w_data is None or w_data.ndim != 2:
        return match_cid

    a_shape = egraph.eclass(a_cid).data.shape
    if a_shape is None or len(a_shape) not in (2, 3):
        return match_cid

    K, N = w_data.shape
    # Conv weight: [N, K, 1, 1]
    conv_weight = w_data.T.reshape(N, K, 1, 1).astype(np.float32)
    conv_w_cid = _add_ndarray_constant(egraph, conv_weight, f"__matmul_conv_w_{id(conv_weight)}")

    if len(a_shape) == 3:
        # [B, M, K] → transpose to [B, K, M], unsqueeze to [B, K, M, 1]
        t1_cid = egraph.add(ENode("Transpose", (a_cid,), attrs=(("perm", (0, 2, 1)),)))
        axes_cid = _add_ndarray_constant(egraph, np.array([3], dtype=np.int64), "__unsq_axes_3", dtype=7)
        us_cid = egraph.add(ENode("Unsqueeze", (t1_cid, axes_cid)))
        # Conv → [B, N, M, 1]
        conv_cid = egraph.add(ENode("Conv", (us_cid, conv_w_cid), attrs=(("kernel_shape", (1, 1)),)))
        # Transpose → [B, M, N, 1]
        t2_cid = egraph.add(ENode("Transpose", (conv_cid,), attrs=(("perm", (0, 2, 1, 3)),)))
        # Reshape → [B, M, N] using [0, 0, -1]
        rs_shape = np.array([0, 0, -1], dtype=np.int64)
        rs_cid = _add_ndarray_constant(egraph, rs_shape, "__reshape_00n1", dtype=7)
        return egraph.add(ENode("Reshape", (t2_cid, rs_cid)))
    else:
        # 2D: [M, K] → reshape to [1, K, M, 1]
        rs1_shape = np.array([1, 0, -1, 1], dtype=np.int64)
        rs1_cid = _add_ndarray_constant(egraph, rs1_shape, "__reshape_10n11", dtype=7)
        rs1_out = egraph.add(ENode("Reshape", (a_cid, rs1_cid)))
        # Conv → [1, N, M, 1]
        conv_cid = egraph.add(ENode("Conv", (rs1_out, conv_w_cid), attrs=(("kernel_shape", (1, 1)),)))
        # Transpose → [1, M, N, 1]
        t2_cid = egraph.add(ENode("Transpose", (conv_cid,), attrs=(("perm", (0, 2, 1, 3)),)))
        # Reshape → [M, N]
        rs2_shape = np.array([-1, N], dtype=np.int64)
        rs2_cid = _add_ndarray_constant(egraph, rs2_shape, f"__reshape_n1_{N}", dtype=7)
        return egraph.add(ENode("Reshape", (t2_cid, rs2_cid)))


# --- Utility: extract initializer data from an eclass ---

def _get_synth_data(egraph: EGraph, cid: EClassId) -> np.ndarray | None:
    """Try to extract numpy array data from a weight eclass.

    Checks both __synth__ attrs (synthetic weights from apply_fn) and
    egraph.initializers (original ONNX initializers).
    """
    ec = egraph.eclass(cid)
    if not ec.data.is_constant:
        return None
    for nid in ec.nodes:
        enode = egraph.enode(nid)
        if enode.op == "weight":
            # Check synthetic data first
            for k, v in enode.attrs:
                if k == "__synth__":
                    dtype_str, shape, data = v
                    return np.frombuffer(data, dtype=np.dtype(dtype_str)).reshape(shape).copy()
            # Check original initializers by __name__
            for k, v in enode.attrs:
                if k == "__name__" and v in egraph.initializers:
                    return egraph.initializers[v].copy()
    return None
