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

from src.common.rules import get_legalization_specs

from .base import RewriteRule
from .wrapper import rulespecs_to_rewrites
from ..egraph.enode import EClassId, ENode
from ..egraph.egraph import EGraph
from ..egraph.pattern import PatternNode, PatternVar, Subst


def get_legalization_rules() -> list[RewriteRule]:
    """Return legalization rewrite rules."""
    x = PatternVar("?x")
    a = PatternVar("?a")
    b = PatternVar("?b")
    e = PatternVar("?e")
    s = PatternVar("?s")
    bn_b = PatternVar("?bn_b")
    bn_m = PatternVar("?bn_m")
    bn_v = PatternVar("?bn_v")
    w = PatternVar("?w")

    rules: list[RewriteRule] = rulespecs_to_rewrites(get_legalization_specs())

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


def _is_close(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-6


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
