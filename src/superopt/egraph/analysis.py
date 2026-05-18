"""E-class analysis: bottom-up propagation of shape/type metadata.

Tensat uses egg's e-class analysis to attach shape and layout info
to each e-class.  We do the same: after adding or merging e-nodes,
the analysis propagates shape/dtype information upward.

This is used for:
- Shape checking before applying rewrite rules
- Cost estimation during extraction
"""

from __future__ import annotations

import numpy as np

from .eclass import AnalysisData
from .enode import EClassId, ENode
from .egraph import EGraph


def _np_dtype_to_onnx(dtype: np.dtype) -> int:
    _MAP = {
        np.dtype("float32"): 1, np.dtype("uint8"): 2, np.dtype("int8"): 3,
        np.dtype("int16"): 5, np.dtype("int32"): 6, np.dtype("int64"): 7,
        np.dtype("bool"): 9, np.dtype("float64"): 11, np.dtype("float16"): 10,
    }
    return _MAP.get(np.dtype(dtype), 1)


def compute_analysis(egraph: EGraph, enode: ENode) -> AnalysisData:
    """Compute analysis data for a single e-node from its children.

    This is the ``make`` function in egg's analysis framework.
    """
    # leaf nodes
    if enode.op in ("input", "weight"):
        return AnalysisData(is_constant=(enode.op == "weight"))

    if enode.op == "noop":
        return AnalysisData()

    # For compute nodes, try to infer shape from children.
    # This is a simplified version; full ONNX shape inference
    # would require op-specific logic.
    child_shapes: list[tuple[int, ...] | None] = []
    for child_cid in enode.children:
        ec = egraph.eclass(child_cid)
        child_shapes.append(ec.data.shape)

    # Helper: get first non-None dtype from children
    def _child_dtype(index: int = 0) -> int | None:
        if index < len(enode.children):
            return egraph.eclass(enode.children[index]).data.dtype
        return None

    def _first_child_dtype() -> int | None:
        for cid in enode.children:
            dt = egraph.eclass(cid).data.dtype
            if dt is not None:
                return dt
        return None

    if enode.op in {"Identity", "Relu", "Sigmoid", "Tanh", "Sqrt", "Neg", "Softmax"}:
        return AnalysisData(shape=child_shapes[0], dtype=_child_dtype())

    if enode.op == "Cast":
        cast_to = None
        for key, value in enode.attrs:
            if key == "to":
                cast_to = int(value)
                break
        return AnalysisData(shape=child_shapes[0], dtype=cast_to if cast_to is not None else _child_dtype())

    if enode.op in {"Add", "Sub", "Mul", "Div", "Max", "Min", "Pow"} and child_shapes:
        reference = next((shape for shape in child_shapes if shape is not None), None)
        return AnalysisData(shape=reference, dtype=_first_child_dtype())

    if enode.op in {"Where"} and child_shapes:
        reference = next((shape for shape in child_shapes if shape is not None), None)
        # Where output dtype matches true/false branches (children[1]), not cond
        return AnalysisData(shape=reference, dtype=_child_dtype(1))

    if enode.op in {"Less", "Greater", "Equal"} and child_shapes:
        reference = next((shape for shape in child_shapes if shape is not None), None)
        return AnalysisData(shape=reference, dtype=9)  # bool

    if enode.op == "MatMul" and len(child_shapes) == 2:
        dtype = _child_dtype()
        lhs, rhs = child_shapes
        if lhs is not None and rhs is not None and len(lhs) >= 2 and len(rhs) >= 2:
            batch = lhs[:-1]
            return AnalysisData(shape=batch + (rhs[-1],), dtype=dtype)
        return AnalysisData(dtype=dtype)

    if enode.op in {"Reshape", "Transpose", "Unsqueeze", "Squeeze", "Concat", "Slice", "Gather"}:
        return AnalysisData(dtype=_child_dtype())

    if enode.op == "ReduceMean" and child_shapes and child_shapes[0] is not None:
        return AnalysisData(dtype=_child_dtype())

    if enode.op == "Conv" and child_shapes:
        return AnalysisData(dtype=_child_dtype())

    if enode.op in {"LayerNormalization", "BatchNormalization"} and child_shapes:
        return AnalysisData(shape=child_shapes[0], dtype=_child_dtype())

    if enode.op == "ConstantOfShape":
        # Output dtype determined by "value" attr; default float32
        cos_dtype = 1
        for key, value in enode.attrs:
            if key == "value":
                if isinstance(value, np.ndarray):
                    cos_dtype = _np_dtype_to_onnx(value.dtype)
                elif isinstance(value, tuple) and len(value) == 3:
                    cos_dtype = _np_dtype_to_onnx(np.dtype(value[0]))
                break
        return AnalysisData(dtype=cos_dtype)

    if enode.op == "Range":
        return AnalysisData()

    if enode.op == "Clip" and child_shapes:
        return AnalysisData(shape=child_shapes[0], dtype=_child_dtype())

    if enode.op == "Gemm":
        return AnalysisData(dtype=_child_dtype())

    # Fallback: propagate dtype from first child if available
    if enode.children:
        return AnalysisData(dtype=_first_child_dtype())

    return AnalysisData()


def check_shape_compatible(
    egraph: EGraph,
    source_cid: EClassId,
    target_enode: ENode,
) -> bool:
    """Check if a target e-node's shape is compatible with the
    source e-class it will be merged into.

    Returns True if shapes are compatible or unknown.
    """
    source_data = egraph.eclass(source_cid).data
    target_data = compute_analysis(egraph, target_enode)

    if source_data.shape is None or target_data.shape is None:
        return True  # unknown shapes are conservatively allowed

    return source_data.shape == target_data.shape


def analyze_egraph(egraph: EGraph) -> None:
    """Refresh bottom-up analysis data for all canonical e-classes."""
    for cid in egraph.canonical_class_ids():
        ec = egraph.eclass(cid)
        merged = ec.data
        for enode in egraph.eclass_nodes(cid):
            merged = AnalysisData.join(merged, compute_analysis(egraph, enode))
        ec.data = merged
