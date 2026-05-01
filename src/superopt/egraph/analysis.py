"""E-class analysis: bottom-up propagation of shape/type metadata.

Tensat uses egg's e-class analysis to attach shape and layout info
to each e-class.  We do the same: after adding or merging e-nodes,
the analysis propagates shape/dtype information upward.

This is used for:
- Shape checking before applying rewrite rules
- Cost estimation during extraction
"""

from __future__ import annotations

from .eclass import AnalysisData
from .enode import EClassId, ENode
from .egraph import EGraph


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

    if enode.op in {"Identity", "Relu", "Sigmoid", "Tanh", "Sqrt", "Cast"}:
        return AnalysisData(shape=child_shapes[0], dtype=egraph.eclass(enode.children[0]).data.dtype)

    if enode.op in {"Add", "Sub", "Mul", "Div", "Where"} and child_shapes:
        reference = next((shape for shape in child_shapes if shape is not None), None)
        return AnalysisData(shape=reference)

    if enode.op == "MatMul" and len(child_shapes) == 2:
        lhs, rhs = child_shapes
        if lhs is not None and rhs is not None and len(lhs) >= 2 and len(rhs) >= 2:
            batch = lhs[:-1]
            return AnalysisData(shape=batch + (rhs[-1],))

    if enode.op in {"Reshape", "Transpose", "Unsqueeze", "Squeeze", "Concat", "Slice"}:
        return AnalysisData()

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
