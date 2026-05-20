"""Lift structural Conv1x1-form e-nodes into ONNX operation blocks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import onnx

from src.common.egraph.egraph import EGraph
from src.common.egraph.enode import EClassId, ENode


_BODY_MUL_SUM = (
    ("mul", ("%0", "%1"), ()),
    ("yield", ("%m",), (("combine", "sum"),)),
)
_ZERO_DIM = ((), 0)


@dataclass
class OnnxNode:
    op_type: str
    inputs: list[str]
    outputs: list[str]
    attrs: dict[str, Any] = field(default_factory=dict)

    def to_proto(self) -> onnx.NodeProto:
        return onnx.helper.make_node(
            self.op_type,
            inputs=self.inputs,
            outputs=self.outputs,
            **self.attrs,
        )


@dataclass
class OnnxSubgraph:
    nodes: list[OnnxNode] = field(default_factory=list)
    initializers: dict[str, np.ndarray] = field(default_factory=dict)
    inputs: list[str] = field(default_factory=list)
    output: str = ""

    def verify_complete(self) -> bool:
        produced = set(self.inputs) | set(self.initializers)
        for node in self.nodes:
            if any(inp not in produced for inp in node.inputs):
                return False
            produced.update(node.outputs)
        return self.output in produced

    def to_node_protos(self) -> list[onnx.NodeProto]:
        return [node.to_proto() for node in self.nodes]


@dataclass(frozen=True)
class Conv1x1Form:
    activation_child: int
    weight_child: int
    n: int
    oc: int
    h: int
    w: int
    ic: int


@dataclass
class Conv1x1LiftCandidate:
    eclass_id: EClassId
    enode: ENode
    form: Conv1x1Form
    subgraph: OnnxSubgraph

    @property
    def rewrites(self) -> tuple[str, ...]:
        return self.enode.rewrites


def find_conv1x1_lift_candidates(egraph: EGraph) -> list[Conv1x1LiftCandidate]:
    """Find structural Conv1x1-form e-nodes that can be materialized as ONNX.

    Dynamic contractions are intentionally rejected here: ONNX Conv requires a
    static weight initializer in our legalization setting.
    """
    candidates: list[Conv1x1LiftCandidate] = []
    for cid in egraph.canonical_class_ids():
        for enode in egraph.eclass_nodes(cid):
            form = is_conv1x1_generic(enode)
            if form is None:
                continue
            subgraph = lift_conv1x1_generic(egraph, cid, enode, form)
            if subgraph is None:
                continue
            candidates.append(Conv1x1LiftCandidate(cid, enode, form, subgraph))
    return candidates


def is_conv1x1_generic(enode: ENode) -> Conv1x1Form | None:
    """Return structural Conv1x1 metadata if an e-node has Conv1x1 IR form."""
    if enode.op != "generic":
        return None

    attrs = dict(enode.attrs)
    if attrs.get("body") != _BODY_MUL_SUM:
        return None

    maps = attrs.get("indexing_maps", ())
    iterators = attrs.get("iterators", ())
    if len(enode.children) != 2 or len(maps) != 3:
        return None

    output = _simple_indices(maps[-1])
    if output is None or len(output) != 4:
        return None
    n_idx, oc_idx, h_idx, w_idx = output

    output_set = set(output)
    reductions = [i for i in range(len(iterators)) if i not in output_set]
    if len(reductions) != 1:
        return None
    ic_idx = reductions[0]

    activation_map = (_dim(n_idx), _dim(ic_idx), _dim(h_idx), _dim(w_idx))
    weight_map = (_dim(oc_idx), _dim(ic_idx), _ZERO_DIM, _ZERO_DIM)

    if maps[0] == activation_map and maps[1] == weight_map:
        return Conv1x1Form(
            activation_child=0,
            weight_child=1,
            n=n_idx,
            oc=oc_idx,
            h=h_idx,
            w=w_idx,
            ic=ic_idx,
        )
    if maps[0] == weight_map and maps[1] == activation_map:
        return Conv1x1Form(
            activation_child=1,
            weight_child=0,
            n=n_idx,
            oc=oc_idx,
            h=h_idx,
            w=w_idx,
            ic=ic_idx,
        )
    return None


def lift_conv1x1_generic(
    egraph: EGraph,
    eclass_id: EClassId,
    enode: ENode,
    form: Conv1x1Form,
) -> OnnxSubgraph | None:
    """Materialize a structural Conv1x1 generic as an ONNX op block."""
    attrs = dict(enode.attrs)
    iterators = attrs.get("iterators", ())
    bounds = _iterator_bounds(iterators)
    if bounds is None:
        return None

    act_cid = enode.children[form.activation_child]
    weight_cid = enode.children[form.weight_child]
    act_data = egraph.eclass(act_cid).data
    weight_data = egraph.eclass(weight_cid).data
    out_data = egraph.eclass(eclass_id).data

    act_name = act_data.preferred_name
    weight_name = weight_data.preferred_name
    output_name = out_data.preferred_name
    if act_name is None or weight_name is None or output_name is None:
        return None
    if not weight_data.is_constant or weight_name not in egraph.initializers:
        return None
    output_shape = _output_shape(egraph, eclass_id)
    if act_data.shape is None or output_shape is None:
        return None

    N = bounds[form.n]
    OC = bounds[form.oc]
    H = bounds[form.h]
    W = bounds[form.w]
    IC = bounds[form.ic]
    if W != 1:
        return None

    weight_arr = egraph.initializers[weight_name]
    if not isinstance(weight_arr, np.ndarray):
        return None
    conv_weight = _make_conv_weight(weight_arr, OC, IC)
    if conv_weight is None:
        return None

    sg = OnnxSubgraph(inputs=[act_name], output=output_name)
    conv_weight_name = f"__conv_w_{output_name}"
    sg.initializers[conv_weight_name] = conv_weight

    act_shape = tuple(int(d) for d in act_data.shape)
    act_layout = _activation_layout(act_shape, N, H, IC)
    if act_layout is None:
        return None

    reshape_in_name = f"__reshape_in_shape_{output_name}"
    reshape_out_name = f"__reshape_out_shape_{output_name}"
    sg.initializers[reshape_in_name] = np.array(act_layout.reshape_shape, dtype=np.int64)
    sg.initializers[reshape_out_name] = np.array(output_shape, dtype=np.int64)

    conv_input = f"__conv_in_{output_name}"
    sg.nodes.append(OnnxNode(
        op_type="Reshape",
        inputs=[act_name, reshape_in_name],
        outputs=[conv_input],
    ))

    if act_layout.transpose_perm is not None:
        transposed = f"__conv_in_nchw_{output_name}"
        sg.nodes.append(OnnxNode(
            op_type="Transpose",
            inputs=[conv_input],
            outputs=[transposed],
            attrs={"perm": act_layout.transpose_perm},
        ))
        conv_input = transposed

    conv_out = f"__conv_out_{output_name}"
    sg.nodes.append(OnnxNode(
        op_type="Conv",
        inputs=[conv_input, conv_weight_name],
        outputs=[conv_out],
        attrs={"kernel_shape": [1, 1]},
    ))

    sg.nodes.append(OnnxNode(
        op_type="Reshape",
        inputs=[conv_out, reshape_out_name],
        outputs=[output_name],
    ))

    return sg if sg.verify_complete() else None


@dataclass(frozen=True)
class _ActivationLayout:
    reshape_shape: tuple[int, int, int, int]
    transpose_perm: list[int] | None


def _activation_layout(
    shape: tuple[int, ...],
    n: int,
    h: int,
    ic: int,
) -> _ActivationLayout | None:
    # Original activation already follows [..., IC, H].
    if shape == (ic, h):
        return _ActivationLayout((1, ic, h, 1), None)
    if shape == (n, ic, h):
        return _ActivationLayout((n, ic, h, 1), None)

    # Original activation follows [..., H, IC], so transpose after reshape.
    if shape == (h, ic):
        return _ActivationLayout((1, h, ic, 1), [0, 2, 1, 3])
    if shape == (n, h, ic):
        return _ActivationLayout((n, h, ic, 1), [0, 2, 1, 3])

    return None


def _make_conv_weight(weight: np.ndarray, oc: int, ic: int) -> np.ndarray | None:
    if weight.shape == (oc, ic):
        return weight.reshape(oc, ic, 1, 1).copy()
    if weight.shape == (ic, oc):
        return weight.T.reshape(oc, ic, 1, 1).copy()
    return None


def _output_shape(egraph: EGraph, eclass_id: EClassId) -> tuple[int, ...] | None:
    data_shape = egraph.eclass(eclass_id).data.shape
    if data_shape is not None:
        return data_shape

    for enode in egraph.eclass_nodes(eclass_id):
        if enode.op != "generic" or enode.rewrites:
            continue
        attrs = dict(enode.attrs)
        shape = _shape_from_generic_output(attrs)
        if shape is not None:
            return shape
    return None


def _shape_from_generic_output(attrs: dict[str, object]) -> tuple[int, ...] | None:
    iterators = attrs.get("iterators", ())
    maps = attrs.get("indexing_maps", ())
    bounds = _iterator_bounds(iterators)
    if bounds is None or not maps:
        return None

    shape: list[int] = []
    for terms, offset in maps[-1]:
        if offset != 0 or len(terms) != 1:
            return None
        coeff, idx = terms[0]
        if coeff != 1:
            return None
        shape.append(bounds[idx])
    return tuple(shape)


def _iterator_bounds(iterators: tuple[object, ...]) -> list[int] | None:
    bounds: list[int] = []
    for bound in iterators:
        try:
            bounds.append(int(bound))
        except (TypeError, ValueError):
            return None
    return bounds


def _dim(idx: int) -> tuple[tuple[tuple[int, int], ...], int]:
    return (((1, idx),), 0)


def _simple_indices(index_map: tuple) -> list[int] | None:
    indices: list[int] = []
    for terms, offset in index_map:
        if offset != 0 or len(terms) != 1:
            return None
        coeff, idx = terms[0]
        if coeff != 1:
            return None
        indices.append(idx)
    return indices
