"""ONNX ↔ IR conversion.

``onnx_to_ir`` converts an ONNX ModelProto into an IRGraph.
``ir_to_onnx`` converts an IRGraph back into an ONNX ModelProto.

Round-trip correctness (onnx → ir → onnx) is the first validation
milestone.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

from .graph import IRGraph
from .node import IRNode, OP_INPUT, OP_NOOP, OP_WEIGHT


def onnx_to_ir(model: onnx.ModelProto) -> IRGraph:
    """Convert an ONNX model to an IRGraph.

    Steps:
    1. Run shape inference.
    2. Collect initializers as weight leaf nodes.
    3. Create input leaf nodes for graph inputs.
    4. Convert each ONNX node to an IRNode.
    5. Add a noop root that combines graph outputs.
    """
    model = onnx.shape_inference.infer_shapes(model)
    graph = model.graph
    ir = IRGraph()

    # --- initializers ---
    init_names: set[str] = set()
    for init in graph.initializer:
        arr = numpy_helper.to_array(init)
        ir.add_initializer(init.name, arr)
        init_names.add(init.name)
        ir.add_node(IRNode(
            id=init.name,
            op=OP_WEIGHT,
            inputs=(),
            shape=tuple(arr.shape),
            dtype=init.data_type,
        ))

    # --- graph inputs (exclude initializers) ---
    for inp in graph.input:
        if inp.name in init_names:
            continue
        shape = _extract_shape(inp)
        dtype = inp.type.tensor_type.elem_type if inp.type.HasField("tensor_type") else None
        ir.add_node(IRNode(
            id=inp.name,
            op=OP_INPUT,
            inputs=(),
            shape=shape,
            dtype=dtype,
        ))

    # --- nodes ---
    for node in graph.node:
        attrs = _extract_attrs(node)
        # TODO: handle multi-output nodes with projection nodes
        output_id = node.output[0]
        shape = _get_value_info_shape(graph, output_id)
        ir.add_node(IRNode(
            id=output_id,
            op=node.op_type,
            inputs=tuple(node.input),
            attrs=tuple(sorted(attrs.items())),
            shape=shape,
        ))

    # --- noop root ---
    output_ids = tuple(o.name for o in graph.output)
    noop_id = "__noop_root__"
    ir.add_node(IRNode(id=noop_id, op=OP_NOOP, inputs=output_ids))
    ir.root = noop_id

    return ir


def ir_to_onnx(ir: IRGraph, ref_model: onnx.ModelProto) -> onnx.ModelProto:
    """Convert an IRGraph back to an ONNX ModelProto.

    Uses ``ref_model`` for opset version, metadata, and original
    graph inputs/outputs spec.
    """
    ref_graph = ref_model.graph

    # Collect graph input and output specs from ref model.
    graph_inputs = list(ref_graph.input)
    graph_outputs = list(ref_graph.output)

    # Build initializers from IRGraph.
    initializers: list[TensorProto] = []
    for name, arr in ir.initializers.items():
        initializers.append(numpy_helper.from_array(arr, name=name))

    # Build nodes in topological order, skipping leaf/noop ops.
    nodes: list[onnx.NodeProto] = []
    for nid in ir.topo_order():
        node = ir.nodes[nid]
        if node.op in (OP_INPUT, OP_WEIGHT, OP_NOOP):
            continue
        attrs = node.attrs_dict
        onnx_node = onnx.helper.make_node(
            node.op,
            inputs=list(node.inputs),
            outputs=[node.id],
            **_attrs_to_kwargs(attrs),
        )
        nodes.append(onnx_node)

    graph_def = onnx.helper.make_graph(
        nodes,
        ref_graph.name,
        graph_inputs,
        graph_outputs,
        initializer=initializers,
    )

    model = onnx.helper.make_model(graph_def)
    model.ir_version = ref_model.ir_version
    del model.opset_import[:]
    for opset in ref_model.opset_import:
        new_opset = model.opset_import.add()
        new_opset.domain = opset.domain
        new_opset.version = opset.version

    onnx.checker.check_model(model)
    return model


def _attrs_to_kwargs(attrs: dict[str, Any]) -> dict[str, Any]:
    """Convert IR attribute dict to onnx.helper.make_node kwargs."""
    kwargs: dict[str, Any] = {}
    for key, val in attrs.items():
        if isinstance(val, np.ndarray):
            kwargs[key] = numpy_helper.from_array(val)
        else:
            kwargs[key] = val
    return kwargs


# --- helpers ---

def _extract_shape(value_info: onnx.ValueInfoProto) -> tuple[int, ...] | None:
    """Extract shape from a ValueInfoProto, returning None if unknown."""
    if not value_info.type.HasField("tensor_type"):
        return None
    tp = value_info.type.tensor_type
    if not tp.HasField("shape"):
        return None
    dims: list[int] = []
    for d in tp.shape.dim:
        dims.append(d.dim_value if d.dim_value > 0 else -1)
    return tuple(dims)


def _get_value_info_shape(
    graph: onnx.GraphProto, name: str
) -> tuple[int, ...] | None:
    """Look up shape for a tensor name from value_info or output."""
    for vi in list(graph.value_info) + list(graph.output):
        if vi.name == name:
            return _extract_shape(vi)
    return None


def _extract_attrs(node: onnx.NodeProto) -> dict[str, Any]:
    """Extract ONNX node attributes into a plain dict."""
    attrs: dict[str, Any] = {}
    for attr in node.attribute:
        if attr.type == onnx.AttributeProto.INT:
            attrs[attr.name] = attr.i
        elif attr.type == onnx.AttributeProto.FLOAT:
            attrs[attr.name] = attr.f
        elif attr.type == onnx.AttributeProto.STRING:
            attrs[attr.name] = attr.s.decode("utf-8")
        elif attr.type == onnx.AttributeProto.INTS:
            attrs[attr.name] = tuple(attr.ints)
        elif attr.type == onnx.AttributeProto.FLOATS:
            attrs[attr.name] = tuple(attr.floats)
        elif attr.type == onnx.AttributeProto.TENSOR:
            attrs[attr.name] = numpy_helper.to_array(attr.t)
        # skip graphs, sparse tensors, etc. for now
    return attrs
