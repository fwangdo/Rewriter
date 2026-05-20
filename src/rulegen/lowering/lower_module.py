"""lowering from onnx to small ir"""
from __future__ import annotations

import src.common.spec.constant as Cons 
import onnx

import logging
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from src.rulegen.lowering.lower_rules import lower 

logger = logging.getLogger(__name__)

from src.rulegen.sir.sir_graph import SirGraph, SirNode, OP_INPUT, OP_NOOP, OP_PROJ, OP_WEIGHT

_MULTI_OUTPUTS_ATTR = "__outputs"
_INPUT_SLOTS_ATTR = "__input_slots"
_NODE_NAME_ATTR = "__node_name"

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


def onnx_to_sir(model: onnx.ModelProto) -> SirGraph:
    """Convert an ONNX model to an SirGraph.

    The IR is value-oriented: ordinary ONNX nodes become IR nodes whose id is
    the output tensor value. Multi-output ONNX nodes become a base operation
    node plus one OP_PROJ node per output value.

    Steps:
    1. Run shape inference.
    2. Collect initializers as weight leaf nodes.
    3. Create input leaf nodes for graph inputs.
    4. Convert each ONNX node to an IRNode.
    5. Add a noop root that combines graph outputs.
    """
    model = onnx.shape_inference.infer_shapes(model)
    graph = model.graph
    sir = SirGraph()
    sir.opset_version = next(
        (o.version for o in model.opset_import if o.domain == ""), 18
    )

    # input, which is runtime value. 
    sir.inputs = tuple(inp.name for inp in graph.input if inp.name not in {init.name for init in graph.initializer})
    sir.outputs = tuple(o.name for o in graph.output)

    # --- initializers ---
    init_names: set[str] = set()
    for init in graph.initializer:
        arr = numpy_helper.to_array(init)
        sir.add_initializer(init.name, arr)
        init_names.add(init.name)
        # weight. 
        sir.add_node(SirNode(
            id=init.name,
            op=OP_WEIGHT,
            inputs=(),
            shape=tuple(arr.shape),
            dtype=init.data_type,
        ))

    # --- graph inputs (exclude initializers) ---
    for inp in graph.input:
        # exclude something in intializer. 
        if inp.name in init_names:
            continue
        shape = _extract_shape(inp)
        dtype = inp.type.tensor_type.elem_type if inp.type.HasField("tensor_type") else None
        # runtime inputs. 
        sir.add_node(SirNode(
            id=inp.name,
            op=OP_INPUT,
            inputs=(),
            shape=shape,
            dtype=dtype,
        ))

    # --- nodes ---
    # key factors for lowering. 
    for node_index, node in enumerate(graph.node):
        attrs = _extract_attrs(node)
        if node.name:
            attrs[_NODE_NAME_ATTR] = node.name
        if any(input_name == "" for input_name in node.input):
            # ONNX uses "" to preserve omitted optional input positions.
            # IRNode.inputs stores only real value ids, so keep the original
            # slots in attrs for exact ONNX reconstruction.
            attrs[_INPUT_SLOTS_ATTR] = tuple(node.input)
        live_outputs = tuple(output for output in node.output if output)

        # multi inputs. 
        if len(live_outputs) > 1:
            base_id = node.name or f"{node.op_type}_{node_index}__multi"
            if base_id in sir.nodes:
                raise Exception(f'[ERROR]: {base_id} already exists.')

            base_attrs = dict(attrs)
            base_attrs[_MULTI_OUTPUTS_ATTR] = live_outputs
            sir_node = SirNode(
                id=base_id,
                op=node.op_type,
                inputs=tuple(input_name for input_name in node.input if input_name),
                attrs=tuple(sorted(base_attrs.items())),
            )
            lower(sir, sir_node)
            # sir.add_node()

            # The base node represents the ONNX operation invocation. Each
            # OP_PROJ node below represents one concrete output tensor value.
            for output_index, output_id in enumerate(live_outputs):
                sir_node = SirNode(
                    id=output_id,
                    op=OP_PROJ,
                    inputs=(base_id,),
                    attrs=(("index", output_index),),
                    shape=_get_value_info_shape(graph, output_id),
                    dtype=_get_value_info_dtype(graph, output_id),
                )
                lower(sir, sir_node) 
                # sir.add_node(sir_node)
            continue

        if len(live_outputs) != 1:
            raise ValueError(f"expected exactly 1 live output, got {len(live_outputs)}: {live_outputs}")
        output_id = live_outputs[0]  
        sir_node = SirNode(
            id=output_id,
            op=node.op_type,
            inputs=tuple(input_name for input_name in node.input if input_name),
            attrs=tuple(sorted(attrs.items())),
            shape=_get_value_info_shape(graph, output_id),
            dtype=_get_value_info_dtype(graph, output_id),
        )
        lower(sir, sir_node)

    # --- noop root ---
    output_ids = tuple(o.name for o in graph.output)
    noop_id = "__noop_root__"
    sir.add_node(SirNode(id=noop_id, op=OP_NOOP, inputs=output_ids))
    sir.root = noop_id

    return sir


# TODO the code below...  
def ir_to_onnx(ir: SirGraph, ref_model: onnx.ModelProto) -> onnx.ModelProto:
    """Convert an IRGraph back to an ONNX ModelProto.

    Uses ``ref_model`` for opset version, metadata, and original
    graph inputs/outputs spec.
    """
    ref_graph = ref_model.graph
    graph_inputs = _graph_inputs_for_ir(ir, ref_graph)
    graph_outputs = _graph_outputs_for_ir(ir, ref_graph)
    initializers = _build_initializers(ir)

    nodes = _build_onnx_nodes(ir)
    value_info = _live_value_info(
        ref_graph,
        nodes,
        graph_inputs,
        graph_outputs,
        initializers,
    )

    graph_def = onnx.helper.make_graph(
        nodes,
        ref_graph.name,
        graph_inputs,
        graph_outputs,
        initializer=initializers,
        value_info=value_info,
    )
    model = _make_model_like_ref(graph_def, ref_model)
    return _checked_model_or_warn(model)


# TODO 
def _build_initializers(ir: SirGraph) -> list[TensorProto]:
    return [
        numpy_helper.from_array(arr, name=name)
        for name, arr in ir.initializers.items()
    ]


# TODO 
def _build_proj_remap(ir: SirGraph) -> dict[str, str]:
    # Build proj -> parent output name mapping.
    # Proj nodes reference a multi-output parent and select one output by index.
    # We resolve each proj id to the parent's actual ONNX output name.
    proj_remap: dict[str, str] = {}
    for nid in ir.topo_order():
        node = ir.nodes[nid]
        if node.op == OP_PROJ:
            parent_id = node.inputs[0]
            parent = ir.nodes[parent_id]
            parent_attrs = dict(parent.attrs_dict)
            parent_outputs = parent_attrs.get(_MULTI_OUTPUTS_ATTR)
            if parent_outputs:
                idx = dict(node.attrs).get("index", 0)
                proj_remap[nid] = parent_outputs[idx]
            else:
                proj_remap[nid] = parent_id
    return proj_remap


# --- translation from ir graph to onnx. 

def _resolve_inputs(
    inputs: tuple[str, ...],
    proj_remap: dict[str, str],
) -> list[str]:
    return [proj_remap.get(inp, inp) for inp in inputs]


# TODO 
def _resolve_node_inputs(
    node: SirNode,
    attrs: dict[str, Any],
    proj_remap: dict[str, str],
) -> list[str]:
    resolved = _resolve_inputs(node.inputs, proj_remap)
    input_slots = attrs.pop(_INPUT_SLOTS_ATTR, None)
    if input_slots is None:
        return resolved

    # ``resolved`` is compact because IRNode.inputs omits optional ONNX slots
    # encoded as "". Walk the original slot list and consume one real input
    # each time the slot was present.
    resolved_iter = iter(resolved)
    restored: list[str] = []
    for slot in input_slots:
        if slot == "":
            restored.append("")
        else:
            restored.append(next(resolved_iter))
    return restored


def _build_onnx_nodes(ir: SirGraph) -> list[onnx.NodeProto]:
    proj_remap = _build_proj_remap(ir)

    # Build nodes in topological order, skipping leaf/noop/proj ops.
    nodes: list[onnx.NodeProto] = []
    for nid in ir.topo_order():
        node = ir.nodes[nid]
        if node.op in (OP_INPUT, OP_WEIGHT, OP_NOOP, OP_PROJ):
            continue
        attrs = dict(node.attrs_dict)
        node_name = attrs.pop(_NODE_NAME_ATTR, "")
        if _MULTI_OUTPUTS_ATTR in attrs:
            outputs = list(attrs.pop(_MULTI_OUTPUTS_ATTR))
        else: 
            outputs = [node.id]

        inputs = _resolve_node_inputs(node, attrs, proj_remap)
        onnx_node = onnx.helper.make_node(
            node.op,
            inputs=inputs,
            outputs=outputs,
            name=node_name,
            **_attrs_to_kwargs(attrs),
        )
        nodes.append(onnx_node)
    return nodes


def _live_value_info(
    ref_graph: onnx.GraphProto,
    nodes: list[onnx.NodeProto],
    graph_inputs: list[onnx.ValueInfoProto],
    graph_outputs: list[onnx.ValueInfoProto],
    initializers: list[TensorProto],
) -> list[onnx.ValueInfoProto]:
    # Get all names in rewritten graph.  
    live_value_names = set()
    for node in nodes:
        live_value_names.update(node.input)
        live_value_names.update(node.output)
    live_value_names.update(init.name for init in initializers)
    live_value_names.update(inp.name for inp in graph_inputs)
    live_value_names.update(out.name for out in graph_outputs)

    return [
        vi for vi in ref_graph.value_info
        if vi.name in live_value_names
    ]


def _make_model_like_ref(
    graph_def: onnx.GraphProto,
    ref_model: onnx.ModelProto,
) -> onnx.ModelProto:
    # To unify model version. 
    model = onnx.helper.make_model(graph_def)
    model.ir_version = ref_model.ir_version
    del model.opset_import[:]
    for opset in ref_model.opset_import:
        new_opset = model.opset_import.add()
        new_opset.domain = opset.domain
        new_opset.version = opset.version
    return model


def _checked_model_or_warn(model: onnx.ModelProto) -> onnx.ModelProto:
    try:
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
    except Exception as e:
        # Post-passes (constant folding + cleanup) may fix remaining issues,
        # but log so failures are visible during debugging.
        logger.warning("ir_to_onnx: checker/shape_inference failed: %s", e)
    return model


def _attrs_to_kwargs(attrs: dict[str, Any]) -> dict[str, Any]:
    """Convert IR attribute dict to onnx.helper.make_node kwargs."""
    kwargs: dict[str, Any] = {}
    for key, val in attrs.items():
        # Skip internal e-graph metadata attrs.
        if key.startswith("__"):
            continue
        if isinstance(val, np.ndarray):
            # from_array: np.array -> onnx.TensorProto
            kwargs[key] = numpy_helper.from_array(val)
        elif _is_hashable_ndarray(val):
            # Reconstruct numpy array from hashable form (dtype_str, shape, bytes).
            dtype_str, shape, data = val
            arr = np.frombuffer(data, dtype=np.dtype(dtype_str)).reshape(shape)
            kwargs[key] = numpy_helper.from_array(arr)
        else:
            kwargs[key] = val
    return kwargs


def _is_hashable_ndarray(val: Any) -> bool:
    """Check if val is a (dtype_str, shape, bytes) tuple from _hashable_attrs."""
    return (
        isinstance(val, tuple)
        and len(val) == 3
        and isinstance(val[0], str)
        and isinstance(val[1], tuple)
        and isinstance(val[2], bytes)
    )


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


def _get_value_info_dtype(graph: onnx.GraphProto, name: str) -> int | None:
    """Look up tensor dtype for a tensor name from graph metadata."""
    for vi in list(graph.value_info) + list(graph.output) + list(graph.input):
        if vi.name == name and vi.type.HasField("tensor_type"):
            return int(vi.type.tensor_type.elem_type)
    for init in graph.initializer:
        if init.name == name:
            return int(init.data_type)
    return None


def _graph_inputs_for_ir(
    ir: IRGraph,
    ref_graph: onnx.GraphProto,
) -> list[onnx.ValueInfoProto]:
    ref_inputs = {inp.name: inp for inp in ref_graph.input}
    # If ir.inputs is not set, infer from OP_INPUT nodes.
    input_names = ir.inputs if ir.inputs else tuple(
        nid for nid, n in ir.nodes.items() if n.op == OP_INPUT
    )

    inputs: list[onnx.ValueInfoProto] = []
    for name in input_names:
        if name not in ir.nodes or ir.nodes[name].op != OP_INPUT:
            continue
        if name in ref_inputs:
            inputs.append(ref_inputs[name])
        else:
            node = ir.nodes[name]
            inputs.append(_make_value_info(name, node.dtype, node.shape))
    return inputs


def _graph_outputs_for_ir(
    ir: IRGraph,
    ref_graph: onnx.GraphProto,
) -> list[onnx.ValueInfoProto]:
    ref_outputs = {output.name: output for output in ref_graph.output}
    outputs: list[onnx.ValueInfoProto] = []
    for name in ir.output_ids():
        if name in ref_outputs:
            outputs.append(ref_outputs[name])
            continue
        node = ir.nodes.get(name)
        dtype = node.dtype if node is not None else None
        shape = node.shape if node is not None else None
        outputs.append(_make_value_info(name, dtype, shape))
    return outputs


def _make_value_info(
    name: str,
    dtype: int | None,
    shape: tuple[int, ...] | None,
) -> onnx.ValueInfoProto:
    elem_type = dtype if dtype is not None else TensorProto.FLOAT
    dims: list[int | str] | None = None
    if shape is not None:
        dims = [dim if dim >= 0 else f"unk_{index}" for index, dim in enumerate(shape)]
    return helper.make_tensor_value_info(name, elem_type, dims)
