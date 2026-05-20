"""End-to-end superoptimization pipeline.

ONNX → pre-passes → IR → e-graph saturation → greedy extraction → ONNX.
"""

from __future__ import annotations

import sys
sys.setrecursionlimit(10**7)

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnx

from src.common.egraph.eclass import AnalysisData
from src.common.egraph.egraph import EGraph
from src.common.egraph.enode import EClassId, ENode
from src.common.explore.explorer import explore, ExploreStats
from src.common.extract.cost import CostModel
from src.common.extract.ilp import ILPStats, extract_ilp

from src.common.compat import run_post_passes, run_pre_passes
from src.common.ir.convert import ir_to_onnx, onnx_to_ir
from src.common.ir.graph import IRGraph
from src.common.ir.node import OP_INPUT, OP_NOOP, OP_PROJ, OP_WEIGHT
from src.common.rules import get_all_specs

#.. 
import sys 

_BOUNDARY_OPS = (OP_INPUT, OP_WEIGHT, OP_NOOP, OP_PROJ)


def _hashable_attrs(
    attrs: tuple[tuple[str, object], ...],
) -> tuple[tuple[str, object], ...]:
    """Make IR attrs hashable for ENode memo dedup.

    numpy arrays are converted to (dtype, shape, bytes) tuples.
    """
    result: list[tuple[str, object]] = []
    for k, v in attrs:
        if isinstance(v, np.ndarray):
            result.append((k, (str(v.dtype), v.shape, v.tobytes())))
        else:
            result.append((k, v))

    return tuple(result)


def ir_to_egraph(ir: IRGraph) -> tuple[EGraph, EClassId]:
    """Convert an IRGraph into the hand-rolled e-graph.

    Returns the e-graph and the e-class id of the root node.
    """
    egraph = EGraph()
    node_to_cid: dict[str, EClassId] = {} # just mapping, name -> EclassId.  

    for nid in ir.topo_order():
        node = ir.nodes[nid]
        children = tuple(node_to_cid[inp] for inp in node.inputs)
        # Leaf nodes (weight, input) have no children and no attrs,
        # so they'd all dedup to the same e-node. Tag them with their
        # id so each leaf gets its own e-class.
        attrs = node.attrs
        if node.op in (OP_INPUT, OP_WEIGHT):
            attrs = (("__name__", nid),)
        attrs = _hashable_attrs(attrs)
        enode = ENode(op=node.op, children=children, attrs=attrs)
        cid = egraph.add(enode)
        node_to_cid[nid] = cid

        # Propagate analysis data. add() may return an existing e-class, so
        # join instead of overwriting facts from an equivalent node.
        scalar_value = None
        if node.op == OP_WEIGHT and nid in ir.initializers:
            arr = ir.initializers[nid]
            if arr.size == 1:
                scalar_value = float(arr.reshape(-1)[0])

        egraph.update_analysis(
            cid,
            AnalysisData(
                shape=node.shape,
                dtype=node.dtype,
                is_constant=(node.op == OP_WEIGHT),
                preferred_name=nid,
                scalar_value=scalar_value,
            ),
        )

    # Store initializer data on e-graph so rules can access weight arrays.
    egraph.initializers = dict(ir.initializers)

    if ir.root is None:
        raise ValueError("IRGraph has no root node")
    root_cid = node_to_cid[ir.root]
    return egraph, root_cid


def _attach_initializers(opt_ir: IRGraph, source_ir: IRGraph) -> None:
    """Carry over initializer leaves that survived extraction."""
    for name, node in opt_ir.nodes.items():
        if node.op != OP_WEIGHT:
            continue
        if name in source_ir.initializers:
            opt_ir.add_initializer(name, source_ir.initializers[name])
            continue

        source_name = node.attrs_dict.get("__name__")
        if source_name in source_ir.initializers:
            opt_ir.add_initializer(name, source_ir.initializers[source_name])
            continue

        # Synthetic weights created by legalization apply_fn carry a
        # __synth__ attr with (dtype_str, shape, bytes).
        synth = node.attrs_dict.get("__synth__")
        if synth is not None:
            dtype_str, shape, data = synth
            arr = np.frombuffer(data, dtype=np.dtype(dtype_str)).reshape(shape)
            # the meaning of copy? frombuffer is just view. so, we need to copy to make the arr has it's own memory. 
            opt_ir.add_initializer(name, arr.copy())
            continue

        raise KeyError(f"missing initializer payload for extracted weight: {name}")


def _make_output_model(
    opt_ir: IRGraph,
    source_ir: IRGraph,
    ref_model: onnx.ModelProto,
) -> onnx.ModelProto:
    _attach_initializers(opt_ir, source_ir)
    opt_model = ir_to_onnx(opt_ir, ref_model)
    return run_post_passes(opt_model)


def _count_compute_nodes(ir: IRGraph) -> int:
    return sum(1 for n in ir.nodes.values() if n.op not in _BOUNDARY_OPS)


def _load_preprocessed_ir(input_path: str) -> tuple[onnx.ModelProto, IRGraph]:
    model = onnx.load(input_path)
    model = run_pre_passes(model)
    return model, onnx_to_ir(model)


def _run_egraph_saturation(
    ir: IRGraph,
    max_iter: int,
    max_nodes: int,
) -> tuple[EGraph, EClassId, set[int], ExploreStats]:
    """Saturate the e-graph with all rewrite rules. No extraction."""
    egraph, root_cid = ir_to_egraph(ir)
    stats = ExploreStats()
    blacklist: set[int] = set()

    all_rules = get_all_specs()

    if all_rules and max_iter > 0:
        stats, blacklist = explore(
            egraph,
            all_rules,
            root_cid=root_cid,
            max_iter=max_iter,
            max_nodes=max_nodes,
        )

    return egraph, root_cid, blacklist, stats


# phase ordering. 
def generate_rule(
) -> None:
    """Run rule generation based on e-graph.

    ONNX ops remain extractable, but a supported-op contract can be attached
    as a post-materialization legality gate.
    """
    # 0. load model 
    model, ir = _load_preprocessed_ir(input_path)

    max_iters = 15
    max_nodes = max(len(ir.nodes) * 5, 2000)

    # 1. lowering 

    # 2. Saturation based on eq saturation 
    egraph, root_cid, blacklist, stats = _run_egraph_saturation(
        ir, max_iters, max_nodes,
    )

    # 3. lifting. 
    
    return 