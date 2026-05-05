"""End-to-end superoptimization pipeline.

ONNX load → shape inference → onnx_to_ir → ir_to_egraph → explore
→ extract_greedy → ir_to_onnx → save.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import onnx

from .egraph.eclass import AnalysisData
from .egraph.egraph import EGraph
from .egraph.enode import EClassId, ENode
from .explore.explorer import ExploreStats, explore
from .extract.cost import CostModel
from .extract.greedy import extract_greedy
from .ir.convert import ir_to_onnx, onnx_to_ir
from .ir.graph import IRGraph
from .ir.node import OP_INPUT, OP_NOOP, OP_PROJ, OP_WEIGHT
from .rules.layout import get_layout_rules
from .rules.legalization import get_legalization_rules

import numpy as np

logger = logging.getLogger(__name__)


def _hashable_attrs(
    attrs: tuple[tuple[str, object], ...],
) -> tuple[tuple[str, object], ...]:
    """Make IR attrs hashable for ENode memo dedup.

    numpy arrays are converted to (dtype, shape, bytes) tuples.
    """
    # TODO: too simple, we need to check it up. 
    result = []
    for k, v in attrs:
        if isinstance(v, np.ndarray):
            result.append((k, (str(v.dtype), v.shape, v.tobytes())))
        # TODO: we need to see all attrs. 
        else:
            result.append((k, v))
    
    return tuple(result)


@dataclass
class SuperoptResult:
    """Summary of a superoptimization run."""

    input_path: str
    output_path: str
    original_nodes: int = 0
    optimized_nodes: int = 0
    explore_stats: ExploreStats = field(default_factory=ExploreStats)


def ir_to_egraph(ir: IRGraph) -> tuple[EGraph, EClassId]:
    """Convert an IRGraph into an e-graph.

    Returns the e-graph and the e-class id of the root node.
    """
    egraph = EGraph()
    node_to_cid: dict[str, EClassId] = {} 

    # topological order. 
    for nid in ir.topo_order():
        node = ir.nodes[nid]
        children = tuple(node_to_cid[inp] for inp in node.inputs)
        # Leaf nodes (weight, input) have no children and no attrs,
        # so they'd all dedup to the same e-node. Tag them with their
        # id so each leaf gets its own e-class.
        attrs = node.attrs
        if node.op in (OP_INPUT, OP_WEIGHT):
            attrs = (("__name__", nid),)
        # ENode must be hashable for memo dedup. Convert any numpy
        # arrays in attrs to bytes so the tuple is hashable.

        # TODO: we have to handle this problem very clearly.
        attrs = _hashable_attrs(attrs)
        enode = ENode(op=node.op, children=children, attrs=attrs)
        cid = egraph.add(enode) # cid means e-class Id. 
        node_to_cid[nid] = cid 

        # Propagate analysis data. add() may return an existing e-class, so
        # join instead of overwriting facts from an equivalent node.
        scalar_value = None
        if node.op == OP_WEIGHT and nid in ir.initializers:
            arr = ir.initializers[nid]
            if arr.size == 1:
                scalar_value = float(arr.reshape(-1)[0])
        egraph.update_analysis(cid, AnalysisData(
            shape=node.shape,
            dtype=node.dtype,
            is_constant=(node.op == OP_WEIGHT),
            preferred_name=nid,
            scalar_value=scalar_value,
        ))

    # Store initializer data on e-graph so rules can access weight arrays.
    egraph.initializers = dict(ir.initializers)

    assert ir.root is not None
    root_cid = node_to_cid[ir.root]

    # sys.exit(1)
    return egraph, root_cid


def superoptimize(
    input_path: str | Path,
    output_path: str | Path,
    supported_ops: frozenset[str] | None = None,
    max_iter: int = 15,
    max_nodes: int = 50_000,
) -> SuperoptResult:
    """Run the full superoptimization pipeline on an ONNX model.

    All ONNX ops are extractable. The extraction objective is the
    profiled ONNX Runtime latency cost model.
    """
    input_path = str(input_path)
    output_path = str(output_path)

    # Load model. onnx_to_ir performs shape inference once and keeps that
    # responsibility localized to conversion.
    model = onnx.load(input_path)

    # Pre-pass: lower deep-pattern ops (DecoderMask, Trilu) at ONNX level.
    from .compat import run_pre_passes
    model = run_pre_passes(model) # constant folding -> decoder mask -> trilu -> constant folding.  

    # ONNX → IR.,
    ir = onnx_to_ir(model)
    # print(ir)
    # sys.exit(1)

    original_nodes = sum(
        1 for n in ir.nodes.values()
        # consider meaningful operation only 
        if n.op not in (OP_INPUT, OP_WEIGHT, OP_NOOP, OP_PROJ)
    )

    # IR → e-graph.
    # TODO: checkpoint 2. 
    egraph, root_cid = ir_to_egraph(ir)

    # Phase 1: Legalization (decompose complex ops into simpler ones).
    # These rules are targeted and don't cause combinatorial explosion.
    legalization_rules = get_legalization_rules()
    explore_stats, blacklist = explore(
        egraph, legalization_rules,
        max_iter=max_iter, max_nodes=max_nodes, root_cid=root_cid,
    )

    # Phase 2: layout-only cleanup (bounded).
    # Arithmetic associativity/commutativity is not bit-exact for floating
    # point tensors and can violate the correctness gate.
    opt_rules = get_layout_rules()
    opt_iter = min(3, max_iter)
    opt_stats, opt_blacklist = explore(
        egraph, opt_rules,
        max_iter=opt_iter, max_nodes=max_nodes, root_cid=root_cid,
    )
    blacklist |= opt_blacklist
    explore_stats.iterations += opt_stats.iterations
    explore_stats.total_matches += opt_stats.total_matches
    explore_stats.total_applied += opt_stats.total_applied
    explore_stats.final_eclasses = opt_stats.final_eclasses
    explore_stats.final_enodes = opt_stats.final_enodes

    # Extract best program using profiled latency cost model.
    cost_model = CostModel()
    opt_ir = extract_greedy(egraph, root_cid, cost_model, blacklist=blacklist)

    # Carry over only initializer leaves that survived extraction.
    # Synthetic weights (created by legalization apply_fn) carry a
    # __synth__ attr with (dtype_str, shape, bytes) for reconstruction.
    for name, node in opt_ir.nodes.items():
        if node.op == OP_WEIGHT:
            if name in ir.initializers:
                opt_ir.add_initializer(name, ir.initializers[name])
            else:
                # Try to reconstruct from __synth__ attr.
                synth = node.attrs_dict.get("__synth__")
                if synth is not None:
                    dtype_str, shape, data = synth
                    arr = np.frombuffer(data, dtype=np.dtype(dtype_str)).reshape(shape)
                    opt_ir.add_initializer(name, arr.copy())
                else:
                    raise KeyError(
                        f"missing initializer payload for extracted weight: {name}"
                    )

    # IR → ONNX.
    opt_model = ir_to_onnx(opt_ir, model)

    # Post-pass: constant folding + cleanup to remove Constant/ConstantOfShape/Shape nodes.
    from .compat import run_post_passes
    opt_model = run_post_passes(opt_model)

    onnx.save(opt_model, output_path)

    optimized_nodes = sum(
        1 for n in opt_ir.nodes.values()
        if n.op not in (OP_INPUT, OP_WEIGHT, OP_NOOP, OP_PROJ)
    )

    return SuperoptResult(
        input_path=input_path,
        output_path=output_path,
        original_nodes=original_nodes,
        optimized_nodes=optimized_nodes,
        explore_stats=explore_stats,
    )
