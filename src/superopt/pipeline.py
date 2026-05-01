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
from .ir.node import OP_INPUT, OP_NOOP, OP_WEIGHT
from .rules.arithmetic import get_arithmetic_rules
from .rules.fusion import get_fusion_rules
from .rules.layout import get_layout_rules
from .rules.legalization import get_legalization_rules

logger = logging.getLogger(__name__)


@dataclass
class SuperoptResult:
    """Summary of a superoptimization run."""

    input_path: str
    output_path: str
    original_nodes: int = 0
    optimized_nodes: int = 0
    explore_stats: ExploreStats = field(default_factory=ExploreStats)
    legality_ok: bool = True


def ir_to_egraph(ir: IRGraph) -> tuple[EGraph, EClassId]:
    """Convert an IRGraph into an e-graph.

    Returns the e-graph and the e-class id of the root node.
    """
    egraph = EGraph()
    node_to_cid: dict[str, EClassId] = {}

    for nid in ir.topo_order():
        node = ir.nodes[nid]
        children = tuple(node_to_cid[inp] for inp in node.inputs)
        enode = ENode(op=node.op, children=children, attrs=node.attrs)
        cid = egraph.add(enode)
        node_to_cid[nid] = cid

        # Propagate analysis data.
        egraph.set_analysis(cid, AnalysisData(
            shape=node.shape,
            dtype=node.dtype,
            is_constant=(node.op == OP_WEIGHT),
            preferred_name=nid,
        ))

    assert ir.root is not None
    root_cid = node_to_cid[ir.root]
    return egraph, root_cid


def superoptimize(
    input_path: str | Path,
    output_path: str | Path,
    supported_ops: frozenset[str],
    max_iter: int = 15,
    max_nodes: int = 50_000,
) -> SuperoptResult:
    """Run the full superoptimization pipeline on an ONNX model."""
    input_path = str(input_path)
    output_path = str(output_path)

    # Load and shape-infer.
    model = onnx.load(input_path)
    model = onnx.shape_inference.infer_shapes(model)

    # ONNX → IR.
    ir = onnx_to_ir(model)
    original_nodes = sum(
        1 for n in ir.nodes.values()
        if n.op not in (OP_INPUT, OP_WEIGHT, OP_NOOP)
    )

    # IR → e-graph.
    egraph, root_cid = ir_to_egraph(ir)

    # Gather all rewrite rules.
    rules = (
        get_arithmetic_rules()
        + get_layout_rules()
        + get_fusion_rules()
        + get_legalization_rules()
    )

    # Explore (equality saturation).
    explore_stats = explore(egraph, rules, max_iter=max_iter, max_nodes=max_nodes)

    # Extract best program.
    cost_model = CostModel(supported_ops)
    opt_ir = extract_greedy(egraph, root_cid, cost_model)

    # Carry over initializers from original IR.
    for name, arr in ir.initializers.items():
        if name in opt_ir.nodes:
            opt_ir.add_initializer(name, arr)

    # IR → ONNX.
    opt_model = ir_to_onnx(opt_ir, model)
    onnx.save(opt_model, output_path)

    optimized_nodes = sum(
        1 for n in opt_ir.nodes.values()
        if n.op not in (OP_INPUT, OP_WEIGHT, OP_NOOP)
    )

    # Check legality.
    unsupported = {
        n.op for n in opt_ir.nodes.values()
        if n.op not in (OP_INPUT, OP_WEIGHT, OP_NOOP)
        and n.op not in supported_ops
    }
    legality_ok = len(unsupported) == 0
    if unsupported:
        logger.warning("unsupported ops in output: %s", unsupported)

    return SuperoptResult(
        input_path=input_path,
        output_path=output_path,
        original_nodes=original_nodes,
        optimized_nodes=optimized_nodes,
        explore_stats=explore_stats,
        legality_ok=legality_ok,
    )
