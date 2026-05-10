"""End-to-end superoptimization pipeline.

ONNX → pre-passes → IR → e-graph saturation (legalization) → egglog
(pure algebraic rules + cost-aware extraction) → IR → ONNX.

Legalization rules run on the hand-rolled e-graph (they need Python
callbacks for weight inspection / constant synthesis).  Pure algebraic
rules (commutativity, associativity) run in egglog where cycle-creating
rewrites are handled natively by the equality saturation engine.
"""

from __future__ import annotations
from typing     import List

import sys
sys.setrecursionlimit(10**7)

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnx

from .backends.egglog import EgglogBackend
from .compat import run_post_passes, run_pre_passes
from .contracts import Contract, check_contract
from .egraph.eclass import AnalysisData
from .egraph.egraph import EGraph
from .egraph.enode import EClassId, ENode
from .explore.explorer import explore, ExploreStats
from .extract.cost import CostModel
from .extract.greedy import extract_greedy
from .ir.convert import ir_to_onnx, onnx_to_ir
from .ir.graph import IRGraph
from .ir.node import OP_INPUT, OP_NOOP, OP_PROJ, OP_WEIGHT

# rules.
from src.common.rules.arithmetic import get_arithmetic_specs
from src.common.rules.fusion import get_fusion_specs
from src.common.rules.layout import get_layout_specs
from src.common.rules.legalization import RuleSpec, get_legalization_specs

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


@dataclass
class SuperoptResult:
    """Summary of a superoptimization run."""

    input_path: str
    output_path: str
    original_nodes: int = 0
    optimized_nodes: int = 0
    explore_stats: ExploreStats = field(default_factory=ExploreStats)
    estimated_cost: float | None = None
    contract_result: dict[str, object] | None = None


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

        # Synthetic weights created by legalization apply_fn carry a
        # __synth__ attr with (dtype_str, shape, bytes).
        synth = node.attrs_dict.get("__synth__")
        if synth is not None:
            dtype_str, shape, data = synth
            arr = np.frombuffer(data, dtype=np.dtype(dtype_str)).reshape(shape)
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


def _merge_stats(dst: ExploreStats, src: ExploreStats) -> None:
    dst.iterations += src.iterations
    dst.total_matches += src.total_matches
    dst.total_applied += src.total_applied
    dst.final_eclasses = src.final_eclasses
    dst.final_enodes = src.final_enodes
    dst.saturated = dst.saturated and src.saturated
    return 


def _run_egraph_saturation(
    ir: IRGraph,
    max_iter: int,
    max_nodes: int,
    supported_ops: frozenset[str] | None = None,
) -> tuple[IRGraph, ExploreStats]:
    """Run all rewrite rules through the hand-rolled e-graph."""
    cumulative_stats = ExploreStats()
    if max_iter <= 0:
        return ir, cumulative_stats

    all_rules: List[RuleSpec] = (
        get_legalization_specs()
        + get_arithmetic_specs()
        + get_layout_specs()
        + get_fusion_specs()
    )

    if not all_rules:
        return ir, cumulative_stats

    # TODO: make this one reasonable. egg should be helpful for industry problems. 
    cost_model = CostModel(supported_ops=supported_ops)

    for _ in range(max_iter):
        egraph, root_cid = ir_to_egraph(ir)
        stats, blacklist = explore(
            egraph,
            all_rules,
            root_cid=root_cid,
            max_iter=max_iter,
            max_nodes=max_nodes,
        )
        _merge_stats(cumulative_stats, stats)
        if stats.total_applied == 0:
            break
        # Never blacklist nodes in the root e-class — the extractor must
        # be able to pick the root to build a valid program.

        # TODO.   
        root_canon = egraph.find(root_cid)
        root_nodes = set(egraph.eclass(root_canon).nodes)
        blacklist -= root_nodes
        bridged_ir = extract_greedy(egraph, root_cid, cost_model, blacklist=blacklist)
        _attach_initializers(bridged_ir, ir)
        ir = bridged_ir

    return ir, cumulative_stats


def _load_egglog_for_extraction(ir: IRGraph) -> EgglogBackend:
    """Load a saturated IR into egglog for cost-aware extraction."""
    return EgglogBackend(ir)


def superoptimize(
    input_path: str | Path,
    output_path: str | Path,
    supported_ops: frozenset[str] | None = None,
    max_iter: int = 15,
    max_nodes: int = 50_000,
) -> SuperoptResult:
    """Run the full superoptimization pipeline on an ONNX model.

    ONNX ops remain extractable, but a supported-op contract can be attached
    as a post-materialization legality gate.
    """
    input_path = str(input_path)
    output_path = str(output_path)
    contract = (
        Contract(name="custom", supported_ops=frozenset(supported_ops))
        if supported_ops is not None
        else None
    )

    model, ir = _load_preprocessed_ir(input_path)
    original_nodes = _count_compute_nodes(ir)
    ir, stats = _run_egraph_saturation(ir, max_iter, max_nodes, supported_ops=supported_ops)

    cost_model = CostModel(supported_ops=supported_ops)
    backend, egg_stats = _load_egglog_for_extraction(ir, max_iter=min(3, max_iter), max_nodes=max_nodes)
    _merge_stats(stats, egg_stats)
    extracted = backend.extract_best(cost_model)
    opt_ir = extracted.ir

    opt_model = _make_output_model(opt_ir, ir, model)
    contract_result = check_contract(opt_model, contract) if contract else None
    onnx.save(opt_model, output_path)

    return SuperoptResult(
        input_path=input_path,
        output_path=output_path,
        original_nodes=original_nodes,
        optimized_nodes=_count_compute_nodes(opt_ir),
        explore_stats=stats,
        estimated_cost=extracted.estimated_cost,
        contract_result=contract_result,
    )