"""End-to-end superoptimization pipeline.

ONNX load → shape inference → onnx_to_ir → egglog saturation/extraction
→ ir_to_onnx → save.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnx

from .backends.egglog import EgglogBackend
from .egraph.eclass import AnalysisData
from .egraph.egraph import EGraph
from .egraph.enode import EClassId, ENode
from .explore.explorer import ExploreStats
from .extract.cost import CostModel
from .ir.convert import ir_to_onnx, onnx_to_ir
from .ir.graph import IRGraph
from .ir.node import OP_INPUT, OP_NOOP, OP_PROJ, OP_WEIGHT
from .rules.arithmetic import get_arithmetic_rules
from .rules.fusion import get_fusion_rules
from .rules.layout import get_layout_rules
from .rules.legalization import get_legalization_rules

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


def ir_to_egraph(ir: IRGraph) -> tuple[EGraph, EClassId]:
    """Convert an IRGraph into the legacy hand-rolled e-graph.

    The main pipeline no longer calls this path; egglog is used instead.
    Keep this only while old rewrite/check/apply_fn helpers still depend on
    the hand-rolled e-graph types.

    Returns the e-graph and the e-class id of the root node.
    """
    egraph = EGraph()
    node_to_cid: dict[str, EClassId] = {}

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

    assert ir.root is not None
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
    from .compat import run_post_passes

    return run_post_passes(opt_model)


def _count_compute_nodes(ir: IRGraph) -> int:
    return sum(1 for n in ir.nodes.values() if n.op not in _BOUNDARY_OPS)


def _load_preprocessed_ir(input_path: str) -> tuple[onnx.ModelProto, IRGraph]:
    model = onnx.load(input_path)
    from .compat import run_pre_passes

    model = run_pre_passes(model)
    return model, onnx_to_ir(model)


def _run_egglog(
    ir: IRGraph,
    max_iter: int,
    max_nodes: int,
) -> tuple[EgglogBackend, ExploreStats]:
    backend = EgglogBackend(ir)

    explore_stats = backend.run_rules(
        get_legalization_rules(),
        max_iter=max_iter,
        max_nodes=max_nodes,
    )
    opt_stats = backend.run_rules(
        get_arithmetic_rules() + get_layout_rules() + get_fusion_rules(),
        max_iter=min(3, max_iter),
        max_nodes=max_nodes,
    )
    explore_stats.iterations += opt_stats.iterations
    explore_stats.total_matches += opt_stats.total_matches
    explore_stats.total_applied += opt_stats.total_applied
    explore_stats.final_eclasses = opt_stats.final_eclasses
    explore_stats.final_enodes = opt_stats.final_enodes
    return backend, explore_stats


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
    del supported_ops
    input_path = str(input_path)
    output_path = str(output_path)

    model, ir = _load_preprocessed_ir(input_path)
    original_nodes = _count_compute_nodes(ir)
    backend, explore_stats = _run_egglog(ir, max_iter, max_nodes)

    extracted = backend.extract_best(CostModel())
    opt_ir = extracted.ir

    opt_model = _make_output_model(opt_ir, ir, model)
    onnx.save(opt_model, output_path)

    return SuperoptResult(
        input_path=input_path,
        output_path=output_path,
        original_nodes=original_nodes,
        optimized_nodes=_count_compute_nodes(opt_ir),
        explore_stats=explore_stats,
        estimated_cost=extracted.estimated_cost,
    )


def superoptimize_topk(
    input_path: str | Path,
    output_dir: str | Path,
    supported_ops: frozenset[str] | None = None,
    max_iter: int = 15,
    max_nodes: int = 50_000,
    k: int = 5,
) -> list[SuperoptResult]:
    """Materialize the top-k estimated-cost extraction candidates."""
    del supported_ops
    input_path = str(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, ir = _load_preprocessed_ir(input_path)
    original_nodes = _count_compute_nodes(ir)
    backend, explore_stats = _run_egglog(ir, max_iter, max_nodes)

    programs = backend.extract_topk(k=k)

    results: list[SuperoptResult] = []
    for index, program in enumerate(programs):
        opt_ir = program.ir
        opt_model = _make_output_model(opt_ir, ir, model)
        output_path = output_dir / f"candidate_{index}.onnx"
        onnx.save(opt_model, output_path)
        results.append(
            SuperoptResult(
                input_path=input_path,
                output_path=str(output_path),
                original_nodes=original_nodes,
                optimized_nodes=_count_compute_nodes(opt_ir),
                explore_stats=explore_stats,
                estimated_cost=program.estimated_cost,
            )
        )
    return results
