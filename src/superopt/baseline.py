"""IR-based rule baseline for fair comparison with superopt.

This baseline shares the same ONNX -> IR lowering and IR -> ONNX restoration
path as superopt. The only intended difference is rewrite strategy:
manual destructive ordering here, e-graph saturation/extraction in superopt.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import onnx

from src.common.rules import RuleSpec, get_all_specs
from src.common.rules.spec import VarCheck
from src.superopt.egraph.pattern import PatternNode, PatternVar

from .compat import run_post_passes, run_pre_passes
from .ir.convert import ir_to_onnx, onnx_to_ir
from .ir.graph import IRGraph
from .ir.node import IRNode, OP_INPUT, OP_NOOP, OP_PROJ, OP_WEIGHT
from .rules.base import check_vars

logger = logging.getLogger(__name__)


_MANUAL_RULE_ORDER = (
    "eliminate_identity",
    "shape_fold",
    "constantofshape_fold",
    "cos_fold",
    "sin_fold",
    "equal_fold",
    "less_fold",
    "ceil_fold",
    "floor_fold",
    "greater_to_less",
    "sub_to_add_neg",
    "neg_to_mul",
    "reciprocal_to_div",
    "not_to_sub",
    "abs_decompose",
    "pow_to_identity",
    "pow_to_sqrt",
    "pow_to_mul",
    "pow_to_cube",
    "pow_to_reciprocal",
    "pow_to_rsqrt",
    "squeeze_to_reshape",
    "unsqueeze_to_reshape",
    "flatten_to_reshape",
    "reshape_reshape",
    "transpose_cancel_perm_0_1",
    "transpose_cancel_perm_1_0",
    "layernorm_decompose",
    "where_mask_decompose",
    "where_to_arithmetic",
    "range_decompose",
    "bn_decompose",
    "gemm_decompose",
    "gemm_decompose_no_bias",
    "matmul_to_conv",
    "expand_to_mul_ones",
    # Cleanup for nodes introduced by decompositions above. This is manual
    # ordering, not a fixpoint loop.
    "sub_to_add_neg",
    "neg_to_mul",
    "reciprocal_to_div",
)

_BOUNDARY_OPS = frozenset({OP_INPUT, OP_WEIGHT, OP_NOOP, OP_PROJ})


@dataclass(frozen=True)
class IRBaselineResult:
    input_path: str
    output_path: str
    original_nodes: int
    optimized_nodes: int
    logs: list[str] = field(default_factory=list)


def optimize_ir_baseline(
    input_path: str | Path,
    output_path: str | Path,
) -> IRBaselineResult:
    """Run the fair-comparison IR baseline and save an ONNX model."""
    input_path = str(input_path)
    output_path = str(output_path)

    model = onnx.load(input_path)
    model = run_pre_passes(model)
    ir = onnx_to_ir(model)
    original_nodes = _count_compute_nodes(ir)

    logs = rewrite_ir_manual(ir)
    _prune_dead_ir(ir)

    opt_model = ir_to_onnx(ir, model)
    opt_model = run_post_passes(opt_model)
    onnx.save(opt_model, output_path)

    return IRBaselineResult(
        input_path=input_path,
        output_path=output_path,
        original_nodes=original_nodes,
        optimized_nodes=_count_compute_nodes(ir),
        logs=logs,
    )


def rewrite_ir_manual(ir: IRGraph) -> list[str]:
    """Apply shared RuleSpecs to IRGraph with a fixed manual order."""
    specs_by_name = {spec.name: spec for spec in get_all_specs()}
    logs: list[str] = []

    for rule_name in _MANUAL_RULE_ORDER:
        spec = specs_by_name.get(rule_name)
        if spec is None:
            raise KeyError(f"unknown manual baseline rule: {rule_name}")

        applied = _apply_rule_once(ir, spec)
        if applied:
            _prune_dead_ir(ir)
            logs.append(f"{rule_name}: applied {applied}")

    return logs


def _apply_rule_once(ir: IRGraph, spec: RuleSpec) -> int:
    applied = 0
    for node_id in list(ir.topo_order()):
        node = ir.nodes.get(node_id)
        if node is None or node.op in _BOUNDARY_OPS:
            continue

        # it's okay, spec has PatternNode. 
        match = _match_node(ir, node_id, spec.source) # type: ignore  
        if match is None:
            continue

        subst, _matched_ids = match
        if not _passes_checks(ir, spec.checks, subst):
            continue

        builder = IRRewriteBuilder(ir, node_id, subst)
        final_value = spec.build_fn(builder, dict(subst))
        if not isinstance(final_value, str):
            raise TypeError(
                f"{spec.name} returned non-IR value handle: "
                f"{type(final_value).__name__}"
            )
        if final_value == node_id:
            continue

        _replace_value(ir, node_id, final_value)
        applied += 1

    return applied


def _match_node(
    ir: IRGraph,
    node_id: str,
    pattern: PatternNode,
) -> tuple[dict[str, str], list[str]] | None:
    node = ir.nodes.get(node_id)
    if node is None:
        return None
    if node.op != pattern.op or len(node.inputs) != len(pattern.children):
        return None
    if pattern.attrs is not None:
        node_attrs = node.attrs_dict
        for key, expected in pattern.attrs:
            if node_attrs.get(key) != expected:
                return None

    subst: dict[str, str] = {}
    matched_ids = [node_id]
    for child, input_name in zip(pattern.children, node.inputs):
        if isinstance(child, PatternVar):
            previous = subst.get(child.name)
            if previous is not None and previous != input_name:
                return None
            subst[child.name] = input_name
            continue

        if isinstance(child, PatternNode):
            child_match = _match_node(ir, input_name, child)
            if child_match is None:
                return None
            child_subst, child_ids = child_match
            for var_name, value_id in child_subst.items():
                if var_name in subst and subst[var_name] != value_id:
                    return None
                subst[var_name] = value_id
            matched_ids.extend(child_ids)
            continue

        return None

    return subst, matched_ids


def _passes_checks(
    ir: IRGraph,
    checks: tuple[VarCheck, ...],
    subst: dict[str, str],
) -> bool:
    for check in checks:
        value_id = subst.get(check.var)
        if value_id is None:
            return False
        scalar_value = _scalar_value(ir, value_id)
        if check.scalar_close is not None:
            if scalar_value is None:
                return False
            if abs(scalar_value - check.scalar_close) > 1e-6:
                return False
        if check.scalar_abs_lt is not None:
            if scalar_value is None or abs(scalar_value) >= check.scalar_abs_lt:
                return False
        if check.scalar_lte is not None:
            if scalar_value is None or scalar_value > check.scalar_lte:
                return False
        if check.is_constant is not None:
            is_constant = _is_constant(ir, value_id)
            if is_constant != check.is_constant:
                return False
        if check.has_shape is not None:
            has_shape = _shape_of(ir, value_id) is not None
            if has_shape != check.has_shape:
                return False
    return True


def _replace_value(ir: IRGraph, old: str, new: str) -> None:
    for node_id, node in list(ir.nodes.items()):
        if old not in node.inputs:
            continue
        ir.nodes[node_id] = IRNode(
            id=node.id,
            op=node.op,
            inputs=tuple(new if value_id == old else value_id for value_id in node.inputs),
            attrs=node.attrs,
            shape=node.shape,
            dtype=node.dtype,
        )
    ir.outputs = tuple(new if value_id == old else value_id for value_id in ir.outputs)
    return 


def _prune_dead_ir(ir: IRGraph) -> None:
    # def-use chain. 
    roots = [ir.root] if ir.root is not None else list(ir.output_ids())
    reachable: set[str] = set()

    def visit(value_id: str | None) -> None:
        if value_id is None or value_id in reachable or value_id not in ir.nodes:
            return
        reachable.add(value_id)
        for input_id in ir.nodes[value_id].inputs:
            visit(input_id)

    for root in roots:
        visit(root)

    ir.nodes = {
        node_id: node
        for node_id, node in ir.nodes.items()
        if node_id in reachable
    }
    used_initializers = {
        node_id
        for node_id, node in ir.nodes.items()
        if node.op == OP_WEIGHT
    }
    ir.initializers = {
        name: value
        for name, value in ir.initializers.items()
        if name in used_initializers
    }
    return 


class IRRewriteBuilder:
    """GraphBuilder adapter that emits IR nodes and initializers."""

    def __init__(self, ir: IRGraph, source_id: str, subst: dict[str, str]) -> None:
        self.ir = ir
        self.source_id = source_id
        self.subst = subst
        self._index = 0

    def add_op(
        self,
        op: str,
        inputs: list[Any],
        attrs: dict[str, Any] | None = None,
    ) -> str:
        output_id = self._fresh_value_id(op.lower())
        self.ir.add_node(
            IRNode(
                id=output_id,
                op=op,
                inputs=tuple(_as_value_id(value) for value in inputs),
                attrs=tuple((attrs or {}).items()),
            )
        )
        return output_id

    def add_scalar(self, value: float, name: str = "") -> str:
        return self.add_array(
            np.array(value, dtype=np.float32),
            name=name or f"__const_{value}",
            dtype_code=1,
        )

    def add_array(
        self,
        arr: np.ndarray,
        name: str,
        dtype_code: int = 1,
    ) -> str:
        arr = np.ascontiguousarray(arr)
        value_id = self._fresh_weight_id(name)
        self.ir.add_initializer(value_id, arr)
        self.ir.add_node(
            IRNode(
                id=value_id,
                op=OP_WEIGHT,
                inputs=(),
                shape=tuple(arr.shape),
                dtype=dtype_code,
            )
        )
        return value_id

    def get_weight_data(self, var: str) -> np.ndarray | None:
        value_id = self.subst[var]
        data = self.ir.initializers.get(value_id)
        return None if data is None else data.copy()

    def get_shape(self, var: str) -> tuple[int, ...] | None:
        return _shape_of(self.ir, self.subst[var])

    def get_matched_shape(self) -> tuple[int, ...] | None:
        return _shape_of(self.ir, self.source_id)

    def get_matched_attr(self, key: str) -> Any:
        node = self.ir.nodes.get(self.source_id)
        if node is None:
            return None
        return node.attrs_dict.get(key)

    def get_match(self) -> str:
        return self.source_id

    def _fresh_value_id(self, role: str) -> str:
        while True:
            value_id = f"{self.source_id}__ir_{role}_{self._index}"
            self._index += 1
            if value_id not in self.ir.nodes and value_id not in self.ir.initializers:
                return value_id

    def _fresh_weight_id(self, name: str) -> str:
        clean = _clean_name(name)
        while True:
            value_id = f"{self.source_id}__ir_{clean}_{self._index}"
            self._index += 1
            if value_id not in self.ir.nodes and value_id not in self.ir.initializers:
                return value_id


def _shape_of(ir: IRGraph, value_id: str) -> tuple[int, ...] | None:
    node = ir.nodes.get(value_id)
    return None if node is None else node.shape


def _scalar_value(ir: IRGraph, value_id: str) -> float | None:
    value = ir.initializers.get(value_id)
    if value is None or value.size != 1:
        return None
    return float(value.reshape(-1)[0])


def _is_constant(ir: IRGraph, value_id: str) -> bool:
    return value_id in ir.initializers


def _count_compute_nodes(ir: IRGraph) -> int:
    return sum(1 for node in ir.nodes.values() if node.op not in _BOUNDARY_OPS)


def _as_value_id(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError(f"expected IR value id string, got {type(value).__name__}")
    return value


def _clean_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the IR-based manual baseline on an ONNX model.",
    )
    parser.add_argument("-i", "--input", required=True, help="Path to input ONNX model")
    parser.add_argument("-o", "--output", required=True, help="Path for rewritten ONNX model")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(name)s %(levelname)s: %(message)s",
    )

    result = optimize_ir_baseline(args.input, args.output)
    logger.info(
        "done: %d -> %d IR nodes",
        result.original_nodes,
        result.optimized_nodes,
    )
    for line in result.logs:
        logger.info(line)


if __name__ == "__main__":
    main()
