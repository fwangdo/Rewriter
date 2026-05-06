"""egglog-backed equality saturation for Superopt.

This module replaces the hand-rolled e-graph engine on the main pipeline
path. It keeps ONNX/IR-specific materialization in Python, but delegates
congruence closure, rule application, and extraction to egglog.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from egglog import EGraph, Expr, String, StringLike, greedy_dag_cost_model
from egglog import rewrite, var
from egglog.deconstruct import get_callable_args, get_literal_value

from ..egraph.enode import ENode
from ..egraph.pattern import Pattern, PatternNode, PatternVar
from ..explore.explorer import ExploreStats
from ..extract.cost import CostModel
from ..ir.graph import IRGraph
from ..ir.node import IRNode, OP_INPUT, OP_NOOP, OP_PROJ, OP_WEIGHT
from ..rules.base import RewriteRule


class TensorExpr(Expr):
    """Generic tensor term language used by egglog."""

    @classmethod
    def op0(cls, op: StringLike, attrs: StringLike) -> TensorExpr: ...

    @classmethod
    def op1(cls, op: StringLike, attrs: StringLike, a: TensorExpr) -> TensorExpr: ...

    @classmethod
    def op2(
        cls,
        op: StringLike,
        attrs: StringLike,
        a: TensorExpr,
        b: TensorExpr,
    ) -> TensorExpr: ...

    @classmethod
    def op3(
        cls,
        op: StringLike,
        attrs: StringLike,
        a: TensorExpr,
        b: TensorExpr,
        c: TensorExpr,
    ) -> TensorExpr: ...

    @classmethod
    def op4(
        cls,
        op: StringLike,
        attrs: StringLike,
        a: TensorExpr,
        b: TensorExpr,
        c: TensorExpr,
        d: TensorExpr,
    ) -> TensorExpr: ...

    @classmethod
    def op5(
        cls,
        op: StringLike,
        attrs: StringLike,
        a: TensorExpr,
        b: TensorExpr,
        c: TensorExpr,
        d: TensorExpr,
        e: TensorExpr,
    ) -> TensorExpr: ...

    @classmethod
    def op6(
        cls,
        op: StringLike,
        attrs: StringLike,
        a: TensorExpr,
        b: TensorExpr,
        c: TensorExpr,
        d: TensorExpr,
        e: TensorExpr,
        f: TensorExpr,
    ) -> TensorExpr: ...

    @classmethod
    def op7(
        cls,
        op: StringLike,
        attrs: StringLike,
        a: TensorExpr,
        b: TensorExpr,
        c: TensorExpr,
        d: TensorExpr,
        e: TensorExpr,
        f: TensorExpr,
        g: TensorExpr,
    ) -> TensorExpr: ...

    @classmethod
    def op8(
        cls,
        op: StringLike,
        attrs: StringLike,
        a: TensorExpr,
        b: TensorExpr,
        c: TensorExpr,
        d: TensorExpr,
        e: TensorExpr,
        f: TensorExpr,
        g: TensorExpr,
        h: TensorExpr,
    ) -> TensorExpr: ...


_OP_CTORS: dict[int, Callable[..., TensorExpr]] = {
    0: TensorExpr.op0,
    1: TensorExpr.op1,
    2: TensorExpr.op2,
    3: TensorExpr.op3,
    4: TensorExpr.op4,
    5: TensorExpr.op5,
    6: TensorExpr.op6,
    7: TensorExpr.op7,
    8: TensorExpr.op8,
}


@dataclass
class EgglogResult:
    ir: IRGraph
    estimated_cost: float | None = None


class OrtCost:
    """egglog cost callback backed by FLOPs-based cost model."""

    def __init__(self, cost_model: CostModel, backend: "EgglogBackend") -> None:
        self.cost_model = cost_model
        self.backend = backend

    def __call__(
        self,
        egraph: EGraph,
        expr: TensorExpr,
        children_costs: list[float],
    ) -> float:
        del egraph
        args = get_callable_args(expr)
        if args is None or len(args) < 2:
            return sum(children_costs) + 1.0
        op = get_literal_value(args[0])
        if not isinstance(op, str):
            return sum(children_costs) + 1.0
        attrs_key = get_literal_value(args[1])
        output_shape = None
        attrs: tuple[tuple[str, object], ...] = ()
        if isinstance(attrs_key, str):
            output_shape = self.backend._shape_by_key.get(attrs_key)
            attrs = self.backend._key_to_attrs.get(attrs_key, ())
        return sum(children_costs) + self.cost_model.node_cost(
            ENode(op, (), attrs=attrs), output_shape=output_shape,
        )


class EgglogBackend:
    """IRGraph <-> egglog adapter."""

    _IR_NODE_ID_ATTR = "__name__"
    _TUPLE_OP = "__tuple__"

    def __init__(self, ir: IRGraph) -> None:
        self.ir = ir
        self.egraph = EGraph()
        self._attrs_to_key: dict[tuple[tuple[str, Any], ...], str] = {}
        self._key_to_attrs: dict[str, tuple[tuple[str, Any], ...]] = {}
        self._shape_by_key: dict[str, tuple[int, ...] | None] = {}
        self._dtype_by_key: dict[str, int | None] = {}
        self._next_attr_id = 0
        self.root: TensorExpr | None = None
        self._load_ir()

    def run_rules(
        self,
        rules: list[RewriteRule],
        max_iter: int,
        max_nodes: int,
    ) -> ExploreStats:
        del max_nodes
        stats = ExploreStats()
        if max_iter <= 0:
            stats.saturated = False
            return stats

        commands = []
        skipped = 0
        for rule in rules:
            # Rules that need Python-side value inspection or graph synthesis
            # are not pure egglog rewrites yet. They are ported separately.
            if rule.check is not None or rule.apply_fn is not None:
                skipped += 1
                continue
            source = self._pattern_to_expr(
                rule.source,
                attr_prefix=f"{rule.name}_s",
                wildcard_attrs=True,
            )
            target = self._pattern_to_expr(
                rule.target,
                attr_prefix=f"{rule.name}_t",
                wildcard_attrs=False,
            )
            commands.append(rewrite(source).to(target))

        stats.total_matches = len(commands)
        if commands:
            self.egraph.register(*commands)
            self.egraph.run(max_iter)
            stats.iterations = max_iter
            stats.total_applied = len(commands)
        stats.saturated = skipped == 0
        return stats

    def extract_best(self, cost_model: CostModel) -> EgglogResult:
        if self.root is None:
            raise ValueError("cannot extract: missing root")
        _ensure_recursion_limit()
        expr, cost = self.egraph.extract(
            self.root,
            include_cost=True,
            cost_model=greedy_dag_cost_model(OrtCost(cost_model, self)),
        )
        return EgglogResult(
            ir=self._expr_to_ir(expr),
            estimated_cost=_cost_to_float(cost),
        )

    def extract_topk(self, k: int) -> list[EgglogResult]:
        if self.root is None:
            raise ValueError("cannot extract: missing root")
        if k <= 1:
            return [self.extract_best(CostModel())]
        exprs = self.egraph.extract_multiple(self.root, k)
        results: list[EgglogResult] = []
        seen: set[str] = set()
        for expr in exprs:
            ir = self._expr_to_ir(expr)
            sig = repr(ir)
            if sig in seen:
                continue
            seen.add(sig)
            results.append(EgglogResult(ir=ir))
        return results

    def _load_ir(self) -> None:
        node_to_expr: dict[str, TensorExpr] = {}
        for nid in self.ir.topo_order():
            node = self.ir.nodes[nid]
            attrs = tuple((k, v) for k, v in node.attrs if k != self._IR_NODE_ID_ATTR)
            # Preserve the original IR node id so extracted graphs can keep
            # ONNX output names and initializer names stable.
            attrs = attrs + ((self._IR_NODE_ID_ATTR, nid),)
            attr_key = self._attrs_key(_hashable_attrs(attrs), node.shape, node.dtype)
            children = tuple(node_to_expr[inp] for inp in node.inputs)
            expr = self._make_expr(node.op, attr_key, children)
            node_to_expr[nid] = self.egraph.let(self._safe_let_name(nid), expr)

        if self.ir.root is None:
            raise ValueError("IRGraph has no root")
        self.root = node_to_expr[self.ir.root]

    def _attrs_key(
        self,
        attrs: tuple[tuple[str, Any], ...],
        shape: tuple[int, ...] | None = None,
        dtype: int | None = None,
    ) -> str:
        # egglog terms can carry strings cheaply; Python attrs can be numpy
        # objects or tuples. Keep attrs in Python and store only a stable key
        # in the egglog term.
        key_attrs = tuple(attrs)
        existing = self._attrs_to_key.get(key_attrs)
        if existing is not None:
            return existing
        key = f"a{self._next_attr_id}"
        self._next_attr_id += 1
        self._attrs_to_key[key_attrs] = key
        self._key_to_attrs[key] = key_attrs
        self._shape_by_key[key] = shape
        self._dtype_by_key[key] = dtype
        return key

    def _make_expr(
        self,
        op: str,
        attrs_key: str,
        children: tuple[TensorExpr, ...],
    ) -> TensorExpr:
        if len(children) > max(_OP_CTORS):
            return self._make_expr(op, attrs_key, (self._pack_children(children),))
        try:
            ctor = _OP_CTORS[len(children)]
        except KeyError as exc:
            raise ValueError(
                f"unsupported op arity for egglog backend: {op}/{len(children)}"
            ) from exc
        return ctor(op, attrs_key, *children)

    def _pack_children(self, children: tuple[TensorExpr, ...]) -> TensorExpr:
        if len(children) == 1:
            return children[0]
        mid = len(children) // 2
        empty_attrs = self._attrs_key(())
        return self._make_expr(
            self._TUPLE_OP,
            empty_attrs,
            (
                self._pack_children(children[:mid]),
                self._pack_children(children[mid:]),
            ),
        )

    def _pattern_to_expr(
        self,
        pattern: Pattern,
        attr_prefix: str,
        wildcard_attrs: bool,
    ) -> TensorExpr:
        if isinstance(pattern, PatternVar):
            return var(pattern.name.lstrip("?"), TensorExpr)
        if not isinstance(pattern, PatternNode):
            raise TypeError(f"unsupported pattern: {pattern!r}")

        children = tuple(
            self._pattern_to_expr(child, attr_prefix, wildcard_attrs)
            for child in pattern.children
        )
        if pattern.attrs is None and wildcard_attrs:
            attrs_key = var(f"{attr_prefix}_attrs_{abs(hash(pattern))}", String)
        elif pattern.attrs is None:
            attrs_key = self._attrs_key(())
        else:
            attrs_key = self._attrs_key(pattern.attrs)
        return self._make_expr(pattern.op, attrs_key, children)

    def _expr_to_ir(self, expr: TensorExpr) -> IRGraph:
        ir = IRGraph()
        ir.inputs = self.ir.inputs
        ir.outputs = self.ir.outputs
        ir.initializers = dict(self.ir.initializers)
        built: dict[TensorExpr, str] = {}
        counter = 0

        def visit(cur: TensorExpr) -> str:
            nonlocal counter
            if cur in built:
                return built[cur]

            args = get_callable_args(cur)
            if args is None or len(args) < 2:
                raise ValueError(f"cannot decode egglog expression: {cur!r}")
            op = get_literal_value(args[0])
            attrs_key = get_literal_value(args[1])
            if not isinstance(op, str) or not isinstance(attrs_key, str):
                raise ValueError(f"cannot decode egglog expression args: {cur!r}")
            attrs = self._key_to_attrs.get(attrs_key, ())
            attrs_dict = dict(attrs)
            child_ids = tuple(visit(child) for child in self._unpack_args(args[2:]))

            if op in (OP_INPUT, OP_WEIGHT):
                node_id = attrs_dict.get(self._IR_NODE_ID_ATTR)
                if not isinstance(node_id, str):
                    node_id = f"{op}_{counter}"
            elif op == OP_NOOP:
                node_id = "__noop_root__"
            else:
                node_id = attrs_dict.get(self._IR_NODE_ID_ATTR)
                if not isinstance(node_id, str):
                    node_id = f"_egg_{counter}_{op}"
            counter += 1

            if node_id in ir.nodes:
                node_id = f"{node_id}_{counter}"
            ir.add_node(
                IRNode(
                    id=node_id,
                    op=op,
                    inputs=child_ids,
                    attrs=tuple((k, v) for k, v in attrs if k != self._IR_NODE_ID_ATTR),
                    shape=self._shape_by_key.get(attrs_key),
                    dtype=self._dtype_by_key.get(attrs_key),
                )
            )
            built[cur] = node_id
            return node_id

        ir.root = visit(expr)
        return ir

    def _unpack_args(self, args: tuple[TensorExpr, ...]) -> tuple[TensorExpr, ...]:
        unpacked: list[TensorExpr] = []
        for arg in args:
            unpacked.extend(self._unpack_expr(arg))
        return tuple(unpacked)

    def _unpack_expr(self, expr: TensorExpr) -> tuple[TensorExpr, ...]:
        args = get_callable_args(expr)
        if args is None or len(args) < 2:
            return (expr,)
        op = get_literal_value(args[0])
        if op != self._TUPLE_OP:
            return (expr,)
        return self._unpack_args(args[2:])

    @staticmethod
    def _safe_let_name(name: str) -> str:
        return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)


def _hashable_attrs(
    attrs: tuple[tuple[str, Any], ...],
) -> tuple[tuple[str, Any], ...]:
    result = []
    for key, value in attrs:
        if isinstance(value, np.ndarray):
            result.append((key, (str(value.dtype), value.shape, value.tobytes())))
        else:
            result.append((key, value))
    return tuple(result)


def _cost_to_float(cost: Any) -> float:
    total = getattr(cost, "total", cost)
    return float(total)


def _ensure_recursion_limit() -> None:
    # egglog's Python binding reconstructs extracted term DAGs recursively.
    # Transformer-style graphs can exceed Python's default recursion limit.
    if sys.getrecursionlimit() < 10_000:
        sys.setrecursionlimit(10_000)
