"""Convert common RuleSpec objects into superopt RewriteRule objects."""

from __future__ import annotations

from typing import Any

import numpy as np
from src.common.rules import PatternSpec, RuleSpec, VarCheck

from ..egraph.eclass import AnalysisData
from ..egraph.enode import EClassId, ENode
from ..egraph.egraph import EGraph
from ..egraph.pattern import Pattern, PatternNode, PatternVar, Subst
from .base import RewriteRule


def rulespec_to_rewrite(spec: RuleSpec) -> RewriteRule:
    """Convert a backend-agnostic RuleSpec to the legacy RewriteRule shape."""
    if spec.target is None and spec.build_fn is None:
        raise ValueError(f"RuleSpec without target is not pattern-convertible: {spec.name}")
    return RewriteRule(
        name=spec.name,
        source=_patternspec_to_pattern(spec.source),
        target=_target_to_pattern(spec.target) if spec.target is not None else _patternspec_to_pattern(spec.source),
        check=_build_check(spec.checks),
        apply_fn=_build_apply(spec) if spec.build_fn is not None else None,
    )


def rulespecs_to_rewrites(specs: list[RuleSpec]) -> list[RewriteRule]:
    return [rulespec_to_rewrite(spec) for spec in specs]


def _target_to_pattern(target: PatternSpec | str) -> Pattern:
    if isinstance(target, str):
        return PatternVar(target)
    return _patternspec_to_pattern(target)


def _patternspec_to_pattern(pattern: PatternSpec | str) -> Pattern:
    if isinstance(pattern, str):
        return PatternVar(pattern)
    children = tuple(_patternspec_to_pattern(arg) for arg in pattern.args)
    return PatternNode(pattern.op, children, attrs=pattern.attrs)


def _build_check(checks: tuple[VarCheck, ...]):
    if not checks:
        return None

    def _check(egraph: EGraph, subst: Subst) -> bool:
        for check in checks:
            cid = subst.get(check.var)
            if cid is None:
                return False
            data = egraph.eclass(cid).data
            if check.scalar_close is not None:
                if data.scalar_value is None:
                    return False
                if abs(data.scalar_value - check.scalar_close) > 1e-6:
                    return False
            if check.scalar_abs_lt is not None:
                if data.scalar_value is None or abs(data.scalar_value) >= check.scalar_abs_lt:
                    return False
            if check.scalar_lte is not None:
                if data.scalar_value is None or data.scalar_value > check.scalar_lte:
                    return False
            if check.is_constant is not None and data.is_constant != check.is_constant:
                return False
            if check.has_shape is not None and (data.shape is not None) != check.has_shape:
                return False
        return True

    return _check


def _build_apply(spec: RuleSpec):
    assert spec.build_fn is not None

    def _apply(egraph: EGraph, match_cid: EClassId, subst: Subst) -> EClassId:
        builder = EGraphBuilder(egraph, match_cid, subst)
        return spec.build_fn(builder, dict(subst))

    return _apply


class EGraphBuilder:
    """GraphBuilder implementation for the legacy EGraph bridge."""

    def __init__(self, egraph: EGraph, match_cid: EClassId, subst: Subst) -> None:
        self.egraph = egraph
        self.match_cid = match_cid
        self.subst = subst

    def add_op(
        self,
        op: str,
        inputs: list[Any],
        attrs: dict[str, Any] | None = None,
    ) -> EClassId:
        return self.egraph.add(
            ENode(
                op,
                tuple(_as_cid(value) for value in inputs),
                attrs=tuple((attrs or {}).items()),
            )
        )

    def add_scalar(self, value: float, name: str = "") -> EClassId:
        arr = np.array(value, dtype=np.float32)
        return self.add_array(arr, name=name or f"__const_{value}", dtype_code=1)

    def add_array(
        self,
        arr: np.ndarray,
        name: str,
        dtype_code: int = 1,
    ) -> EClassId:
        arr = np.ascontiguousarray(arr)
        hashable = (str(arr.dtype), arr.shape, arr.tobytes())
        cid = self.egraph.add(
            ENode(
                "weight",
                (),
                attrs=(
                    ("__name__", name),
                    ("__synth__", hashable),
                ),
            )
        )
        scalar_value = float(arr.reshape(-1)[0]) if arr.size == 1 else None
        self.egraph.update_analysis(
            cid,
            AnalysisData(
                shape=tuple(arr.shape),
                dtype=dtype_code,
                is_constant=True,
                scalar_value=scalar_value,
            ),
        )
        return cid

    def get_weight_data(self, var: str) -> np.ndarray | None:
        cid = self.subst[var]
        ec = self.egraph.eclass(cid)
        if not ec.data.is_constant:
            return None
        for nid in ec.nodes:
            enode = self.egraph.enode(nid)
            if enode.op != "weight":
                continue
            for key, value in enode.attrs:
                if key == "__synth__":
                    dtype_str, shape, data = value
                    return np.frombuffer(data, dtype=np.dtype(dtype_str)).reshape(shape).copy()
            for key, value in enode.attrs:
                if key == "__name__" and value in self.egraph.initializers:
                    return self.egraph.initializers[value].copy()
        return None

    def get_shape(self, var: str) -> tuple[int, ...] | None:
        return self.egraph.eclass(self.subst[var]).data.shape

    def get_matched_shape(self) -> tuple[int, ...] | None:
        return self.egraph.eclass(self.match_cid).data.shape

    def get_matched_attr(self, key: str) -> Any:
        for nid in self.egraph.eclass(self.match_cid).nodes:
            for attr_key, attr_value in self.egraph.enode(nid).attrs:
                if attr_key == key:
                    return attr_value
        return None

    def get_match(self) -> EClassId:
        return self.match_cid


def _as_cid(value: Any) -> EClassId:
    if not isinstance(value, int):
        raise TypeError(f"expected EClassId-compatible int, got {type(value).__name__}")
    return value
