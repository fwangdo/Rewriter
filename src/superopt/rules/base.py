"""Rule application for the e-graph exploration phase.

Applies RuleSpec rules directly on the hand-rolled e-graph.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.common.rules.legalization import RuleSpec
from src.common.rules.spec import VarCheck, GraphBuilder

from ..egraph.eclass import AnalysisData
from ..egraph.enode import EClassId, ENode
from ..egraph.egraph import EGraph
from ..egraph.pattern import Subst


def apply_rule(
    egraph: EGraph,
    rule: RuleSpec,
    match_cid: EClassId,
    subst: Subst,
) -> EClassId | None:
    """Apply a RuleSpec to a matched e-class.

    Returns the merged e-class id, or None if check failed or shapes
    are incompatible.
    """
    if not _check_vars(egraph, rule.checks, subst):
        return None

    builder = EGraphBuilder(egraph, match_cid, subst)
    target_cid = rule.build_fn(builder, dict(subst))

    # Skip merge if shapes are incompatible.
    s1 = egraph.eclass(match_cid).data.shape
    s2 = egraph.eclass(target_cid).data.shape
    if s1 is not None and s2 is not None and s1 != s2:
        return None

    return egraph.merge(match_cid, target_cid)


def _check_vars(egraph: EGraph, checks: tuple[VarCheck, ...], subst: Subst) -> bool:
    """Evaluate VarCheck guards against the e-graph."""
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


class EGraphBuilder:
    """GraphBuilder implementation for the hand-rolled e-graph."""

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
