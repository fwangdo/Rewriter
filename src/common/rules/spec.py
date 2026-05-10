"""Common rewrite rule format shared by rewrite backends."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from src.superopt.egraph.pattern import Pattern


@dataclass(frozen=True)
class PatternSpec:
    """Backend-independent tree pattern."""

    op: str
    # forward referecne. 
    args: tuple[str | "PatternSpec", ...] # type: ignore 
    attrs: tuple[tuple[str, Any], ...] | None = None


@dataclass(frozen=True)
class VarCheck:
    """Small declarative guard over a matched pattern variable."""
    # for marking specific properties of vars. 

    var: str
    scalar_close: float | None = None  # scalar 값이 이 값과 근사해야 함 (|v - x| <= 1e-6)
    scalar_abs_lt: float | None = None  # scalar 절댓값이 이 값 미만이어야 함 (|v| < x)
    scalar_lte: float | None = None  # scalar 값이 이 값 이하여야 함 (v <= x)
    is_constant: bool | None = None  # 상수(weight) 노드인지 여부
    has_shape: bool | None = None  # shape 정보가 존재하는지 여부


class GraphBuilder(Protocol):
    """Backend-independent graph synthesis surface for complex rules."""

    def add_op(
        self,
        op: str,
        inputs: list[Any],
        attrs: dict[str, Any] | None = None,
    ) -> Any: ...

    def add_scalar(self, value: float, name: str = "") -> Any: ...

    def add_array(
        self,
        arr: np.ndarray,
        name: str,
        dtype_code: int = 1,
    ) -> Any: ...

    def get_weight_data(self, var: str) -> np.ndarray | None: ...

    def get_shape(self, var: str) -> tuple[int, ...] | None: ...

    def get_matched_shape(self) -> tuple[int, ...] | None: ...

    def get_matched_attr(self, key: str) -> Any: ...

    def get_match(self) -> Any: ...


BuildFn = Callable[[GraphBuilder, dict[str, Any]], Any]


@dataclass(frozen=True)
class RuleSpec:
    """Backend-independent rewrite rule specification."""

    name: str
    source: Pattern
    build_fn: BuildFn
    checks: tuple[VarCheck, ...] = ()