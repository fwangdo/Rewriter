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
    scalar_close: float | None = None
    scalar_abs_lt: float | None = None
    scalar_lte: float | None = None
    is_constant: bool | None = None
    has_shape: bool | None = None


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