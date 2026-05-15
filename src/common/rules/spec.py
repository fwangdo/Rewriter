"""Common rewrite rule format shared by rewrite backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from src.superopt.egraph.pattern import Pattern, Subst


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


class GraphBuilder(ABC):
    """Backend-independent graph synthesis surface for complex rules."""

    # -- 노드 생성 --

    @abstractmethod
    def add_op(
        self,
        op: str,
        inputs: list[Any],
        shape: tuple[int, ...] | None = None,
        dtype: int | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> Any:
        """연산 노드를 추가하고 출력 핸들(value id 또는 e-class id)을 반환."""
        raise NotImplementedError

    @abstractmethod
    def add_scalar(self, value: float, var: str, name: str = "") -> Any:
        """var의 dtype에 맞춰 scalar 상수 노드를 추가."""
        raise NotImplementedError

    @abstractmethod
    def add_array(
        self,
        arr: np.ndarray,
        name: str,
        dtype_code: int = 1,
    ) -> Any:
        """numpy 배열을 weight 노드로 추가."""
        raise NotImplementedError

    # -- 매치된 노드 조회 --

    @abstractmethod
    def get_dtype(self, var: str) -> int | None:
        """패턴 변수 var에 매치된 노드의 ONNX dtype 코드를 반환."""
        raise NotImplementedError

    @abstractmethod
    def get_weight_data(self, var: str) -> np.ndarray | None:
        """패턴 변수 var에 매치된 weight의 numpy 데이터를 반환. 상수가 아니면 None."""
        raise NotImplementedError

    @abstractmethod
    def get_shape(self, var: str) -> tuple[int, ...] | None:
        """패턴 변수 var에 매치된 노드의 shape를 반환."""
        raise NotImplementedError

    @abstractmethod
    def get_matched_shape(self) -> tuple[int, ...] | None:
        """매치된 소스 패턴 루트 노드의 shape를 반환."""
        raise NotImplementedError

    @abstractmethod
    def get_matched_attr(self, key: str) -> Any:
        """매치된 소스 패턴 루트 노드에서 attribute를 이름으로 조회."""
        raise NotImplementedError

    @abstractmethod
    def get_match(self, fn: str = "", reason: str = "") -> Any:
        """매치된 소스 패턴 루트의 핸들을 그대로 반환. rewrite 포기 시 사용.
        fn: fallback이 발생한 build 함수명, reason: 실패 원인."""
        raise NotImplementedError

    @abstractmethod
    def get_opset_version(self) -> int:
        """그래프의 ONNX opset 버전을 반환."""
        raise NotImplementedError

    # -- 편의 메서드 (concrete) --

    def add_scalar_float(self, value: float, name: str = "") -> Any:
        return self.add_array(
            np.array(value, dtype=np.float32),
            name=name or f"__const_{value}",
            dtype_code=1,
        )

    def add_scalar_int64(self, value: int, name: str = "") -> Any:
        return self.add_array(
            np.array(value, dtype=np.int64),
            name=name or f"__const_{value}",
            dtype_code=7,
        )


BuildFn = Callable[[GraphBuilder, dict[str, Any]], Any]


@dataclass(frozen=True)
class RuleSpec:
    """Backend-independent rewrite rule specification."""

    name: str
    source: Pattern
    build_fn: BuildFn
    checks: tuple[VarCheck, ...] = ()
