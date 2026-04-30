from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

import onnx

from ..specs import SUPPORTED_OPS


class OpChecker:
    """Validate that a graph contains only supported ops."""

    @staticmethod
    def _normalize_supported_ops(supported_ops: Iterable[str] | None) -> frozenset[str]:
        return frozenset(supported_ops) if supported_ops is not None else SUPPORTED_OPS

    @classmethod
    def histogram(cls, model: onnx.ModelProto) -> dict[str, int]:
        return dict(sorted(Counter(node.op_type for node in model.graph.node).items()))

    @classmethod
    def get_unsupported_ops(
        cls,
        model: onnx.ModelProto,
        supported_ops: Iterable[str] | None = None,
    ) -> set[str]:
        supported = cls._normalize_supported_ops(supported_ops)
        return {node.op_type for node in model.graph.node if node.op_type not in supported}

    @classmethod
    def get_violations(
        cls,
        model: onnx.ModelProto,
        supported_ops: Iterable[str] | None = None,
    ) -> list[str]:
        supported = cls._normalize_supported_ops(supported_ops)
        violations: list[str] = []
        for node in model.graph.node:
            if node.op_type not in supported:
                name = node.name or (node.output[0] if node.output else node.op_type)
                violations.append(f"{name}: unsupported op '{node.op_type}'")
        return violations

    @classmethod
    def assert_supported_only(
        cls,
        model: onnx.ModelProto,
        supported_ops: Iterable[str] | None = None,
    ) -> None:
        violations = cls.get_violations(model, supported_ops=supported_ops)
        if violations:
            raise AssertionError(
                "graph is not supported-op-only:\n" + "\n".join(f"  - {item}" for item in violations)
            )

    @classmethod
    def summarize(
        cls,
        model: onnx.ModelProto,
        path: str = "",
        supported_ops: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        supported = cls._normalize_supported_ops(supported_ops)
        histogram = cls.histogram(model)
        unsupported = {op: count for op, count in histogram.items() if op not in supported}
        return {
            "path": path,
            "total_nodes": sum(histogram.values()),
            "supported_ops": sorted(supported),
            "op_histogram": histogram,
            "unsupported_histogram": unsupported,
            "is_supported_only": not unsupported,
        }
