from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import onnx

from src.common.contracts import SUPPORTED_OPS


@dataclass(frozen=True)
class AuditSummary:
    path: str
    total_nodes: int
    supported_ops: list[str]
    op_histogram: dict[str, int]
    unsupported_histogram: dict[str, int]

    @property
    def is_supported_only(self) -> bool:
        return not self.unsupported_histogram

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["is_supported_only"] = self.is_supported_only
        return data


def audit_model(
    model: onnx.ModelProto,
    supported_ops: Iterable[str] | None = None,
) -> AuditSummary:
    supported = frozenset(supported_ops) if supported_ops is not None else SUPPORTED_OPS
    histogram = Counter(node.op_type for node in model.graph.node)
    unsupported = {op: count for op, count in sorted(histogram.items()) if op not in supported}
    return AuditSummary(
        path="",
        total_nodes=sum(histogram.values()),
        supported_ops=sorted(supported),
        op_histogram=dict(sorted(histogram.items())),
        unsupported_histogram=unsupported,
    )


def audit_path(
    path: str | Path,
    supported_ops: Iterable[str] | None = None,
) -> AuditSummary:
    model_path = Path(path)
    model = onnx.load(str(model_path))
    summary = audit_model(model, supported_ops=supported_ops)
    return AuditSummary(
        path=str(model_path),
        total_nodes=summary.total_nodes,
        supported_ops=summary.supported_ops,
        op_histogram=summary.op_histogram,
        unsupported_histogram=summary.unsupported_histogram,
    )
