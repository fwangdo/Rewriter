from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class OnnxTerm:
    """Base class for ONNX pattern terms."""


@dataclass
class OnnxVar(OnnxTerm):
    name: str

@dataclass
class OnnxExpr(OnnxTerm):
    op_name: str
    children: list[OnnxTerm]
    attrs: dict[str, Any] | None = None