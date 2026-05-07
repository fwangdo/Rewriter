"""Internal IR node for e-graph representation.

Tensat represents each operator as a node whose value is the output tensor.
We follow the same convention: one IRNode = one output tensor.

Multi-output ONNX ops (e.g. Split) are decomposed into a base node
plus projection nodes (split_0, split_1, ...) so every IRNode has
exactly one output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class IRNode:
    """A single operation in the IR graph.

    Each node produces exactly one output tensor.
    ``inputs`` are references to other nodes' output ids.
    """

    id: str
    op: str
    inputs: tuple[str, ...]
    attrs: tuple[tuple[str, Any], ...] = ()
    shape: tuple[int, ...] | None = None
    dtype: int | None = None  # onnx.TensorProto.DataType

    @property
    def attrs_dict(self) -> dict[str, Any]:
        return dict(self.attrs)


# Sentinel ops for graph boundary and multi-output handling.
OP_INPUT = "input"      # graph input placeholder
OP_WEIGHT = "weight"    # initializer (constant weight) leaf
OP_NOOP = "noop"        # combines multiple graph outputs into one root
OP_PROJ = "proj"        # projection for multi-output ops: proj_0, proj_1, ...
