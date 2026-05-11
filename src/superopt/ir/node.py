"""Internal IR node for e-graph representation.

The IR is value-oriented: an ordinary IRNode represents one tensor value
and the operation that produces it. For example, ``IRNode(id="z", op="Add",
inputs=("x", "y"))`` means tensor value ``z`` is produced by ``Add(x, y)``.

Multi-output ONNX ops are the exception. They are represented as one base
operation node plus one ``proj`` node per output value. The base node is an
operation invocation handle; the ``proj`` nodes are the tensor values that
downstream IR nodes consume.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class IRNode:
    """A value node in the IR graph.

    For ordinary ops, ``id`` is the produced tensor value name and ``op`` is
    the producer operation. ``inputs`` are references to other value ids.
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
