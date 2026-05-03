"""E-class: a set of equivalent e-nodes.

Each e-class carries optional analysis data (shape, dtype, etc.)
computed bottom-up from its members.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .enode import ENodeId


@dataclass
class AnalysisData:
    """Per-e-class analysis results, propagated bottom-up.

    Used for shape checking during exploration and cost estimation
    during extraction.
    """

    shape: tuple[int, ...] | None = None
    dtype: int | None = None
    is_constant: bool = False
    preferred_name: str | None = None

    @staticmethod
    def join(a: AnalysisData, b: AnalysisData) -> AnalysisData:
        """Merge analysis data when two e-classes are merged.

        Shapes must be compatible (equal or one is None).
        """
        shape = a.shape if a.shape is not None else b.shape
        if a.shape is not None and b.shape is not None and a.shape != b.shape:
            raise ValueError(
                f"shape conflict during e-class merge: {a.shape} vs {b.shape}"
            )
        dtype = a.dtype if a.dtype is not None else b.dtype
        is_constant = a.is_constant or b.is_constant
        preferred_name = (
            a.preferred_name
            if a.preferred_name is not None
            else b.preferred_name
        )
        return AnalysisData(
            shape=shape,
            dtype=dtype,
            is_constant=is_constant,
            preferred_name=preferred_name,
        )


@dataclass
class EClass:
    """An equivalence class of e-nodes."""

    id: int
    nodes: set[ENodeId] = field(default_factory=set)
    parents: set[ENodeId] = field(default_factory=set)
    data: AnalysisData = field(default_factory=AnalysisData)
