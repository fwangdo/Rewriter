"""Rewrite rules for the SIR-level e-graph.

These are domain-independent normalization rules. When applied to
saturation, different high-level ops (MatMul, Conv, etc.) that have
been lowered to generic form will converge to the same canonical form
and be automatically merged by the e-graph.
"""

from src.rulegen.rewrite.normalize import (
    eliminate_trivial_iterators,
    canonicalize_iterator_order,
    commute_inputs,
    introduce_conv1x1_contraction_layout,
    saturate,
)

__all__ = [
    "eliminate_trivial_iterators",
    "canonicalize_iterator_order",
    "commute_inputs",
    "introduce_conv1x1_contraction_layout",
    "saturate",
]
