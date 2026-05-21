"""Rewrite rules for the SIR-level e-graph.

These are target-independent normalization rules over lowered generic
contractions. Backend-specific ops are recognized later by structural lifting.
"""

from src.rulegen.rewrite.normalize import (
    eliminate_trivial_iterators,
    canonicalize_iterator_order,
    commute_inputs,
    introduce_rank4_unit_contraction_view,
    saturate,
)

__all__ = [
    "eliminate_trivial_iterators",
    "canonicalize_iterator_order",
    "commute_inputs",
    "introduce_rank4_unit_contraction_view",
    "saturate",
]
