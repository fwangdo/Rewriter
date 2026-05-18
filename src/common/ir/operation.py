"""NumPy-like operation vocabulary for the next IR.

The important design unit is not this finite list itself.  The main boundary
is the lowering contract:

    ONNX op -> small NumPy-like tensor expression

This module records the first operation vocabulary we are willing to lower
into.  It is intentionally close to a NumPy subset because that keeps
constant folding and reference evaluation straightforward.

Grammar sketch, adapted from STENSO:

    F ::= full(S, FScalar)
        | triu(F) | tril(F)
        | sum(F, D) | transpose(F, D)
        | sqrt(F) | add(F, F) | subtract(F, F)
        | multiply(F, F) | divide(F, F) | power(F, F)
        | dot(F, F) | tensordot(F, F, D, D)
        | where(B, F, F)
        | FArg | FConst

    B ::= full(S, BScalar)
        | triu(B) | tril(B)
        | less(F, F)
        | BArg | BConst

    D ::= DConst   # axis, axes, permutation, contraction dimensions
    S ::= SConst   # shape tuple

Notes:
    - F/B are expression sorts, not precise ONNX dtypes.
    - D/S are attributes, not tensor values.
    - dot/tensordot are kept as primitives for now to avoid lowering too far
      before lifting is stable.
    - triu/tril/where are boundary primitives: they can be rewritten, but
      lowering them further can obscure mask semantics or introduce dtype/NaN
      corner cases.
"""
from __future__ import annotations

from typing import Final


FLOAT_OPS: Final[frozenset[str]] = frozenset(
    {
        "full",
        "triu",
        "tril",
        "sum",
        "transpose",
        "sqrt",
        "add",
        "subtract",
        "multiply",
        "divide",
        "power",
        "dot",
        "tensordot",
        "where",
    }
)

BOOL_OPS: Final[frozenset[str]] = frozenset(
    {
        "full",
        "triu",
        "tril",
        "less",
    }
)

BOUNDARY_OPS: Final[frozenset[str]] = frozenset(
    {
        "dot",
        "tensordot",
        "triu",
        "tril",
        "where",
    }
)

ATTRIBUTE_SORTS: Final[frozenset[str]] = frozenset({"D", "S"})


__all__ = [
    "ATTRIBUTE_SORTS",
    "BOOL_OPS",
    "BOUNDARY_OPS",
    "FLOAT_OPS",
]
