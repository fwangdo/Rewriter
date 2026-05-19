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


# Numeric / floating tensor ops
FULL = "full"
TRIU = "triu"
TRIL = "tril"
SUM = "sum"
TRANSPOSE = "transpose"
SQRT = "sqrt"
ADD = "add"
SUBTRACT = "subtract"
MULTIPLY = "multiply"
DIVIDE = "divide"
POWER = "power"
DOT = "dot"
TENSOR_DOT = "tensordot"
WHERE = "where"

# Boolean tensor ops
LESS = "less"


FLOAT_OPS: Final[frozenset[str]] = frozenset(
    {
        FULL,
        TRIU,
        TRIL,
        SUM,
        TRANSPOSE,
        SQRT,
        ADD,
        SUBTRACT,
        MULTIPLY,
        DIVIDE,
        POWER,
        DOT,
        TENSOR_DOT,
        WHERE,
    }
)

BOOL_OPS: Final[frozenset[str]] = frozenset(
    {
        FULL,
        TRIU,
        TRIL,
        LESS,
    }
)

BOUNDARY_OPS: Final[frozenset[str]] = frozenset(
    {
        DOT,
        TENSOR_DOT,
        TRIU,
        TRIL,
        WHERE,
    }
)

ATTRIBUTE_SORTS: Final[frozenset[str]] = frozenset({"D", "S"})


__all__ = [
    "ADD",
    "ATTRIBUTE_SORTS",
    "BOOL_OPS",
    "BOUNDARY_OPS",
    "DIVIDE",
    "DOT",
    "FLOAT_OPS",
    "FULL",
    "LESS",
    "MULTIPLY",
    "POWER",
    "SQRT",
    "SUBTRACT",
    "SUM",
    "TENSOR_DOT",
    "TRANSPOSE",
    "TRIL",
    "TRIU",
    "WHERE",
]
