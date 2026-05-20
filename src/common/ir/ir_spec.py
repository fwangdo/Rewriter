"""NumPy-like operation vocabulary for the next IR.

The important design unit is not this finite list itself.  The main boundary
is the lowering contract:

    ONNX op -> small NumPy-like tensor expression

This module records the first operation vocabulary we are willing to lower
into.  It is intentionally close to a NumPy subset because that keeps
constant folding and reference evaluation straightforward.


Current scope.  
    - onnx level operations. 
        add, mul, sub, div, neg 
        equal, less, where
        reshape, transpose, resize, slice, split
        cast, clip, concat, expand
        full, 
    # additional? 
        reduce, sum, mean, min, max  
    # not now 
        triu, tril, 

    # composed? 
        conv2d 
        conv2dtrans 
        reduceSum 

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
ADD = "add"
MUL = "multiply"
SUB = "subtract"
DIV = "divide" 
NEG = "neg" 

EQUAL = "equal"
LESS = "less"
WHERE = "where"

RESHAPE = "reshape"
TRANSPOSE = "transpose"
RESIZE = "resize"
SLICE = "slice"
SPLIT = "split"

CAST = "cast"
CLIP = "clip"
CONCAT = "concat"
EXPAND = "expand"

FULL = "full"
TRIU = "triu"
TRIL = "tril"

REDUCE = "reduce"
SUM = "sum"
MEAN = "mean"
MIN = "min"
MIN = "max"

SQRT = "sqrt"
POWER = "power"

SOFTMAX = "softmax"
SIGMOID = "sigmoid"
SHAPE = "shape"


# DOT = "dot"
# TENSOR_DOT = "tensordot" # matmul 