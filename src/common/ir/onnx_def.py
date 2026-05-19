from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import src.common.spec.constant as OnnxOp


# consider factors to represent something symbolic. 
# sort, tensor, scalar, axis, axes, shape, dim, dtype 

# sort
class OnnxSort:
    """Base class for ONNX pattern sorts."""


@dataclass
class FPSort(OnnxSort):
    """FP"""


@dataclass
class BoolSort(OnnxSort):
    """BOOL"""


@dataclass
class SymbolSort(OnnxSort):
    name: str 


# tensor(var / const). 
class OnnxTerm:
    """Base class for ONNX pattern terms."""

@dataclass
class OnnxVar(OnnxTerm):
    name: str
    sort: OnnxSort
    attrs: dict[str, Any] = field(default_factory=dict) 


@dataclass
class OnnxConstVar(OnnxTerm):
    symbol: str 

@dataclass
class OnnxConst(OnnxTerm):
    name: str
    value: bool | int | float | OnnxConstVar 
    sort: OnnxSort 
    attrs: dict[str, Any] = field(default_factory=dict) 


@dataclass
class OnnxExpr(OnnxTerm):
    op_name: str 
    sort: OnnxSort | None 
    children: list[OnnxTerm] 
    attrs: dict[str, Any] = field(default_factory=dict)


def onnx_expr(
    op_name: str,
    sort: OnnxSort,
    children: list[OnnxTerm],
    **attrs: Any,
) -> OnnxExpr:
    return OnnxExpr(op_name, sort, children, attrs or dict())


# ---onnx vocabulary patterns.
HOLE = "hole" # signal to fill out. 

# sorts. 
FP_SORT   = FPSort()
BOOL_SORT = BoolSort()
SORT_A = SymbolSort("a")
SORT_B = SymbolSort("b")

# atom  
# F_A = OnnxVar("f_a", FP_SORT)
# F_B = OnnxVar("f_b", FP_SORT)

# B_A = OnnxVar("b_a", BOOL_SORT)
# B_B = OnnxVar("b_b", BOOL_SORT)


# ONNX ops observed in all benchmarks after ConstantFolding.
#
# Abs, Add, BatchNormalization, Cast, Clip, Concat, ConstantOfShape, Conv,
# Cos, Div, Equal, Erf, Expand, Flatten, Gather, GatherElements, Gemm,
# GlobalAveragePool, Greater, Identity, LayerNormalization, Less, MatMul,
# MaxPool, Mod, Mul, Neg, Pad, Pow, Range, ReduceMax, ReduceMean, ReduceSum,
# Relu, Reshape, Resize, ScatterND, Shape, Sigmoid, Sin, Slice, Softmax,
# Split, Sqrt, Squeeze, Sub, Tile, TopK, Transpose, Trilu, Unsqueeze, Where.
#
# ONNX ops observed in tinyllama_15m after ConstantFolding.
#
# Add, Cast, Concat, ConstantOfShape, Div, Equal, Expand, Gather, Less,
# MatMul, Mul, Neg, Pow, Range, ReduceMean, Reshape, Shape, Sigmoid, Slice,
# Softmax, Sqrt, Squeeze, Sub, Transpose, Unsqueeze, Where.
#

# TinyLlama ONNX op patterns.
#
# These are ONNX-side templates only. They do not define lowering semantics;
# rules.py should connect them to IR expressions.
VAR_A = OnnxVar("a", SORT_A)
VAR_B = OnnxVar("b", SORT_A)
VAR_C = OnnxVar("c", SORT_A)
VAR_COND = OnnxVar("cond", BOOL_SORT)
VAR_SHAPE = OnnxVar("shape", SymbolSort("shape"))
VAR_AXES = OnnxVar("axes", SymbolSort("axes"))
VAR_START = OnnxVar("start", SymbolSort("scalar"))
VAR_LIMIT = OnnxVar("limit", SymbolSort("scalar"))
VAR_DELTA = OnnxVar("delta", SymbolSort("scalar"))
VAR_ENDS = OnnxVar("ends", SymbolSort("tensor"))
VAR_STEPS = OnnxVar("steps", SymbolSort("tensor"))
VAR_INDICES = OnnxVar("indices", SymbolSort("tensor"))

ADD = onnx_expr(OnnxOp.ADD, SORT_A, [VAR_A, VAR_B])
CAST = onnx_expr(OnnxOp.CAST, SORT_A, [VAR_A], to=HOLE) # to means translated dtype. 
CONCAT = onnx_expr(OnnxOp.CONCAT, SORT_A, [VAR_A, VAR_B], axis=HOLE)
CONSTANT_OF_SHAPE = onnx_expr(OnnxOp.CONSTANT_OF_SHAPE, SORT_A, [VAR_SHAPE], value=HOLE)
DIV = onnx_expr(OnnxOp.DIV, SORT_A, [VAR_A, VAR_B])
EQUAL = onnx_expr(OnnxOp.EQUAL, BOOL_SORT, [VAR_A, VAR_B])
EXPAND = onnx_expr(OnnxOp.EXPAND, SORT_A, [VAR_A, VAR_SHAPE])
GATHER = onnx_expr(OnnxOp.GATHER, SORT_A, [VAR_A, VAR_INDICES], axis=HOLE)
LESS = onnx_expr(OnnxOp.LESS, BOOL_SORT, [VAR_A, VAR_B])
MAT_MUL = onnx_expr(OnnxOp.MAT_MUL, SORT_A, [VAR_A, VAR_B])
MUL = onnx_expr(OnnxOp.MUL, SORT_A, [VAR_A, VAR_B])
NEG = onnx_expr(OnnxOp.NEG, SORT_A, [VAR_A])
POW = onnx_expr(OnnxOp.POW, SORT_A, [VAR_A, VAR_B])
RANGE = onnx_expr(OnnxOp.RANGE, SORT_A, [VAR_START, VAR_LIMIT, VAR_DELTA])
REDUCE_MEAN = onnx_expr(OnnxOp.REDUCE_MEAN, SORT_A, [VAR_A, VAR_AXES], keepdims=HOLE)
RESHAPE = onnx_expr(OnnxOp.RESHAPE, SORT_A, [VAR_A, VAR_SHAPE])
SHAPE = onnx_expr(OnnxOp.SHAPE, SymbolSort("shape_tensor"), [VAR_A])
SIGMOID = onnx_expr(OnnxOp.SIGMOID, SORT_A, [VAR_A])
SLICE = onnx_expr(OnnxOp.SLICE, SORT_A, [VAR_A, VAR_START, VAR_ENDS, VAR_AXES, VAR_STEPS])
SOFTMAX = onnx_expr(OnnxOp.SOFTMAX, SORT_A, [VAR_A], axis=HOLE)
SQRT = onnx_expr(OnnxOp.SQRT, SORT_A, [VAR_A])
SQUEEZE = onnx_expr(OnnxOp.SQUEEZE, SORT_A, [VAR_A, VAR_AXES])
SUB = onnx_expr(OnnxOp.SUB, SORT_A, [VAR_A, VAR_B])
TRANSPOSE = onnx_expr(OnnxOp.TRANSPOSE, SORT_A, [VAR_A], perm=HOLE)
UNSQUEEZE = onnx_expr(OnnxOp.UNSQUEEZE, SORT_A, [VAR_A, VAR_AXES])
WHERE = onnx_expr(OnnxOp.WHERE, SORT_A, [VAR_COND, VAR_A, VAR_B])