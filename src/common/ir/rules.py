"""Bidirectional definitions between ONNX ops and the NumPy-like IR.

This module is intentionally declarative.  It describes symbolic expression
patterns; lowering/lifting code should decide which direction to use them in.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing      import Any 

from .ir_def import IRExpr, IRTerm
from .onnx_def import OnnxExpr

import src.common.spec.constant as OnnxOp
import src.common.ir.ir_def as IRDef
import src.common.ir.onnx_def as OnnxDef


@dataclass
class OpDefinition:
    name: str
    onnx: OnnxExpr
    ir: IRExpr
    constraints: dict[str, Any] 


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


def _ir_placeholder(op_name: str, *children: IRTerm, **attrs: Any) -> IRExpr:
    return IRDef.ir(op_name, IRDef.FP_SORT, *children, **attrs)


# --- op definitions. 

ADD = OpDefinition(
    name=OnnxOp.ADD,
    onnx=OnnxDef.ADD,  
    ir=IRDef.IR_ADD,
    constraints={},
)

CAST = OpDefinition(
    name=OnnxOp.CAST,
    onnx=OnnxDef.CAST,
    ir=_ir_placeholder("cast", IRDef.F_X, to=IRDef.HOLE),
    constraints={},
)

CONCAT = OpDefinition(
    name=OnnxOp.CONCAT,
    onnx=OnnxDef.CONCAT,
    ir=_ir_placeholder("concat", IRDef.F_X, IRDef.F_Y, axis=IRDef.HOLE),
    constraints={},
)

CONSTANT_OF_SHAPE = OpDefinition(
    name=OnnxOp.CONSTANT_OF_SHAPE,
    onnx=OnnxDef.CONSTANT_OF_SHAPE,
    ir=IRDef.IR_F_FULL,
    constraints={},
)

DIV = OpDefinition(
    name=OnnxOp.DIV,
    onnx=OnnxDef.DIV,
    ir=IRDef.IR_DIVIDE,
    constraints={},
)

EQUAL = OpDefinition(
    name=OnnxOp.EQUAL,
    onnx=OnnxDef.EQUAL,
    ir=IRDef.ir("equal", IRDef.BOOL_SORT, IRDef.F_X, IRDef.F_Y),
    constraints={},
)

EXPAND = OpDefinition(
    name=OnnxOp.EXPAND,
    onnx=OnnxDef.EXPAND,
    ir=_ir_placeholder("expand", IRDef.F_X, shape=IRDef.HOLE),
    constraints={},
)

GATHER = OpDefinition(
    name=OnnxOp.GATHER,
    onnx=OnnxDef.GATHER,
    ir=_ir_placeholder("gather", IRDef.F_X, indices=IRDef.HOLE, axis=IRDef.HOLE),
    constraints={},
)

LESS = OpDefinition(
    name=OnnxOp.LESS,
    onnx=OnnxDef.LESS,
    ir=IRDef.IR_LESS,
    constraints={},
)

MAT_MUL = OpDefinition(
    name=OnnxOp.MAT_MUL,
    onnx=OnnxDef.MAT_MUL,
    ir=IRDef.IR_DOT,
    constraints={"todo": "batched MatMul should lower to tensordot with batch semantics"},
)

MUL = OpDefinition(
    name=OnnxOp.MUL,
    onnx=OnnxDef.MUL,
    ir=IRDef.IR_MULTIPLY,
    constraints={},
)

NEG = OpDefinition(
    name=OnnxOp.NEG,
    onnx=OnnxDef.NEG,
    ir=_ir_placeholder("neg", IRDef.F_X),
    constraints={},
)

POW = OpDefinition(
    name=OnnxOp.POW,
    onnx=OnnxDef.POW,
    ir=IRDef.IR_POWER,
    constraints={},
)

RANGE = OpDefinition(
    name=OnnxOp.RANGE,
    onnx=OnnxDef.RANGE,
    ir=_ir_placeholder("range", start=IRDef.HOLE, limit=IRDef.HOLE, delta=IRDef.HOLE),
    constraints={},
)

REDUCE_MEAN = OpDefinition(
    name=OnnxOp.REDUCE_MEAN,
    onnx=OnnxDef.REDUCE_MEAN,
    ir=_ir_placeholder("mean", IRDef.F_X, axes=IRDef.HOLE, keepdims=IRDef.HOLE),
    constraints={},
)

RESHAPE = OpDefinition(
    name=OnnxOp.RESHAPE,
    onnx=OnnxDef.RESHAPE,
    ir=_ir_placeholder("reshape", IRDef.F_X, shape=IRDef.HOLE),
    constraints={},
)

SHAPE = OpDefinition(
    name=OnnxOp.SHAPE,
    onnx=OnnxDef.SHAPE,
    ir=_ir_placeholder("shape", IRDef.F_X),
    constraints={},
)

SIGMOID = OpDefinition(
    name=OnnxOp.SIGMOID,
    onnx=OnnxDef.SIGMOID,
    ir=_ir_placeholder("sigmoid", IRDef.F_X),
    constraints={},
)

SLICE = OpDefinition(
    name=OnnxOp.SLICE,
    onnx=OnnxDef.SLICE,
    ir=_ir_placeholder("slice", IRDef.F_X, starts=IRDef.HOLE, ends=IRDef.HOLE, axes=IRDef.HOLE),
    constraints={},
)

SOFTMAX = OpDefinition(
    name=OnnxOp.SOFTMAX,
    onnx=OnnxDef.SOFTMAX,
    ir=_ir_placeholder("softmax", IRDef.F_X, axis=IRDef.HOLE),
    constraints={},
)

SQRT = OpDefinition(
    name=OnnxOp.SQRT,
    onnx=OnnxDef.SQRT,
    ir=IRDef.IR_SQRT,
    constraints={},
)

SQUEEZE = OpDefinition(
    name=OnnxOp.SQUEEZE,
    onnx=OnnxDef.SQUEEZE,
    ir=_ir_placeholder("squeeze", IRDef.F_X, axes=IRDef.HOLE),
    constraints={},
)

SUB = OpDefinition(
    name=OnnxOp.SUB,
    onnx=OnnxDef.SUB,
    ir=IRDef.IR_SUBTRACT,
    constraints={},
)

TRANSPOSE = OpDefinition(
    name=OnnxOp.TRANSPOSE,
    onnx=OnnxDef.TRANSPOSE,
    ir=IRDef.IR_TRANSPOSE,
    constraints={},
)

UNSQUEEZE = OpDefinition(
    name=OnnxOp.UNSQUEEZE,
    onnx=OnnxDef.UNSQUEEZE,
    ir=_ir_placeholder("unsqueeze", IRDef.F_X, axes=IRDef.HOLE),
    constraints={},
)

WHERE = OpDefinition(
    name=OnnxOp.WHERE,
    onnx=OnnxDef.WHERE,
    ir=IRDef.IR_WHERE,
    constraints={},
)

TINYLLAMA_DEFINITIONS = (
    ADD,
    CAST,
    CONCAT,
    CONSTANT_OF_SHAPE,
    DIV,
    EQUAL,
    EXPAND,
    GATHER,
    LESS,
    MAT_MUL,
    MUL,
    NEG,
    POW,
    RANGE,
    REDUCE_MEAN,
    RESHAPE,
    SHAPE,
    SIGMOID,
    SLICE,
    SOFTMAX,
    SQRT,
    SQUEEZE,
    SUB,
    TRANSPOSE,
    UNSQUEEZE,
    WHERE,
)
