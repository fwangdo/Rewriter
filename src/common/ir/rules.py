"""Bidirectional definitions between ONNX ops and the NumPy-like IR.

This module is intentionally declarative.  It describes symbolic expression
patterns; lowering/lifting code should decide which direction to use them in.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing      import Any 

from .ir_def import *
from .onnx_def import *

import src.common.spec.constant as OnnxOp
import src.common.ir.ir_def as IRDef


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


# --- our onnx IR for lowering and lifting.
ABS = OpDefinition(

)
