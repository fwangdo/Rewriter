from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import src.common.ir.operation as IROp

# --- sort. 
class IRSort:
    """IR sort class"""

@dataclass
class FPSort(IRSort):
    """FP """


@dataclass
class BoolSort(IRSort):
    """BoolSort """


# --- term. 
@dataclass
class IRTerm:
    """Base class for IR pattern terms."""

@dataclass
class IRVar(IRTerm):
    name: str
    sort: IRSort 
    attrs: dict[str, Any] = field(default_factory=dict) 

@dataclass
class IRConst(IRTerm): # for concrete value in IR. 
    """const template"""
    # value: float | int | bool | np.ndarray 
    # sort: IRSort 
    # attrs: dict[str, Any] = dict() 

@dataclass
class IRScalarConst(IRTerm):
    name: str
    value: bool | int | float | None
    sort: IRSort
    attrs: dict[str, Any] = field(default_factory=dict) 

@dataclass
class IRTensorConst(IRTerm):
    name: str 
    value: np.ndarray | None 
    sort: IRSort
    attrs: dict[str, Any] = field(default_factory=dict) 


@dataclass
class IRExpr(IRTerm):
    op_name: str
    sort: IRSort
    children: list[IRTerm]
    attrs: dict[str, Any] = field(default_factory=dict) 


# note that, we have to define what attrs would be in our ir.. 
def ir(op_name: str, sort: IRSort, *children: IRTerm, **attrs: Any) -> IRExpr:
    temp = dict()
    return IRExpr(op_name, sort, list(children), attrs or temp)

# --- NumPy-like IR vocabulary patterns.
HOLE = "hole" # signal to fill out. 

FP_SORT   = FPSort()
BOOL_SORT = BoolSort()

# atom  
F_X = IRVar("x", FP_SORT)
F_Y = IRVar("y", FP_SORT)
F_SCALAR_CONST = IRScalarConst("fp_scalar", None, FP_SORT)
F_TENSOR_CONST = IRTensorConst("fp_tensor", None, FP_SORT) 

B = IRVar("b", BOOL_SORT)
B_SCALAR_CONST = IRScalarConst("b_scalar", None, BOOL_SORT)
B_TENSOR_CONST = IRTensorConst("b_tensor", None, BOOL_SORT)

IR_F_FULL = ir(IROp.FULL, FP_SORT, shape=HOLE, value=HOLE)
IR_F_TRIU = ir(IROp.TRIU, FP_SORT, F_X)
IR_F_TRIL = ir(IROp.TRIL, FP_SORT, F_X)
IR_SUM = ir(IROp.SUM, FP_SORT, F_X, axis=HOLE)
IR_TRANSPOSE = ir(IROp.TRANSPOSE, FP_SORT, F_X, axes=HOLE)
IR_SQRT = ir(IROp.SQRT, FP_SORT, F_X)
IR_ADD = ir(IROp.ADD, FP_SORT, F_X, F_Y)
IR_SUBTRACT = ir(IROp.SUBTRACT, FP_SORT, F_X, F_Y)
IR_MULTIPLY = ir(IROp.MULTIPLY, FP_SORT, F_X, F_Y)
IR_DIVIDE = ir(IROp.DIVIDE, FP_SORT, F_X, F_Y)
IR_POWER = ir(IROp.POWER, FP_SORT, F_X, F_Y)
IR_DOT = ir(IROp.DOT, FP_SORT, F_X, F_Y)
IR_TENSORDOT = ir(IROp.TENSOR_DOT, FP_SORT, F_X, F_Y, lhs_axes=HOLE, rhs_axes=HOLE)
IR_WHERE = ir(IROp.WHERE, FP_SORT, B, F_X, F_Y)

IR_B_FULL = ir(IROp.FULL, BOOL_SORT, shape=HOLE, value=HOLE)
IR_B_TRIU = ir(IROp.TRIU, BOOL_SORT, B)
IR_B_TRIL = ir(IROp.TRIL, BOOL_SORT, B)
IR_LESS = ir(IROp.LESS, BOOL_SORT, F_X, F_Y)


# FLOAT   = 1   # float32
# UINT8   = 2
# INT8    = 3
# UINT16  = 4
# INT16   = 5
# INT32   = 6
# INT64   = 7
# BOOL    = 9
# FLOAT16 = 10
# DOUBLE  = 11
# UINT32  = 12
# UINT64  = 13
# BFLOAT16 = 16
# dtype: int # same as onnx 