from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

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
    attrs: dict[str, Any] = dict() 

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
    attrs: dict[str, Any] = dict() 

@dataclass
class IRTensorConst(IRTerm):
    name: str 
    value: np.ndarray | None 
    sort: IRSort
    attrs: dict[str, Any] = dict() 


@dataclass
class IRExpr(IRTerm):
    op_name: str
    sort: IRSort
    children: list[IRTerm]
    attrs: dict[str, Any] = dict()  


# note that, we have to define what attrs would be in our ir.. 
def ir(op_name: str, sort: IRSort, *children: IRTerm, **attrs: Any) -> IRExpr:
    return IRExpr(op_name, sort, list(children), attrs or dict())


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