""" Our ir which is prirmitive operation in our project's scope
add, mul, sub, div, neg 
equal, where, reduce
reshape, transpose, resize, slice, split
cast, clip, concat, expand, where
conv2d
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# import numpy as np
# import src.rulegen.sir.ir_spec as IROp

# --- sort. 
class IRSort:
    """IR sort class"""

@dataclass
class FPSort(IRSort):
    """FP """


@dataclass
class BoolSort(IRSort):
    """BoolSort """


@dataclass
class SymbolSort(IRSort):
    symbol: str  


# --- term. 
@dataclass
class SirTerm:
    """Base class for IR pattern terms."""

@dataclass
class SirLeaf(SirTerm):
    sort: IRSort 
    attrs: dict[str, Any] = field(default_factory=dict) # if it is constant, it would be represented asd "value." 


@dataclass
class SirExpr(SirTerm):
    op_name: str
    sort: IRSort
    children: list[SirTerm]
    attrs: dict[str, Any] = field(default_factory=dict) 


# --- NumPy-like IR vocabulary patterns.
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
