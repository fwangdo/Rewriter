"""definition of onnx operations which consists of ir"""
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class OnnxVar:
    name: str

@dataclass
class OnnxExpr:
    op_name: str 
    children: list[OnnxVar | OnnxExpr]

@dataclass
class IRVar:
    name: str

@dataclass
class IRExpr:
    op_name: str 
    children: list[IRVar | IRExpr]

@dataclass
class OpDefinition:
    name: str
    onnx: OnnxExpr
    ir: IRExpr 
    constraints: dict[str, int] 


# definitions of IR.  