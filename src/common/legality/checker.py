# legality checker. 
from __future__ import annotations
from dataclasses import dataclass

from src.common.spec.qnn import QNN_SUPPORTED_OPS, QNN_LIMITED_OPS
from src.common.spec.constant import *

import onnx 

from enum import Enum 


@dataclass(unsafe_hash=True)
class LegalityResult:
    legal_ops: dict[str, int]
    illegal_ops: dict[str, int]
    limited_ops: dict[str, int]


class LegalitySignal(Enum):
    LEGAL = "Legal"
    ILLEGAL = "Illegal"
    LIMITED = "Limited"

# rules. 
def _check_gather(node: onnx.NodeProto) -> LegalitySignal:
    return 


def _check_lp_norm(node: onnx.NodeProto) -> LegalitySignal:
    return 


def _handle_limited_ops(node: onnx.NodeProto) -> LegalitySignal:
    op_type = node.op_type
    if op_type == GATHER: 
        return _check_gather(node)
    elif op_type == LP_NORMALIZATION:
        return _check_lp_norm(node)
    else:
        raise Exception(f'[ERROR]: {op_type} is considered as limited operation. ')


def check_legality(node: onnx.NodeProto) -> LegalitySignal:
    # return True when it is legal if not so, 
    op_name = node.op_type

    if op_name in QNN_LIMITED_OPS:
        return _handle_limited_ops(node) 
    elif op_name in QNN_SUPPORTED_OPS:
        return LegalitySignal.LEGAL 
    else:
        return LegalitySignal.ILLEGAL 


def _organize_res(res: LegalitySignal, node: onnx.NodeProto, lr) -> None:
    return 
    

def run(model: onnx.ModelProto) -> LegalityResult:
    lr = LegalityResult({}, {}, {})
    for node in model.graph.node:
        res = check_legality(node)
        _organize_res(res, node, lr)
        
    return lr  