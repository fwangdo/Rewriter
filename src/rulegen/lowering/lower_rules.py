"""lowering from onnx to small ir"""
from __future__ import annotations

import src.common.spec.constant as Cons 
from src.rulegen.sir.sir_graph import SirGraph, SirNode

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# primitive

# matmul
def lower_matmul(graph: SirGraph, node: SirNode) -> SirNode:
    return node

# gemm


# bn 


# gather


def lower(graph: SirGraph, node: SirNode) -> SirNode:
    """lowering and enrollment.  
    Note that, all generated nodes in this procedure should be enrolled in sir. 
    """
    res = node
    match node.op:
        case Cons.MAT_MUL: 
            pass
            # res = 
        case Cons.GEMM:
            pass
        case Cons.BATCH_NORMALIZATION:
            pass
        case Cons.GATHER:
            pass 
        case _:
            res = node  
    
    graph.add_node(node)
    return res  
