"""Cost model for e-graph extraction.

Assigns a cost to each e-node and checks legality against a
supported-op contract.
"""

from __future__ import annotations

from ..egraph.enode import ENode
from ..ir.node import OP_INPUT, OP_NOOP, OP_PROJ, OP_WEIGHT

_FREE_OPS = frozenset({OP_INPUT, OP_WEIGHT, OP_NOOP, OP_PROJ})


class CostModel:
    """Uniform cost model with legality filter.

    Boundary/projection ops have zero cost.
    All other ops have cost 1.0.
    An e-node is legal if its op is in ``supported_ops`` or is a boundary op.
    """

    def __init__(self, supported_ops: frozenset[str]) -> None:
        self.supported_ops = supported_ops

    def node_cost(self, enode: ENode) -> float:
        if enode.op in _FREE_OPS:
            return 0.0
        return 1.0

    def is_legal(self, enode: ENode) -> bool:
        if enode.op in _FREE_OPS:
            return True
        return enode.op in self.supported_ops
