"""SIR graph: a DAG of SirNodes with linalg-like tensor operations.

The IR is value-oriented: each node represents one tensor value.
All nodes use a single SirNode type (linalg.generic-like).
Leaf nodes (input/weight) and structural nodes (noop/proj) simply
have empty iterators/indexing_maps/body.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Affine index expression: coeff_1*var_1 + coeff_2*var_2 + ... + offset
# e.g.  i        = AffineExpr(((1, "i"),))
#       oh + kh  = AffineExpr(((1, "oh"), (1, "kh")))
#       2*i + 1  = AffineExpr(((2, "i"),), offset=1)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AffineExpr:
    terms: tuple[tuple[int, str], ...]   # ((coeff, var_name), ...)
    offset: int = 0


def dim(name: str) -> AffineExpr:
    """Shorthand: single iterator variable with coefficient 1."""
    return AffineExpr(terms=((1, name),))


def dim_add(*names: str) -> AffineExpr:
    """Shorthand: sum of iterator variables.  e.g. dim_add("oh", "kh")."""
    return AffineExpr(terms=tuple((1, n) for n in names))


# ---------------------------------------------------------------------------
# Linalg-like components
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SirIterator:
    """One dimension of the iteration domain.

    name:  loop variable name ("i", "k", ...)
    bound: symbolic range   ("M", "K", ...)
    kind:  "parallel" or "reduction"
    """
    name: str
    bound: str
    kind: str       # "parallel" | "reduction"


@dataclass(frozen=True)
class SirIndexMap:
    """How one tensor is accessed within the iteration domain.

    tensor:  tensor id (must match an input id or the output id)
    indices: one AffineExpr per tensor dimension
    """
    tensor: str
    indices: tuple[AffineExpr, ...]


@dataclass(frozen=True)
class SirScalarOp:
    """One scalar operation inside the body.

    The body receives pre-loaded scalar values (%0, %1, ... for inputs
    in order) and yields a result.

    id:     SSA-style value name ("%m", "%out", ...)
    op:     scalar operation ("mul", "add", "yield", ...)
    inputs: references to other SirScalarOp ids or block args ("%0", "%1")
    attrs:  e.g. (("combine", "sum"),) on yield
    """
    id: str
    op: str
    inputs: tuple[str, ...]
    attrs: tuple[tuple[str, Any], ...] = ()


# ---------------------------------------------------------------------------
# Common body patterns
# ---------------------------------------------------------------------------

# matmul / conv body: mul two inputs, accumulate with sum
BODY_MUL_SUM = (
    SirScalarOp("%m", "mul", ("%0", "%1")),
    SirScalarOp("%out", "yield", ("%m",), attrs=(("combine", "sum"),)),
)


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SirNode:
    """Unified node type for the SIR graph.

    For compute ops (matmul, conv, elementwise, ...):
        iterators, indexing_maps, body describe the computation.
    For leaf/structural ops (input, weight, noop, proj):
        iterators, indexing_maps, body are empty tuples.
    """
    id: str
    op: str
    inputs: tuple[str, ...]
    attrs: tuple[tuple[str, Any], ...] = ()
    shape: tuple[int, ...] | None = None
    dtype: int | None = None
    iterators: tuple[SirIterator, ...] = ()
    indexing_maps: tuple[SirIndexMap, ...] = ()
    body: tuple[SirScalarOp, ...] = ()

    @property
    def attrs_dict(self) -> dict[str, Any]:
        return dict(self.attrs)


# ---------------------------------------------------------------------------
# Sentinel ops
# ---------------------------------------------------------------------------

OP_INPUT = "input"
OP_WEIGHT = "weight"
OP_NOOP = "noop"
OP_PROJ = "proj"


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

@dataclass
class SirGraph:
    """A directed acyclic graph of SirNodes.

    Invariants:
    - Every node id is unique.
    - ``root`` is the single output node (usually a noop combining outputs).
    - ``initializers`` maps weight tensor ids to numpy arrays.
    """

    nodes: dict[str, SirNode] = field(default_factory=dict)
    root: str | None = None
    initializers: dict[str, np.ndarray] = field(default_factory=dict)
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    opset_version: int = 18

    # --- mutation ---

    def add_node(self, node: SirNode) -> None:
        if node.id in self.nodes:
            raise ValueError(f"duplicate node id: {node.id}")
        self.nodes[node.id] = node

    def add_initializer(self, name: str, value: np.ndarray) -> None:
        self.initializers[name] = value

    # --- queries ---

    def topo_order(self) -> list[str]:
        """Node ids in topological order (inputs first)."""
        visited: set[str] = set()
        order: list[str] = []

        def _visit(nid: str) -> None:
            if nid in visited or nid not in self.nodes:
                return
            visited.add(nid)
            for inp in self.nodes[nid].inputs:
                _visit(inp)
            order.append(nid)

        for nid in self.nodes:
            _visit(nid)
        return order

    def output_ids(self) -> tuple[str, ...]:
        """Declared graph outputs, falling back to root's inputs."""
        if self.outputs:
            return self.outputs
        if self.root is None or self.root not in self.nodes:
            return ()
        return self.nodes[self.root].inputs

    # --- debug ---

    def __repr__(self) -> str:
        if self.root is None:
            return "SirGraph(empty)"

        visited: dict[str, str] = {}

        def dfs(node_id: str) -> str:
            if node_id in visited:
                return visited[node_id]

            if node_id in self.initializers:
                res = f"(Const {node_id})"
            elif node_id in self.inputs:
                res = f"(Input {node_id})"
            elif node_id not in self.nodes:
                res = f"(?{node_id})"
            else:
                node = self.nodes[node_id]
                if not node.inputs:
                    res = f"({node.op})"
                else:
                    args = " ".join(dfs(inp) for inp in node.inputs)
                    res = f"({node.op} {args})"

            visited[node_id] = res
            return res

        return dfs(self.root)

    def show_nodes(self) -> None:
        for nid, node in self.nodes.items():
            print(f"{nid} -> {node}")
