"""IR graph: a DAG of IRNodes with initializers.

The IR is value-oriented: an ordinary SIRNode represents one tensor value
and the operation that produces it. For example, ``IRNode(id="z", op="Add",
inputs=("x", "y"))`` means tensor value ``z`` is produced by ``Add(x, y)``.

Multi-output ONNX ops are the exception. They are represented as one base
operation node plus one ``proj`` node per output value. The base node is an
operation invocation handle; the ``proj`` nodes are the tensor values that
downstream IR nodes consume.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from typing import Any


@dataclass(frozen=True)
class SirNode:
    """A value node in the IR graph.

    For ordinary ops, ``id`` is the produced tensor value name and ``op`` is
    the producer operation. ``inputs`` are references to other value ids.
    """

    id: str
    op: str
    inputs: tuple[str, ...]
    attrs: tuple[tuple[str, Any], ...] = ()
    shape: tuple[int, ...] | None = None
    dtype: int | None = None  # onnx.TensorProto.DataType

    @property
    def attrs_dict(self) -> dict[str, Any]:
        return dict(self.attrs)


# Sentinel ops for graph boundary and multi-output handling.
OP_INPUT = "input"      # graph input placeholder
OP_WEIGHT = "weight"    # initializer (constant weight) leaf
OP_NOOP = "noop"        # combines multiple graph outputs into one root
OP_PROJ = "proj"        # projection for multi-output ops: proj_0, proj_1, ...


@dataclass
class SirGraph:
    """A directed acyclic graph of IR nodes.

    Invariants
    ----------
    - Every node id is unique.
    - ``root`` is the single output node (usually a noop that
      combines all graph outputs).
    - ``initializers`` maps weight tensor ids to numpy arrays.
    """

    nodes: dict[str, SirNode] = field(default_factory=dict)
    root: str | None = None
    initializers: dict[str, np.ndarray] = field(default_factory=dict)
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    opset_version: int = 18

    # for debugging. 
    def __repr__(self) -> str:
        if self.root is None:
            return "IRGraph(empty)"

        visited = {}  # memo: node_id -> sexpr string

        def dfs(node_id: str) -> str:
            # 이미 계산된 노드는 재사용 (DAG 공유 처리)
            if node_id in visited:
                return visited[node_id]

            # initializer (constant)
            if node_id in self.initializers:
                res = f"(Const {node_id})"
                visited[node_id] = res
                return res

            # input node
            if node_id in self.inputs:
                res = f"(Input {node_id})"
                visited[node_id] = res
                return res

            node = self.nodes[node_id]

            # leaf (no inputs)
            if not node.inputs:
                res = f"({node.op})"
                visited[node_id] = res
                return res

            # 일반 케이스
            args = " ".join(dfs(inp) for inp in node.inputs)
            res = f"({node.op} {args})"

            visited[node_id] = res
            return res

        return dfs(self.root)

    # --- mutation helpers ---

    def add_node(self, node: SirNode) -> None:
        if node.id in self.nodes:
            raise ValueError(f"duplicate node id: {node.id}")
        self.nodes[node.id] = node
        return 

    def add_initializer(self, name: str, value: np.ndarray) -> None:
        # already fixed weight value in onnx.  
        self.initializers[name] = value
        return  

    # --- query helpers ---

    def topo_order(self) -> list[str]:
        """Return node ids in topological order (inputs first)."""
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
        """Return declared graph outputs, falling back to root children."""
        if self.outputs:
            return self.outputs
        if self.root is None or self.root not in self.nodes:
            return ()
        return self.nodes[self.root].inputs


    # --- utils 
    def show_nodes(self) -> None:
        for idx, node in self.nodes.items():
            print(f'{idx} -> {node}')
        return 