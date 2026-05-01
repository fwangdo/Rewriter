"""IR graph: a DAG of IRNodes with initializers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .node import IRNode


@dataclass
class IRGraph:
    """A directed acyclic graph of IR nodes.

    Invariants
    ----------
    - Every node id is unique.
    - ``root`` is the single output node (usually a noop that
      combines all graph outputs).
    - ``initializers`` maps weight tensor ids to numpy arrays.
    """

    nodes: dict[str, IRNode] = field(default_factory=dict)
    root: str | None = None
    initializers: dict[str, np.ndarray] = field(default_factory=dict)
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()

    # --- mutation helpers ---

    def add_node(self, node: IRNode) -> None:
        if node.id in self.nodes:
            raise ValueError(f"duplicate node id: {node.id}")
        self.nodes[node.id] = node

    def add_initializer(self, name: str, value: np.ndarray) -> None:
        self.initializers[name] = value

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
