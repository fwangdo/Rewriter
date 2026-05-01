"""E-graph: the core equality saturation data structure.

Implements union-find based e-class management, e-node deduplication,
and the rebuild operation that restores invariants after merges.

Reference: egg (Willsey et al., 2020).
"""

from __future__ import annotations

from .eclass import AnalysisData, EClass
from .enode import EClassId, ENode, ENodeId


class EGraph:
    """An e-graph over tensor computation nodes."""

    def __init__(self) -> None:
        # union-find parent map (canonical id → itself)
        self._parent: dict[EClassId, EClassId] = {}
        # e-class storage
        self._classes: dict[EClassId, EClass] = {}
        # all e-nodes, keyed by id
        self._nodes: dict[ENodeId, ENode] = {}
        # dedup memo: canonical ENode → e-class id
        self._memo: dict[ENode, EClassId] = {}
        # counters
        self._next_class_id: int = 0
        self._next_node_id: int = 0
        # pending merges for rebuild
        self._pending: list[tuple[EClassId, EClassId]] = []

    # --- public API ---

    def add(self, enode: ENode) -> EClassId:
        """Add an e-node to the e-graph.

        If an identical e-node already exists, return its e-class.
        Otherwise create a new e-class containing this e-node.
        """
        canon = enode.canonicalize(
            {cid: self.find(cid) for cid in enode.children}
        )
        if canon in self._memo:
            return self.find(self._memo[canon])

        # allocate new e-class
        cid = self._next_class_id
        self._next_class_id += 1
        self._parent[cid] = cid
        ec = EClass(id=cid)
        self._classes[cid] = ec

        # allocate new e-node
        nid = self._next_node_id
        self._next_node_id += 1
        self._nodes[nid] = canon
        ec.nodes.add(nid)
        self._memo[canon] = cid

        return cid

    def merge(self, id1: EClassId, id2: EClassId) -> EClassId:
        """Merge two e-classes, returning the canonical id."""
        id1 = self.find(id1)
        id2 = self.find(id2)
        if id1 == id2:
            return id1
        # merge smaller into larger (union by size)
        c1, c2 = self._classes[id1], self._classes[id2]
        if len(c1.nodes) < len(c2.nodes):
            id1, id2 = id2, id1
            c1, c2 = c2, c1
        self._parent[id2] = id1
        c1.nodes |= c2.nodes
        c1.data = AnalysisData.join(c1.data, c2.data)
        self._pending.append((id1, id2))
        return id1

    def find(self, cid: EClassId) -> EClassId:
        """Find the canonical e-class id (with path compression)."""
        root = cid
        while self._parent[root] != root:
            root = self._parent[root]
        # path compression
        while self._parent[cid] != root:
            self._parent[cid], cid = root, self._parent[cid]
        return root

    def rebuild(self) -> None:
        """Restore e-graph invariants after merges.

        Re-canonicalizes all e-nodes whose children's canonical ids
        may have changed, and updates the memo table.
        """
        while self._pending:
            _, merged_id = self._pending.pop()
            merged_class = self._classes.get(merged_id)
            if merged_class is None:
                continue
            for nid in merged_class.nodes:
                old = self._nodes[nid]
                new = old.canonicalize(
                    {c: self.find(c) for c in old.children}
                )
                self._nodes[nid] = new
                if new in self._memo:
                    self.merge(self._memo[new], self.find(merged_id))
                else:
                    self._memo[new] = self.find(merged_id)

    # --- query ---

    def __len__(self) -> int:
        """Number of canonical e-classes."""
        return sum(1 for c in self._parent if self._parent[c] == c)

    @property
    def num_enodes(self) -> int:
        return len(self._nodes)

    def eclass(self, cid: EClassId) -> EClass:
        return self._classes[self.find(cid)]

    def enode(self, nid: ENodeId) -> ENode:
        return self._nodes[nid]

    def eclass_nodes(self, cid: EClassId) -> list[ENode]:
        """Return all e-nodes in the given e-class."""
        ec = self.eclass(cid)
        return [self._nodes[nid] for nid in ec.nodes]

    def canonical_class_ids(self) -> list[EClassId]:
        """Return all canonical e-class ids."""
        return [cid for cid, parent in self._parent.items() if cid == parent]

    def set_analysis(self, cid: EClassId, data: AnalysisData) -> None:
        """Overwrite analysis data for an e-class."""
        self.eclass(cid).data = data
