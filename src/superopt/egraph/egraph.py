"""E-graph data structure.

Implements union-find based e-class management, e-node deduplication,
and the rebuild operation that restores invariants after merges.

Reference: egg (Willsey et al., 2020).
"""

from __future__ import annotations

from collections import deque

from .eclass import AnalysisData, EClass
from .enode import EClassId, ENode, ENodeId


class EGraph:
    """An e-graph over tensor computation nodes."""

    def __init__(self) -> None:
        # union-find parent map (canonical id → itself)
        self._parent: dict[EClassId, EClassId] = {} # M 
        # e-class storage
        self._classes: dict[EClassId, EClass] = {} # U. 
        # all e-nodes, keyed by id
        self._nodes: dict[ENodeId, ENode] = {} # mapping. 
        # dedup memo: canonical ENode → e-class id
        self._memo: dict[ENode, EClassId] = {}
        # reverse map: enode id → owning e-class id
        self._node_to_class: dict[ENodeId, EClassId] = {}
        # counters
        self._next_class_id: int = 0
        self._next_node_id: int = 0
        self._version: int = 0
        # pending merges for rebuild
        self._pending: list[EClassId] = [] # worklist. 
        # initializer data: weight node name → numpy array
        self.initializers: dict[str, object] = {}

    # --- public API ---

    def add(self, enode: ENode) -> EClassId:
        """Add an e-node to the e-graph.

        If an identical e-node already exists, return its e-class.
        Otherwise create a new e-class containing this e-node.
        """
        # the most important part in e-graph. 
        canon = enode.canonicalize({cid: self.find(cid) for cid in enode.children})
        if canon in self._memo:
            return self.find(self._memo[canon])

        # allocate new e-class
        cid = self._next_class_id
        self._next_class_id += 1
        self._parent[cid] = cid 
        ec = EClass(id=cid)
        self._classes[cid] = ec # U. 

        # allocate new e-node
        nid = self._next_node_id
        self._next_node_id += 1
        self._nodes[nid] = canon
        ec.nodes.add(nid)
        self._memo[canon] = cid
        self._node_to_class[nid] = cid 

        # Register parent links: this enode is a parent of each child eclass
        for child_cid in canon.children:
            self._classes[self.find(child_cid)].parents.add(nid)

        self._version += 1
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
        c1.parents |= c2.parents
        c1.data = AnalysisData.join(c1.data, c2.data)
        # Update _node_to_class for merged nodes
        for nid in c2.nodes:
            self._node_to_class[nid] = id1
        self._pending.append(id1)
        self._version += 1
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

        Implements egg's rebuild algorithm: re-canonicalize parent enodes,
        detect new congruences, and re-propagate analysis data upward.
        """
        worklist = deque(self._pending)
        self._pending.clear()

        while worklist:
            cid = self.find(worklist.popleft())
            self._repair(cid, worklist)
        return 

    def _repair(self, cid: EClassId, worklist: deque) -> None:
        # upward merge.
        ec = self._classes[cid]

        # Phase 1: Re-canonicalize parent enodes, update memo, detect congruences
        old_parents = list(ec.parents)
        new_parents: dict[ENode, tuple[ENodeId, EClassId]] = {}

        for nid in old_parents:
            old_enode = self._nodes[nid]
            new_enode = old_enode.canonicalize(
                {c: self.find(c) for c in old_enode.children}
            )
            self._nodes[nid] = new_enode

            # Remove stale memo entry
            self._memo.pop(old_enode, None)

            parent_cid = self.find(self._node_to_class[nid])

            if new_enode in new_parents:
                # Congruence: two parent enodes now have the same canonical form
                existing_nid, existing_cid = new_parents[new_enode]
                merged = self.merge(existing_cid, parent_cid)
                new_parents[new_enode] = (existing_nid, merged)
                worklist.append(merged)
            else:
                new_parents[new_enode] = (nid, self.find(parent_cid))

            self._memo[new_enode] = self.find(new_parents[new_enode][1])

        # Rebuild the parents set with surviving nids
        ec.parents = {nid for _, (nid, _) in new_parents.items()}

        # Phase 2: Re-compute analysis (make + join)
        # Shape inference is approximate, so skip update on conflicts.
        from .analysis import compute_analysis

        new_data = AnalysisData()
        try:
            for nid in ec.nodes:
                enode = self._nodes[nid]
                new_data = AnalysisData.join(new_data, compute_analysis(self, enode))
        except ValueError:
            return
        if new_data != ec.data:
            ec.data = new_data
            # Propagate upward: parents of this eclass need re-check
            for nid in ec.parents:
                p_cid = self.find(self._node_to_class[nid])
                worklist.append(p_cid)
        
        return 

    # --- query ---

    def __len__(self) -> int:
        """Number of canonical e-classes."""
        return sum(1 for c in self._parent if self._parent[c] == c)

    @property
    def num_enodes(self) -> int:
        return len(self._nodes)

    @property
    def version(self) -> int:
        """Monotonic counter bumped when add/merge changes e-graph structure."""
        return self._version

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

    def update_analysis(self, cid: EClassId, data: AnalysisData) -> None:
        """Join analysis data into an e-class without discarding prior facts."""
        ec = self.eclass(cid)
        ec.data = AnalysisData.join(ec.data, data)

    def show_format(self, cid: EClassId) -> ENode: 
        elem = self.eclass_nodes(cid)[0]
        return elem
