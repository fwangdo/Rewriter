"""Greedy bottom-up extraction from an e-graph.

Selects the lowest-cost legal e-node per e-class to reconstruct an IRGraph.
"""

from __future__ import annotations

from ..egraph.egraph import EGraph
from ..egraph.enode import EClassId, ENode
from ..ir.graph import IRGraph
from ..ir.node import IRNode, OP_INPUT, OP_NOOP, OP_WEIGHT
from .cost import CostModel

_LEAF_OPS = frozenset({OP_INPUT, OP_WEIGHT, OP_NOOP})


def extract_greedy(
    egraph: EGraph,
    root_cid: EClassId,
    cost_model: CostModel,
) -> IRGraph:
    """Extract the lowest-cost program from *egraph* rooted at *root_cid*.

    Algorithm (bottom-up):
    1. Topologically sort e-classes by dependency.
    2. For each e-class pick the best (legal, lowest-cost) e-node.
    3. If no legal e-node exists, fall back to the lowest-cost one.
    4. Build an IRGraph from chosen e-nodes.
    """
    # Map each canonical e-class id → chosen e-node.
    best: dict[EClassId, tuple[float, ENode]] = {}

    # Collect all reachable e-class ids from root via BFS.
    reachable: list[EClassId] = []
    visited: set[EClassId] = set()
    stack = [egraph.find(root_cid)]
    while stack:
        cid = egraph.find(stack.pop())
        if cid in visited:
            continue
        visited.add(cid)
        reachable.append(cid)
        for enode in egraph.eclass_nodes(cid):
            for child in enode.children:
                child_canon = egraph.find(child)
                if child_canon not in visited:
                    stack.append(child_canon)

    # Process in reverse (leaves first) for bottom-up cost propagation.
    reachable.reverse()

    for cid in reachable:
        enodes = egraph.eclass_nodes(cid)
        best_legal: tuple[float, ENode] | None = None
        best_any: tuple[float, ENode] | None = None

        for enode in enodes:
            # Total cost = node cost + sum of children costs.
            child_cost = sum(
                best[egraph.find(c)][0] for c in enode.children if egraph.find(c) in best
            )
            total = cost_model.node_cost(enode) + child_cost

            if best_any is None or total < best_any[0]:
                best_any = (total, enode)
            if cost_model.is_legal(enode):
                if best_legal is None or total < best_legal[0]:
                    best_legal = (total, enode)

        best[cid] = best_legal if best_legal is not None else best_any  # type: ignore[assignment]

    # Reconstruct IRGraph from chosen e-nodes.
    ir = IRGraph()
    built: set[EClassId] = set()

    def _build(cid: EClassId) -> str:
        cid = egraph.find(cid)
        if cid in built:
            return _node_id(cid)
        built.add(cid)

        _, enode = best[cid]
        # Build children first.
        child_ids = tuple(_build(egraph.find(c)) for c in enode.children)

        nid = _node_id(cid)
        ec = egraph.eclass(cid)
        ir.add_node(IRNode(
            id=nid,
            op=enode.op,
            inputs=child_ids,
            attrs=enode.attrs,
            shape=ec.data.shape,
            dtype=ec.data.dtype,
        ))
        return nid

    def _node_id(cid: EClassId) -> str:
        ec = egraph.eclass(cid)
        if ec.data.preferred_name:
            return ec.data.preferred_name
        return f"_e{cid}"

    root_id = _build(egraph.find(root_cid))
    ir.root = root_id
    return ir
