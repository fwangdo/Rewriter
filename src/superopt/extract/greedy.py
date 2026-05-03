"""Greedy bottom-up extraction from an e-graph.

Selects the lowest-cost legal e-node per e-class to reconstruct an IRGraph.
"""

from __future__ import annotations

import logging

from ..egraph.egraph import EGraph
from ..egraph.enode import EClassId, ENode
from ..ir.graph import IRGraph
from ..ir.node import IRNode
from .cost import CostModel

logger = logging.getLogger(__name__)


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

    # Collect all reachable e-class ids from root in topological order
    # (leaves first) via DFS post-order.
    reachable: list[EClassId] = []
    visited: set[EClassId] = set()

    def _dfs_postorder(cid: EClassId) -> None:
        cid = egraph.find(cid)
        if cid in visited:
            return
        visited.add(cid)
        for enode in egraph.eclass_nodes(cid):
            for child in enode.children:
                _dfs_postorder(egraph.find(child))
        reachable.append(cid)

    _dfs_postorder(egraph.find(root_cid))

    for cid in reachable:
        enodes = egraph.eclass_nodes(cid)
        best_legal: tuple[float, ENode] | None = None
        best_any: tuple[float, ENode] | None = None

        for enode in enodes:
            # Total cost = node cost + sum of children costs.
            child_cost = 0.0
            for child in enode.children:
                child_cid = egraph.find(child)
                if child_cid not in best:
                    raise ValueError(
                        f"cannot extract e-class {cid}: child {child_cid} has no cost"
                    )
                child_cost += best[child_cid][0]
            total = cost_model.node_cost(enode) + child_cost

            if best_any is None or total < best_any[0]:
                best_any = (total, enode)
            if cost_model.is_legal(enode):
                if best_legal is None or total < best_legal[0]:
                    best_legal = (total, enode)

        if best_legal is not None:
            chosen = best_legal
        elif best_any is not None:
            logger.warning(
                "e-class %d: no legal e-node, falling back to op=%r",
                cid, best_any[1].op,
            )
            chosen = best_any
        else:
            raise ValueError(f"cannot extract e-class {cid}: no e-nodes available")
        best[cid] = chosen

    # Reconstruct IRGraph from chosen e-nodes.
    ir = IRGraph()

    def _node_id(cid: EClassId) -> str:
        ec = egraph.eclass(cid)
        if ec.data.preferred_name:
            return ec.data.preferred_name
        return f"_e{cid}"

    cid_to_node_id: dict[EClassId, str] = {}
    for cid in reachable:
        _, enode = best[cid]
        child_ids: list[str] = []
        for child in enode.children:
            child_cid = egraph.find(child)
            if child_cid not in cid_to_node_id:
                raise ValueError(
                    f"cannot build e-class {cid}: child {child_cid} was not built"
                )
            child_ids.append(cid_to_node_id[child_cid])

        nid = _node_id(cid)
        if nid in ir.nodes:
            nid = f"{nid}__e{cid}"
        ec = egraph.eclass(cid)
        ir.add_node(IRNode(
            id=nid,
            op=enode.op,
            inputs=tuple(child_ids),
            attrs=enode.attrs,
            shape=ec.data.shape,
            dtype=ec.data.dtype,
        ))
        cid_to_node_id[cid] = nid

    root_id = cid_to_node_id[egraph.find(root_cid)]
    ir.root = root_id
    return ir
