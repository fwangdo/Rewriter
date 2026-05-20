"""Greedy extraction for the e-graph.

Selects the lowest-cost legal e-node per e-class to reconstruct an IRGraph.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..egraph.egraph import EGraph
from ..egraph.enode import EClassId, ENode
from src.common.ir.graph import IRGraph
from src.common.ir.node import IRNode
from .cost import CostModel

import logging

logger = logging.getLogger(__name__)


_Candidate = tuple[float, dict[EClassId, ENode]]
BestType = dict[EClassId, _Candidate]


@dataclass(frozen=True)
class ExtractedProgram:
    """One extracted candidate program."""

    cost: float
    ir: IRGraph


def extract_greedy(
    egraph: EGraph,
    root_cid: EClassId,
    cost_model: CostModel,
    blacklist: set[int],
) -> IRGraph:
    """Extract the lowest-cost program from *egraph* rooted at *root_cid*.

    Algorithm (bottom-up iterative):
    1. Collect all reachable e-classes from root.
    2. Iteratively resolve e-classes whose children are already resolved.
    3. For each e-class pick the best (lowest-cost) e-node.
    4. Build an IRGraph from chosen e-nodes.

    Blacklisted e-node ids (from cycle post-processing) are skipped.
    """
    if blacklist is None:
        blacklist = set()

    programs = extract_best(egraph, root_cid, cost_model, blacklist)
    if not programs:
        raise ValueError("cannot extract: no candidate programs")
    return programs.ir


def extract_best(
    egraph: EGraph,
    root_cid: EClassId,
    cost_model: CostModel,
    blacklist: set[int],
) -> ExtractedProgram:
    """Extract the k lowest estimated-cost programs.

    This is a bounded bottom-up k-best extractor. It is still greedy at each
    e-class boundary, but it keeps multiple alternatives so later validation
    can choose the fastest candidate that passes correctness.

    Uses iterative fixpoint instead of topological sort so that cycles in the
    e-graph (common after rule application) are handled correctly.
    """
    # Collect all reachable e-class ids from root. post order. 
    reachable = _reachable_eclasses(egraph, root_cid)

    # Map canonical e-class id -> top candidates for that e-class.
    # Each candidate is (cost, choices), where choices maps every reachable
    # e-class in the subtree to the selected e-node for that e-class.
    best: BestType = {} # tuple[float, Dict[]].. 

    # Iterative fixpoint: keep resolving e-classes whose children are ready
    # until no more progress is made.  This handles cycles correctly —
    # e-classes in cycles will remain unresolved but non-cycle classes
    # will all be extracted.
    changed = True
    while changed:
        changed = False
        for cid in reachable:
            if cid in best:
                continue
            candidate = _try_extract_class(
                egraph=egraph,
                cid=cid,
                cost_model=cost_model,
                blacklist=blacklist,
                best=best,
            )
            if candidate:
                best[cid] = candidate
                changed = True

    root = egraph.find(root_cid)
    if not best.get(root):
        raise ValueError("cannot extract: root e-class has no candidates")

    cost, choice = best[root]
    return ExtractedProgram(
            cost=cost, ir=_build_ir_from_choices(egraph, choice, root)
        )


def _try_extract_class(
    egraph: EGraph,
    cid: EClassId,
    cost_model: CostModel,
    blacklist: set[int],
    best: BestType,
) -> _Candidate | None:
    ec = egraph.eclass(cid)
    candidates: list[_Candidate] = []

    for nid in ec.nodes:
        if nid in blacklist:
            continue

        enode = egraph.enode(nid)
        child_lists = _child_lists(egraph, enode, best)
        if child_lists is None:
            continue

        translated = _merge_child_lists(child_lists)
        if translated is None:
            continue

        translated[cid] = enode
        children_cost = sum([s for s, _ in child_lists ])
        candidates.append((
            _node_cost(egraph, cid, enode, cost_model) + children_cost,
            translated,
        ))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0]


def _child_lists(
    egraph: EGraph,
    enode: ENode,
    best: BestType,
) -> list[_Candidate] | None:
    child_lists: list[_Candidate] = []
    for child in enode.children:
        child_cid = egraph.find(child)
        child_candidate = best.get(child_cid)
        if not child_candidate:
            # error case, it should be. 
            return None
        child_lists.append(child_candidate)
    return child_lists


def _merge_child_lists(
    child_lists: list[_Candidate], 
) -> dict[EClassId, ENode] | None: 
    """Make class -> node mapping."""
    choices: dict[EClassId, ENode] = {}
    for _child_cost, child_choices in child_lists:
        for child_choice_cid, child_choice_enode in child_choices.items():
            existing = choices.get(child_choice_cid)
            if existing is not None and existing != child_choice_enode:
                logger.warning("choice map conflict at e-class %s: %s vs %s", child_choice_cid, existing, child_choice_enode)
                return None
            choices[child_choice_cid] = child_choice_enode
    return choices


def _node_cost(
    egraph: EGraph,
    cid: EClassId,
    enode: ENode,
    cost_model: CostModel,
) -> float:
    ec_data = egraph.eclass(cid).data
    child_shapes = [
        egraph.eclass(egraph.find(child)).data.shape
        for child in enode.children
    ]
    # Use node-local cost for ranking.  Adding full subtree costs causes
    # double-counting when children share sub-DAGs, which inflates decomposed
    # alternatives and defeats legality-driven extraction.
    return cost_model.node_cost(
        enode,
        output_shape=ec_data.shape,
        input_shapes=child_shapes,
    )


def _reachable_eclasses(egraph: EGraph, root_cid: EClassId) -> list[EClassId]:
    """Collect all e-class ids reachable from root."""
    # note that, this is post order. 
    reachable: list[EClassId] = []
    visited: set[EClassId] = set()

    def _dfs(cid: EClassId) -> None:
        cid = egraph.find(cid)
        if cid in visited:
            return
        visited.add(cid)
        for enode in egraph.eclass_nodes(cid):
            for child in enode.children:
                _dfs(egraph.find(child))
        reachable.append(cid)

    _dfs(egraph.find(root_cid))
    return reachable


def _build_ir_from_choices(
    egraph: EGraph,
    choice: dict[EClassId, ENode],
    root_cid: EClassId,
) -> IRGraph:
    """Reconstruct an IRGraph from selected e-nodes.

    Builds nodes in dependency order by following chosen e-node children.
    """

    ir = IRGraph()
    cid_to_node_id: dict[EClassId, str] = {}
    _build_node_from_choices(egraph, choice, ir, cid_to_node_id, root_cid)
    ir.root = cid_to_node_id[egraph.find(root_cid)]
    return ir


def _build_node_from_choices(
    egraph: EGraph,
    choices: dict[EClassId, ENode],
    ir: IRGraph,
    cid_to_node_id: dict[EClassId, str],
    cid: EClassId,
) -> str:
    cid = egraph.find(cid)
    if cid in cid_to_node_id:
        return cid_to_node_id[cid]
    if cid not in choices:
        raise ValueError(
            f"cannot build e-class {cid}: not in extraction choices"
        )
    enode = choices[cid]

    # Build children first (handles arbitrary ordering).
    child_ids: list[str] = []
    for child in enode.children:
        child_ids.append(
            _build_node_from_choices(
                egraph, choices, ir, cid_to_node_id, egraph.find(child),
            )
        )

    nid = _node_id(egraph, cid)
    if nid in ir.nodes:
        nid = f"{nid}__e{cid}"
    ec = egraph.eclass(cid)
    ir.add_node(
        IRNode(
            id=nid,
            op=enode.op,
            inputs=tuple(child_ids),
            attrs=enode.attrs,
            shape=ec.data.shape,
            dtype=ec.data.dtype,
        )
    )
    cid_to_node_id[cid] = nid
    return nid


def _node_id(egraph: EGraph, cid: EClassId) -> str:
    ec = egraph.eclass(cid)
    if ec.data.preferred_name:
        return ec.data.preferred_name
    return f"_e{cid}"
