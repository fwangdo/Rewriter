"""Greedy extraction for the e-graph.

Selects the lowest-cost legal e-node per e-class to reconstruct an IRGraph.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..egraph.egraph import EGraph
from ..egraph.enode import EClassId, ENode
from ..ir.graph import IRGraph
from ..ir.node import IRNode
from .cost import CostModel


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

    programs = extract_topk(egraph, root_cid, cost_model, blacklist, k=1)
    if not programs:
        raise ValueError("cannot extract: no candidate programs")
    return programs[0].ir


def extract_topk(
    egraph: EGraph,
    root_cid: EClassId,
    cost_model: CostModel,
    blacklist: set[int],
    k: int = 5,
) -> list[ExtractedProgram]:
    """Extract the k lowest estimated-cost programs.

    This is a bounded bottom-up k-best extractor. It is still greedy at each
    e-class boundary, but it keeps multiple alternatives so later validation
    can choose the fastest candidate that passes correctness.

    Uses iterative fixpoint instead of topological sort so that cycles in the
    e-graph (common after rule application) are handled correctly.
    """
    # Collect all reachable e-class ids from root.
    reachable = _reachable_eclasses(egraph, root_cid)

    # Map canonical e-class id -> top candidates for that e-class.
    # Each candidate is (cost, choices), where choices maps every reachable
    # e-class in the subtree to the selected e-node for that e-class.
    best: dict[EClassId, list[tuple[float, dict[EClassId, ENode]]]] = {}

    def _signature(
        choices: dict[EClassId, ENode],
    ) -> tuple[tuple[EClassId, ENode], ...]:
        return tuple(sorted(choices.items(), key=lambda item: item[0]))

    def _try_extract_class(cid: EClassId) -> list[tuple[float, dict[EClassId, ENode]]]:
        ec = egraph.eclass(cid)
        candidates: list[tuple[float, dict[EClassId, ENode]]] = []
        seen: set[tuple[tuple[EClassId, ENode], ...]] = set()

        for nid in ec.nodes:
            if nid in blacklist:
                continue
            enode = egraph.enode(nid)
            child_lists = []
            extractable = True
            for child in enode.children:
                child_cid = egraph.find(child)
                child_candidates = best.get(child_cid)
                if not child_candidates:
                    extractable = False
                    break
                child_lists.append(child_candidates)
            if not extractable:
                continue

            # Limit combinations to avoid exponential blowup.
            if not child_lists:
                all_combos: list[tuple] = [()]
            else:
                all_combos = []
                base = tuple(cl[0] for cl in child_lists)
                all_combos.append(base)
                for ci, cl in enumerate(child_lists):
                    for alt in cl[1:k]:
                        variant = list(base)
                        variant[ci] = alt
                        all_combos.append(tuple(variant))
            for combo in all_combos:
                ec_data = egraph.eclass(cid).data
                child_shapes = [
                    egraph.eclass(egraph.find(c)).data.shape
                    for c in enode.children
                ]
                # Use node-local cost for ranking.  Adding full subtree
                # costs causes double-counting when children share
                # sub-DAGs, which inflates decomposed alternatives and
                # defeats legality-driven extraction.
                node_cost = cost_model.node_cost(
                    enode,
                    output_shape=ec_data.shape,
                    input_shapes=child_shapes,
                )
                choices: dict[EClassId, ENode] = {}
                conflict = False
                for _child_cost, child_choices in combo:
                    for child_choice_cid, child_choice_enode in child_choices.items():
                        existing = choices.get(child_choice_cid)
                        if existing is not None and existing != child_choice_enode:
                            conflict = True
                            break
                        choices[child_choice_cid] = child_choice_enode
                    if conflict:
                        break
                if conflict:
                    continue
                choices[cid] = enode
                sig = _signature(choices)
                if sig in seen:
                    continue
                seen.add(sig)
                candidates.append((node_cost, choices))

        candidates.sort(key=lambda item: item[0])
        return candidates[:k]

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
            candidates = _try_extract_class(cid)
            if candidates:
                best[cid] = candidates
                changed = True

    root = egraph.find(root_cid)
    if not best.get(root):
        raise ValueError("cannot extract: root e-class has no candidates")
    return [
        ExtractedProgram(
            cost=cost, ir=_build_ir_from_choices(egraph, best, choices, root)
        )
        for cost, choices in best[root][:k]
    ]


def _reachable_eclasses(egraph: EGraph, root_cid: EClassId) -> list[EClassId]:
    """Collect all e-class ids reachable from root."""
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
    best: dict[EClassId, list[tuple[float, dict[EClassId, ENode]]]],
    choices: dict[EClassId, ENode],
    root_cid: EClassId,
) -> IRGraph:
    """Reconstruct an IRGraph from selected e-nodes.

    Builds nodes in dependency order by following chosen e-node children.
    """

    ir = IRGraph()

    def _node_id(cid: EClassId) -> str:
        ec = egraph.eclass(cid)
        if ec.data.preferred_name:
            return ec.data.preferred_name
        return f"_e{cid}"

    cid_to_node_id: dict[EClassId, str] = {}

    def _build(cid: EClassId) -> str:
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
            child_ids.append(_build(egraph.find(child)))

        nid = _node_id(cid)
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

    _build(root_cid)
    ir.root = cid_to_node_id[egraph.find(root_cid)]
    return ir
