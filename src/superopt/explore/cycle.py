"""Cycle filtering for the exploration phase.

Valid rewrite rules can introduce cycles into the e-graph.
Since the extracted graph must be a DAG, we filter out
rewrites that would create cycles.

Two-layer strategy (Tensat Algorithm 2, Section 5.2):

Layer 1 (pre-filtering): Precompute a descendant map once per
iteration, then O(1) lookup per match. Sound but not complete:
matches applied earlier in the same iteration can change
descendant relations, so later matches may still create cycles.

Layer 2 (post-processing): After all matches are applied, DFS
from root to find actual cycles. For each cycle, blacklist the
most recently added e-node. Repeat until cycle-free.
"""

from __future__ import annotations

from src.common.rules.legalization import RuleSpec

from ..egraph.enode import EClassId
from ..egraph.egraph import EGraph
from ..egraph.pattern import PatternNode, PatternVar, Subst

DescendantMap = dict[EClassId, set[EClassId]]


def build_descendant_map(
    egraph: EGraph,
    blacklist: set[int] | None = None,
) -> DescendantMap:
    """Precompute the set of descendant e-classes for every canonical e-class.

    Uses iterative post-order DFS.  O(V + E) total.
    Blacklisted e-node ids are skipped.
    """
    if blacklist is None:
        blacklist = set()
    cache: DescendantMap = {}

    for start in egraph.canonical_class_ids():
        if start in cache:
            continue
        # iterative post-order DFS
        stack: list[tuple[EClassId, bool]] = [(start, False)]
        in_progress: set[EClassId] = set()
        while stack:
            cid, processed = stack.pop()
            cid = egraph.find(cid)
            if cid in cache:
                continue
            if processed:
                desc: set[EClassId] = set()
                ec = egraph.eclass(cid)
                for nid in ec.nodes:
                    if nid in blacklist:
                        continue
                    enode = egraph.enode(nid)
                    for child in enode.children:
                        child = egraph.find(child)
                        desc.add(child)
                        if child in cache:
                            desc |= cache[child]
                cache[cid] = desc
                in_progress.discard(cid)
                continue
            if cid in in_progress:
                cache[cid] = set()
                continue
            in_progress.add(cid)
            stack.append((cid, True))
            ec = egraph.eclass(cid)
            for nid in ec.nodes:
                if nid in blacklist:
                    continue
                enode = egraph.enode(nid)
                for child in enode.children:
                    child = egraph.find(child)
                    if child not in cache and child not in in_progress:
                        stack.append((child, False))
    return cache


def will_create_cycle(
    egraph: EGraph,
    rule: RuleSpec,
    match_cid: EClassId,
    subst: Subst,
    descendant_map: DescendantMap,
) -> bool:
    """Check if applying this rule would introduce a cycle.

    O(1) per call thanks to precomputed descendant_map.
    """
    target_refs = _collect_var_refs(rule.source, subst)

    match_canon = egraph.find(match_cid)
    for ref_cid in target_refs:
        ref_cid = egraph.find(ref_cid)
        if ref_cid == match_canon:
            continue
        if match_canon in descendant_map.get(ref_cid, set()):
            return True
    return False


def remove_cycles(
    egraph: EGraph,
    root_cid: EClassId,
    blacklist: set[int],
) -> None:
    """Layer 2: find and blacklist e-nodes that form cycles.

    DFS from root. When a back-edge is found, the cycle is
    collected and the most recently added e-node (highest nid)
    is blacklisted. Repeat until no cycles remain.

    Blacklisted e-nodes remain in the e-graph but are skipped
    during extraction and descendant-map computation.
    """
    while True:
        cycle = _find_one_cycle(egraph, root_cid, blacklist)
        if cycle is None:
            break
        # Blacklist the newest node in the cycle (highest nid = added last).
        newest = max(cycle)
        blacklist.add(newest)


def _find_one_cycle(
    egraph: EGraph,
    root_cid: EClassId,
    blacklist: set[int],
) -> list[int] | None:
    """DFS from root, return the nids forming a cycle, or None."""
    root_cid = egraph.find(root_cid)

    # States: 0 = unvisited, 1 = in-progress, 2 = done
    state: dict[EClassId, int] = {}
    # For each in-progress eclass, record which nid we used to enter it
    path_nids: dict[EClassId, int] = {}

    def _dfs(cid: EClassId) -> list[int] | None:
        cid = egraph.find(cid)
        s = state.get(cid, 0)
        if s == 2:
            return None
        if s == 1:
            # Back-edge: cycle found. Collect nids along the path
            # from cid back to cid via the recursion stack.
            return []  # sentinel: caller will build the cycle

        state[cid] = 1
        ec = egraph.eclass(cid)
        for nid in sorted(ec.nodes):
            if nid in blacklist:
                continue
            enode = egraph.enode(nid)
            path_nids[cid] = nid
            for child in enode.children:
                child = egraph.find(child)
                child_state = state.get(child, 0)
                if child_state == 2:
                    continue
                if child_state == 1:
                    # Back-edge to an ancestor. Return cycle.
                    return [nid]
                result = _dfs(child)
                if result is not None:
                    result.append(nid)
                    return result
        state[cid] = 2
        return None

    return _dfs(root_cid)


def _collect_var_refs(
    pattern, subst: Subst
) -> set[EClassId]:
    """Collect all e-class ids referenced by variables in a pattern."""
    refs: set[EClassId] = set()
    if isinstance(pattern, PatternVar):
        if pattern.name in subst:
            refs.add(subst[pattern.name])
    elif isinstance(pattern, PatternNode):
        for child in pattern.children:
            refs |= _collect_var_refs(child, subst)
    return refs
