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

from ..egraph.enode import EClassId, ENodeId
from ..egraph.egraph import EGraph
from ..egraph.pattern import PatternNode, PatternVar, Subst

DescendantMap = dict[EClassId, set[EClassId]]


def build_descendant_map(
    egraph: EGraph,
    blacklist: set[ENodeId],
) -> DescendantMap:
    """Precompute the set of descendant e-classes for every canonical e-class.

    Recursive post-order DFS.  O(V + E) total.
    Blacklisted e-node ids are skipped.
    """
    cache: DescendantMap = {}
    in_progress: set[EClassId] = set()

    def _dfs(cid: EClassId) -> set[EClassId]:
        cid = egraph.find(cid)
        if cid in cache:
            return cache[cid]
        if cid in in_progress:
            cache[cid] = set()
            return cache[cid]

        in_progress.add(cid)
        desc: set[EClassId] = set()
        for nid in egraph.eclass(cid).nodes:
            if nid in blacklist:
                continue
            for child in egraph.enode(nid).children:
                child = egraph.find(child)
                desc.add(child)
                desc |= _dfs(child)
        cache[cid] = desc
        in_progress.discard(cid)
        return desc

    for cid in egraph.canonical_class_ids():
        _dfs(cid)

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
    blacklist: set[ENodeId],
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
        # Blacklist the newest node in the cycle (highest nid = added last),
        # but never blacklist the last live enode of any e-class — doing so
        # would make that e-class permanently unextractable.
        blacklisted_one = False
        for candidate in sorted(cycle, reverse=True):
            cid = egraph._node_to_class.get(candidate)
            if cid is not None:
                cid = egraph.find(cid)
                live = egraph.eclass(cid).nodes - blacklist - {candidate}
                if not live:
                    continue
            blacklist.add(candidate)
            blacklisted_one = True
            break
        if not blacklisted_one:
            break

    return


def _find_one_cycle(
    egraph: EGraph,
    root_cid: EClassId,
    blacklist: set[ENodeId],
) -> list[ENodeId] | None:
    """DFS from root, return the nids forming a cycle, or None.

    3-color DFS (CLRS): white=unvisited, gray=in-progress, black=done.
    Back-edge to a gray node means a cycle exists.
    """
    root_cid = egraph.find(root_cid)
    # current stack to detect recursion. 
    in_progress: set[EClassId] = set()
    # alreay done. it is not in current scpe. 
    done: set[EClassId] = set()

    # 3 color dfs. 
    def _dfs(cid: EClassId) -> list[ENodeId] | None:
        cid = egraph.find(cid)
        if cid in done:
            return None
        if cid in in_progress:
            return []  # back-edge found, caller appends nids

        in_progress.add(cid)
        for nid in sorted(egraph.eclass(cid).nodes):
            if nid in blacklist:
                continue
            for child in egraph.enode(nid).children:
                child = egraph.find(child)
                if child in done:
                    continue
                if child in in_progress:
                    return [nid]
                result = _dfs(child)
                if result is not None:
                    result.append(nid)
                    return result

        in_progress.discard(cid)
        done.add(cid)
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
