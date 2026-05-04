"""Cycle filtering for the exploration phase.

Valid rewrite rules can introduce cycles into the e-graph.
Since the extracted graph must be a DAG, we filter out
rewrites that would create cycles.

Strategy: precompute a descendant map once per iteration,
then O(1) lookup per match.  Total cost per iteration is
O(V + E) for the map build.
"""

from __future__ import annotations

from ..egraph.enode import EClassId
from ..egraph.egraph import EGraph
from ..egraph.pattern import Subst
from ..rules.base import RewriteRule

DescendantMap = dict[EClassId, set[EClassId]]


def build_descendant_map(egraph: EGraph) -> DescendantMap:
    """Precompute the set of descendant e-classes for every canonical e-class.

    Uses iterative post-order DFS.  O(V + E) total.
    """
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
                for enode in egraph.eclass_nodes(cid):
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
            for enode in egraph.eclass_nodes(cid):
                for child in enode.children:
                    child = egraph.find(child)
                    if child not in cache and child not in in_progress:
                        stack.append((child, False))
    return cache


def will_create_cycle(
    egraph: EGraph,
    rule: RewriteRule,
    match_cid: EClassId,
    subst: Subst,
    descendant_map: DescendantMap,
) -> bool:
    """Check if applying this rule would introduce a cycle.

    O(1) per call thanks to precomputed descendant_map.
    """
    target_refs = _collect_var_refs(rule.target, subst)

    match_canon = egraph.find(match_cid)
    for ref_cid in target_refs:
        ref_cid = egraph.find(ref_cid)
        if ref_cid == match_canon:
            continue
        if match_canon in descendant_map.get(ref_cid, set()):
            return True
    return False


def _collect_var_refs(
    pattern, subst: Subst
) -> set[EClassId]:
    """Collect all e-class ids referenced by variables in a pattern."""
    from ..egraph.pattern import PatternVar, PatternNode

    refs: set[EClassId] = set()
    if isinstance(pattern, PatternVar):
        if pattern.name in subst:
            refs.add(subst[pattern.name])
    elif isinstance(pattern, PatternNode):
        for child in pattern.children:
            refs |= _collect_var_refs(child, subst)
    return refs
