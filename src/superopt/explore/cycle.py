"""Cycle filtering for the exploration phase.

Valid rewrite rules can introduce cycles into the e-graph.
Since the extracted graph must be a DAG, we filter out
rewrites that would create cycles.

Two strategies from Tensat:

1. Vanilla: check full e-graph for cycles before each apply.
   O(n_m * N) per iteration.

2. Efficient: pre-compute descendant map, use it for fast
   pre-filtering, then DFS post-processing for remaining cycles.
   Up to 2000x faster than vanilla.

We start with vanilla and upgrade to efficient when needed.
"""

from __future__ import annotations

from ..egraph.enode import EClassId
from ..egraph.egraph import EGraph
from ..egraph.pattern import Subst
from ..rules.base import RewriteRule


def will_create_cycle(
    egraph: EGraph,
    rule: RewriteRule,
    match_cid: EClassId,
    subst: Subst,
) -> bool:
    """Check if applying this rule with the given match would
    introduce a cycle in the e-graph.

    Uses depth-bounded BFS: only checks descendants up to MAX_DEPTH
    hops from the target reference. This trades off completeness for
    speed — deep cycles may slip through but will be caught during
    extraction (which must produce a DAG).
    """
    target_refs = _collect_var_refs(rule.target, subst)

    match_canon = egraph.find(match_cid)
    for ref_cid in target_refs:
        ref_cid = egraph.find(ref_cid)
        if ref_cid == match_canon:
            continue
        if _is_descendant_bounded(egraph, ref_cid, match_canon):
            return True
    return False


_MAX_CYCLE_DEPTH = 5


def _is_descendant_bounded(
    egraph: EGraph, ancestor: EClassId, target: EClassId,
) -> bool:
    """Bounded-depth descendant check."""
    visited: set[EClassId] = set()
    # (cid, depth)
    stack: list[tuple[EClassId, int]] = [(ancestor, 0)]
    while stack:
        cid, depth = stack.pop()
        if depth > _MAX_CYCLE_DEPTH:
            continue
        cid = egraph.find(cid)
        if cid in visited:
            continue
        visited.add(cid)
        for enode in egraph.eclass_nodes(cid):
            for child in enode.children:
                child = egraph.find(child)
                if child == target:
                    return True
                stack.append((child, depth + 1))
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
