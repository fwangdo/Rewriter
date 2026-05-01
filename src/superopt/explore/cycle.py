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

    Vanilla implementation: conservative check using the descendant
    relationship.  If the matched e-class is a descendant of any
    variable binding in the substitution, applying the rule would
    create a cycle.
    """
    # Collect all e-class ids referenced in the target pattern
    target_refs = _collect_var_refs(rule.target, subst)

    # Check if match_cid is an ancestor of any target reference
    for ref_cid in target_refs:
        ref_cid = egraph.find(ref_cid)
        if ref_cid == egraph.find(match_cid):
            continue  # self-reference is ok
        if _is_descendant(egraph, egraph.find(match_cid), ref_cid):
            return True
    return False


def _is_descendant(
    egraph: EGraph, ancestor: EClassId, target: EClassId
) -> bool:
    """Check if ``target`` is a descendant of ``ancestor`` in the e-graph."""
    visited: set[EClassId] = set()
    stack = [ancestor]
    while stack:
        cid = egraph.find(stack.pop())
        if cid in visited:
            continue
        visited.add(cid)
        for enode in egraph.eclass_nodes(cid):
            for child in enode.children:
                child = egraph.find(child)
                if child == target:
                    return True
                stack.append(child)
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
