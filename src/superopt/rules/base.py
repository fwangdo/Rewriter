"""Rewrite rule base class.

A rewrite rule specifies a source pattern and a target pattern.
When the source matches an e-class, the target is instantiated
and merged into the same e-class.

Single-pattern rules have one source and one target.
Multi-pattern rules have multiple sources/targets
(e.g. merging two matmuls into one).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..egraph.enode import EClassId, ENode
from ..egraph.egraph import EGraph
from ..egraph.pattern import Pattern, PatternNode, PatternVar, Subst


@dataclass(frozen=True)
class RewriteRule:
    """A single rewrite rule: source → target.

    Attributes
    ----------
    name : str
        Human-readable name for debugging/logging.
    source : Pattern
        The pattern to search for in the e-graph.
    target : Pattern
        The pattern to instantiate and merge.
    check : Callable | None
        Optional precondition on the substitution.
        If provided, the rule only fires when check(egraph, subst)
        returns True. Use this for attribute/shape constraints.
    """

    name: str
    source: Pattern
    target: Pattern
    check: Callable[[EGraph, Subst], bool] | None = None
    apply_fn: Callable[[EGraph, EClassId, Subst], EClassId] | None = None


def apply_rule(
    egraph: EGraph,
    rule: RewriteRule,
    match_cid: EClassId,
    subst: Subst,
) -> EClassId | None:
    """Instantiate the target pattern with the given substitution
    and merge it into the matched e-class.

    Returns the merged e-class id, or None if the check failed.
    """
    if rule.check is not None and not rule.check(egraph, subst):
        return None

    if rule.apply_fn is not None:
        target_cid = rule.apply_fn(egraph, match_cid, subst)
    else:
        target_cid = _instantiate(egraph, rule.target, subst)

    # Skip merge if shapes are incompatible.
    s1 = egraph.eclass(match_cid).data.shape
    s2 = egraph.eclass(target_cid).data.shape
    if s1 is not None and s2 is not None and s1 != s2:
        return None

    return egraph.merge(match_cid, target_cid)


def _instantiate(
    egraph: EGraph, pattern: Pattern, subst: Subst
) -> EClassId:
    """Recursively instantiate a pattern into the e-graph."""
    if isinstance(pattern, PatternVar):
        if pattern.name not in subst:
            raise ValueError(f"unbound variable in target: {pattern.name}")
        return subst[pattern.name]

    assert isinstance(pattern, PatternNode)
    children = tuple(
        _instantiate(egraph, child, subst) for child in pattern.children
    )
    enode = ENode(
        op=pattern.op,
        children=children,
        attrs=pattern.attrs if pattern.attrs is not None else (),
    )
    return egraph.add(enode)
