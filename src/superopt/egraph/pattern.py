"""Pattern matching for e-graph rewrite rules.

Patterns are trees with concrete operator nodes and variable
placeholders.  Matching a pattern against an e-graph produces
a set of substitutions (variable → e-class id).

Follows Tensat's S-expression representation:
  (matmul ?x ?y)          — matches any matmul with two inputs
  (add ?x (mul ?x ?y))    — matches add where first input equals
                             the first input of the mul
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Union, Tuple

from .enode import EClassId
from .egraph import EGraph

import logging 

logger = logging.getLogger(__name__)


# --- Pattern AST ---

@dataclass(frozen=True)
class PatternVar:
    """A variable that matches any e-class."""
    name: str


@dataclass(frozen=True)
class PatternNode:
    """A concrete operator that must match an e-node's op."""
    op: str
    children: Tuple[ PatternVar | PatternNode, ... ] 
    attrs: tuple[tuple[str, Any], ...] | None = None  # None = don't care


Pattern = Union[PatternVar, PatternNode]

# A substitution maps variable names to e-class ids.
Subst = dict[str, EClassId]


# search and matching algorithm.
def search(
    egraph: EGraph, pattern: Pattern
) -> list[tuple[EClassId, Subst]]:
    """Find all matches of ``pattern`` in the e-graph.

    Returns a list of (matched e-class id, substitution) pairs.
    """
    results: list[tuple[EClassId, Subst]] = []
    for cid in _canonical_classes(egraph):
        for subst in _match_eclass(egraph, pattern, cid):
            # get cid, and check pattern. 
            results.append((cid, subst))
    return results


def _match_eclass(
    egraph: EGraph, pattern: Pattern, cid: EClassId
) -> list[Subst]:
    """Try to match ``pattern`` against e-class ``cid``."""
    # main algorithm of matching. 
    cid = egraph.find(cid) # we dont have to do this. but it's safe. 

    if isinstance(pattern, PatternVar):
        return [{pattern.name: cid}]

    if not isinstance(pattern, PatternNode):
        raise TypeError(f"expected PatternNode, got {type(pattern).__name__}")

    results: list[Subst] = []
    for enode in egraph.eclass_nodes(cid):
        # three conds. 
        if enode.op != pattern.op:
            continue
        if len(enode.children) != len(pattern.children):
            continue
        if pattern.attrs is not None and enode.attrs != pattern.attrs:
            continue

        # logging e-class.. 
        logger.debug(f'[E-class]: {egraph.show_format(cid)}')
        logger.debug(f'[Pattern]: {pattern}\n')

        # recursively match children
        child_substs = _match_children(egraph, pattern.children, enode.children)
        results.extend(child_substs)
    return results


def _match_children(
    egraph: EGraph,
    patterns: tuple[Pattern, ...],
    children: tuple[EClassId, ...],
) -> list[Subst]:
    """Match a sequence of patterns against a sequence of children.

    Returns all consistent substitutions (variables with the same name
    must map to the same e-class).
    """
    # base case(tail). 
    if not patterns:
        return [{}]

    first_pattern = patterns[0]
    first_child = children[0]
    rest_patterns = patterns[1:]
    rest_children = children[1:]

    results: list[Subst] = []
    # if first child is not matched with first pattern, it's done and return []. 
    for subst in _match_eclass(egraph, first_pattern, first_child):
        for rest_subst in _match_children(egraph, rest_patterns, rest_children):
            merged = _merge_substs(subst, rest_subst)
            if merged is not None:
                results.append(merged)

    logger.debug(f'[Children]: {results}\n')
    return results


def _merge_substs(a: Subst, b: Subst) -> Subst | None:
    """Merge two substitutions. Returns None if inconsistent."""
    merged = dict(a)
    for k, v in b.items():
        if k in merged:
            if merged[k] != v:
                return None
        else:
            merged[k] = v
    return merged


def _canonical_classes(egraph: EGraph) -> list[EClassId]:
    """Return all canonical e-class ids."""
    return [cid for cid in egraph._parent if egraph._parent[cid] == cid]
