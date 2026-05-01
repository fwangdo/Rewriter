"""Pattern matcher for the exploration phase.

Wraps the core pattern search with rule-level bookkeeping:
collecting all matches for all rules, and reporting statistics.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..egraph.enode import EClassId
from ..egraph.egraph import EGraph
from ..egraph.pattern import Subst, search
from ..rules.base import RewriteRule


@dataclass
class Match:
    """A single match of a rewrite rule."""

    rule: RewriteRule
    eclass_id: EClassId
    subst: Subst


def find_all_matches(
    egraph: EGraph, rules: list[RewriteRule]
) -> list[Match]:
    """Search for all matches of all rules in the e-graph.

    Returns a flat list of Match objects.
    """
    matches: list[Match] = []
    for rule in rules:
        for cid, subst in search(egraph, rule.source):
            matches.append(Match(rule=rule, eclass_id=cid, subst=subst))
    return matches
