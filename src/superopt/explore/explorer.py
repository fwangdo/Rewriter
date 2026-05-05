"""Exploration phase: grow the e-graph by applying rewrite rules.

This is the main loop of equality saturation.
Each iteration:
1. Find all matches of all rules.
2. Filter out matches that would create cycles.
3. Apply surviving matches (add target e-nodes, merge e-classes).
4. Rebuild the e-graph to restore invariants.
5. Stop if saturated or limits are reached.

Reference: Tensat Algorithm 2.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..egraph.egraph import EGraph
from ..egraph.enode import EClassId
from ..rules.base import RewriteRule, apply_rule
from .cycle import build_descendant_map, remove_cycles, will_create_cycle
from .matcher import find_all_matches

logger = logging.getLogger(__name__)


@dataclass
class ExploreStats:
    """Statistics from the exploration phase."""

    iterations: int = 0
    total_matches: int = 0
    total_applied: int = 0
    final_eclasses: int = 0
    final_enodes: int = 0
    saturated: bool = False


def explore(
    egraph: EGraph,
    rules: list[RewriteRule],
    max_iter: int = 15,
    max_nodes: int = 50_000,
    root_cid: EClassId | None = None,
) -> tuple[ExploreStats, set[int]]:
    """Run the exploration phase on the e-graph.

    Modifies ``egraph`` in place.  Returns (statistics, blacklist).

    Two-layer cycle filtering (Tensat Algorithm 2):
    - Layer 1: pre-filter with descendant map (sound, not complete)
    - Layer 2: post-process to blacklist e-nodes in remaining cycles
    """
    stats = ExploreStats()
    blacklist: set[int] = set()

    for iteration in range(max_iter):
        stats.iterations = iteration + 1

        if egraph.num_enodes >= max_nodes:
            logger.info(
                "exploration stopped: node limit %d reached", max_nodes
            )
            break

        # 1. find all matches
        matches = find_all_matches(egraph, rules)
        stats.total_matches += len(matches)

        if not matches:
            stats.saturated = True
            logger.info("exploration saturated at iteration %d", iteration)
            break

        # 2. Layer 1: pre-filter with descendant map
        desc_map = build_descendant_map(egraph, blacklist)
        applied = 0
        for match in matches:
            if will_create_cycle(
                egraph, match.rule, match.eclass_id, match.subst, desc_map
            ):
                continue
            result = apply_rule(
                egraph, match.rule, match.eclass_id, match.subst
            )
            if result is not None:
                applied += 1

        stats.total_applied += applied

        # 3. rebuild
        egraph.rebuild()

        # 4. Layer 2: post-process to remove any cycles that slipped through
        if root_cid is not None:
            remove_cycles(egraph, root_cid, blacklist)

        logger.debug(
            "iter %d: matches=%d applied=%d eclasses=%d enodes=%d blacklisted=%d",
            iteration,
            len(matches),
            applied,
            len(egraph),
            egraph.num_enodes,
            len(blacklist),
        )

    stats.final_eclasses = len(egraph)
    stats.final_enodes = egraph.num_enodes
    return stats, blacklist
