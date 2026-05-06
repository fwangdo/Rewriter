"""Legacy ILP extraction for the hand-rolled e-graph.

The main superopt pipeline no longer uses this extractor.  It is kept as a
reference implementation while extraction logic moves to egglog.

Formulates extraction as an integer linear program:
- Variables: t_c ∈ {0,1} for each e-class (active), x_n ∈ {0,1} for each e-node (selected)
- Objective: minimize Σ cost(n) · x_n
- Constraints:
  (1) t_root = 1
  (2) For each e-class c: Σ x_n = t_c  (exactly one enode if active)
  (3) For each e-node n with child e-class c': t_c' ≥ x_n
  (4) Illegal enodes: x_n = 0 (unless eclass has no legal enodes → allow all)

Reference: Tensat §4.2
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import LinearConstraint, milp
from scipy.sparse import csc_array

from ..egraph.egraph import EGraph
from ..egraph.enode import EClassId, ENode
from ..ir.graph import IRGraph
from ..ir.node import IRNode
from .cost import CostModel

logger = logging.getLogger(__name__)


def extract_ilp(
    egraph: EGraph,
    root_cid: EClassId,
    cost_model: CostModel,
) -> IRGraph:
    """Extract the globally optimal program via ILP.

    Unlike greedy extraction, this considers subgraph sharing
    and finds the true minimum-cost DAG.
    """
    root_cid = egraph.find(root_cid)

    # 1. DFS from root to collect reachable eclasses.
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

    _dfs(root_cid)

    # 2. Assign variable indices.
    # Collect enodes per eclass (as list of ENode).
    eclass_enodes: dict[EClassId, list[ENode]] = {}
    for cid in reachable:
        eclass_enodes[cid] = egraph.eclass_nodes(cid)

    n_classes = len(reachable)
    # t variables: indices [0, n_classes)
    t_idx: dict[EClassId, int] = {}
    for i, cid in enumerate(reachable):
        t_idx[cid] = i

    # x variables: indices [n_classes, n_classes + total_enodes)
    x_idx: dict[tuple[EClassId, int], int] = {}
    idx = n_classes
    for cid in reachable:
        for j in range(len(eclass_enodes[cid])):
            x_idx[(cid, j)] = idx
            idx += 1
    n_vars = idx

    # 3. Cost vector: 0 for t vars, node_cost for x vars.
    c = np.zeros(n_vars)
    for cid in reachable:
        for j, enode in enumerate(eclass_enodes[cid]):
            ec_data = egraph.eclass(cid).data
            child_shapes = [
                egraph.eclass(egraph.find(ch)).data.shape
                for ch in enode.children
            ]
            c[x_idx[(cid, j)]] = cost_model.node_cost(
                enode, output_shape=ec_data.shape, input_shapes=child_shapes,
            )

    # 4. Equality constraints: for each eclass, Σ x_n - t_c = 0.
    eq_rows = []
    eq_cols = []
    eq_data = []
    for row, cid in enumerate(reachable):
        # -t_c
        eq_rows.append(row)
        eq_cols.append(t_idx[cid])
        eq_data.append(-1.0)
        # +x_n for each enode
        for j in range(len(eclass_enodes[cid])):
            eq_rows.append(row)
            eq_cols.append(x_idx[(cid, j)])
            eq_data.append(1.0)

    A_eq = csc_array(
        (eq_data, (eq_rows, eq_cols)),
        shape=(n_classes, n_vars),
    )
    b_eq = np.zeros(n_classes)

    # 5. Inequality constraints: x_n - t_{child} ≤ 0 for each (enode, child).
    ub_rows = []
    ub_cols = []
    ub_data = []
    ub_row = 0
    for cid in reachable:
        for j, enode in enumerate(eclass_enodes[cid]):
            for child in enode.children:
                child_cid = egraph.find(child)
                if child_cid not in t_idx:
                    continue  # unreachable child (shouldn't happen)
                # x_n - t_{child} ≤ 0
                ub_rows.append(ub_row)
                ub_cols.append(x_idx[(cid, j)])
                ub_data.append(1.0)
                ub_rows.append(ub_row)
                ub_cols.append(t_idx[child_cid])
                ub_data.append(-1.0)
                ub_row += 1

    n_ub = ub_row
    A_ub = csc_array(
        (ub_data, (ub_rows, ub_cols)),
        shape=(n_ub, n_vars),
    ) if n_ub > 0 else csc_array((0, n_vars))
    b_ub = np.zeros(n_ub)

    # 6. Bounds.
    lb = np.zeros(n_vars)
    ub = np.ones(n_vars)

    # Root must be active.
    lb[t_idx[root_cid]] = 1.0

    # Fix illegal enodes to 0, unless eclass has no legal enodes (fallback).
    for cid in reachable:
        enodes = eclass_enodes[cid]
        has_legal = any(cost_model.is_legal(en) for en in enodes)
        if has_legal:
            for j, enode in enumerate(enodes):
                if not cost_model.is_legal(enode):
                    ub[x_idx[(cid, j)]] = 0.0

    # 7. Solve.
    integrality = np.ones(n_vars)

    constraints = []
    # Equality: A_eq x = b_eq
    constraints.append(LinearConstraint(A_eq, b_eq, b_eq))
    # Inequality: A_ub x ≤ b_ub
    if n_ub > 0:
        constraints.append(LinearConstraint(A_ub, -np.inf, b_ub))

    from scipy.optimize import Bounds
    bounds = Bounds(lb=lb, ub=ub)

    result = milp(
        c=c,
        constraints=constraints,
        integrality=integrality,
        bounds=bounds,
    )

    if not result.success:
        raise RuntimeError(f"ILP extraction failed: {result.message}")

    sol = np.round(result.x).astype(int)

    # 8. Read solution → build IRGraph.
    chosen: dict[EClassId, ENode] = {}
    for cid in reachable:
        for j, enode in enumerate(eclass_enodes[cid]):
            if sol[x_idx[(cid, j)]] == 1:
                chosen[cid] = enode
                break

    ir = IRGraph()

    def _node_id(cid: EClassId) -> str:
        ec = egraph.eclass(cid)
        if ec.data.preferred_name:
            return ec.data.preferred_name
        return f"_e{cid}"

    cid_to_node_id: dict[EClassId, str] = {}
    for cid in reachable:
        if cid not in chosen:
            continue  # inactive eclass
        enode = chosen[cid]
        child_ids: list[str] = []
        for child in enode.children:
            child_cid = egraph.find(child)
            if child_cid not in cid_to_node_id:
                raise ValueError(
                    f"cannot build e-class {cid}: child {child_cid} was not built"
                )
            child_ids.append(cid_to_node_id[child_cid])

        nid = _node_id(cid)
        if nid in ir.nodes:
            nid = f"{nid}__e{cid}"
        ec = egraph.eclass(cid)
        ir.add_node(IRNode(
            id=nid,
            op=enode.op,
            inputs=tuple(child_ids),
            attrs=enode.attrs,
            shape=ec.data.shape,
            dtype=ec.data.dtype,
        ))
        cid_to_node_id[cid] = nid

    root_id = cid_to_node_id[root_cid]
    ir.root = root_id
    return ir
