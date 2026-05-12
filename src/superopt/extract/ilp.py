"""ILP extraction for the hand-rolled e-graph.

Formulates extraction as an integer linear program:
- Variables: t_c ∈ {0,1} for each e-class (active), x_n ∈ {0,1} for each e-node (selected)
- Objective: minimize Σ cost(n) · x_n
- Constraints:
  (1) t_root = 1
  (2) For each e-class c: Σ x_n = t_c  (exactly one enode if active)
  (3) For each e-node n with child e-class c': t_c' ≥ x_n

Uses OR-Tools SCIP solver.

Reference: Tensat §4.2
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
from ortools.linear_solver import pywraplp

from ..egraph.egraph import EGraph
from ..egraph.enode import EClassId, ENode
from ..ir.graph import IRGraph
from ..ir.node import IRNode
from .cost import CostModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ILPProblem:
    """Internal representation of the ILP to solve."""

    reachable: list[EClassId]
    eclass_enodes: dict[EClassId, list[tuple[int, ENode]]]
    t_idx: dict[EClassId, int]
    x_idx: dict[tuple[EClassId, int], int]
    cost: np.ndarray        # objective coefficients, length n_vars
    lb: np.ndarray           # lower bounds, length n_vars
    ub: np.ndarray           # upper bounds, length n_vars
    n_vars: int
    n_eq: int                # number of equality constraints
    n_ub: int                # number of inequality constraints
    # Sparse triplets for equality constraints (Σ x_n - t_c = 0)
    eq_rows: list[int] = field(default_factory=list)
    eq_cols: list[int] = field(default_factory=list)
    eq_data: list[float] = field(default_factory=list)
    # Sparse triplets for inequality constraints (x_n - t_child ≤ 0)
    ub_rows: list[int] = field(default_factory=list)
    ub_cols: list[int] = field(default_factory=list)
    ub_data: list[float] = field(default_factory=list)


@dataclass
class ILPStats:
    """Statistics from an ILP extraction run."""

    reachable_eclasses: int = 0
    candidate_enodes: int = 0
    variables: int = 0
    equality_constraints: int = 0
    inequality_constraints: int = 0
    blacklisted_enodes: int = 0
    illegal_candidate_enodes: int = 0
    build_time_s: float = 0.0
    solve_time_s: float = 0.0
    objective_value: float | None = None
    status: str = ""


# ---------------------------------------------------------------------------
# Problem construction
# ---------------------------------------------------------------------------

def _build_problem(
    egraph: EGraph,
    root_cid: EClassId,
    cost_model: CostModel,
    blacklist: set[int],
    soft_legalization: bool,
) -> tuple[ILPProblem, int, int]:
    """Build the ILP problem. Returns (problem, blacklisted_count, illegal_count)."""
    root_cid = egraph.find(root_cid)
    blacklisted_count = 0

    # 1. DFS from root, skipping blacklisted enodes during traversal.
    reachable: list[EClassId] = []
    visited: set[EClassId] = set()

    def _dfs(cid: EClassId) -> None:
        cid = egraph.find(cid)
        if cid in visited:
            return
        visited.add(cid)
        ec = egraph.eclass(cid)
        for nid in ec.nodes:
            if nid in blacklist:
                continue
            enode = egraph.enode(nid)
            for child in enode.children:
                _dfs(egraph.find(child))
        reachable.append(cid)

    _dfs(root_cid)

    # 2. Collect candidate enodes, excluding blacklisted.
    eclass_enodes: dict[EClassId, list[tuple[int, ENode]]] = {}
    for cid in reachable:
        ec = egraph.eclass(cid)
        candidates = []
        for nid in ec.nodes:
            if nid in blacklist:
                blacklisted_count += 1
                continue
            candidates.append((nid, egraph.enode(nid)))
        if not candidates:
            raise RuntimeError(
                f"e-class {cid} has no candidates after blacklist removal "
                f"(all {len(ec.nodes)} enodes blacklisted)"
            )
        eclass_enodes[cid] = candidates

    # In strict mode, filter illegal candidates where a legal alternative exists.
    illegal_count = 0
    if not soft_legalization:
        for cid in reachable:
            enodes = eclass_enodes[cid]
            has_legal = any(cost_model.is_legal(en) for _nid, en in enodes)
            if has_legal:
                filtered = [(nid, en) for nid, en in enodes if cost_model.is_legal(en)]
                illegal_count += len(enodes) - len(filtered)
                eclass_enodes[cid] = filtered
            else:
                illegal_count += len(enodes)
    else:
        for cid in reachable:
            for _nid, en in eclass_enodes[cid]:
                if not cost_model.is_legal(en):
                    illegal_count += 1

    # 3. Assign variable indices.
    n_classes = len(reachable)
    t_idx: dict[EClassId, int] = {}
    for i, cid in enumerate(reachable):
        t_idx[cid] = i

    x_idx: dict[tuple[EClassId, int], int] = {}
    idx = n_classes
    for cid in reachable:
        for j in range(len(eclass_enodes[cid])):
            x_idx[(cid, j)] = idx
            idx += 1
    n_vars = idx

    # 4. Cost vector.
    cost = np.zeros(n_vars)
    for cid in reachable:
        for j, (_nid, enode) in enumerate(eclass_enodes[cid]):
            ec_data = egraph.eclass(cid).data
            child_shapes = [
                egraph.eclass(egraph.find(ch)).data.shape
                for ch in enode.children
            ]
            cost[x_idx[(cid, j)]] = cost_model.node_cost(
                enode, output_shape=ec_data.shape, input_shapes=child_shapes,
            )

    # 5. Equality constraints: Σ x_n - t_c = 0.
    eq_rows: list[int] = []
    eq_cols: list[int] = []
    eq_data: list[float] = []
    for row, cid in enumerate(reachable):
        eq_rows.append(row)
        eq_cols.append(t_idx[cid])
        eq_data.append(-1.0)
        for j in range(len(eclass_enodes[cid])):
            eq_rows.append(row)
            eq_cols.append(x_idx[(cid, j)])
            eq_data.append(1.0)

    # 6. Inequality constraints: x_n - t_child ≤ 0.
    ub_rows: list[int] = []
    ub_cols: list[int] = []
    ub_data: list[float] = []
    ub_row = 0
    for cid in reachable:
        for j, (_nid, enode) in enumerate(eclass_enodes[cid]):
            for child in enode.children:
                child_cid = egraph.find(child)
                if child_cid not in t_idx:
                    continue
                ub_rows.append(ub_row)
                ub_cols.append(x_idx[(cid, j)])
                ub_data.append(1.0)
                ub_rows.append(ub_row)
                ub_cols.append(t_idx[child_cid])
                ub_data.append(-1.0)
                ub_row += 1

    # 7. Bounds.
    lb = np.zeros(n_vars)
    ub_vec = np.ones(n_vars)
    lb[t_idx[root_cid]] = 1.0  # root must be active

    problem = ILPProblem(
        reachable=reachable,
        eclass_enodes=eclass_enodes,
        t_idx=t_idx,
        x_idx=x_idx,
        cost=cost,
        lb=lb,
        ub=ub_vec,
        n_vars=n_vars,
        n_eq=n_classes,
        n_ub=ub_row,
        eq_rows=eq_rows,
        eq_cols=eq_cols,
        eq_data=eq_data,
        ub_rows=ub_rows,
        ub_cols=ub_cols,
        ub_data=ub_data,
    )
    return problem, blacklisted_count, illegal_count


# ---------------------------------------------------------------------------
# SCIP solver
# ---------------------------------------------------------------------------

def _solve_scip(
    problem: ILPProblem,
    time_limit_s: float | None,
    mip_gap: float | None,
) -> tuple[np.ndarray, str, float | None]:
    """Solve with OR-Tools SCIP backend. Returns (solution, status, objective)."""
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if solver is None:
        raise RuntimeError("SCIP solver not available in OR-Tools")

    if time_limit_s is not None:
        solver.SetTimeLimit(int(time_limit_s * 1000))

    if mip_gap is not None:
        solver.SetSolverSpecificParametersAsString(
            f"limits/gap = {mip_gap}"
        )

    # Create binary variables.
    variables = [
        solver.BoolVar(f"v{i}") for i in range(problem.n_vars)
    ]

    # Apply bounds.
    for i in range(problem.n_vars):
        if problem.lb[i] == 1.0 and problem.ub[i] == 1.0:
            solver.Add(variables[i] == 1)
        elif problem.ub[i] == 0.0:
            solver.Add(variables[i] == 0)

    # Objective.
    objective = solver.Objective()
    for i in range(problem.n_vars):
        if problem.cost[i] != 0.0:
            objective.SetCoefficient(variables[i], problem.cost[i])
    objective.SetMinimization()

    # Equality constraints: build row-wise.
    eq_row_map: dict[int, list[tuple[int, float]]] = {}
    for r, c, d in zip(problem.eq_rows, problem.eq_cols, problem.eq_data):
        eq_row_map.setdefault(r, []).append((c, d))
    for row_entries in eq_row_map.values():
        ct = solver.Constraint(0.0, 0.0)
        for col, val in row_entries:
            ct.SetCoefficient(variables[col], val)

    # Inequality constraints: x_n - t_child ≤ 0.
    ub_row_map: dict[int, list[tuple[int, float]]] = {}
    for r, c, d in zip(problem.ub_rows, problem.ub_cols, problem.ub_data):
        ub_row_map.setdefault(r, []).append((c, d))
    for row_entries in ub_row_map.values():
        ct = solver.Constraint(-solver.infinity(), 0.0)
        for col, val in row_entries:
            ct.SetCoefficient(variables[col], val)

    # Solve.
    status = solver.Solve()

    status_map = {
        pywraplp.Solver.OPTIMAL: "optimal",
        pywraplp.Solver.FEASIBLE: "feasible",
        pywraplp.Solver.INFEASIBLE: "infeasible",
        pywraplp.Solver.UNBOUNDED: "unbounded",
        pywraplp.Solver.ABNORMAL: "abnormal",
        pywraplp.Solver.NOT_SOLVED: "not_solved",
    }
    status_str = status_map.get(status, f"unknown({status})")

    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        raise RuntimeError(
            f"ILP extraction failed: {status_str}"
        )

    sol = np.array([int(v.solution_value()) for v in variables])
    obj_val = solver.Objective().Value()
    return sol, status_str, obj_val


# ---------------------------------------------------------------------------
# IRGraph reconstruction
# ---------------------------------------------------------------------------

def _build_node_from_choices(
    egraph: EGraph,
    choices: dict[EClassId, ENode],
    ir: IRGraph,
    cid_to_node_id: dict[EClassId, str],
    cid: EClassId,
) -> str:
    cid = egraph.find(cid)
    if cid in cid_to_node_id:
        return cid_to_node_id[cid]
    if cid not in choices:
        raise ValueError(f"cannot build e-class {cid}: not selected by ILP")

    enode = choices[cid]
    child_ids = [
        _build_node_from_choices(
            egraph, choices, ir, cid_to_node_id, egraph.find(child),
        )
        for child in enode.children
    ]

    nid = _node_id(egraph, cid)
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
    return nid


def _node_id(egraph: EGraph, cid: EClassId) -> str:
    ec = egraph.eclass(cid)
    if ec.data.preferred_name:
        return ec.data.preferred_name
    return f"_e{cid}"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def extract_ilp(
    egraph: EGraph,
    root_cid: EClassId,
    cost_model: CostModel,
    blacklist: set[int] | None = None,
    soft_legalization: bool = True,
    time_limit_s: float | None = 600,
    mip_gap: float | None = None,
) -> tuple[IRGraph, ILPStats]:
    """Extract the globally optimal program via ILP (SCIP solver).

    Returns (ir_graph, stats).

    Parameters
    ----------
    soft_legalization : if True, illegal enodes get high cost but remain
        selectable. If False, illegal enodes are removed when a legal
        alternative exists in the same e-class.
    time_limit_s : solver time limit in seconds (default 600).
    mip_gap : relative MIP gap tolerance (None = solver default).
    """
    if blacklist is None:
        blacklist = set()

    # Build.
    t_build = time.monotonic()
    problem, blacklisted_count, illegal_count = _build_problem(
        egraph, egraph.find(root_cid), cost_model, blacklist, soft_legalization,
    )
    build_time = time.monotonic() - t_build

    total_enodes = sum(len(v) for v in problem.eclass_enodes.values())

    stats = ILPStats(
        reachable_eclasses=len(problem.reachable),
        candidate_enodes=total_enodes,
        variables=problem.n_vars,
        equality_constraints=problem.n_eq,
        inequality_constraints=problem.n_ub,
        blacklisted_enodes=blacklisted_count,
        illegal_candidate_enodes=illegal_count,
        build_time_s=round(build_time, 4),
    )

    logger.info(
        "ILP built: %d eclasses, %d enodes, %d vars, %d eq, %d ub "
        "(%d blacklisted, %d illegal) [%.2fs]",
        stats.reachable_eclasses, stats.candidate_enodes, stats.variables,
        stats.equality_constraints, stats.inequality_constraints,
        stats.blacklisted_enodes, stats.illegal_candidate_enodes,
        stats.build_time_s,
    )

    # Solve.
    t_solve = time.monotonic()
    sol, status, obj_val = _solve_scip(problem, time_limit_s, mip_gap)
    solve_time = time.monotonic() - t_solve

    stats.solve_time_s = round(solve_time, 4)
    stats.objective_value = obj_val
    stats.status = status

    logger.info(
        "ILP solved: status=%s obj=%.1f [%.2fs]",
        status, obj_val if obj_val is not None else -1, solve_time,
    )

    # Reconstruct IRGraph from solution.
    chosen: dict[EClassId, ENode] = {}
    for cid in problem.reachable:
        for j, (_nid, enode) in enumerate(problem.eclass_enodes[cid]):
            if sol[problem.x_idx[(cid, j)]] == 1:
                chosen[cid] = enode
                break

    ir = IRGraph()
    cid_to_node_id: dict[EClassId, str] = {}
    root_cid = egraph.find(root_cid)
    root_id = _build_node_from_choices(egraph, chosen, ir, cid_to_node_id, root_cid)
    ir.root = root_id
    return ir, stats
