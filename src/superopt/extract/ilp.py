"""ILP-based extraction from a saturated e-graph.

Given a saturated e-graph (where every reachable e-class contains one or
more equivalent e-nodes), extraction picks exactly one e-node per active
e-class such that the resulting DAG has minimum total cost.

This is formulated as a binary integer linear program:

  Variables
  ---------
  t_c ∈ {0,1}  for each e-class c    — "is c part of the final DAG?"
  x_n ∈ {0,1}  for each e-node  n    — "is n the chosen implementation for its e-class?"

  Objective
  ---------
  minimize  Σ cost(n) · x_n

  Constraints
  -----------
  (1) t_root = 1
      The root e-class must be active — the extracted program starts here.

  (2) Σ_{n ∈ c} x_n  =  t_c          for every e-class c
      If an e-class is active (t_c=1), exactly one of its e-nodes is
      selected.  If inactive (t_c=0), none is selected.

  (3) x_n  ≤  t_{child}              for every (e-node n, child e-class child)
      Selecting an e-node forces all of its children e-classes to be
      active.  This propagates "demand" downward through the DAG.

  No cycle constraints are added.  Cycles are handled during the
  exploration phase (see explore/cycle.py) by blacklisting e-nodes that
  would create structural cycles.  Blacklisted e-nodes are excluded
  before variable creation, so the ILP never sees them.  This follows
  the Tensat approach (§4.2) and avoids the exponential blowup that
  explicit cycle constraints would cause.

  Legality
  --------
  Two modes control how unsupported (illegal) operators are treated:

  - soft (default): illegal e-nodes remain in the candidate set but
    receive a large penalty cost (1e9) from CostModel.  The solver
    avoids them when a cheaper legal path exists, but can still pick
    them if no legal alternative is reachable.

  - strict: if an e-class has at least one legal candidate, all illegal
    candidates in that e-class are removed before variable creation.
    If it has no legal candidate, illegal ones are kept (otherwise the
    e-class would be empty and the problem infeasible).

Solver: OR-Tools SCIP (via pywraplp).

Reference: Tensat (Yang et al., 2021) §4.2
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
    """Fully constructed ILP ready to be handed to a solver.

    Separating construction from solving makes it easier to log problem
    size, swap solvers, or inspect the formulation for debugging.

    Fields
    ------
    reachable : e-classes reachable from root (post-order DFS).
    eclass_enodes : candidate (nid, ENode) pairs per e-class,
        after blacklist and (optionally) strict-legality filtering.
    t_idx : maps e-class id → variable index for t_c.
    x_idx : maps (e-class id, local index j) → variable index for x_n.
    cost : objective coefficient vector.  cost[i] is nonzero only for
        x-variables; t-variables have zero cost.
    lb, ub : per-variable bounds.  lb[root_t]=1 forces the root active.
    n_vars : total number of variables  (= |eclasses| + |enodes|).
    n_eq : number of equality constraints  (= |eclasses|).
    n_ub : number of inequality constraints  (= Σ |children(n)| for all n).
    eq_*, ub_* : COO-format sparse triplets for constraint matrices.
    """

    reachable: list[EClassId]
    eclass_enodes: dict[EClassId, list[tuple[int, ENode]]]
    t_idx: dict[EClassId, int]
    x_idx: dict[tuple[EClassId, int], int]
    cost: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    n_vars: int
    n_eq: int
    n_ub: int
    eq_rows: list[int] = field(default_factory=list)
    eq_cols: list[int] = field(default_factory=list)
    eq_data: list[float] = field(default_factory=list)
    ub_rows: list[int] = field(default_factory=list)
    ub_cols: list[int] = field(default_factory=list)
    ub_data: list[float] = field(default_factory=list)


@dataclass
class ILPStats:
    """Diagnostics collected from a single ILP extraction run.

    Logged after every extraction and optionally included in benchmark
    JSON output for systematic comparison.
    """

    reachable_eclasses: int = 0
    candidate_enodes: int = 0
    variables: int = 0
    equality_constraints: int = 0
    inequality_constraints: int = 0
    blacklisted_enodes: int = 0       # removed by cycle handling
    illegal_candidate_enodes: int = 0  # unsupported-op candidates remaining
    build_time_s: float = 0.0         # time to construct ILPProblem
    solve_time_s: float = 0.0         # time inside SCIP solver
    objective_value: float | None = None
    status: str = ""                  # "optimal" | "feasible" | error


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
    """Translate the e-graph into an ILPProblem.

    Returns (problem, blacklisted_count, illegal_count).

    The construction proceeds in seven steps:
      1. Collect reachable e-classes via DFS from root.
      2. Collect candidate e-nodes per e-class (minus blacklisted).
      3. Assign variable indices (t first, then x).
      4. Build cost vector.
      5. Build equality constraints.
      6. Build inequality constraints.
      7. Set bounds (root forced active).
    """
    root_cid = egraph.find(root_cid)
    blacklisted_count = 0

    # -- Step 1: Reachability --
    # DFS from root.  Blacklisted e-nodes are skipped so that e-classes
    # reachable *only* through blacklisted paths are excluded entirely.
    # This reduces the ILP size.
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
        reachable.append(cid)  # post-order: children before parents

    _dfs(root_cid)

    # -- Step 2: Candidate collection --
    # Each reachable e-class gets a list of (nid, ENode) candidates.
    # Blacklisted e-nodes (from cycle handling) are structurally invalid
    # and must never appear in the ILP — they are excluded here rather
    # than merely bounded to zero.
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

    # -- Legality filtering --
    # In strict mode, remove illegal candidates from e-classes that have
    # at least one legal alternative.  In soft mode, just count them —
    # the cost model's 1e9 penalty steers the solver away.
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
                # No legal alternative — keep all to avoid infeasibility.
                illegal_count += len(enodes)
    else:
        # standard mode. 
        for cid in reachable:
            for _nid, en in eclass_enodes[cid]:
                if not cost_model.is_legal(en):
                    illegal_count += 1

    # -- Step 3: Variable index assignment --
    # Variable layout:  [ t_0, t_1, ..., t_{C-1},  x_0, x_1, ..., x_{N-1} ]
    # where C = |reachable eclasses|, N = total candidate enodes.
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

    # -- Step 4: Cost vector --
    # Only x-variables carry cost.  t-variables have zero cost because
    # we want to minimize the total cost of *selected e-nodes*, not the
    # number of active e-classes.
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

    # -- Step 5: Equality constraints --
    # For each e-class c:   Σ_{n ∈ c} x_n  -  t_c  =  0
    # Stored as COO sparse triplets (row, col, value).
    eq_rows: list[int] = []
    eq_cols: list[int] = []
    eq_data: list[float] = []
    for row, cid in enumerate(reachable):
        # -t_c term
        eq_rows.append(row)
        eq_cols.append(t_idx[cid])
        eq_data.append(-1.0)
        # +x_n terms
        for j in range(len(eclass_enodes[cid])):
            eq_rows.append(row)
            eq_cols.append(x_idx[(cid, j)])
            eq_data.append(1.0)

    # -- Step 6: Inequality constraints --
    # For each (e-node n in e-class c, child e-class c'):
    #     x_n  -  t_{c'}  ≤  0
    # i.e. selecting e-node n forces its child e-class c' to be active.
    ub_rows: list[int] = []
    ub_cols: list[int] = []
    ub_data: list[float] = []
    ub_row = 0
    for cid in reachable:
        for j, (_nid, enode) in enumerate(eclass_enodes[cid]):
            for child in enode.children:
                child_cid = egraph.find(child)
                if child_cid not in t_idx:
                    continue  # child not reachable (shouldn't happen)
                # x_n coefficient = +1
                ub_rows.append(ub_row)
                ub_cols.append(x_idx[(cid, j)])
                ub_data.append(1.0)
                # t_{child} coefficient = -1
                ub_rows.append(ub_row)
                ub_cols.append(t_idx[child_cid])
                ub_data.append(-1.0)
                ub_row += 1

    # -- Step 7: Bounds --
    # All variables are binary [0, 1].
    # The root e-class is forced active: lb[t_root] = 1.
    lb = np.zeros(n_vars)
    ub_vec = np.ones(n_vars)
    lb[t_idx[root_cid]] = 1.0

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
    """Feed ILPProblem to SCIP and return the solution.

    Returns
    -------
    sol : binary solution vector of length n_vars.
    status : "optimal" if the solver proved optimality, "feasible" if
        it found a valid solution but ran out of time before proving
        optimality.
    obj_val : objective function value of the returned solution.
    """
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if solver is None:
        raise RuntimeError("SCIP solver not available in OR-Tools")

    if time_limit_s is not None:
        solver.SetTimeLimit(int(time_limit_s * 1000))

    if mip_gap is not None:
        solver.SetSolverSpecificParametersAsString(
            f"limits/gap = {mip_gap}"
        )

    # Binary variables: one per t_c and one per x_n.
    variables = [
        solver.BoolVar(f"v{i}") for i in range(problem.n_vars)
    ]

    # Fix variables whose bounds collapse to a single value.
    # lb=1, ub=1 → force to 1 (used for root t_c).
    # ub=0       → force to 0 (currently unused after blacklist removal,
    #              but kept for safety).
    # just simplification. 
    for i in range(problem.n_vars):
        if problem.lb[i] == 1.0 and problem.ub[i] == 1.0:
            solver.Add(variables[i] == 1)
        elif problem.ub[i] == 0.0:
            solver.Add(variables[i] == 0)

    # Objective: minimize Σ cost[i] * v[i].
    objective = solver.Objective()
    for i in range(problem.n_vars):
        if problem.cost[i] != 0.0:
            objective.SetCoefficient(variables[i], problem.cost[i]) # Give cost. 
    objective.SetMinimization()

    # Equality constraints (Σ x_n - t_c = 0).
    # Convert COO triplets to row-grouped form for OR-Tools API.
    eq_row_map: dict[int, list[tuple[int, float]]] = {}
    for r, c, d in zip(problem.eq_rows, problem.eq_cols, problem.eq_data):
        eq_row_map.setdefault(r, []).append((c, d))
    for row_entries in eq_row_map.values():
        ct = solver.Constraint(0.0, 0.0)  # lower == upper == 0 → equality
        for col, val in row_entries:
            ct.SetCoefficient(variables[col], val)

    # Inequality constraints (x_n - t_child ≤ 0).
    ub_row_map: dict[int, list[tuple[int, float]]] = {}
    for r, c, d in zip(problem.ub_rows, problem.ub_cols, problem.ub_data):
        ub_row_map.setdefault(r, []).append((c, d))
    for row_entries in ub_row_map.values():
        ct = solver.Constraint(-solver.infinity(), 0.0)  # ≤ 0
        for col, val in row_entries:
            ct.SetCoefficient(variables[col], val)

    # Run the solver.
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

    # OPTIMAL and FEASIBLE both provide a usable solution.
    # FEASIBLE means the solver hit the time limit before proving
    # optimality, but the incumbent solution satisfies all constraints.
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
    """Recursively build IRGraph nodes from the ILP solution.

    Walks child-first (DFS) from the given e-class, creating an IRNode
    for each active e-class using the chosen e-node.  DAG sharing is
    preserved: if an e-class is visited twice (shared subgraph), the
    second visit returns the existing node id via cid_to_node_id cache.
    """
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
    """Pick a human-readable node id for an e-class."""
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
    """Extract a minimum-cost DAG from the saturated e-graph via ILP.

    This is the main entry point for ILP extraction.  It builds the ILP
    problem from the e-graph, solves it with SCIP, and reconstructs the
    resulting IRGraph.

    Parameters
    ----------
    egraph : the saturated e-graph.
    root_cid : e-class id of the root (output) node.
    cost_model : assigns cost to each e-node.  Supported ops get cost 1,
        unsupported ops get cost 1e9 (soft penalty).
    blacklist : e-node ids to exclude (from cycle handling).
    soft_legalization : if True, illegal e-nodes stay in the candidate
        set with high penalty.  If False, they are removed when a legal
        alternative exists.
    time_limit_s : SCIP solver time limit (default 600s).  If the solver
        cannot prove optimality within this limit, it returns the best
        feasible solution found so far.
    mip_gap : relative MIP gap tolerance.  The solver stops when
        (incumbent - bound) / incumbent < mip_gap.

    Returns
    -------
    (ir_graph, stats) : the extracted program and solver diagnostics.
    """
    if blacklist is None:
        blacklist = set()

    # -- Build the ILP problem from the e-graph --
    t_build = time.monotonic() # handle the problem taht time changes in os. 
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

    # -- Solve --
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

    # -- Reconstruct IRGraph from the binary solution vector --
    # For each e-class, find the x_n variable that is 1 and record the
    # corresponding e-node as the chosen implementation.
    chosen: dict[EClassId, ENode] = {}
    for cid in problem.reachable:
        for j, (_nid, enode) in enumerate(problem.eclass_enodes[cid]):
            if sol[problem.x_idx[(cid, j)]] == 1:
                chosen[cid] = enode
                break

    # Build the IRGraph by walking chosen e-nodes from root.
    ir = IRGraph()
    cid_to_node_id: dict[EClassId, str] = {}
    root_cid = egraph.find(root_cid)
    root_id = _build_node_from_choices(egraph, chosen, ir, cid_to_node_id, root_cid)
    ir.root = root_id
    return ir, stats
