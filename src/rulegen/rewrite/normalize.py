"""Domain-independent normalization rules for generic e-nodes.

These rules transform the structural metadata (iterators, indexing maps)
of generic e-nodes into equivalent forms.  Target-specific names are kept
out of rewrite attrs; backend ops are recognized later by structural lifting.

Rules:
  1. eliminate_trivial_iterators — remove bound="1" iterators
  2. canonicalize_iterator_order — sort iterators by canonical ordering
  3. commute_inputs — swap inputs when body is commutative (mul)
  4. introduce_rank4_unit_contraction_view — expose a contraction as a
     rank-4 unit-spatial contraction view
"""

from __future__ import annotations

import logging
from src.common.egraph.egraph import EGraph
from src.common.egraph.eclass import AnalysisData
from src.common.egraph.enode import ENode

logger = logging.getLogger(__name__)

# Body pattern: mul two inputs, accumulate with sum.
_BODY_MUL_SUM = (
    ("mul", ("%0", "%1"), ()),
    ("yield", ("%m",), (("combine", "sum"),)),
)

_CONST_ZERO_DIM = ((), 0)
_UNIT_BOUND = "1"


def _dim(idx: int) -> tuple[tuple[tuple[int, int], ...], int]:
    return (((1, idx),), 0)


def _extra_attrs(attrs: dict) -> tuple[tuple[str, object], ...]:
    return tuple(
        (k, v)
        for k, v in attrs.items()
        if k not in {"iterators", "indexing_maps", "body"}
    )


def _derived(enode: ENode, rule_name: str) -> tuple[str, ...]:
    return enode.rewrites + (rule_name,)


# ---------------------------------------------------------------------------
# Rule 1: Eliminate trivial iterators (bound = "1")
# ---------------------------------------------------------------------------

def eliminate_trivial_iterators(egraph: EGraph) -> int:
    """Remove iterators with bound="1" from generic nodes.

    A trivial parallel iterator contributes a size-1 dimension (removable).
    A trivial reduction iterator sums over a single element (identity).

    The resulting generic node computes the same values (up to trivial
    size-1 dimensions), so it is merged into the same e-class.
    """
    count = 0
    for cid in egraph.canonical_class_ids():
        for enode in egraph.eclass_nodes(cid):
            if enode.op != "generic":
                continue
            attrs = dict(enode.attrs)
            iterators = attrs.get("iterators", ())
            if not iterators:
                continue

            # Find trivial iterators.
            trivial_idxs = [i for i, b in enumerate(iterators) if b == "1"]
            if not trivial_idxs:
                continue

            new_node = _remove_iterators(attrs, enode, trivial_idxs)
            if new_node is None:
                continue

            new_cid = egraph.add(new_node)
            # Compute reduced shape (remove size-1 dims for trivial parallel).
            maps = attrs.get("indexing_maps", ())
            output_map = maps[-1] if maps else ()
            output_indices = {idx for terms, _ in output_map for _, idx in terms}
            old_shape = egraph.eclass(cid).data.shape
            if old_shape is not None:
                # Remove dims corresponding to trivial parallel iterators.
                trivial_parallel = [i for i in trivial_idxs if i in output_indices]
                # Map trivial parallel idx → position in output_map
                out_dim_positions = []
                for pos, (terms, _) in enumerate(output_map):
                    for _, idx in terms:
                        if idx in trivial_parallel:
                            out_dim_positions.append(pos)
                new_shape = tuple(
                    d for i, d in enumerate(old_shape) if i not in out_dim_positions
                )
                egraph.update_analysis(new_cid, AnalysisData(
                    shape=new_shape if new_shape else None,
                    dtype=egraph.eclass(cid).data.dtype,
                ))

            if egraph.find(new_cid) != egraph.find(cid):
                egraph.merge(cid, new_cid)
                count += 1

    if count > 0:
        egraph.rebuild()
    return count


def _remove_iterators(
    attrs: dict, enode: ENode, remove_idxs: list[int],
) -> ENode | None:
    """Build a new generic e-node with specified iterators removed."""
    iterators = attrs["iterators"]
    maps = attrs.get("indexing_maps", ())
    body = attrs.get("body", ())

    remove_set = set(remove_idxs)
    # Build old→new index mapping.
    idx_map: dict[int, int] = {}
    new_pos = 0
    for old_pos in range(len(iterators)):
        if old_pos not in remove_set:
            idx_map[old_pos] = new_pos
            new_pos += 1

    new_iterators = tuple(b for i, b in enumerate(iterators) if i not in remove_set)
    if not new_iterators:
        # All iterators eliminated → scalar; skip.
        return None

    # Rebuild indexing maps: remove dimensions that reference only trivial iterators.
    new_maps = []
    for m in maps:
        new_dims = []
        for terms, offset in m:
            if not terms:
                # Constant index dim (e.g., broadcast 0). Keep only if it
                # doesn't correspond to a removed iterator's map position.
                # Actually constant dims don't reference any iterator,
                # but they appear when a trivial iterator was added.
                # Heuristic: if this is a dim added by a trivial iter, drop it.
                # A constant dim in input maps added by trivial iters is
                # only removable if it maps to a removed output dim.
                # For simplicity: keep constant dims.
                new_dims.append((terms, offset))
                continue
            # Remap iterator indices. Drop dims where ALL referenced
            # iterators are trivial.
            new_terms = tuple(
                (coeff, idx_map[idx])
                for coeff, idx in terms
                if idx not in remove_set
            )
            if new_terms:
                new_dims.append((new_terms, offset))
            # else: all iterators in this dim are trivial → dim removed.
        new_maps.append(tuple(new_dims))

    new_attrs = (
        ("iterators", tuple(new_iterators)),
        ("indexing_maps", tuple(new_maps)),
        ("body", body),
    ) + _extra_attrs(attrs)
    return ENode(
        op="generic",
        children=enode.children,
        attrs=new_attrs,
        rewrites=_derived(enode, "eliminate_trivial_iterators"),
    )


# ---------------------------------------------------------------------------
# Rule 2: Canonicalize iterator order
# ---------------------------------------------------------------------------

def canonicalize_iterator_order(egraph: EGraph) -> int:
    """Reorder iterators to a canonical ordering.

    Canonical order: output-map order first (matching output dim positions),
    then reduction iterators sorted by bound descending.

    This ensures that two generic nodes computing the same thing with
    different iterator orderings are recognized as equivalent.
    """
    count = 0
    for cid in egraph.canonical_class_ids():
        for enode in egraph.eclass_nodes(cid):
            if enode.op != "generic":
                continue
            attrs = dict(enode.attrs)
            iterators = attrs.get("iterators", ())
            maps = attrs.get("indexing_maps", ())
            body = attrs.get("body", ())
            if not iterators or not maps:
                continue

            output_map = maps[-1]
            # Determine parallel (in output map) vs reduction.
            output_indices = []
            for terms, _ in output_map:
                for _, idx in terms:
                    output_indices.append(idx)

            reduction = [i for i in range(len(iterators)) if i not in output_indices]

            # Canonical ordering: output_indices in their output-map order,
            # then reduction sorted by bound descending (stable by index).
            canonical_order = list(output_indices) + sorted(
                reduction, key=lambda i: (-_bound_value(iterators[i]), i)
            )

            if canonical_order == list(range(len(iterators))):
                continue  # Already canonical.

            # Build permutation: new_pos → old_pos.
            # And inverse: old_pos → new_pos.
            inv_perm: dict[int, int] = {}
            for new_pos, old_pos in enumerate(canonical_order):
                inv_perm[old_pos] = new_pos

            new_iterators = tuple(iterators[old] for old in canonical_order)
            new_maps = tuple(_permute_map(m, inv_perm) for m in maps)

            new_attrs = (
                ("iterators", new_iterators),
                ("indexing_maps", new_maps),
                ("body", body),
            ) + _extra_attrs(attrs)
            new_node = ENode(
                op="generic",
                children=enode.children,
                attrs=new_attrs,
                rewrites=_derived(enode, "canonicalize_iterator_order"),
            )
            new_cid = egraph.add(new_node)

            data = egraph.eclass(cid).data
            egraph.update_analysis(new_cid, AnalysisData(
                shape=data.shape, dtype=data.dtype,
            ))

            if egraph.find(new_cid) != egraph.find(cid):
                egraph.merge(cid, new_cid)
                count += 1

    if count > 0:
        egraph.rebuild()
    return count


def _permute_map(
    m: tuple, inv_perm: dict[int, int],
) -> tuple:
    """Remap iterator indices in an indexing map according to inv_perm."""
    new_dims = []
    for terms, offset in m:
        new_terms = tuple(
            (coeff, inv_perm[idx]) for coeff, idx in terms
        )
        new_dims.append((new_terms, offset))
    return tuple(new_dims)


def _bound_value(b: str) -> int:
    """Parse bound string to int for sorting (0 if not numeric)."""
    try:
        return int(b)
    except ValueError:
        return 0


# ---------------------------------------------------------------------------
# Rule 3: Commute inputs (mul commutativity)
# ---------------------------------------------------------------------------

def commute_inputs(egraph: EGraph) -> int:
    """Swap input order for generic nodes with commutative body.

    For mul+sum body: generic(A, B) with maps [mapA, mapB, mapOut]
    is equivalent to generic(B, A) with maps [mapB, mapA, mapOut].

    To converge to a unique canonical form, we impose an ordering:
    the first input's map should be lexicographically <= the second's.
    """
    count = 0
    for cid in egraph.canonical_class_ids():
        for enode in egraph.eclass_nodes(cid):
            if enode.op != "generic":
                continue
            attrs = dict(enode.attrs)
            body = attrs.get("body", ())
            if body != _BODY_MUL_SUM:
                continue
            if len(enode.children) != 2:
                continue

            maps = attrs.get("indexing_maps", ())
            if len(maps) != 3:
                continue

            map_a, map_b, map_out = maps

            # Canonical: map of first input <= map of second (lexicographic).
            if map_a <= map_b:
                continue  # Already canonical or equal.

            # Swap inputs and their maps.
            new_children = (enode.children[1], enode.children[0])
            new_maps = (map_b, map_a, map_out)
            new_attrs = (
                ("iterators", attrs["iterators"]),
                ("indexing_maps", new_maps),
                ("body", body),
            ) + _extra_attrs(attrs)
            new_node = ENode(
                op="generic",
                children=new_children,
                attrs=new_attrs,
                rewrites=_derived(enode, "commute_inputs"),
            )
            new_cid = egraph.add(new_node)

            data = egraph.eclass(cid).data
            egraph.update_analysis(new_cid, AnalysisData(
                shape=data.shape, dtype=data.dtype,
            ))

            if egraph.find(new_cid) != egraph.find(cid):
                egraph.merge(cid, new_cid)
                count += 1

    if count > 0:
        egraph.rebuild()
    return count


# ---------------------------------------------------------------------------
# Rule 4: Introduce a rank-4 unit contraction view
# ---------------------------------------------------------------------------

def introduce_rank4_unit_contraction_view(egraph: EGraph) -> int:
    """Expose a rank-2/3 contraction as a rank-4 unit-spatial view.

    This rule does not inspect the original ONNX op or any target backend op.
    It only recognizes the lower-level contraction form:

        out[..., m, n] = sum_k lhs[..., m, k] * rhs[..., k, n]

    and adds an equivalent generic with explicit outer/channel/spatial/unit
    axes:

        out4[o, c, s, u] = sum_r lhs4[o, r, s, u] * rhs4[c, r, 0, 0]

    The inserted unit ``w`` dimension and optional unit ``n`` dimension are
    view structure. Later lifting may recognize this structure as a supported
    backend op, but the rewrite itself is just a contraction view expansion.
    """
    count = 0
    for cid in egraph.canonical_class_ids():
        for enode in egraph.eclass_nodes(cid):
            if enode.op != "generic":
                continue

            attrs = dict(enode.attrs)
            if attrs.get("body", ()) != _BODY_MUL_SUM:
                continue
            if len(enode.children) != 2:
                continue

            variants = _rank4_unit_contraction_variants(attrs)
            for new_attrs in variants:
                new_node = ENode(
                    op="generic",
                    children=enode.children,
                    attrs=new_attrs,
                    rewrites=_derived(enode, "introduce_rank4_unit_contraction_view"),
                )
                new_cid = egraph.add(new_node)
                data = egraph.eclass(cid).data
                egraph.update_analysis(new_cid, AnalysisData(
                    dtype=data.dtype,
                    preferred_name=data.preferred_name,
                ))
                if egraph.find(new_cid) != egraph.find(cid):
                    egraph.merge(cid, new_cid)
                    count += 1

    if count > 0:
        egraph.rebuild()
    return count


def _rank4_unit_contraction_variants(attrs: dict) -> list[tuple[tuple[str, object], ...]]:
    iterators = attrs.get("iterators", ())
    maps = attrs.get("indexing_maps", ())
    body = attrs.get("body", ())
    if len(maps) != 3:
        return []

    output_map = maps[-1]
    output_indices = _simple_map_indices(output_map)
    if output_indices is None:
        return []

    reduction = [idx for idx in range(len(iterators)) if idx not in output_indices]
    if len(reduction) != 1:
        return []
    k_idx = reduction[0]

    # This rank-4 view has one outer dimension. For now, do not flatten
    # multiple batch iterators into one because that needs an explicit
    # non-affine view.
    if len(output_indices) not in (2, 3):
        return []

    variants: list[tuple[tuple[str, object], ...]] = []
    variants.extend(_rank4_right_factor_view(iterators, maps, output_indices, k_idx, body))
    variants.extend(_rank4_left_factor_view(iterators, maps, output_indices, k_idx, body))
    return variants


def _rank4_right_factor_view(
    iterators: tuple[str, ...],
    maps: tuple,
    output_indices: list[int],
    k_idx: int,
    body: tuple,
) -> list[tuple[tuple[str, object], ...]]:
    # lhs[..., s, r], rhs[r, c] -> lhs4[o, r, s, u], rhs4[c, r, 0, 0]
    lhs_map, rhs_map, _ = maps
    lhs_indices = _map_variable_indices(lhs_map)
    rhs_indices = _map_variable_indices(rhs_map)
    if lhs_indices is None or rhs_indices is None:
        return []

    has_batch = len(output_indices) == 3
    batch_idx = output_indices[0] if has_batch else None
    h_idx = output_indices[-2]
    oc_idx = output_indices[-1]

    expected_lhs = ([batch_idx] if has_batch else []) + [h_idx, k_idx]
    expected_rhs = [k_idx, oc_idx]
    if lhs_indices != expected_lhs or rhs_indices != expected_rhs:
        return []

    return [_make_rank4_unit_contraction_attrs(
        iterators=iterators,
        batch_idx=batch_idx,
        oc_idx=oc_idx,
        h_idx=h_idx,
        k_idx=k_idx,
        body=body,
    )]


def _rank4_left_factor_view(
    iterators: tuple[str, ...],
    maps: tuple,
    output_indices: list[int],
    k_idx: int,
    body: tuple,
) -> list[tuple[tuple[str, object], ...]]:
    # lhs[c, r], rhs[..., r, s] -> lhs4[c, r, 0, 0], rhs4[o, r, s, u].
    # This keeps child order unchanged, so the left factor map is still first.
    lhs_map, rhs_map, _ = maps
    lhs_indices = _map_variable_indices(lhs_map)
    rhs_indices = _map_variable_indices(rhs_map)
    if lhs_indices is None or rhs_indices is None:
        return []

    has_batch = len(output_indices) == 3
    batch_idx = output_indices[0] if has_batch else None
    oc_idx = output_indices[-2]
    h_idx = output_indices[-1]

    expected_lhs = [oc_idx, k_idx]
    expected_rhs = ([batch_idx] if has_batch else []) + [k_idx, h_idx]
    if lhs_indices != expected_lhs or rhs_indices != expected_rhs:
        return []

    return [_make_rank4_unit_contraction_attrs(
        iterators=iterators,
        batch_idx=batch_idx,
        oc_idx=oc_idx,
        h_idx=h_idx,
        k_idx=k_idx,
        body=body,
        weight_first=True,
    )]


def _make_rank4_unit_contraction_attrs(
    iterators: tuple[str, ...],
    batch_idx: int | None,
    oc_idx: int,
    h_idx: int,
    k_idx: int,
    body: tuple,
    weight_first: bool = False,
) -> tuple[tuple[str, object], ...]:
    new_iterators: list[str] = []
    old_to_new: dict[int, int] = {}

    def add_old(old_idx: int) -> int:
        if old_idx in old_to_new:
            return old_to_new[old_idx]
        old_to_new[old_idx] = len(new_iterators)
        new_iterators.append(iterators[old_idx])
        return old_to_new[old_idx]

    def add_unit() -> int:
        new_iterators.append(_UNIT_BOUND)
        return len(new_iterators) - 1

    n_new = add_old(batch_idx) if batch_idx is not None else add_unit()
    oc_new = add_old(oc_idx)
    h_new = add_old(h_idx)
    w_new = add_unit()
    k_new = add_old(k_idx)

    activation_map = (_dim(n_new), _dim(k_new), _dim(h_new), _dim(w_new))
    weight_map = (_dim(oc_new), _dim(k_new), _CONST_ZERO_DIM, _CONST_ZERO_DIM)
    output_map = (_dim(n_new), _dim(oc_new), _dim(h_new), _dim(w_new))

    if weight_first:
        new_maps = (weight_map, activation_map, output_map)
    else:
        new_maps = (activation_map, weight_map, output_map)

    return (
        ("iterators", tuple(new_iterators)),
        ("indexing_maps", new_maps),
        ("body", body),
    )


def _simple_map_indices(index_map: tuple) -> list[int] | None:
    indices: list[int] = []
    for terms, offset in index_map:
        if offset != 0 or len(terms) != 1:
            return None
        coeff, idx = terms[0]
        if coeff != 1:
            return None
        indices.append(idx)
    return indices


def _map_variable_indices(index_map: tuple) -> list[int] | None:
    """Return non-constant simple iterator indices from an input map.

    Broadcast dimensions appear as constant zero in the lowered SIR maps. They
    do not participate in the contraction structure, so layout recognition can
    ignore them while preserving the original tensor child.
    """
    indices: list[int] = []
    for terms, offset in index_map:
        if not terms:
            if offset != 0:
                return None
            continue
        if len(terms) != 1:
            return None
        coeff, idx = terms[0]
        if coeff != 1 or offset != 0:
            return None
        indices.append(idx)
    return indices


# ---------------------------------------------------------------------------
# Saturation: apply all rules until fixpoint
# ---------------------------------------------------------------------------

def saturate(egraph: EGraph, max_iterations: int = 10) -> int:
    """Apply all normalization rules until no more changes."""
    total = 0
    for _ in range(max_iterations):
        n = 0
        n += eliminate_trivial_iterators(egraph)
        n += canonicalize_iterator_order(egraph)
        n += commute_inputs(egraph)
        n += introduce_rank4_unit_contraction_view(egraph)
        if n == 0:
            break
        total += n
    return total
