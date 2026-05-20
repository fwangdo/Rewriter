"""Convert a SirGraph into an e-graph.

Each SirNode becomes one ENode.  Linalg-like metadata (iterators,
indexing maps, body) is packed into the ENode's attrs as hashable
tuples.  This keeps one SirNode = one ENode, so equivalence discovery
operates at the operation level.

Encoding is fully normalized:
- Variable names are replaced by integer indices (position in iterator list).
- Iterator kind (parallel/reduction) is omitted — it's recoverable from
  whether the index appears in the output's indexing map.
- Iterators are stored as a tuple of bounds only.
"""

from __future__ import annotations

from src.common.egraph.egraph import EGraph
from src.common.egraph.eclass import AnalysisData
from src.common.egraph.enode import EClassId, ENode
from src.rulegen.sir.sir_graph import (
    SirGraph, SirNode, SirIterator, SirIndexMap, SirScalarOp, AffineExpr,
    OP_INPUT, OP_WEIGHT,
)


def sir_to_egraph(sir: SirGraph) -> tuple[EGraph, EClassId]:
    """Convert a SirGraph into an e-graph.

    Returns the e-graph and the e-class id of the root node.
    """
    egraph = EGraph()
    node_to_cid: dict[str, EClassId] = {}

    for nid in sir.topo_order():
        node = sir.nodes[nid]
        children = tuple(node_to_cid[inp] for inp in node.inputs)

        if node.iterators:
            attrs = _encode_linalg_attrs(node)
        elif node.op in (OP_INPUT, OP_WEIGHT):
            attrs = (("__name__", node.id),)
        else:
            attrs = node.attrs

        cid = egraph.add(ENode(op=node.op, children=children, attrs=attrs))
        node_to_cid[nid] = cid

        egraph.update_analysis(cid, AnalysisData(
            shape=node.shape,
            dtype=node.dtype,
            is_constant=(node.op == OP_WEIGHT),
            preferred_name=nid,
        ))

    # Copy initializers so rewrite rules can access weight data.
    for name, arr in sir.initializers.items():
        egraph.initializers[name] = arr

    if sir.root is None:
        raise ValueError("SirGraph has no root node")
    return egraph, node_to_cid[sir.root]


# ---------------------------------------------------------------------------
# Linalg attrs encoding
# ---------------------------------------------------------------------------

def _encode_linalg_attrs(node: SirNode) -> tuple[tuple[str, object], ...]:
    """Pack iterators, indexing_maps, body into minimal hashable attrs.

    - Variable names -> integer index in iterator list.
    - Iterator kind (P/R) omitted (derivable from output map).
    - Iterators stored as tuple of bounds only.
    """
    # Map variable name -> position index
    var_to_idx = {it.name: i for i, it in enumerate(node.iterators)}

    iters = tuple(it.bound for it in node.iterators)

    maps = tuple(
        _encode_index_map(m.indices, var_to_idx)
        for m in node.indexing_maps
    )

    body = tuple(
        (s.op, s.inputs, s.attrs)
        for s in node.body
    )

    return (
        ("iterators", iters),
        ("indexing_maps", maps),
        ("body", body),
    )


def _encode_index_map(
    indices: tuple[AffineExpr, ...],
    var_to_idx: dict[str, int],
) -> tuple:
    """Encode one tensor's indexing map as ((coeff, iter_index), ..., offset) per dim."""
    result = []
    for expr in indices:
        terms = tuple(
            (coeff, var_to_idx[var])
            for coeff, var in expr.terms
            if var in var_to_idx
        )
        result.append((terms, expr.offset))
    return tuple(result)
