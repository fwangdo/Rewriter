"""Lowering rules: ONNX op -> linalg-like SirNode."""
from __future__ import annotations

import logging
from typing import Any

import src.common.spec.constant as Cons
from src.rulegen.sir.sir_graph import (
    SirGraph, SirNode, SirIterator, SirIndexMap, SirScalarOp,
    AffineExpr, dim, dim_add
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Common body patterns
# ---------------------------------------------------------------------------

# matmul / conv body: mul two inputs, accumulate with sum
BODY_MUL_SUM = (
    SirScalarOp("%m", "mul", ("%0", "%1")),
    SirScalarOp("%out", "yield", ("%m",), attrs=(("combine", "sum"),)),
)


# ---------------------------------------------------------------------------
# MatMul lowering
#
# ONNX MatMul follows numpy.matmul semantics (4 cases by input rank):
#   1. 2D x 2D:  (M,K) @ (K,N) -> (M,N)
#   2. 1D x 2D:  (K,) @ (K,N)  -> (N,)
#   3. 2D x 1D:  (M,K) @ (K,)  -> (M,)
#   4. ND x MD:  batch dims broadcast, last 2 dims do matmul
# ---------------------------------------------------------------------------

def lower_matmul(graph: SirGraph, node: SirNode) -> None:
    a_id, b_id = node.inputs
    a_shape = graph.nodes[a_id].shape
    b_shape = graph.nodes[b_id].shape

    if a_shape is None or b_shape is None:
        # shape unknown: pass through as-is
        graph.add_node(node)
        return

    a_rank = len(a_shape)
    b_rank = len(b_shape)

    if a_rank == 2 and b_rank == 2:
        _lower_matmul_2d_2d(graph, node, a_shape, b_shape)
    elif a_rank == 1 and b_rank == 2:
        _lower_matmul_1d_2d(graph, node, a_shape, b_shape)
    elif a_rank == 2 and b_rank == 1:
        _lower_matmul_2d_1d(graph, node, a_shape, b_shape)
    else:
        _lower_matmul_batched(graph, node, a_shape, b_shape)


def _lower_matmul_2d_2d(
    graph: SirGraph, node: SirNode,
    a_shape: tuple[int, ...], b_shape: tuple[int, ...],
) -> None:
    """(M,K) @ (K,N) -> (M,N)"""
    a_id, b_id = node.inputs
    M, K = str(a_shape[0]), str(a_shape[1])
    N = str(b_shape[1])

    graph.add_node(SirNode(
        id=node.id,
        op="generic",
        inputs=(a_id, b_id),
        shape=node.shape,
        dtype=node.dtype,
        iterators=(
            SirIterator("i", M, "parallel"),
            SirIterator("j", N, "parallel"),
            SirIterator("k", K, "reduction"),
        ),
        indexing_maps=(
            SirIndexMap(a_id, (dim("i"), dim("k"))),
            SirIndexMap(b_id, (dim("k"), dim("j"))),
            SirIndexMap(node.id, (dim("i"), dim("j"))),
        ),
        body=BODY_MUL_SUM,
    ))


def _lower_matmul_1d_2d(
    graph: SirGraph, node: SirNode,
    a_shape: tuple[int, ...], b_shape: tuple[int, ...],
) -> None:
    """(K,) @ (K,N) -> (N,)"""
    a_id, b_id = node.inputs
    K = str(a_shape[0])
    N = str(b_shape[1])

    graph.add_node(SirNode(
        id=node.id,
        op="generic",
        inputs=(a_id, b_id),
        shape=node.shape,
        dtype=node.dtype,
        iterators=(
            SirIterator("j", N, "parallel"),
            SirIterator("k", K, "reduction"),
        ),
        indexing_maps=(
            SirIndexMap(a_id, (dim("k"),)),
            SirIndexMap(b_id, (dim("k"), dim("j"))),
            SirIndexMap(node.id, (dim("j"),)),
        ),
        body=BODY_MUL_SUM,
    ))


def _lower_matmul_2d_1d(
    graph: SirGraph, node: SirNode,
    a_shape: tuple[int, ...], b_shape: tuple[int, ...],
) -> None:
    """(M,K) @ (K,) -> (M,)"""
    a_id, b_id = node.inputs
    M = str(a_shape[0])
    K = str(a_shape[1])

    graph.add_node(SirNode(
        id=node.id,
        op="generic",
        inputs=(a_id, b_id),
        shape=node.shape,
        dtype=node.dtype,
        iterators=(
            SirIterator("i", M, "parallel"),
            SirIterator("k", K, "reduction"),
        ),
        indexing_maps=(
            SirIndexMap(a_id, (dim("i"), dim("k"))),
            SirIndexMap(b_id, (dim("k"),)),
            SirIndexMap(node.id, (dim("i"),)),
        ),
        body=BODY_MUL_SUM,
    ))


def _lower_matmul_batched(
    graph: SirGraph, node: SirNode,
    a_shape: tuple[int, ...], b_shape: tuple[int, ...],
) -> None:
    """(...,M,K) @ (...,K,N) -> (...,M,N) with batch broadcast."""
    a_id, b_id = node.inputs

    # batch dims: everything except last 2
    a_batch = a_shape[:-2]
    b_batch = b_shape[:-2]
    # broadcast: take max rank, resolve per-dim
    batch_len = max(len(a_batch), len(b_batch))
    a_batch_padded = (1,) * (batch_len - len(a_batch)) + a_batch
    b_batch_padded = (1,) * (batch_len - len(b_batch)) + b_batch

    iterators = []
    a_batch_dims = []
    b_batch_dims = []
    out_batch_dims = []

    for idx in range(batch_len):
        name = f"b{idx}"
        ad, bd = a_batch_padded[idx], b_batch_padded[idx]
        bound = str(max(ad, bd))
        iterators.append(SirIterator(name, bound, "parallel"))
        out_batch_dims.append(dim(name))
        # broadcast: size-1 dims don't index (map to constant 0)
        a_batch_dims.append(dim(name) if ad != 1 else AffineExpr(terms=(), offset=0))
        b_batch_dims.append(dim(name) if bd != 1 else AffineExpr(terms=(), offset=0))

    M = str(a_shape[-2])
    K = str(a_shape[-1])
    N = str(b_shape[-1])

    iterators.extend([
        SirIterator("i", M, "parallel"),
        SirIterator("j", N, "parallel"),
        SirIterator("k", K, "reduction"),
    ])

    graph.add_node(SirNode(
        id=node.id,
        op="generic",
        inputs=(a_id, b_id),
        shape=node.shape,
        dtype=node.dtype,
        iterators=tuple(iterators),
        indexing_maps=(
            SirIndexMap(a_id, tuple(a_batch_dims) + (dim("i"), dim("k"))),
            SirIndexMap(b_id, tuple(b_batch_dims) + (dim("k"), dim("j"))),
            SirIndexMap(node.id, tuple(out_batch_dims) + (dim("i"), dim("j"))),
        ),
        body=BODY_MUL_SUM,
    ))

# gemm 

# bn 

# gather

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_LOWERING_TABLE = {
    Cons.MAT_MUL: lower_matmul,
}


def lower(graph: SirGraph, node: SirNode) -> None:
    """Lower an ONNX-level SirNode and register it in the graph.

    If a lowering rule exists, the op is decomposed into linalg-like form.
    Otherwise the node is registered as-is.
    """
    fn = _LOWERING_TABLE.get(node.op)
    if fn is not None:
        fn(graph, node)
    else:
        graph.add_node(node)
