"""Running rule generation pipeline demo.

Usage:
    python -m src.rulegen.run           # run all test cases
    python -m src.rulegen.run -v        # verbose
    python -m src.rulegen.run -i model  # single ONNX file
"""

from __future__ import annotations

import argparse
import logging

from src.rulegen.sir.tests.matmul import (
    matmul_static_right_2d,
    matmul_static_right_3d,
    matmul_static_right_4d,
    matmul_static_left_2d,
    matmul_static_left_3d,
    matmul_dynamic,
)
from src.rulegen.lowering.lower_module import onnx_to_sir
from src.rulegen.sir.sir_to_egraph import sir_to_egraph
from src.rulegen.rewrite import saturate
from src.rulegen.lifting import find_conv1x1_lift_candidates, is_conv1x1_generic

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Test cases
# -----------------------------------------------------------------------

_TEST_CASES = [
    ("matmul_static_right_2d", matmul_static_right_2d),
    ("matmul_static_right_3d", matmul_static_right_3d),
    ("matmul_static_right_4d", matmul_static_right_4d),
    ("matmul_static_left_2d",  matmul_static_left_2d),
    ("matmul_static_left_3d",  matmul_static_left_3d),
    ("matmul_dynamic",         matmul_dynamic),
]


def _show_sir(sir):
    """Print SIR graph summary."""
    for nid, node in sir.nodes.items():
        line = f"    {nid}: op={node.op}, inputs={node.inputs}"
        if node.shape:
            line += f", shape={node.shape}"
        print(line)
        if node.iterators:
            iters = ", ".join(
                f"{it.name}:{it.bound}({'R' if it.kind == 'reduction' else 'P'})"
                for it in node.iterators
            )
            print(f"      iterators: [{iters}]")
            for m in node.indexing_maps:
                idx_str = ", ".join(_fmt_affine(e) for e in m.indices)
                print(f"      {m.tensor}[{idx_str}]")
            body_str = " -> ".join(f"{s.op}({','.join(s.inputs)})" for s in node.body)
            print(f"      body: {body_str}")


def _fmt_affine(expr):
    """Format an AffineExpr for display."""
    parts = []
    for coeff, var in expr.terms:
        if coeff == 1:
            parts.append(var)
        else:
            parts.append(f"{coeff}*{var}")
    s = "+".join(parts) if parts else "0"
    if expr.offset != 0:
        s += f"+{expr.offset}"
    return s


def _show_egraph(egraph, root_cid):
    """Print e-graph summary and structure."""
    print(f"    e-classes: {len(egraph)}, e-nodes: {egraph.num_enodes}, root: {root_cid}")

    for cid in egraph.canonical_class_ids():
        for en in egraph.eclass_nodes(cid):
            if en.op == "generic":
                a = dict(en.attrs)
                iters = a.get("iterators", ())
                maps = a.get("indexing_maps", ())
                # Recover P/R from output map (last map)
                out_indices = set()
                if maps:
                    for terms, _ in maps[-1]:
                        for _, idx in terms:
                            out_indices.add(idx)
                iter_strs = [
                    f"d{i}:{b}({'P' if i in out_indices else 'R'})"
                    for i, b in enumerate(iters)
                ]
                print(f"    generic [{', '.join(iter_strs)}]")
                for map_data in maps:
                    dims = []
                    for terms, offset in map_data:
                        parts = []
                        for coeff, idx in terms:
                            parts.append(f"d{idx}" if coeff == 1 else f"{coeff}*d{idx}")
                        s = "+".join(parts) if parts else "0"
                        if offset:
                            s += f"+{offset}"
                        dims.append(s)
                    print(f"      [{', '.join(dims)}]")


def _show_lifting(egraph):
    structural = [
        en
        for cid in egraph.canonical_class_ids()
        for en in egraph.eclass_nodes(cid)
        if is_conv1x1_generic(en) is not None
    ]
    candidates = find_conv1x1_lift_candidates(egraph)

    print("  [Lifting]")
    print(f"    structural Conv1x1-form IR: {len(structural)}")
    print(f"    ONNX Conv materializable: {len(candidates)}")
    if structural and not candidates:
        print("    reason: no candidate has a static weight initializer")

    for idx, candidate in enumerate(candidates, start=1):
        print(f"    candidate {idx}: eclass={candidate.eclass_id}")
        print(f"      rewrites: {candidate.rewrites}")
        for node in candidate.subgraph.nodes:
            attrs = f", attrs={node.attrs}" if node.attrs else ""
            print(f"      {node.op_type}: {node.inputs} -> {node.outputs}{attrs}")
        if candidate.subgraph.initializers:
            init_names = ", ".join(sorted(candidate.subgraph.initializers))
            print(f"      initializers: {init_names}")


def run_test_case(name, model_fn):
    """Run ONNX -> SIR -> EGraph for one test case."""
    print(f"{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    model = model_fn()

    # ONNX
    print(f"  [ONNX] {len(model.graph.node)} node(s)")
    for n in model.graph.node:
        print(f"    {n.op_type}: {list(n.input)} -> {list(n.output)}")

    # SIR
    sir = onnx_to_sir(model)
    print(f"  [SIR] {len(sir.nodes)} node(s)")
    _show_sir(sir)

    # EGraph
    egraph, root_cid = sir_to_egraph(sir)
    print(f"  [EGraph] (before rewrite)")
    _show_egraph(egraph, root_cid)

    # Normalize
    count = saturate(egraph)
    print(f"  [Normalize] {count} rewrites applied")
    if count > 0:
        print(f"  [EGraph] (after normalization)")
        _show_egraph(egraph, root_cid)
    _show_lifting(egraph)

    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Rule generation pipeline demo.")
    parser.add_argument("-i", "--input", help="Path to input ONNX model")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(name)s %(levelname)s: %(message)s",
    )

    if args.input:
        import onnx
        model = onnx.load(args.input)
        sir = onnx_to_sir(model)
        egraph, root_cid = sir_to_egraph(sir)
        count = saturate(egraph)
        print(f"Loaded {args.input}: {len(sir.nodes)} SIR nodes -> {len(egraph)} e-classes, {egraph.num_enodes} e-nodes")
        print(f"Normalize: {count} rewrites applied")
        _show_lifting(egraph)
        return

    # Default: run all test cases
    for name, fn in _TEST_CASES:
        run_test_case(name, fn)


if __name__ == "__main__":
    main()
