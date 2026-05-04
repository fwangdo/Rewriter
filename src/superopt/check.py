"""Superopt smoke check: tinyllama_15m end-to-end.

Each checkpoint prints a status line. Failures are caught and reported.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

DEFAULT_MODEL = "benchmarks/onnx/nlp/tinyllama_15m/onnx/model.onnx"
DEFAULT_OUTPUT = "artifacts/superopt/tinyllama_15m.onnx"


def run_checkpoint(label: str, fn) -> bool:
    """Run fn inside try/except, print OK/FAIL."""
    print(label)
    try:
        fn()
        print("  OK\n")
        return True
    except Exception:
        traceback.print_exc()
        print("  FAIL\n")
        return False


def cp1_onnx_to_ir(model_path: str):
    """ONNX → IR conversion."""
    import onnx
    from src.superopt.ir.convert import onnx_to_ir
    from src.superopt.ir.node import OP_INPUT, OP_WEIGHT, OP_NOOP, OP_PROJ

    model = onnx.load(model_path)
    ir = onnx_to_ir(model)
    skip = {OP_INPUT, OP_WEIGHT, OP_NOOP, OP_PROJ}
    n_ops = sum(1 for n in ir.nodes.values() if n.op not in skip)
    ops = sorted(set(n.op for n in ir.nodes.values() if n.op not in skip))
    print(f"  nodes={len(ir.nodes)}  compute_ops={n_ops}  inputs={len(ir.inputs)}  initializers={len(ir.initializers)}")
    print(f"  op_types={ops}")


def cp2_ir_to_egraph(model_path: str):
    """IR → e-graph."""
    import onnx
    from src.superopt.ir.convert import onnx_to_ir
    from src.superopt.pipeline import ir_to_egraph

    model = onnx.load(model_path)
    ir = onnx_to_ir(model)
    egraph, root = ir_to_egraph(ir)
    print(f"  eclasses={len(egraph)}  enodes={egraph.num_enodes}  root={root}")


def cp3_basic_rules():
    """Load basic rewrite rules."""
    from src.superopt.rules.arithmetic import get_arithmetic_rules
    from src.superopt.rules.layout import get_layout_rules
    from src.superopt.rules.fusion import get_fusion_rules

    rules = get_arithmetic_rules() + get_layout_rules() + get_fusion_rules()
    print(f"  total_rules={len(rules)}")
    for r in rules:
        print(f"    {r.name}")


def cp4_legalization_rules():
    """Load legalization rules."""
    from src.superopt.rules.legalization import get_legalization_rules

    rules = get_legalization_rules()
    print(f"  legalization_rules={len(rules)}")
    for r in rules:
        print(f"    {r.name}")


def cp5_exploration(model_path: str):
    """Exploration (equality saturation, basic rules only)."""
    import onnx
    from src.superopt.ir.convert import onnx_to_ir
    from src.superopt.pipeline import ir_to_egraph
    from src.superopt.explore.explorer import explore
    from src.superopt.rules.arithmetic import get_arithmetic_rules
    from src.superopt.rules.layout import get_layout_rules
    from src.superopt.rules.fusion import get_fusion_rules

    model = onnx.load(model_path)
    ir = onnx_to_ir(model)
    egraph, root = ir_to_egraph(ir)
    rules = get_arithmetic_rules() + get_layout_rules() + get_fusion_rules()
    stats = explore(egraph, rules, max_iter=3, max_nodes=10000)
    print(f"  iterations={stats.iterations}  matches={stats.total_matches}  applied={stats.total_applied}")
    print(f"  eclasses={stats.final_eclasses}  enodes={stats.final_enodes}  saturated={stats.saturated}")


def cp6_extraction(model_path: str):
    """Greedy extraction (legality-aware)."""
    import onnx
    from src.superopt.ir.convert import onnx_to_ir
    from src.superopt.pipeline import ir_to_egraph
    from src.superopt.explore.explorer import explore
    from src.superopt.extract.cost import CostModel
    from src.superopt.extract.greedy import extract_greedy
    from src.superopt.rules.arithmetic import get_arithmetic_rules
    from src.superopt.rules.layout import get_layout_rules
    from src.superopt.rules.fusion import get_fusion_rules
    from src.common.contracts import LLM_SUPPORTED_OPS

    model = onnx.load(model_path)
    ir = onnx_to_ir(model)
    egraph, root = ir_to_egraph(ir)
    rules = get_arithmetic_rules() + get_layout_rules() + get_fusion_rules()
    stats = explore(egraph, rules, max_iter=3, max_nodes=10000)

    cost_model = CostModel(LLM_SUPPORTED_OPS)
    opt_ir = extract_greedy(egraph, root, cost_model)
    skip = {"input", "weight", "noop", "proj"}
    n_ops = sum(1 for n in opt_ir.nodes.values() if n.op not in skip)
    ops = sorted(set(n.op for n in opt_ir.nodes.values() if n.op not in skip))
    illegal = [op for op in ops if op not in LLM_SUPPORTED_OPS]
    print(f"  extracted_ops={n_ops}  op_types={ops}")
    print(f"  illegal_ops={illegal}")


def cp7_full_pipeline(model_path: str, output_path: str):
    """Full pipeline: ONNX → superopt → ONNX."""
    from pathlib import Path
    from src.common.contracts import LLM_SUPPORTED_OPS
    from src.superopt.pipeline import superoptimize

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result = superoptimize(
        model_path, output_path, LLM_SUPPORTED_OPS,
        max_iter=15, max_nodes=50000,
    )
    print(f"  {result.original_nodes} → {result.optimized_nodes} nodes  legality_ok={result.legality_ok}")

    # Compare with baseline
    import onnx as _onnx
    from collections import Counter as _Counter
    opt_model = _onnx.load(output_path)
    ops = _Counter(n.op_type for n in opt_model.graph.node)
    illegal = {op: cnt for op, cnt in ops.items() if op not in LLM_SUPPORTED_OPS}
    print(f"  total_onnx_ops={sum(ops.values())}  illegal={dict(sorted(illegal.items()))}")


def main():
    parser = argparse.ArgumentParser(description="Superopt smoke check")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL, help="Path to ONNX model")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT, help="Path for output ONNX")
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        print(f"FAIL: model not found at {args.model}")
        print("      Run: python scripts/fetch_benchmark_models.py --only tinyllama_15m")
        sys.exit(1)

    print("=== superopt check: tinyllama_15m ===\n")

    results = []
    results.append(run_checkpoint("[1/7] ONNX → IR conversion...", lambda: cp1_onnx_to_ir(args.model)))
    results.append(run_checkpoint("[2/7] IR → e-graph...", lambda: cp2_ir_to_egraph(args.model)))
    results.append(run_checkpoint("[3/7] Load basic rewrite rules...", lambda: cp3_basic_rules()))
    results.append(run_checkpoint("[4/7] Load legalization rules...", lambda: cp4_legalization_rules()))
    results.append(run_checkpoint("[5/7] Exploration (equality saturation, basic rules only)...", lambda: cp5_exploration(args.model)))
    results.append(run_checkpoint("[6/7] Greedy extraction (legality-aware)...", lambda: cp6_extraction(args.model)))
    results.append(run_checkpoint("[7/7] Full pipeline: ONNX → superopt → ONNX...", lambda: cp7_full_pipeline(args.model, args.output)))

    passed = sum(results)
    failed = len(results) - passed
    print(f"=== results: {passed} passed, {failed} failed ===")
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
