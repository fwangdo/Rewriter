"""Run baseline + superopt on all benchmark models and print comparison."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import onnx

# (domain, name, max_iter, max_nodes)
MODELS = [
    ("nlp", "tinyllama_15m", 15, 50_000),
    ("nlp", "smollm_135m", 15, 50_000),
    ("nlp", "pythia_70m", 15, 50_000),
    ("vision", "mobilenetv2", 15, 50_000),
    ("vision", "mobilevit_xxs", 15, 50_000),
    ("vision", "yolo26_nano", 15, 50_000),
]

# Map domain to supported ops contract
DOMAIN_OPS = {
    "nlp": "LLM_SUPPORTED_OPS",
    "vision": "VISION_SUPPORTED_OPS",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run benchmark models through baseline, superopt, or both.",
    )
    parser.add_argument(
        "--mode",
        choices=("baseline", "superopt", "comp"),
        default="comp",
        help="Run only baseline, only superopt, or both for comparison.",
    )
    parser.add_argument("--ilp-time-limit", type=float, default=600,
                        help="ILP solver time limit in seconds (default: 600)")
    return parser.parse_args()


def get_supported_ops(domain: str):
    from src.common.contracts import LLM_SUPPORTED_OPS, VISION_SUPPORTED_OPS
    return LLM_SUPPORTED_OPS if domain == "nlp" else VISION_SUPPORTED_OPS


def run_baseline(model_path: str, domain: str):
    """Run IR-based fair baseline pipeline, return op counts."""
    from src.superopt.baseline import optimize_ir_baseline

    t0 = time.time()
    model_name = Path(model_path).parent.parent.name
    output_path = Path("artifacts/superopt") / f"{model_name}_ir_baseline.onnx"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    optimize_ir_baseline(model_path, output_path)
    elapsed = time.time() - t0

    model = onnx.load(output_path)
    ops = Counter(n.op_type for n in model.graph.node)
    supported = get_supported_ops(domain)
    illegal = {op: cnt for op, cnt in ops.items() if op not in supported}
    return {
        "total_ops": sum(ops.values()),
        "illegal": dict(sorted(illegal.items())),
        "illegal_count": sum(illegal.values()),
        "ops": dict(sorted(ops.items())),
        "time_s": round(elapsed, 2),
    }


def run_superopt(model_path: str, output_path: str, domain: str,
                 max_iter: int = 15, max_nodes: int = 50_000,
                 ilp_time_limit_s: float | None = 600):
    """Run superopt pipeline, return op counts."""
    from dataclasses import asdict
    from src.superopt.pipeline import superoptimize

    supported = get_supported_ops(domain)
    t0 = time.time()
    result = superoptimize(
        model_path, output_path, supported,
        max_iter=max_iter, max_nodes=max_nodes,
        ilp_time_limit_s=ilp_time_limit_s,
    )
    elapsed = time.time() - t0

    model = onnx.load(output_path)
    ops = Counter(n.op_type for n in model.graph.node)
    illegal = {op: cnt for op, cnt in ops.items() if op not in supported}
    out = {
        "total_ops": sum(ops.values()),
        "illegal": dict(sorted(illegal.items())),
        "illegal_count": sum(illegal.values()),
        "ops": dict(sorted(ops.items())),
        "time_s": round(elapsed, 2),
        "ir_original": result.original_nodes,
        "ir_optimized": result.optimized_nodes,
        "explore_iters": result.explore_stats.iterations,
        "explore_matches": result.explore_stats.total_matches,
        "explore_applied": result.explore_stats.total_applied,
    }
    if result.ilp_stats is not None:
        out["ilp_stats"] = asdict(result.ilp_stats)
    return out


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent.parent.parent
    results = []

    for domain, name, max_iter, max_nodes in MODELS:
        model_path = root / f"benchmarks/onnx/{domain}/{name}/onnx/model.onnx"
        output_path = root / f"artifacts/superopt/{name}.onnx"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not model_path.exists():
            print(f"SKIP {name}: model not found at {model_path}")
            continue

        print(f"\n{'='*60}")
        print(f"  {domain}/{name}  (max_iter={max_iter}, max_nodes={max_nodes})")
        print(f"{'='*60}")

        bl = None
        if args.mode in ("baseline", "comp"):
            print("  [baseline] running...")
            try:
                bl = run_baseline(str(model_path), domain)
                print(f"  [baseline] {bl['total_ops']} ops, {bl['illegal_count']} illegal, {bl['time_s']}s")
            except Exception as e:
                import traceback
                print(f"  [baseline] FAILED: {e}")
                traceback.print_exc()

        so = None
        if args.mode in ("superopt", "comp"):
            print("  [superopt] running...")
            try:
                so = run_superopt(str(model_path), str(output_path), domain,
                                  max_iter=max_iter, max_nodes=max_nodes,
                                  ilp_time_limit_s=args.ilp_time_limit)
                print(f"  [superopt] {so['total_ops']} ops, {so['illegal_count']} illegal, {so['time_s']}s")
            except Exception as e:
                import traceback
                print(f"  [superopt] FAILED: {e}")
                traceback.print_exc()

        results.append({
            "domain": domain,
            "name": name,
            "contract": DOMAIN_OPS[domain],
            "mode": args.mode,
            "baseline": bl,
            "superopt": so,
        })

    # Dump raw JSON for report generation
    output_name = "bench_results.json" if args.mode == "comp" else f"bench_results_{args.mode}.json"
    out_json = root / "artifacts/superopt" / output_name
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_json}")

    # Print summary table
    print(f"\n{'='*80}")
    print(f"{'Model':<20} {'Contract':<20} {'Baseline':>10} {'Superopt':>10} {'Delta':>8} {'BL illegal':>12} {'SO illegal':>12}")
    print(f"{'-'*80}")
    for r in results:
        bl = r["baseline"]
        so = r["superopt"]
        bl_ops = bl["total_ops"] if bl else "FAIL"
        so_ops = so["total_ops"] if so else "FAIL"
        delta = ""
        if bl and so:
            d = so["total_ops"] - bl["total_ops"]
            delta = f"{d:+d}"
        bl_ill = bl["illegal_count"] if bl else "-"
        so_ill = so["illegal_count"] if so else "-"
        print(f"{r['name']:<20} {r['contract']:<20} {bl_ops:>10} {so_ops:>10} {delta:>8} {bl_ill:>12} {so_ill:>12}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
