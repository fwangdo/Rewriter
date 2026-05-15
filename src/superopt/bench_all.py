"""Run baseline + superopt on all benchmark models and print comparison."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import onnx

from src.common.validation.correctness import validate_correctness

# (domain, name, max_iter)
MODELS = [
    ("nlp", "tinyllama_15m", 15),
    ("nlp", "smollm_135m", 15),
    ("nlp", "pythia_70m", 15),
    ("vision", "mobilenetv2", 15),
    ("vision", "mobilevit_xxs", 15),
    ("vision", "yolo26_nano", 15),
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


def _make_rewritten_model_path(root: Path, model_name: str, mode: str) -> Path:
    output_path = root / "artifacts" / mode / f"{model_name}.onnx"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def _correctness_summary(result: dict[str, object] | None) -> str:
    if result is None:
        return "-"
    if result.get("ok"):
        max_abs = result.get("max_abs_diff")
        return f"OK({max_abs:.2e})" if isinstance(max_abs, float) else "OK"
    stage = result.get("stage", "?")
    return f"FAIL({stage})"


def _validate_candidate(
    original_path: str | Path,
    candidate_path: str | Path,
    domain: str,
) -> dict[str, object]:
    return validate_correctness(original_path, candidate_path, domain)


def run_baseline(model_path: str, output_path: Path, domain: str):
    """Run IR-based fair baseline pipeline, return op counts."""
    from src.baseline.ir_manual import optimize_ir_baseline

    t0 = time.time()
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
        "output_path": str(output_path),
    }


def run_superopt(model_path: str, output_path: str, domain: str,
                 max_iter: int = 15,
                 ilp_time_limit_s: float | None = 600):
    """Run superopt pipeline, return op counts."""
    from dataclasses import asdict
    from src.superopt.pipeline import superoptimize

    supported = get_supported_ops(domain)
    t0 = time.time()
    result = superoptimize(
        model_path, output_path, supported,
        max_iter=max_iter,
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
        "output_path": str(output_path),
    }
    if result.ilp_stats is not None:
        out["ilp_stats"] = asdict(result.ilp_stats)
    return out


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent.parent.parent
    results = []

    for domain, name, max_iter in MODELS:
        model_path = root / f"benchmarks/onnx/{domain}/{name}/onnx/model.onnx"
        baseline_path = _make_rewritten_model_path(root, name, "baseline")
        superopt_path = _make_rewritten_model_path(root, name, "superopt")

        if not model_path.exists():
            print(f"SKIP {name}: model not found at {model_path}")
            continue

        print(f"\n{'='*60}")
        print(f"  {domain}/{name}  (max_iter={max_iter})")
        print(f"{'='*60}")

        bl = None
        bl_correctness = None
        if args.mode in ("baseline", "comp"):
            print("  [baseline] running...")
            try:
                bl = run_baseline(str(model_path), baseline_path, domain)
                bl_correctness = _validate_candidate(model_path, baseline_path, domain)
                print(
                    f"  [baseline] {bl['total_ops']} ops, "
                    f"{bl['illegal_count']} illegal, {bl['time_s']}s, "
                    f"correctness={_correctness_summary(bl_correctness)}"
                )
            except Exception as e:
                import traceback
                print(f"  [baseline] FAILED: {e}")
                traceback.print_exc()

        so = None
        so_correctness = None
        if args.mode in ("superopt", "comp"):
            print("  [superopt] running...")
            try:
                so = run_superopt(str(model_path), str(superopt_path), domain,
                                  max_iter=max_iter,
                                  ilp_time_limit_s=args.ilp_time_limit)
                so_correctness = _validate_candidate(model_path, superopt_path, domain)
                print(
                    f"  [superopt] {so['total_ops']} ops, "
                    f"{so['illegal_count']} illegal, {so['time_s']}s, "
                    f"correctness={_correctness_summary(so_correctness)}"
                )
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
            "baseline_correctness": bl_correctness,
            "superopt_correctness": so_correctness,
        })

    # Dump raw JSON for report generation
    output_name = "bench_results.json" if args.mode == "comp" else f"bench_results_{args.mode}.json"
    out_json = root / "artifacts/superopt" / output_name
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_json}")

    # Print summary table
    print(f"\n{'='*160}")
    print(
        f"{'Model':<20} {'Contract':<20} {'Baseline':>10} {'Superopt':>10} "
        f"{'Delta':>8} {'BL illegal':>12} {'SO illegal':>12} "
        f"{'BL correct':>14} {'SO correct':>14}"
    )
    print(f"{'-'*160}")
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
        bl_corr = _correctness_summary(r["baseline_correctness"])
        so_corr = _correctness_summary(r["superopt_correctness"])
        print(
            f"{r['name']:<20} {r['contract']:<20} {bl_ops:>10} {so_ops:>10} "
            f"{delta:>8} {bl_ill:>12} {so_ill:>12} "
            f"{bl_corr:>14} {so_corr:>14}"
        )
    print(f"{'='*160}")


if __name__ == "__main__":
    main()
