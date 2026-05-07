"""Benchmark: correctness + latency comparison.

Original model (OnnxRT) vs ONNX optimizer vs Superopt.
Correctness = np.allclose(orig_output, opt_output, atol, rtol=1e-4).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

from src.superopt.validation import (
    TOLERANCES,
    compare_outputs as check_correctness,
    make_inputs,
    make_session,
    measure_latency,
    run_inference,
)

# (domain, name, max_iter, max_nodes)
MODELS = [
    ("nlp", "tinyllama_15m", 10, 20_000),
    ("nlp", "smollm_135m", 5, 10_000),
    ("nlp", "pythia_70m", 10, 20_000),
    ("vision", "mobilenetv2", 15, 50_000),
    ("vision", "mobilevit_xxs", 15, 50_000),
    ("vision", "yolo26_nano", 10, 20_000),
]

ROOT = Path(__file__).resolve().parent.parent.parent


def run_baseline(input_path: str, output_path: str) -> bool:
    """Apply the rule-based onnx_rewrite baseline and save the result."""
    try:
        from src.onnx_rewrite.passes.passer import Passer
        model = onnx.load(input_path)
        model, _ = Passer().optimize(model)
        onnx.save(model, output_path)
        return True
    except Exception as e:
        print(f"    baseline FAIL: {e}")
        return False


def run_onnx_optimizer(input_path: str, output_path: str) -> bool:
    """Apply ORT graph optimization and save the optimized model."""
    try:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.optimized_model_filepath = output_path
        ort.InferenceSession(input_path, opts, providers=["CPUExecutionProvider"])
        return True
    except Exception as e:
        print(f"    onnx-opt FAIL: {e}")
        return False


def run_superopt_topk(input_path: str, output_dir: str, k: int = 5,
                      max_iter: int = 15, max_nodes: int = 50_000) -> list[str] | None:
    """Run superopt top-k pipeline. Returns list of candidate paths, or None."""
    try:
        from src.superopt.pipeline import superoptimize_topk
        results = superoptimize_topk(input_path, output_dir, k=k,
                                     max_iter=max_iter, max_nodes=max_nodes)
        return [r.output_path for r in results]
    except Exception as e:
        print(f"    superopt FAIL: {e}")
        return None


def evaluate_model(tag: str, model_path: str, orig_sess: ort.InferenceSession,
                   orig_outputs: list[np.ndarray], inputs: dict[str, np.ndarray],
                   domain: str) -> dict | None:
    """Load optimized model, check correctness, measure latency."""
    tol = TOLERANCES[domain]
    try:
        sess = make_session(model_path)
    except Exception as e:
        print(f"  [{tag}] FAIL load: {e}")
        return {"status": "FAIL_LOAD", "error": str(e)}

    try:
        opt_outputs = run_inference(sess, inputs)
    except Exception as e:
        print(f"  [{tag}] FAIL inference: {e}")
        return {"status": "FAIL_INFERENCE", "error": str(e)}

    corr = check_correctness(orig_outputs, opt_outputs, atol=tol["atol"], rtol=tol["rtol"])
    if not corr["ok"]:
        print(f"  [{tag}] FAIL correctness: max_diff={corr.get('max_abs_diff', 'N/A')}, "
              f"reason={corr.get('reason', 'exceeds tolerance')}")
        return {"status": "FAIL_CORRECTNESS", **corr}

    lat = measure_latency(sess, inputs)
    print(f"  [{tag}] OK: {lat:.2f} ms, max_diff={corr['max_abs_diff']:.2e}")
    return {"status": "OK", "latency_ms": lat, **corr}


def main():
    artifacts = ROOT / "artifacts" / "superopt" / "latency"
    artifacts.mkdir(parents=True, exist_ok=True)

    results = []

    for domain, name, max_iter, max_nodes in MODELS:
        model_path = ROOT / f"benchmarks/onnx/{domain}/{name}/onnx/model.onnx"
        if not model_path.exists():
            print(f"SKIP {name}: not found")
            continue

        print(f"\n{'='*60}")
        print(f"  {domain}/{name}")
        print(f"{'='*60}")

        # 1. Original model — baseline reference
        print("  [original] loading...")
        try:
            orig_sess = make_session(str(model_path))
            inputs = make_inputs(orig_sess)
            orig_outputs = run_inference(orig_sess, inputs)
            orig_lat = measure_latency(orig_sess, inputs)
            print(f"  [original] {orig_lat:.2f} ms")
        except Exception as e:
            print(f"  [original] FAIL: {e}")
            results.append({"name": name, "domain": domain, "original": {"status": "FAIL"}})
            continue

        # 2. Rule-based baseline (onnx_rewrite)
        bl_path = str(artifacts / f"{name}_baseline.onnx")
        print("  [baseline] optimizing...")
        if run_baseline(str(model_path), bl_path):
            r_bl = evaluate_model("baseline", bl_path, orig_sess, orig_outputs, inputs, domain)
        else:
            r_bl = {"status": "FAIL_OPT"}

        # 3. ORT optimizer
        opt_path = str(artifacts / f"{name}_onnxopt.onnx")
        print("  [onnx-opt] optimizing...")
        if run_onnx_optimizer(str(model_path), opt_path):
            r_opt = evaluate_model("onnx-opt", opt_path, orig_sess, orig_outputs, inputs, domain)
        else:
            r_opt = {"status": "FAIL_OPT"}

        # 4. Superopt top-k
        so_dir = str(artifacts / f"{name}_candidates")
        print(f"  [superopt] optimizing (top-5, iter={max_iter}, nodes={max_nodes})...")
        candidates = run_superopt_topk(str(model_path), so_dir, k=5,
                                       max_iter=max_iter, max_nodes=max_nodes)
        r_so = {"status": "FAIL_OPT"}
        if candidates:
            best_lat = float("inf")
            for ci, cpath in enumerate(candidates):
                r_c = evaluate_model(f"candidate_{ci}", cpath, orig_sess, orig_outputs, inputs, domain)
                if r_c and r_c.get("status") == "OK" and r_c["latency_ms"] < best_lat:
                    best_lat = r_c["latency_ms"]
                    r_so = r_c
            if r_so.get("status") == "OK":
                print(f"  [superopt] BEST: {r_so['latency_ms']:.2f} ms")
            else:
                print("  [superopt] no valid candidate found")

        results.append({
            "name": name,
            "domain": domain,
            "original_ms": orig_lat,
            "baseline": r_bl,
            "onnxopt": r_opt,
            "superopt": r_so,
        })

    # Summary table
    print(f"\n{'='*104}")
    print(f"{'Model':<18} {'Original':>10} {'Baseline':>10} {'ORT-Opt':>10} {'Superopt':>10} {'SO/Orig':>10} {'SO/BL':>10} {'Correct':>10}")
    print(f"{'-'*104}")
    for r in results:
        orig = f"{r.get('original_ms', 0):.2f}"
        bl = r.get("baseline", {})
        oopt = r.get("onnxopt", {})
        so = r.get("superopt", {})
        bl_str = f"{bl['latency_ms']:.2f}" if bl.get("status") == "OK" else bl.get("status", "FAIL")
        oopt_str = f"{oopt['latency_ms']:.2f}" if oopt.get("status") == "OK" else oopt.get("status", "FAIL")
        so_str = f"{so['latency_ms']:.2f}" if so.get("status") == "OK" else so.get("status", "FAIL")
        if so.get("status") == "OK" and r.get("original_ms"):
            ratio = f"{so['latency_ms'] / r['original_ms']:.3f}"
        else:
            ratio = "-"
        if so.get("status") == "OK" and bl.get("status") == "OK":
            bl_ratio = f"{so['latency_ms'] / bl['latency_ms']:.3f}"
        else:
            bl_ratio = "-"
        corr = "PASS" if so.get("status") == "OK" else "FAIL"
        print(f"{r['name']:<18} {orig:>10} {bl_str:>10} {oopt_str:>10} {so_str:>10} {ratio:>10} {bl_ratio:>10} {corr:>10}")
    print(f"{'='*104}")

    out_json = artifacts / "latency_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_json}")


if __name__ == "__main__":
    main()
