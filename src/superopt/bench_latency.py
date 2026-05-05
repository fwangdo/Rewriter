"""Benchmark: correctness + latency comparison.

Original model (OnnxRT) vs ONNX optimizer vs Superopt.
Correctness = np.allclose(orig_output, opt_output, atol, rtol=1e-4).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

MODELS = [
    ("nlp", "tinyllama_15m"),
    ("nlp", "smollm_135m"),
    ("nlp", "pythia_70m"),
    ("vision", "mobilenetv2"),
    ("vision", "mobilevit_xxs"),
    ("vision", "yolo26_nano"),
]

# Tolerances per domain (from Gawee).
TOLERANCES = {
    "nlp": {"atol": 5e-4, "rtol": 1e-4},
    "vision": {"atol": 1e-4, "rtol": 1e-4},
}

ROOT = Path(__file__).resolve().parent.parent.parent


def make_session(model_path: str) -> ort.InferenceSession:
    """Create an ORT session with single-thread, all optimizations."""
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])


def make_inputs(sess: ort.InferenceSession, seed: int = 42) -> dict[str, np.ndarray]:
    """Create reproducible dummy inputs matching session's input specs."""
    rng = np.random.default_rng(seed)
    SEQ_LEN = 128
    PAST_LEN = 0

    inputs = {}
    for inp in sess.get_inputs():
        shape = []
        for d in inp.shape:
            if isinstance(d, str) or d is None:
                if d and "past_sequence_length" in str(d):
                    shape.append(PAST_LEN)
                elif d and "sequence_length" in str(d):
                    shape.append(SEQ_LEN)
                elif d and "batch" in str(d):
                    shape.append(1)
                else:
                    shape.append(1)
            else:
                shape.append(d)

        if "input_ids" in inp.name:
            shape = [1, SEQ_LEN]
        elif "attention_mask" in inp.name:
            shape = [1, PAST_LEN + SEQ_LEN]
        elif "position_ids" in inp.name:
            shape = [1, SEQ_LEN]

        # Determine dtype
        is_int = "int" in inp.type.lower() or any(
            k in inp.name for k in ("input_ids", "attention_mask", "position_ids")
        )
        is_bool = "bool" in inp.type.lower()

        if is_int:
            inputs[inp.name] = rng.integers(0, 1000, size=shape, dtype=np.int64)
        elif is_bool:
            inputs[inp.name] = rng.integers(0, 2, size=shape).astype(np.bool_)
        else:
            inputs[inp.name] = rng.standard_normal(shape).astype(np.float32)
    return inputs


def run_inference(sess: ort.InferenceSession, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
    """Run inference, return list of output arrays."""
    return sess.run(None, inputs)


def check_correctness(
    orig_outputs: list[np.ndarray],
    opt_outputs: list[np.ndarray],
    atol: float,
    rtol: float = 1e-4,
) -> dict:
    """Compare outputs. Returns correctness info."""
    if len(orig_outputs) != len(opt_outputs):
        return {"ok": False, "reason": f"output count mismatch: {len(orig_outputs)} vs {len(opt_outputs)}"}

    max_abs_diff = 0.0
    for i, (orig, opt) in enumerate(zip(orig_outputs, opt_outputs)):
        if orig.shape != opt.shape:
            return {"ok": False, "reason": f"output[{i}] shape mismatch: {orig.shape} vs {opt.shape}"}
        diff = float(np.max(np.abs(orig.astype(np.float64) - opt.astype(np.float64))))
        max_abs_diff = max(max_abs_diff, diff)

    ok = all(
        np.allclose(orig, opt, atol=atol, rtol=rtol)
        for orig, opt in zip(orig_outputs, opt_outputs)
    )
    return {"ok": ok, "max_abs_diff": max_abs_diff, "atol": atol, "rtol": rtol}


def measure_latency(sess: ort.InferenceSession, inputs: dict[str, np.ndarray],
                    warmup: int = 5, runs: int = 20) -> float:
    """Return median latency in ms."""
    for _ in range(warmup):
        sess.run(None, inputs)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, inputs)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


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


def run_superopt(input_path: str, output_path: str) -> bool:
    """Run our superopt pipeline."""
    try:
        from src.superopt.pipeline import superoptimize
        superoptimize(input_path, output_path)
        return True
    except Exception as e:
        print(f"    superopt FAIL: {e}")
        return False


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

    for domain, name in MODELS:
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

        # 2. ORT optimizer
        opt_path = str(artifacts / f"{name}_onnxopt.onnx")
        print("  [onnx-opt] optimizing...")
        if run_onnx_optimizer(str(model_path), opt_path):
            r_opt = evaluate_model("onnx-opt", opt_path, orig_sess, orig_outputs, inputs, domain)
        else:
            r_opt = {"status": "FAIL_OPT"}

        # 3. Superopt
        so_path = str(artifacts / f"{name}_superopt.onnx")
        print("  [superopt] optimizing...")
        if run_superopt(str(model_path), so_path):
            r_so = evaluate_model("superopt", so_path, orig_sess, orig_outputs, inputs, domain)
        else:
            r_so = {"status": "FAIL_OPT"}

        results.append({
            "name": name,
            "domain": domain,
            "original_ms": orig_lat,
            "onnxopt": r_opt,
            "superopt": r_so,
        })

    # Summary table
    print(f"\n{'='*90}")
    print(f"{'Model':<18} {'Original':>10} {'ORT-Opt':>10} {'Superopt':>10} {'SO/Orig':>10} {'Correctness':>12}")
    print(f"{'-'*90}")
    for r in results:
        orig = f"{r.get('original_ms', 0):.2f}"
        oopt = r.get("onnxopt", {})
        so = r.get("superopt", {})
        oopt_str = f"{oopt['latency_ms']:.2f}" if oopt.get("status") == "OK" else oopt.get("status", "FAIL")
        so_str = f"{so['latency_ms']:.2f}" if so.get("status") == "OK" else so.get("status", "FAIL")
        if so.get("status") == "OK" and r.get("original_ms"):
            ratio = f"{so['latency_ms'] / r['original_ms']:.3f}"
        else:
            ratio = "-"
        corr = "PASS" if so.get("status") == "OK" else "FAIL"
        print(f"{r['name']:<18} {orig:>10} {oopt_str:>10} {so_str:>10} {ratio:>10} {corr:>12}")
    print(f"{'='*90}")

    out_json = artifacts / "latency_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_json}")


if __name__ == "__main__":
    main()
