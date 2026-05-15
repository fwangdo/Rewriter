"""Reusable ORT validation helpers for superopt candidates."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

# atol = absolute tolerance, rtol = relatvie tolerance. 
TOLERANCES = {
    "nlp": {"atol": 5e-4, "rtol": 1e-4},
    "llm": {"atol": 5e-4, "rtol": 1e-4},
    "vision": {"atol": 1e-4, "rtol": 1e-4},
}


def make_session(model_path: str | Path) -> ort.InferenceSession:
    """Create a deterministic single-thread ORT CPU session."""
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1 # one op
    opts.inter_op_num_threads = 1 # multi op. 
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])


def _check_model_with_fallback(model_path: str | Path) -> None:
    """Run ONNX checker, falling back to proto check after path parse failure."""
    try:
        onnx.checker.check_model(str(model_path))
        return
    except Exception as path_exc:
        try:
            model = onnx.load(str(model_path))
            onnx.checker.check_model(model)
            return
        except Exception as proto_exc:
            raise RuntimeError(
                f"path check failed: {path_exc}; proto check failed: {proto_exc}"
            ) from proto_exc


def make_inputs(
    sess: ort.InferenceSession,
    seed: int = 42,
    sequence_length: int = 128,
    past_length: int = 0,
) -> dict[str, np.ndarray]:
    """Create reproducible dummy inputs matching an ORT session's inputs."""
    rng = np.random.default_rng(seed)
    inputs = {}
    for inp in sess.get_inputs():
        shape = []
        for dim in inp.shape:
            if isinstance(dim, str) or dim is None:
                if dim and "past_sequence_length" in str(dim): # the length of seq that we've already handled. 
                    shape.append(past_length)
                elif dim and "sequence_length" in str(dim): # the length of sequence that we've taken currently. 
                    shape.append(sequence_length)
                elif dim and "batch" in str(dim):
                    shape.append(1)
                else:
                    shape.append(1)
            else:
                shape.append(dim)

        if "input_ids" in inp.name:
            shape = [1, sequence_length]
        elif "attention_mask" in inp.name:
            shape = [1, past_length + sequence_length]
        elif "position_ids" in inp.name:
            shape = [1, sequence_length]

        is_int = "int" in inp.type.lower() or any(
            key in inp.name
            for key in ("input_ids", "attention_mask", "position_ids")
        )
        is_bool = "bool" in inp.type.lower()

        if is_int:
            if "position_ids" in inp.name:
                inputs[inp.name] = np.arange(shape[-1], dtype=np.int64).reshape(shape)
            elif "attention_mask" in inp.name:
                inputs[inp.name] = np.ones(shape, dtype=np.int64)
            else:
                inputs[inp.name] = rng.integers(0, 100, size=shape, dtype=np.int64)
        elif is_bool:
            inputs[inp.name] = rng.integers(0, 2, size=shape).astype(np.bool_)
        else:
            inputs[inp.name] = rng.standard_normal(shape).astype(np.float32)
    return inputs


def run_inference(
    sess: ort.InferenceSession,
    inputs: dict[str, np.ndarray],
) -> list[np.ndarray]:
    # None means "returning all outputs"
    return sess.run(None, inputs) # type: ignore


def compare_outputs(
    orig_outputs: list[np.ndarray],
    cand_outputs: list[np.ndarray],
    atol: float,
    rtol: float = 1e-4,
) -> dict[str, object]:
    """Compare ORT outputs and return a structured correctness result."""
    if len(orig_outputs) != len(cand_outputs):
        return {
            "ok": False,
            "reason": "output_count_mismatch",
            "output_count": {"original": len(orig_outputs), "candidate": len(cand_outputs)},
            "max_abs_diff": None,
            "failed_output_index": None,
            "atol": atol,
            "rtol": rtol,
        }

    max_abs_diff = 0.0
    for index, (orig, cand) in enumerate(zip(orig_outputs, cand_outputs)):
        if orig.shape != cand.shape:
            return {
                "ok": False,
                "reason": "output_shape_mismatch",
                "output_count": len(orig_outputs),
                "failed_output_index": index,
                "original_shape": tuple(orig.shape),
                "candidate_shape": tuple(cand.shape),
                "max_abs_diff": None,
                "atol": atol,
                "rtol": rtol,
            }
        if orig.dtype != cand.dtype:
            return {
                "ok": False,
                "reason": "output_dtype_mismatch",
                "output_count": len(orig_outputs),
                "failed_output_index": index,
                "original_dtype": str(orig.dtype),
                "candidate_dtype": str(cand.dtype),
                "max_abs_diff": None,
                "atol": atol,
                "rtol": rtol,
            }
        diff = float(np.max(np.abs(orig.astype(np.float64) - cand.astype(np.float64))))
        max_abs_diff = max(max_abs_diff, diff)
        if not np.allclose(orig, cand, atol=atol, rtol=rtol):
            return {
                "ok": False,
                "reason": "value_mismatch",
                "output_count": len(orig_outputs),
                "failed_output_index": index,
                "max_abs_diff": max_abs_diff,
                "atol": atol,
                "rtol": rtol,
            }

    return {
        "ok": True,
        "reason": "ok",
        "output_count": len(orig_outputs),
        "failed_output_index": None,
        "max_abs_diff": max_abs_diff,
        "atol": atol,
        "rtol": rtol,
    }


def validate_correctness(
    original_model_path: str | Path,
    candidate_model_path: str | Path,
    domain: str,
    seed: int = 42,
) -> dict[str, object]:
    """Run ONNX checker, ORT load, and output correctness validation."""
    try:
        _check_model_with_fallback(candidate_model_path)
    except Exception as exc:
        return {"ok": False, "stage": "onnx_checker", "reason": str(exc)}

    try:
        original_sess = make_session(original_model_path)
        candidate_sess = make_session(candidate_model_path)
    except Exception as exc:
        return {"ok": False, "stage": "ort_load", "reason": str(exc)}

    try:
        inputs = make_inputs(original_sess, seed=seed)
        orig_outputs = run_inference(original_sess, inputs)
        cand_outputs = run_inference(candidate_sess, inputs)
    except Exception as exc:
        return {"ok": False, "stage": "ort_inference", "reason": str(exc)}

    tol = TOLERANCES["nlp" if domain == "llm" else domain]
    result = compare_outputs(
        orig_outputs,
        cand_outputs,
        atol=tol["atol"],
        rtol=tol["rtol"],
    )
    result["stage"] = "correctness"
    return result


def measure_latency(
    sess: ort.InferenceSession,
    inputs: dict[str, np.ndarray],
    warmup: int = 5,
    runs: int = 20,
) -> float:
    """Return median latency in milliseconds."""
    for _ in range(warmup):
        sess.run(None, inputs)
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        sess.run(None, inputs)
        times.append((time.perf_counter() - start) * 1000)
    return float(np.median(times))
