"""Profile per-op latency on ONNX Runtime for benchmark models."""

import json
import os
import glob
import numpy as np
import onnx
import onnxruntime as ort
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

MODELS = [
    "benchmarks/onnx/nlp/tinyllama_15m/onnx/model.onnx",
    "benchmarks/onnx/nlp/smollm_135m/onnx/model.onnx",
    "benchmarks/onnx/nlp/pythia_70m/onnx/model.onnx",
    "benchmarks/onnx/vision/mobilenetv2/onnx/model.onnx",
    "benchmarks/onnx/vision/mobilevit_xxs/onnx/model.onnx",
    "benchmarks/onnx/vision/yolo26_nano/onnx/model.onnx",
]

WARMUP = 3
BENCHMARK = 10


def make_dummy_inputs(model_path: str) -> dict:
    """Create dummy inputs by inspecting the ONNX model."""
    model = onnx.load(model_path)
    inputs = {}
    seq_len = 128
    batch = 1

    for inp in model.graph.input:
        name = inp.name
        elem_type = inp.type.tensor_type.elem_type
        dims = inp.type.tensor_type.shape.dim

        shape = []
        for d in dims:
            if d.dim_value > 0:
                shape.append(d.dim_value)
            else:
                param = d.dim_param
                if "batch" in param:
                    shape.append(batch)
                elif "sequence_length" in param and "past" not in param:
                    shape.append(seq_len)
                elif "past_sequence_length" in param:
                    # For prefill: past_sequence_length = 0
                    # "past_sequence_length + 1" -> use seq_len for attention_mask
                    shape.append(seq_len)
                else:
                    shape.append(seq_len)

        # elem_type: 1=float32, 7=int64, 6=int32
        if elem_type == 1:
            inputs[name] = np.random.randn(*shape).astype(np.float32)
        elif elem_type == 7:
            inputs[name] = np.ones(shape, dtype=np.int64)
        elif elem_type == 6:
            inputs[name] = np.ones(shape, dtype=np.int32)
        else:
            inputs[name] = np.random.randn(*shape).astype(np.float32)

    # For NLP models with past_key_values, set past_sequence_length=0 (prefill)
    # and adjust attention_mask and position_ids accordingly
    has_past = any("past_key_values" in name for name in inputs)
    if has_past:
        for name in list(inputs.keys()):
            if "past_key_values" in name:
                # Set past to empty (seq dim = 0)
                old_shape = list(inputs[name].shape)
                # Find the past_sequence_length dim and set to 0
                # Shape is [batch, num_heads, past_seq_len, head_dim]
                old_shape[2] = 0
                inputs[name] = np.zeros(old_shape, dtype=inputs[name].dtype)
        # attention_mask should be [batch, seq_len] for prefill
        if "attention_mask" in inputs:
            inputs["attention_mask"] = np.ones((batch, seq_len), dtype=np.int64)
        if "position_ids" in inputs:
            inputs["position_ids"] = np.arange(seq_len, dtype=np.int64).reshape(1, seq_len)

    return inputs


def profile_model(model_path: str) -> list:
    """Run profiling on a single model, return parsed profiling events."""
    sess_options = ort.SessionOptions()
    sess_options.enable_profiling = True
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1

    sess = ort.InferenceSession(model_path, sess_options, providers=["CPUExecutionProvider"])
    inputs = make_dummy_inputs(model_path)

    # Warmup
    for _ in range(WARMUP):
        sess.run(None, inputs)

    # Benchmark runs (profiling captures the last run's data)
    for _ in range(BENCHMARK):
        sess.run(None, inputs)

    # End profiling and get the file
    profile_file = sess.end_profiling()

    with open(profile_file, "r") as f:
        profile_data = json.load(f)

    # Clean up profile file
    os.remove(profile_file)

    return profile_data


def parse_profile(profile_data: list) -> dict:
    """Parse profiling JSON and aggregate by op_type."""
    op_times = defaultdict(list)

    for event in profile_data:
        if event.get("cat") == "Node" and "dur" in event:
            op_type = event.get("args", {}).get("op_name", "")
            if op_type:
                op_times[op_type].append(event["dur"])

    return op_times


def main():
    # Aggregate across all models
    global_op_times = defaultdict(list)

    for rel_path in MODELS:
        model_path = str(ROOT / rel_path)
        model_name = rel_path.split("/")[-3]
        print(f"Profiling {model_name}...")

        try:
            profile_data = profile_model(model_path)
            op_times = parse_profile(profile_data)
            for op_type, times in op_times.items():
                global_op_times[op_type].extend(times)
            print(f"  {len(op_times)} op types found")
        except Exception as e:
            print(f"  ERROR: {e}")

    # Build cost table
    cost_table = {}
    for op_type, times in global_op_times.items():
        mean_us = float(np.mean(times))
        count = len(times)
        total_us = float(np.sum(times))
        cost_table[op_type] = {
            "mean_us": round(mean_us, 2),
            "count": count,
            "total_us": round(total_us, 2),
        }

    # Sort by total_us descending
    cost_table = dict(
        sorted(cost_table.items(), key=lambda x: x[1]["total_us"], reverse=True)
    )

    # Print summary table
    print(f"\n{'='*70}")
    print(f"{'Op Type':<25} {'Mean (us)':>12} {'Count':>8} {'Total (us)':>14} {'%':>7}")
    print(f"{'-'*70}")
    grand_total = sum(v["total_us"] for v in cost_table.values())
    for op_type, stats in cost_table.items():
        pct = 100.0 * stats["total_us"] / grand_total if grand_total > 0 else 0
        print(f"{op_type:<25} {stats['mean_us']:>12.2f} {stats['count']:>8} {stats['total_us']:>14.2f} {pct:>6.1f}%")
    print(f"{'-'*70}")
    print(f"{'TOTAL':<25} {'':>12} {'':>8} {grand_total:>14.2f}")

    # Save cost table
    out_path = ROOT / "artifacts" / "superopt" / "op_cost_table.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(cost_table, f, indent=2)
    print(f"\nCost table saved to {out_path}")


if __name__ == "__main__":
    main()
