"""Interactive LLM shell for ONNX decoder models.

Usage:
    python3 -m src.test.llm_shell --model artifacts/superopt/tinyllama_15m.onnx
    python3 -m src.test.llm_shell --model artifacts/superopt/pythia_70m.onnx --tokenizer EleutherAI/pythia-70m
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

# Benchmark model name -> HuggingFace tokenizer
_TOKENIZER_MAP = {
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "smollm": "HuggingFaceTB/SmolLM-135M",
    "pythia": "EleutherAI/pythia-70m",
}


def _guess_tokenizer(model_path: str) -> str | None:
    path_lower = model_path.lower()
    for key, hf_name in _TOKENIZER_MAP.items():
        if key in path_lower:
            return hf_name
    return None


def _parse_kv_structure(sess: ort.InferenceSession) -> tuple[int, int, int]:
    """Detect (num_layers, num_heads, head_dim) from past_key_values inputs."""
    layers, heads, head_dim = 0, 0, 0
    for inp in sess.get_inputs():
        if inp.name.startswith("past_key_values.") and inp.name.endswith(".key"):
            idx = int(inp.name.split(".")[1])
            layers = max(layers, idx + 1)
            # shape: [batch, num_heads, past_len, head_dim]
            heads = inp.shape[1]
            head_dim = inp.shape[3]
    return layers, heads, head_dim


def generate(
    sess: ort.InferenceSession,
    input_ids: np.ndarray,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    max_new_tokens: int = 200,
    eos_token_id: int | None = None,
) -> list[int]:
    kv = {}
    for i in range(num_layers):
        kv[f"past_key_values.{i}.key"] = np.zeros(
            (1, num_heads, 0, head_dim), dtype=np.float32
        )
        kv[f"past_key_values.{i}.value"] = np.zeros(
            (1, num_heads, 0, head_dim), dtype=np.float32
        )

    cur_ids = input_ids
    past_len = 0
    generated: list[int] = []

    for _ in range(max_new_tokens):
        cur_len = cur_ids.shape[1]
        feeds = {
            "input_ids": cur_ids,
            "attention_mask": np.ones((1, past_len + cur_len), dtype=np.int64),
            "position_ids": np.arange(
                past_len, past_len + cur_len, dtype=np.int64
            ).reshape(1, -1),
        }
        feeds.update(kv)

        outputs = sess.run(None, feeds)
        logits = outputs[0]
        next_token = int(np.argmax(logits[0, -1, :]))
        generated.append(next_token)

        if eos_token_id is not None and next_token == eos_token_id:
            break

        for i in range(num_layers):
            kv[f"past_key_values.{i}.key"] = outputs[1 + i * 2]
            kv[f"past_key_values.{i}.value"] = outputs[2 + i * 2]
        past_len += cur_len
        cur_ids = np.array([[next_token]], dtype=np.int64)

    return generated


def main():
    parser = argparse.ArgumentParser(description="Interactive LLM shell")
    parser.add_argument("--model", required=True, help="Path to ONNX model")
    parser.add_argument("--tokenizer", default=None, help="HuggingFace tokenizer name")
    parser.add_argument("--max-tokens", type=int, default=200)
    args = parser.parse_args()

    tok_name = args.tokenizer or _guess_tokenizer(args.model)
    if tok_name is None:
        print("Cannot guess tokenizer. Use --tokenizer to specify.")
        sys.exit(1)

    print(f"Loading tokenizer: {tok_name}")
    tokenizer = AutoTokenizer.from_pretrained(tok_name)

    print(f"Loading model: {args.model}")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(args.model, opts, providers=["CPUExecutionProvider"])

    num_layers, num_heads, head_dim = _parse_kv_structure(sess)
    print(f"Model: {num_layers} layers, {num_heads} heads, {head_dim} head_dim")
    print(f"Max new tokens: {args.max_tokens}")
    print("Type a prompt and press Enter. Ctrl+D to exit.\n")

    while True:
        try:
            prompt = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not prompt.strip():
            continue

        input_ids = tokenizer.encode(prompt, return_tensors="np").astype(np.int64)

        import time

        t0 = time.time()
        tokens = generate(
            sess,
            input_ids,
            num_layers,
            num_heads,
            head_dim,
            max_new_tokens=args.max_tokens,
            eos_token_id=tokenizer.eos_token_id,
        )
        elapsed = time.time() - t0

        text = tokenizer.decode(tokens, skip_special_tokens=True)
        tok_per_sec = len(tokens) / elapsed if elapsed > 0 else 0
        print(f"{text}")
        print(f"  [{len(tokens)} tokens, {elapsed:.2f}s, {tok_per_sec:.0f} tok/s]\n")


if __name__ == "__main__":
    main()
