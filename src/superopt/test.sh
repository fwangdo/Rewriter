#!/bin/bash
cd "$(git rev-parse --show-toplevel)"
python3 -m src.superopt.run \
  -i benchmarks/onnx/nlp/tinyllama_15m/onnx/model.onnx \
  -o artifacts/superopt/tinyllama_15m.onnx \
  --contract llm \
  -v
