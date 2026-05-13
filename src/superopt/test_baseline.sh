#!/bin/bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

PYTHON_BIN="${PYTHON:-.venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="python3"
fi

"$PYTHON_BIN" -m src.baseline.ir_manual \
  -i benchmarks/onnx/vision/mobilenetv2/onnx/model.onnx \
  -o artifacts/superopt/mobilenetv2_ir_baseline.onnx \
  -v
