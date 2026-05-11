#!/bin/bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

PYTHON_BIN="${PYTHON:-.venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="python3"
fi

"$PYTHON_BIN" -m src.superopt.run \
  -i benchmarks/onnx/vision/mobilenetv2/onnx/model.onnx \
  -o artifacts/superopt/mobilenetv2.onnx \
  --contract vision \
  -v
