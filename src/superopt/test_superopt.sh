#!/bin/bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

PYTHON_BIN="${PYTHON:-.venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="python3"
fi

"$PYTHON_BIN" -m src.superopt.run \
  -i benchmarks/onnx/nlp/pythia_70m/onnx/model.onnx \
  -o artifacts/superopt/pythia_70m.onnx \
  --contract vision
