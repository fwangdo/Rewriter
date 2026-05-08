#!/bin/bash
cd "$(git rev-parse --show-toplevel)"
python3 -m src.superopt.run \
  -i benchmarks/onnx/vision/mobilenetv2/onnx/model.onnx \
  -o artifacts/superopt/mobilenetv2.onnx \
  --contract vision \
  -v
