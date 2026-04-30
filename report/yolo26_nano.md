# yolo26_nano

## Status

- artifact: `benchmarks/onnx/vision/yolo26_nano/onnx/model.onnx`
- actual ONNX opset: `18`
- current pipeline result: `supported-op-only` 유지
- correctness: `pass`

## Before Rewrite

- total nodes: `390`
- unsupported ops: `{}`

핵심 관찰:

- 현재 scaffold union contract 기준으로 원본 graph는 이미 supported-op-only였다.
- 초기에 `RewriteMatmul`을 적용했을 때 correctness drift가 발생했지만, vision pipeline에서 `MatMul` lowering을 제거한 뒤 정확히 일치했다.

## Applied Rewrite

- 실질적인 legalization rewrite 없음
- `Cleanup`
  - graph normalization
  - ONNX checker 통과

## Correctness

- runtime: `ONNX Runtime CPU`
- cases run: `8`
- worst case: `baseline`
- tolerance: `1e-4`
- max abs diff: `0.0`
- verdict: `pass`

해석:

- 현재 pipeline에서는 원본과 rewritten 모델의 출력이 케이스 전부에서 정확히 일치한다.
- detection 모델에서 불필요한 `MatMul` lowering을 하지 않는 것이 더 안전했다.

## Latency

- runtime: `ONNX Runtime CPU`
- warmup: `5`
- repeat: `20`
- representative input: `seed=42`, `dynamic_size=1`

| model | median (ms) | p95 (ms) |
|---|---:|---:|
| original | 89.821 | 91.648 |
| rewritten | 89.611 | 97.303 |

- delta median: `-0.210 ms`
- speedup ratio: `1.002x`

## Conclusion

- `yolo26_nano`는 현재 vision pipeline에서 correctness를 유지한다.
- detection graph에서는 target contract가 허용하는 op를 굳이 추가 lowering하지 않는 쪽이 baseline으로 더 적절했다.
