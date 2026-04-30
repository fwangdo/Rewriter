# mobilevit_xxs

## Status

- artifact: `benchmarks/onnx/vision/mobilevit_xxs/onnx/model.onnx`
- actual ONNX opset: `18`
- current pipeline result: `supported-op-only` 달성
- correctness: `pass`

## Before Rewrite

- total nodes: `394`
- unsupported ops:
  - `LayerNormalization=21`
  - `Gemm=1`

핵심 관찰:

- hybrid vision graph의 핵심 blocker는 transformer block 내부 `LayerNormalization`이었다.
- `MatMul`은 vision contract에서 허용하므로, 현재 baseline vision pipeline에서는 그대로 유지한다.

## Applied Rewrite

- `RewriteLayerNorm`
  - `LayerNormalization -> ReduceMean + Sub + Mul + Add + Sqrt + Div + Mul + Add`
- `RewriteGemm`
  - classifier tail의 `Gemm -> Conv` lowering
- `Cleanup`
  - unused initializer 제거
  - topological sort
  - ONNX checker 통과

## After Rewrite

- unsupported ops: `{}`

## Correctness

- runtime: `ONNX Runtime CPU`
- cases run: `8`
- worst case: `mask_random`
- tolerance: `1e-4`
- max abs diff: `1.1920928955078125e-07`
- verdict: `pass`

해석:

- layer norm decomposition 이후에도 출력 차이는 tolerance보다 훨씬 작다.
- `mobilevit_xxs`는 hybrid vision benchmark로서 end-to-end correctness를 만족한다.

## Latency

- runtime: `ONNX Runtime CPU`
- warmup: `5`
- repeat: `20`
- representative input: `seed=42`, `dynamic_size=1`

| model | median (ms) | p95 (ms) |
|---|---:|---:|
| original | 14.049 | 30.064 |
| rewritten | 13.573 | 13.887 |

- delta median: `-0.476 ms`
- speedup ratio: `1.035x`

## Conclusion

- `mobilevit_xxs`는 현재 pipeline에서 supported-op-only와 correctness를 모두 만족한다.
- 현재 측정에서는 latency도 소폭 개선됐다.
