# mobilenetv2

## Status

- artifact: `benchmarks/onnx/vision/mobilenetv2/onnx/model.onnx`
- actual ONNX opset: `18`
- current pipeline result: `supported-op-only` 달성
- correctness: `pass`

## Before Rewrite

- total nodes: `100`
- unsupported ops:
  - `Gemm=1`

핵심 관찰:

- 현재 pipeline의 union contract 기준으로 blocker는 classifier tail의 `Gemm` 하나였다.
- 실전형 vision contract 기준으로는 `Clip`도 추가 rewrite 대상이지만, 현재 scaffold에서는 `Clip`을 이미 허용 op로 둔 상태다.

## Applied Rewrite

- `RewriteClip`
  - `Clip -> Max + Min`
  - `ReLU6` 계열 경로를 primitive op로 정규화
- `RewriteGemm`
  - classifier tail의 `Gemm -> Conv` lowering
- `Cleanup`
  - unused initializer 제거
  - topological sort
  - ONNX checker 통과

## After Rewrite

- total nodes: `139`
- unsupported ops: `{}`

## Correctness

- runtime: `ONNX Runtime CPU`
- cases run: `9`
- worst case: `mask_random`
- tolerance: `1e-4`
- max abs diff: `5.773159728050814e-15`
- verdict: `pass`

해석:

- 다양한 seed / dynamic size 조합에서 원본과 rewritten 출력 차이는 machine epsilon 수준이다.
- `mobilenetv2`는 현재 baseline pipeline에서 end-to-end correctness를 만족한다.

## Latency

- runtime: `ONNX Runtime CPU`
- warmup: `5`
- repeat: `20`
- representative input: `seed=42`, `dynamic_size=1`

| model | median (ms) | p95 (ms) |
|---|---:|---:|
| original | 13.747 | 44.842 |
| rewritten | 15.974 | 17.008 |

- delta median: `+2.228 ms`
- speedup ratio: `0.861x`

## Conclusion

- `mobilenetv2`는 vision 3종 중 첫 end-to-end correctness 통과 모델이다.
- legalization은 성공했지만 latency는 소폭 악화됐다.
