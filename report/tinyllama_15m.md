# tinyllama_15m

## Status

- artifact: `benchmarks/onnx/nlp/tinyllama_15m/onnx/model.onnx`
- actual ONNX opset: `14`
- strict target contract: `LLM_SUPPORTED_OPS`
- current strict pipeline result: `not yet supported-op-only`
- current correctness status: `union pipeline pass`, `strict legality pending`

## Before Rewrite

- total nodes: `1152`
- strict unsupported ops:
  - `ConstantOfShape=4`
  - `Equal=2`
  - `Expand=2`
  - `Less=1`
  - `Neg=12`
  - `Pow=13`
  - `Range=1`
  - `Shape=49`
  - `Squeeze=2`
  - `Unsqueeze=111`
  - `Where=5`

핵심 관찰:

- 현재 tinyllama blocker는 두 부류다.
- `Neg`, `Pow` 같은 산술 rewrite는 표준 pass로 바로 줄어든다.
- 남는 큰 덩어리는 causal mask와 shape plumbing이다.

## Applied Rewrite

- `ConstantFolding`
  - `Constant` 393개를 initializer로 fold
- `RewriteReshapeShape`
  - `Shape -> Gather -> Unsqueeze -> Concat -> Reshape` 패턴을 `Reshape` template initializer로 치환
  - 대표 template:
    - `[0, 0, 6, 48]`
    - `[0, 0, 288]`
- `RewriteNeg`
  - `Neg -> Mul(-1)`
- `RewritePow`
  - norm 경로의 `Pow(x, 2)`를 `Mul(x, x)`로 치환
- `Cleanup`
  - dead node 제거
  - unused initializer 제거
  - topological sort
  - ONNX checker 통과

## After Rewrite

- total nodes: `663`
- strict unsupported ops:
  - `ConstantOfShape=4`
  - `Equal=2`
  - `Expand=2`
  - `Less=1`
  - `Range=1`
  - `Shape=37`
  - `Squeeze=2`
  - `Unsqueeze=63`
  - `Where=5`

해석:

- 이번 턴의 핵심 진전은 shape-builder cleanup이다.
- `Shape: 49 -> 37`, `Unsqueeze: 111 -> 63`으로 줄였지만, strict LLM legality까지는 아직 멀다.

## Correctness

- runtime: `ONNX Runtime CPU`
- cases run: `8`
- worst case: `low_band_vocab`
- max abs diff: `3.9696693420410156e-05`
- max rel diff: `3.6597251892089844e-05`
- verdict: `pass`

중요:

- 이 correctness는 현재 pipeline rewrite 결과에 대한 ORT 비교다.
- 하지만 strict `LLM_SUPPORTED_OPS` legality가 아직 남아 있으므로 baseline 완료로 보지 않는다.

## Conclusion

- tinyllama는 산술 축은 정리됐고, 이제 남은 핵심은 `Range / Less / Where / Expand / ConstantOfShape`로 이어지는 causal mask subgraph다.
- 다음 단계는 이 마스크 경로를 표준적인 decoder mask rewrite로 정리하는 것이다.
