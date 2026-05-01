# pythia_70m

## Status

- artifact: `benchmarks/onnx/nlp/pythia_70m/onnx/model.onnx`
- actual ONNX opset: `14`
- strict target contract: `LLM_SUPPORTED_OPS`
- current strict pipeline result: `not yet supported-op-only`
- current correctness status: `union pipeline pass`, `strict legality pending`

## Before Rewrite

- total nodes: `589`
- strict unsupported ops:
  - `ConstantOfShape=2`
  - `Equal=2`
  - `Erf=6`
  - `Expand=2`
  - `Less=1`
  - `Neg=12`
  - `Pow=7`
  - `Range=1`
  - `Shape=27`
  - `Squeeze=2`
  - `Unsqueeze=52`
  - `Where=5`

핵심 관찰:

- `Neg`, `Pow`와 shape-builder 일부는 잘 정리된다.
- 그러나 strict contract 기준으로는 `Erf` 기반 exact GELU와 causal mask 경로가 그대로 남는다.

## Applied Rewrite

- `RewriteReshapeShape`
  - `Shape -> Gather -> Unsqueeze -> Concat -> Reshape`를 template initializer로 치환
  - 대표 template:
    - `[0, -1]`
    - `[0, 0, 8, 192]`
    - `[0, 0, 512]`
- `RewriteNeg`
  - `Neg -> Mul(-1)`
- `RewritePow`
  - LayerNorm 경로의 `Pow(x, 2)`를 `Mul(x, x)`로 치환
- `RewriteRange`
  - `Range(0, limit, 1)`를 precomputed arange table + dynamic `Slice`로 치환
- `RewriteDecoderMask`
  - GPT-NeoX 계열의 `Where + Expand` attention mask 경로를 arithmetic broadcast mask로 치환
- `Cleanup`
  - dead node 제거
  - unused initializer 제거
  - topological sort
  - ONNX checker 통과

## After Rewrite

- total nodes: `564`
- strict unsupported ops:
  - `ConstantOfShape=2`
  - `Erf=6`
  - `Less=1`
  - `Equal=2`
  - `Shape=22`
  - `Unsqueeze=2`
- practical must-remove ops:
  - none

해석:

- `Where / Expand / Range`는 제거되어 현재 `LLM_MUST_REMOVE_OPS`는 0이 됐다.
- `Unsqueeze`는 `52 -> 2`까지 줄었다.
- 다만 `Erf=6`이 그대로 남아 있어, strict legality는 아직 미완료다.
- strict contract의 남은 핵심은 exact `GELU`, `Shape`, `ConstantOfShape`, `Equal`, `Less`다.

## Correctness

- runtime: `ONNX Runtime CPU`
- cases run: `8`
- worst case: `low_band_vocab`
- max abs diff: `0.0006103515625`
- max rel diff: `5.737878723266476e-07`
- verdict: `pass`

중요:

- 현재 correctness는 union pipeline 기준으로는 유지된다.
- 하지만 strict `LLM_SUPPORTED_OPS` legality가 남아 있으므로 baseline 완료는 아니다.

## Conclusion

- pythia는 practical 기준에서 1차 목표를 달성했다.
- 다음 strict blocker는 exact `GELU`의 `Erf`와 shape/meta 축이다.
