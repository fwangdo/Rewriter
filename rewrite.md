# Rewrite Baseline

이 문서는 현재 baseline rewrite가 어떤 well-known rule-based 변환을 포함하는지 정리한다.

## Current Passes

- `ConstantFolding`
  `Constant`, `ConstantOfShape`를 initializer로 접는다.
- `EliminateId`
  `Identity`를 제거하고 edge를 직접 연결한다.
- `RewriteClip`
  static `Clip`을 `Max` / `Min` 조합으로 내린다.
- `RewriteReshapeShape`
  `Shape -> Gather -> Unsqueeze -> Concat -> Reshape` shape-builder를 `Reshape` template initializer로 치환한다.
- `RewriteBN`
  `BatchNormalization`을 제거한다.
- `RewriteNeg`
  `Neg`를 `Mul(-1)`로 내린다.
- `RewritePow`
  scalar constant exponent를 `Mul`, `Sqrt`, `Div` 조합으로 내린다.
- `RewriteGemm`
  `Gemm`을 `Conv` 기반 1x1 lowering으로 바꾼다.
- `Cleanup`
  dead node 제거, unused initializer 제거, topological sort, ONNX checker 검증을 수행한다.

## Near-Term Must-Haves

- `RewriteGather`를 pipeline에 편입할지 재판단
- `RewriteMatmul`을 target contract가 실제로 금지하는 경우에만 재도입할지 재판단
- `LayerNorm` / `RMSNorm` 주변 산술 패턴 정리
- exact `GELU`를 correctness를 깨지 않고 내릴 방법 재검토
- `Transpose + Reshape + Unsqueeze + Squeeze` chain cleanup 강화
- attention mask 주변 `Cast / Equal / Where / Expand` 정리
- benchmark별 unsupported op histogram을 기준으로 missing rewrite를 계속 추가

## Policy

- baseline은 "supported op only 계약을 만족시키는 데 실질적으로 필요한 well-known rewrite"를 우선한다
- novelty보다 재현성과 coverage를 우선한다
- rewrite 추가 여부는 benchmark coverage와 ORT correctness/latency 결과로 판단한다
- 실전 target contract는 `VISION_SUPPORTED_OPS`와 `LLM_SUPPORTED_OPS`를 기준으로 잡고, 현재 `SUPPORTED_OPS`는 migration 중인 scaffold용 union set으로만 본다
- 특히 `LLM_SUPPORTED_OPS`는 dense math 최소 집합만 남기고 `Pow`, `Sin/Cos`, `Where`, `Expand`, `Unsqueeze`, `Shape`, `Trilu`, `ScatterND` 같은 op를 rewrite 대상으로 밀어 넣는 방향으로 유지한다
