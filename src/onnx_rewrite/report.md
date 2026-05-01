# ONNX Rewrite Report

## Scope

이 문서는 `src/onnx_rewrite` rewrite pipeline의 현재 성능/정확도 상태를 정리한다.

측정 대상 모델은 현재 benchmark 6종이다.

- `mobilenetv2`
- `mobilevit_xxs`
- `yolo26_nano`
- `tinyllama_15m`
- `pythia_70m`
- `smollm_135m`

측정 조건:

- rewrite pipeline: `ConstantFolding -> EliminateId -> RewriteClip -> RewriteGather -> RewriteMetaReshape -> RewriteReshapeShape -> RewriteBN -> RewriteNeg -> RewritePow -> RewriteGemm -> RewriteLayerNorm -> Cleanup`
- runtime: ONNX Runtime CPU
- latency setting: `warmup=5`, `repeat=20`
- correctness gate: `max_abs_diff <= 1e-4`

모델별 상세 report는 [report/](../../report/) 디렉토리의 개별 파일을 참조.

## Summary

### Vision (target: `VISION_SUPPORTED_OPS`)

Vision 3종은 모두 `supported-op-only` 달성, correctness 통과.

| model | before nodes | after nodes | unsupported before | unsupported after | max abs diff | correctness | speedup |
| --- | ---: | ---: | --- | --- | ---: | --- | ---: |
| `mobilenetv2` | 100 | 139 | `Gemm=1` | `{}` | `5.77e-15` | pass | 0.861x |
| `mobilevit_xxs` | 394 | — | `LayerNorm=21, Gemm=1` | `{}` | `1.19e-07` | pass | 1.035x |
| `yolo26_nano` | 390 | — | `{}` | `{}` | `0.0` | pass | 1.002x |

### LLM (target: `LLM_SUPPORTED_OPS`)

LLM 3종은 아직 strict legality 미완료. union scaffold 기준 correctness는 확보.

| model | before nodes | after nodes | key unsupported after | correctness (union) | strict legality |
| --- | ---: | ---: | --- | --- | --- |
| `tinyllama_15m` | 1152 | 743 | `Shape=42, Unsqueeze=14, Where=5, ...` | pass | pending |
| `pythia_70m` | 589 | 514 | `Erf=6, Shape=15, Unsqueeze=27, Where=5, ...` | pass | pending |
| `smollm_135m` | 2844 | 2574 | `Sin=1, Cos=1, Trilu=1, ScatterND=1, Equal=64, Where=64, ...` | not verified | pending |

## Current Conclusion

> Vision 3종은 baseline 완료. LLM 3종은 union-contract correctness까지 확보했으나, strict `LLM_SUPPORTED_OPS` legality는 아직 남아 있다.

다음 우선순위:

1. `tinyllama_15m` / `pythia_70m`의 causal mask + GELU rewrite로 strict legality 달성
2. `smollm_135m`의 RoPE / Trilu / ScatterND blocker 정리
3. strict legality 달성 후 LLM correctness / latency 재측정
