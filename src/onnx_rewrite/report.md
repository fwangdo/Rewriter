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

- rewrite pipeline: `ConstantFolding -> RewriteGather -> EliminateId -> RewriteClip -> RewriteCompare -> RewriteMetaReshape -> RewriteReshapeShape -> RewriteLayerNorm -> RewriteBN -> RewriteNeg -> RewriteRange -> RewriteGemm -> RewriteDecoderMask -> RewriteWhereMask -> RewriteTrilu -> RewriteScatterND -> Cleanup`
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

### LLM (baseline target: `SUPPORTED_OPS`)

LLM 3종은 baseline `SUPPORTED_OPS` 기준 supported-op-only + correctness를 달성했다. strict `LLM_SUPPORTED_OPS`는 별도 후속 목표다.

| model | before nodes | after nodes | unsupported after | max abs diff | correctness |
| --- | ---: | ---: | --- | ---: | --- |
| `tinyllama_15m` | 1152 | — | `{}` | `0.0` | pass |
| `pythia_70m` | 589 | — | `{}` | `0.0` | pass |
| `smollm_135m` | 2844 | 2824 | `{}` | `0.0` | pass |

## Current Conclusion

> Benchmark 6종은 baseline `SUPPORTED_OPS` 기준 supported-op-only + correctness를 달성했다.

다음 우선순위:

1. strict `LLM_SUPPORTED_OPS` 기준 잔여 op를 별도 목표로 줄인다
2. latency 측정을 `warmup=5`, `repeat=20` 조건으로 다시 고정 측정한다
3. baseline rewrite를 regression test로 고정한다
