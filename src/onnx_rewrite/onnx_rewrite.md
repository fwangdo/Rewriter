# ONNX Rewrite Module

## What "Done" Means

이 모듈은 아래 조건을 모두 만족해야 완료로 본다.

- benchmark 6종이 준비되어 있다
- vision 3종은 `VISION_SUPPORTED_OPS` 계약을 만족한다
- LLM 3종은 `LLM_SUPPORTED_OPS` 계약을 만족한다
- lowering 전후 correctness 비교가 된다
- rewrite 전후 latency 비교가 된다
- 모델별 unsupported op before/after 표가 있다
- README와 report만 읽어도 문제 정의, 접근 방식, 한계가 선명하다

## Domain Contracts

### Vision Supported Ops

```text
Add, AveragePool, Cast, Concat, Conv, Div, GlobalAveragePool,
HardSigmoid, HardSwish, MatMul, MaxPool, Mul, ReduceMean, Relu,
Reshape, Resize, Sigmoid, Slice, Softmax, Split, Sqrt, Squeeze,
Sub, Transpose
```

### LLM Supported Ops

```text
Add, Cast, Concat, Div, Gather, MatMul, Mul, ReduceMean,
Reshape, Sigmoid, Slice, Softmax, Sqrt, Sub, Tanh, Transpose
```

LLM contract는 dense math 최소 집합만 남기고, `Pow`, `Sin/Cos`, `Where`, `Expand`, `Unsqueeze`, `Shape` 같은 op를 rewrite 대상으로 밀어 넣는 방향으로 빡빡하게 잡았다.

## Benchmark Models

| # | 모델 | 도메인 | Target Contract |
|---|---|---|---|
| 1 | mobilenetv2 | Vision / CNN | VISION_SUPPORTED_OPS |
| 2 | mobilevit_xxs | Vision / hybrid | VISION_SUPPORTED_OPS |
| 3 | yolo26_nano | Vision / detection | VISION_SUPPORTED_OPS |
| 4 | tinyllama_15m | LLM / decoder | LLM_SUPPORTED_OPS |
| 5 | pythia_70m | LLM / decoder | LLM_SUPPORTED_OPS |
| 6 | smollm_135m | LLM / decoder | LLM_SUPPORTED_OPS |

모델 catalog source-of-truth: [specs/catalog.py](specs/catalog.py)

## Immediate Order of Work

1. vision 3종 supported-op-only + correctness + latency baseline 확보 (완료)
2. LLM 3종 strict `LLM_SUPPORTED_OPS` legality + correctness 확보 (진행 중)
3. 전체 6종 baseline report 정리
4. e-graph superoptimization stage 진입

## Rule For Baseline

baseline rewrite pipeline의 현재 핵심 규칙:

> 최종 ONNX graph는 해당 domain의 supported op만 포함해야 한다.

- rewrite 이후 unsupported op가 하나라도 남으면 strict legality 미달
- `onnx.checker.check_model`이 실패하면 pipeline 실패
