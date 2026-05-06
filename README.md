# Legalization-Aware Graph Superoptimization for ONNX

## Overview

ONNX graph의 graph-level 최적화를 다루며, 특히 **변하는 operator 집합(op set)** 환경에서의 최적화를 연구한다.

> *허용되는 operator 집합이 바뀌는 상황에서, ONNX 그래프를 얼마나 효과적으로 "동등하지만 더 빠른 형태"로 변환할 수 있는가?*

기존 연구는 보통 고정된 backend를 가정하지만, 실제 환경에서는 NPU/Edge device, vendor-specific compiler, 제한된 runtime 등으로 인해 **지원 가능한 operator 집합이 계속 변화한다.**
본 프로젝트는 이 현실적인 제약을 직접 모델링한다.

## Problem Definition

**입력:**
- ONNX 그래프 집합 `G = {g1, g2, ..., gn}`
- Operator 제약 집합 `O = {O1, O2, ..., Ok}` — 각 Oi는 특정 타깃 환경에서 허용되는 operator 집합

**목표:** 각 그래프 g와 operator 집합 Oi에 대해, 다음을 만족하는 g'를 찾는다.

1. **Operator 제약 만족** — `ops(g') ⊆ Oi`
2. **의미적 동등성 유지** — `g'(x) ≈ g(x)` (ε tolerance)
3. **성능 최적화** — `latency(g')` 최소화

## Two-Stage Approach

### Stage 1: Rule-Based Baseline (현재 진행 중)

사전에 정의된 rewrite rule(Conv+BatchNorm folding, constant folding, LayerNorm decomposition, GELU lowering 등)을 적용한다.
빠르고 안정적이지만 커버리지가 제한된다.

이 단계의 목적은 **비교 대상(baseline)**을 만드는 것이다.
시니어 엔지니어가 구현할 법한 well-known rule-based rewrite를 충분히 갖추어, 이후 superoptimization의 개선폭을 정량적으로 측정할 수 있는 강한 baseline을 확보한다.

### Stage 2: E-Graph Superoptimization (핵심 기여)

E-graph(equality saturation) 기반 탐색으로, rule-based baseline과 동일한 입력에서 시작해 **유의미하게 더 좋은 동등 그래프**를 합리적인 시간 안에 찾는다.

핵심 차별점은 **legalization과 optimization을 동시에 수행**하는 것이다.
기존 e-graph 연구(Tensat, MLSys 2021)는 고정된 backend를 가정하고 optimization만 하지만, 본 연구는 operator 제약(legality)까지 탐색 공간에 포함시킨다.

## Evaluation

| 항목 | 측정 방법 |
|------|----------|
| 성능 | ONNX Runtime 기준 `latency(g') / latency(g)` |
| 성공률 | 주어진 op set에서 legalization 성공 여부 |
| 최적화 효과 | `latency(rule-based) / latency(superopt)` |
| 탐색 비용 | superopt 수행 시간 및 candidate 수 |

## Benchmark

6종 benchmark로 검증한다. 자세한 정의는 [benchmark.md](benchmark.md)를 참조.

| # | 모델 | 도메인 | Target Contract |
|---|---|---|---|
| 1 | MobileViT-XXS | Vision / hybrid | VISION_SUPPORTED_OPS |
| 2 | MobileNetV2 | Vision / CNN | VISION_SUPPORTED_OPS |
| 3 | YOLO26-Nano | Vision / detection | VISION_SUPPORTED_OPS |
| 4 | TinyLlama-15M | LLM / decoder | LLM_SUPPORTED_OPS |
| 5 | Pythia-70M | LLM / decoder | LLM_SUPPORTED_OPS |
| 6 | SmolLM-135M | LLM / decoder | LLM_SUPPORTED_OPS |

## Key Hypotheses

1. Rule-based 방식은 optimal을 보장하지 않는다
2. Operator 제약이 강할수록 superopt의 장점이 커진다
3. Legalization + optimization을 동시에 탐색하면 rule-based보다 더 좋은 동등 그래프를 발견할 수 있다
4. Graph rewrite는 단순 성능 최적화를 넘어서 **target adaptation** 역할을 한다

## How It Works

아키텍처, 실행 방법, 규칙 추가 방법 등은 [how.md](how.md)를 참조.

## Related Work

- **Tensat** (MLSys 2021): e-graph 기반 tensor graph superoptimization. 고정 backend 가정, legalization 없음.
- **Mind the Abstraction Gap** (OOPSLA 2025): equality saturation을 XLA compiler에 적용.
- **ONNX Simplifier**: rule-based ONNX 최적화. 본 프로젝트의 baseline과 유사한 역할.

---

> **Graph rewrite는 단순한 성능 최적화가 아니라, 변화하는 operator 환경에 대한 "적응(adaptation)" 문제이다.**
