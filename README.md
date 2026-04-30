# ONNX Graph Superoptimization under Operator Constraints

## 개요

ONNX 그래프의 graph-level 최적화를 다루며, 특히 **변하는 operator 집합(op set)** 환경에서의 최적화를 연구한다.
핵심은 기존 rule-based graph rewriting의 한계를 넘어서는 **search 기반 superoptimization**을 개발하는 것이다.

> *허용되는 operator 집합이 바뀌는 상황에서, ONNX 그래프를 얼마나 효과적으로 "동등하지만 더 빠른 형태"로 변환할 수 있는가?*

기존 연구는 보통 고정된 backend를 가정하지만, 실제 환경에서는 NPU/Edge device, vendor-specific compiler, 제한된 runtime 등으로 인해 **지원 가능한 operator 집합이 계속 변화한다.**
본 프로젝트는 이 현실적인 제약을 직접 모델링한다.

## 문제 정의

**입력:**
- ONNX 그래프 집합 `G = {g₁, g₂, ..., gₙ}`
- Operator 제약 집합 `O = {O₁, O₂, ..., Oₖ}` — 각 Oᵢ는 특정 타깃 환경에서 허용되는 operator 집합

**목표:** 각 그래프 g와 operator 집합 Oᵢ에 대해, 다음을 만족하는 g′를 찾는다.

1. **Operator 제약 만족** — `ops(g') ⊆ Oᵢ`
2. **의미적 동등성 유지** — `g'(x) ≈ g(x)` (ε tolerance)
3. **성능 최적화** — `latency(g')` 최소화

## 접근 방법

### Rule-based Optimization

사전에 정의된 rewrite rule(Conv+BatchNorm folding, constant folding, 간단한 fusion 등)을 적용한다.
빠르고 안정적이지만 커버리지가 제한된다.

### Superoptimization (Search-based)

가능한 graph transformation 공간을 탐색하여 다양한 후보 graph를 생성하고, cost 기반으로 선택한다.
더 넓은 탐색 공간에서 새로운 rewrite를 발견할 수 있지만 계산 비용이 높다.

## 평가 방법

| 항목 | 측정 방법 |
|------|----------|
| 성능 | ONNX Runtime 기준 `latency(g') / latency(g)` |
| 성공률 | 주어진 op set에서 legalization 성공 여부 |
| 최적화 효과 | `latency(rule-based) / latency(superopt)` |
| 탐색 비용 | superopt 수행 시간 및 candidate 수 |

## 실험 시나리오

- **Operator 제한**: 특정 operator 제거 (예: BatchNorm, ReduceMean 제거)
- **Backend 특화**: CPU-friendly vs NPU-friendly op set 비교
- **극단적 제약**: Elementwise-only, MatMul-only 등 극단적 제한 환경

## 핵심 가설

1. Rule-based 방식은 optimal을 보장하지 않는다
2. Operator 제약이 강할수록 superopt의 장점이 커진다
3. Superopt는 새로운 rewrite 패턴을 발견할 수 있다
4. Graph rewrite는 단순 성능 최적화를 넘어서 **target adaptation** 역할을 한다

## 기대 효과

- Operator 제약 환경에서의 graph rewriting 문제 정식화
- Rule-based vs Superopt의 정량적 비교
- 새로운 ONNX graph rewrite 패턴 발견
- Target-aware graph optimization의 필요성 제시

## 향후 확장

- e-graph 기반 rewrite 시스템 적용
- 자동 rewrite rule 추출 (offline superopt → rule mining)
- Backend-aware cost model 개선
- MLIR / custom IR로 확장

---

> **Graph rewrite는 단순한 성능 최적화가 아니라, 변화하는 operator 환경에 대한 "적응(adaptation)" 문제이다.**
