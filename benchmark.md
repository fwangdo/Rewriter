# Benchmark 정의

Apple Silicon CPU + ONNX Runtime 환경에서 rewriter를 검증하기 위한
최종 benchmark 6종과 benchmark-driven logical opset을 정의한다.

여기서 "opset"은 두 층으로 나뉜다.

- `ONNX export opset`: 모델 artifact를 만들 때 맞추는 ONNX spec version
- `logical opset`: rewriter가 최종 graph에서 허용하는 operator 부분집합
- `domain supported ops`: 실전 accelerator를 상정한 도메인별 contract

이 문서의 기준은 다음과 같다.

- benchmark artifact의 기본 export 기준은 `ai.onnx opset 17`
- benchmark artifact의 최소 허용 기준은 `ai.onnx opset >= 13`
- benchmark coverage 관리는 아래 `logical opset` 5단계로 한다
- `BatchNormalization`은 입력 benchmark에는 등장해도, 최종 target opset에는 포함하지 않는다

## Current Status

현재 상태는 아래와 같다.

- vision 3종 `mobilenetv2`, `mobilevit_xxs`, `yolo26_nano`는 현재 baseline pipeline에서 ORT correctness를 통과했다
- vision 쪽 baseline rewrite는 `Clip`, `LayerNormalization`, `Gemm` 처리가 들어가 있다
- vision pipeline에서는 target contract가 허용하는 `MatMul`은 굳이 lowering하지 않는다
- LLM 쪽에서는 `Reshape` shape-builder cleanup을 추가해 strict unsupported histogram을 줄이기 시작했다
- 다음 우선순위는 LLM 3종에 대해 `LLM_SUPPORTED_OPS` 기준 legality와 correctness를 같이 확보하는 것이다

중요:

- LLM에서 "완료"의 의미는 `ops(rewritten_llm) ⊆ LLM_SUPPORTED_OPS`를 만족한 뒤 correctness를 통과하는 것이다
- union scaffold 기준 correctness만 통과한 상태는 LLM baseline 완료로 보지 않는다

## Domain Contracts

현재는 benchmark별 세부 contract보다, 먼저 아래 두 개의 실전형 domain contract를 더 중요하게 본다.

### Vision Supported Ops

```text
Add, AveragePool, Cast, Concat, Conv, Div, GlobalAveragePool,
HardSigmoid, HardSwish, MatMul, MaxPool, Mul, ReduceMean, Relu,
Reshape, Resize, Sigmoid, Slice, Softmax, Split, Sqrt, Squeeze,
Sub, Transpose
```

의도:

- 모바일 CNN, hybrid vision, detection graph에서 실제로 기대할 수 있는 연산만 남긴다
- `MatMul`, `Softmax`, `ReduceMean`은 MobileViT류 hybrid vision을 위해 허용한다
- 반대로 `Erf`, `TopK`, `Range`, `Tile`, `GatherElements`, `IsNaN`, `Sin/Cos`, `Where`, `Expand`는 vision target에서 제외한다

### LLM Supported Ops

```text
Add, Cast, Concat, Div, Gather, MatMul, Mul, ReduceMean,
Reshape, Sigmoid, Slice, Softmax, Sqrt, Sub, Tanh, Transpose
```

의도:

- decoder block의 핵심 dense math와 최소한의 tensor layout op만 남긴다
- `Tanh`는 GELU tanh approximation 계열을 위해 허용한다
- `RoPE`, causal mask, shape/index plumbing, cache update 같은 주변 op는 가능한 한 rewrite / folding / compiler-side lowering 대상으로 본다
- 그래서 `Sin`, `Cos`, `Pow`, `Range`, `Where`, `Expand`, `Unsqueeze`, `Squeeze`, `Shape`, `ConstantOfShape`, `Equal`, `Less`, `Neg`, `TopK`, `Trilu`, `ScatterND`는 LLM target contract에서 제외한다

이 설정은 아래 rewrite들이 실제로 필요해지도록 일부러 빡빡하게 잡은 것이다.

- `LayerNorm / RMSNorm decomposition`
- `GELU / SiLU / SwiGLU decomposition`
- `MatMul + bias canonicalization`
- `QKV attention canonicalization`
- `RoPE lowering / canonicalization`
- `causal mask construction rewrite`
- `shape / reshape / transpose plumbing cleanup`
- `constant folding / propagation`
- `Gather / embedding rewrite`
- `KV-cache update / index-scatter canonicalization`

현재 코드의 `SUPPORTED_OPS`는 여전히 scaffold용 union set이고,
실전 목표 contract는 위 `VISION_SUPPORTED_OPS`와 `LLM_SUPPORTED_OPS`다.

---

## 벤치마크 모델 (최종 6종)

| # | 모델 | 도메인 | 크기 | 기본 역할 |
|---|---|---|---:|---|
| 1 | MobileViT-XXS | Vision / hybrid CNN+ViT | 1.3M | hybrid vision stress |
| 2 | MobileNetV2 | Vision / depthwise CNN | 3.4M | mobile CNN baseline |
| 3 | YOLO26-Nano | Vision / detection | ~3M | detection topology stress |
| 4 | TinyLlama-15M | LLM / decoder (MHA+RoPE+RMSNorm+SwiGLU) | 15M | smallest decoder baseline |
| 5 | Pythia-70M | LLM / decoder (parallel attn, LayerNorm, GELU) | 70M | GPT-NeoX style decoder |
| 6 | SmolLM-135M | LLM / decoder (GQA) | 135M | modern decoder extension |

### 1. MobileViT-XXS

| 항목 | 값 |
|---|---|
| 출처 | `timm mobilevit_xxs` local export |
| 실제 artifact opset | 18 |
| 핵심 stress | Conv+BN, MatMul attention, LayerNorm 계열 산술, Reshape/Transpose 체인 |

**선정 근거**: 가장 작은 모델이지만 vision과 transformer operator vocabulary가 한 graph에 같이 들어간다. MobileNet류의 convolutional block과 attention block이 동시에 있으므로 hybrid rewrite 경계면을 보기 좋다.

### 2. MobileNetV2

| 항목 | 값 |
|---|---|
| 출처 | `torchvision mobilenet_v2` local export |
| 실제 artifact opset | 18 |
| 핵심 stress | depthwise Conv, residual Add, mobile CNN 최소 operator 집합 |

**선정 근거**: benchmark 전체의 가장 단순한 mobile CNN baseline이다. 복잡한 transformer 연산 없이 `Conv` 중심 legalization과 latency baseline을 제공한다.

### 3. YOLO26-Nano

| 항목 | 값 |
|---|---|
| 출처 | `onnx-community/yolo26n-ONNX` 또는 Ultralytics export mirror |
| 실제 artifact opset | 18 |
| 핵심 stress | Resize, Concat, Split, Sigmoid, detection head, 비선형 DAG |

**선정 근거**: 분류 모델이 아닌 detection graph를 추가해 feature pyramid, branch/merge, head split을 검증한다. MobileNetV2와 MobileViT가 주지 못하는 detection topology coverage를 담당한다.

### 4. TinyLlama-15M

| 항목 | 값 |
|---|---|
| 출처 | `nickypro/tinyllama-15M-fp32` ONNX export |
| 실제 artifact opset | 14 |
| 핵심 stress | rank-4 attention MatMul, RoPE (`Sin/Cos`), RMSNorm, SwiGLU |

**선정 근거**: 가장 작은 decoder baseline이다. decoder-only LLM에서 처음 필요한 `RoPE`, 4D attention layout, causal path를 가장 낮은 비용으로 검증한다.

### 5. Pythia-70M

| 항목 | 값 |
|---|---|
| 출처 | `Xenova/pythia-70m` ONNX export |
| 실제 artifact opset | 14 |
| 핵심 stress | GPT-NeoX 계열 parallel attention, LayerNorm, GELU |

**선정 근거**: LLaMA 계열과 다른 decoder family를 넣어 architecture bias를 줄인다. TinyLlama가 `RoPE+RMSNorm`을 커버한다면, Pythia는 `parallel attention + LayerNorm + GELU` 조합을 커버한다.

### 6. SmolLM-135M

| 항목 | 값 |
|---|---|
| 출처 | `onnx-community/SmolLM-135M-ONNX` |
| 실제 artifact opset | 14 |
| 핵심 stress | GQA, decoder scaling, modern small-LLM graph topology |

**선정 근거**: benchmark의 최종 modern decoder anchor다. TinyLlama와 Pythia가 각각 MHA 기반 baseline과 GPT-NeoX 계열 baseline을 준다면, SmolLM은 GQA가 들어간 최신 소형 decoder 경로를 담당한다.

---

## Rewrite Pass / Feature 커버리지

| 항목 | MobileViT-XXS | MobileNetV2 | YOLO26-Nano | TinyLlama-15M | Pythia-70M | SmolLM-135M |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Conv / mobile CNN baseline | o | **주** | o | - | - | - |
| Depthwise / inverted residual | o | **주** | - | - | - | - |
| Detection branch-merge topology | - | - | **주** | - | - | - |
| Resize / Concat / Split | - | - | **주** | - | - | - |
| Vision attention / hybrid block | **주** | - | - | - | - | - |
| LayerNorm 계열 산술 | **주** | - | - | - | **주** | o |
| Decoder MHA | - | - | - | **주** | o | - |
| RoPE (`Sin/Cos`) | - | - | - | **주** | - | - |
| RMSNorm / SwiGLU 계열 | - | - | - | **주** | - | o |
| Parallel attention + GELU | - | - | - | - | **주** | - |
| GQA | - | - | - | - | - | **주** |

`주`는 해당 benchmark의 primary stressor이고, `o`는 부수 커버리지를 뜻한다.

---

## Benchmark Artifact Policy

### 공통 ONNX export 기준

- 기본 기준은 `ai.onnx opset 17`
- 최소 허용 기준은 `ai.onnx opset >= 13`
- 이미 검증된 공개 ONNX artifact가 있으면 그대로 사용 가능
- 공개 artifact가 opset 13 미만이면 local re-export로 교체한다
- 단, logical opset 평가는 ONNX spec version이 아니라 최종 op histogram으로 판단

### 왜 opset 17로 통일하는가

1. decoder benchmark 3종을 하나의 export 기준으로 관리하기 쉽다
2. modern ONNX exporter가 내놓는 shape/masking 패턴을 지나치게 오래된 opset으로 왜곡하지 않는다
3. YOLO26와 최신 HF ONNX mirror들이 이미 이 근처 기준으로 관리되는 경우가 많다

---

## Logical Opset 5단계

`Constant`는 모든 단계에서 암묵적으로 허용한다고 본다. 아래 목록은 benchmark 분류에 의미 있는 차별 op만 적는다.

### Opset 1: Mobile-CNN

```
Conv, Add, Relu, AveragePool, GlobalAveragePool, Flatten, Reshape, Gemm
```

**대상**: MobileNetV2 같은 순수 mobile CNN baseline

**의미**: 가장 작은 분류형 vision contract다. depthwise Conv와 residual Add는 들어가지만 detection/transformer 구조는 없다.

### Opset 2: Detection-Vision

```
Opset 1 + MaxPool, Mul, Sigmoid, Pad, Slice, Concat, Resize, Split, Transpose
```

**대상**: YOLO26-Nano

**의미**: detection neck/head에서 필요한 branch/merge와 spatial resize를 추가한다. 이 단계부터 단순 직렬 CNN이 아니라 DAG 구조를 다룬다.

### Opset 3: Hybrid-Transformer

```
Opset 2 + MatMul, Softmax, ReduceMean, Sub, Div, Sqrt, Shape,
         Gather, Unsqueeze, Squeeze, Erf, Gelu
```

**대상**: MobileViT-XXS

**의미**: vision benchmark 안에 transformer block이 섞이기 시작하는 단계다. attention과 LayerNorm 계열 산술이 처음 등장한다.

### Opset 4: Decoder-Core

```
Opset 3 + Cast, Equal, Expand, Neg, Where, Sin, Cos, Tanh, Pow
```

**대상**: TinyLlama-15M, Pythia-70M

**의미**: decoder-only LLM의 공통 기반이다. RoPE, RMSNorm/LayerNorm 파생 산술, causal masking을 포함한다.

### Opset 5: Full-Benchmark

```
Opset 4 + Min, Max, Range, ConstantOfShape, ReduceSum, ReduceMax,
         Less, LessOrEqual, Mod, TopK, Tile, GatherElements,
         HardSigmoid, HardSwish, IsNaN, LeakyRelu
```

**대상**: SmolLM-135M까지 포함한 전체 benchmark contract

**의미**: small modern LLM과 일부 mobile/exporter 잔여 op까지 포함한 최종 지원 집합이다. 현재 코드의 `SUPPORTED_OPS`는 이 단계를 기준으로 둔다.

### Opset 계층 요약

| Opset | 대표 모델 | 초점 |
|---|---|---|
| 1. Mobile-CNN | MobileNetV2 | 최소 mobile CNN |
| 2. Detection-Vision | YOLO26-Nano | detection DAG |
| 3. Hybrid-Transformer | MobileViT-XXS | vision + transformer 혼합 |
| 4. Decoder-Core | TinyLlama-15M, Pythia-70M | decoder 기본 경로 |
| 5. Full-Benchmark | SmolLM-135M | 전체 benchmark contract |

### Current LLM Progress Snapshot

- `tinyllama_15m`
  - correctness는 현재 pipeline에서 유지된다
  - strict unsupported는 `Shape: 49 -> 37`, `Unsqueeze: 111 -> 63`까지 감소
  - 남은 핵심은 causal mask subgraph
- `pythia_70m`
  - correctness는 현재 pipeline에서 유지된다
  - strict unsupported는 `Shape: 27 -> 15`, `Unsqueeze: 52 -> 27`까지 감소
  - 남은 핵심은 exact GELU의 `Erf`와 causal mask subgraph
- `smollm_135m`
  - 아직 strict legality / correctness 모두 미완료
  - `RoPE`, `Trilu`, `ScatterND`, GQA mask plumbing이 남아 있다

---

## 구현 메모

- benchmark model catalog와 logical opset의 코드 source-of-truth는 `src/onnx_rewrite/specs/catalog.py`에 둔다
- `VISION_SUPPORTED_OPS`와 `LLM_SUPPORTED_OPS`가 실전 target contract다
- 현재 `SUPPORTED_OPS`는 migration 중인 scaffold용 union set이다
- `BatchNormalization`은 benchmark 입력에서 허용되지만 rewrite 완료 graph에는 남아 있지 않아야 한다
