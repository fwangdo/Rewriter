# Rewriter

ONNX model을 target backend가 실행할 수 있는 graph로 낮추고, correctness와 latency를 기준으로 후보를 고르는 graph rewrite / superoptimization 프로젝트다.

핵심 질문은 다음이다.

> 지원 가능한 operator 집합이 제한된 backend에서, ONNX graph를 의미적으로 동등하면서 더 실행하기 좋은 형태로 바꿀 수 있는가?

일반적인 ONNX optimizer는 보통 실행 가능한 backend를 이미 가정하고 성능 최적화에 집중한다. 이 프로젝트는 그보다 앞단의 문제를 다룬다. NPU, edge runtime, vendor compiler, 제한된 inference runtime처럼 지원 op set이 좁거나 계속 바뀌는 환경에서 graph를 target contract에 맞게 legalize하고, 가능한 후보 중 실제로 맞고 빠른 것을 선택한다.

## Goals

- ONNX graph를 읽고 rewrite candidate를 생성한다.
- target backend의 supported-op contract를 모델링한다.
- rewrite 결과가 원본과 같은 출력을 내는지 ONNX Runtime으로 검증한다.
- cost model은 최종 답이 아니라 candidate ranking heuristic으로만 사용한다.
- 최종 선택은 checker, contract, ORT load, correctness, measured latency를 통과한 후보 중에서 한다.
- 더 나은 후보가 없으면 원본 또는 baseline을 유지하는 것도 정상 결과로 본다.

## Architecture

현재 코드는 두 축으로 나뉜다.

| 경로 | 역할 |
| --- | --- |
| `src/onnx_rewrite` | rule-based baseline. well-known ONNX graph rewrite를 직접 적용하고 audit / correctness / latency를 측정한다. |
| `src/superopt` | e-graph 기반 후보 생성. ONNX를 IR로 낮추고, egglog와 legacy callback bridge를 통해 rewrite space를 넓힌 뒤 candidate를 추출한다. |

Superopt의 큰 흐름은 다음과 같다.

```text
ONNX model
  -> pre-pass / shape inference
  -> ONNX -> IR
  -> callback legalization bridge
  -> egglog equality saturation
  -> cost-aware candidate extraction
  -> IR -> ONNX materialization
  -> contract / correctness / latency evaluation
```

`egglog`는 pure pattern rewrite의 saturation과 extraction에 사용한다. Python-side shape/value inspection이나 synthetic constant generation이 필요한 rule은 아직 `src/superopt/egraph`의 legacy bridge로 materialize한다.

## Target Contracts

지원 op contract는 `src/common/contracts.py`와 `src/superopt/contracts.py`에 있다.

| Contract | 용도 |
| --- | --- |
| `SUPPORTED_OPS` | scaffold / ORT CPU friendly union set |
| `VISION_SUPPORTED_OPS` | vision model target |
| `LLM_SUPPORTED_OPS` | decoder LLM target |
| `LLM_MUST_REMOVE_OPS` | LLM path에서 반드시 제거해야 하는 op |

현재 superopt extraction은 unsupported op에 큰 cost penalty를 주고 contract check 결과를 기록한다. 최종 candidate selection에서 contract를 hard gate로 완전히 닫는 작업은 진행 중이다.

## Benchmark Models

자세한 모델 정의는 [benchmark.md](benchmark.md)를 본다.

| # | Model | Domain | Target |
| --- | --- | --- | --- |
| 1 | MobileViT-XXS | Vision / hybrid | `VISION_SUPPORTED_OPS` |
| 2 | MobileNetV2 | Vision / CNN | `VISION_SUPPORTED_OPS` |
| 3 | YOLO26-Nano | Vision / detection | `VISION_SUPPORTED_OPS` |
| 4 | TinyLlama-15M | LLM / decoder | `LLM_SUPPORTED_OPS` |
| 5 | Pythia-70M | LLM / decoder | `LLM_SUPPORTED_OPS` |
| 6 | SmolLM-135M | LLM / decoder | `LLM_SUPPORTED_OPS` |

## Evaluation

Correctness와 latency는 ONNX Runtime CPU 기준으로 측정한다.

- `intra_op_num_threads=1`
- `inter_op_num_threads=1`
- `ORT_SEQUENTIAL`
- deterministic input generation
- domain별 tolerance
- median latency

주요 metric:

| 항목 | 의미 |
| --- | --- |
| Contract result | unsupported / must-remove op가 남았는지 |
| Correctness | output count, shape, dtype, value tolerance |
| Latency | original / rule baseline / ORT optimizer / superopt candidate 비교 |
| Candidate count | e-graph extraction에서 나온 후보 수 |
| Estimated cost | extraction heuristic 값. 최종 성능값으로 보지 않는다. |

## Usage

의존성 설치:

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Rule-based ONNX rewrite:

```bash
python3.11 -m src.onnx_rewrite.run_onnx_rewrite \
  --input benchmarks/onnx/vision/mobilenetv2/onnx/model.onnx \
  --output artifacts/rewrite/mobilenetv2.onnx
```

Rewrite + correctness + latency:

```bash
python3.11 -m src.onnx_rewrite.eval_rewrite \
  --input benchmarks/onnx/vision/mobilenetv2/onnx/model.onnx \
  --output artifacts/rewrite/mobilenetv2.onnx \
  --report artifacts/rewrite/mobilenetv2_eval.json
```

Superopt latency benchmark:

```bash
python3.11 -m src.superopt.bench_latency
```

## Current Status

이 프로젝트는 production optimizer가 아니라, 현업형 문제 정의를 가진 research / portfolio prototype이다.

잘 되어 있는 부분:

- ONNX rewrite baseline과 e-graph superopt path가 분리되어 있다.
- ONNX graph를 IR로 낮춘 뒤 다시 ONNX로 materialize하는 경로가 있다.
- egglog 기반 equality saturation을 main path로 사용한다.
- callback이 필요한 legalization rule을 별도 bridge로 처리한다.
- candidate를 실제 ORT correctness와 latency로 평가하는 방향이 잡혀 있다.
- CPU budget은 single-thread / sequential ORT 실행을 기본으로 둔다.

아직 닫아야 할 부분:

- contract violation을 최종 selection의 hard failure로 만드는 정책
- original / pre-pass를 candidate 0으로 포함하는 fallback
- materialized graph hash dedup
- 모델당 timeout과 node budget
- rule metadata와 correctness failure bisect
- 자동 regression test suite
- JSON / Markdown report schema

## Related Work

- **Tensat**: tensor graph superoptimization with equality saturation.
- **egglog**: e-graph / equality saturation backend.
- **ONNX Runtime graph optimizer**: baseline optimizer and execution reference.
- **ONNX Simplifier**: rule-based ONNX simplification baseline.

## Project Thesis

Graph rewrite는 단순히 op 수를 줄이는 작업이 아니다. 제한된 backend contract, numerical correctness, 실제 runtime latency 사이에서 실행 가능한 graph를 고르는 target adaptation 문제다.
