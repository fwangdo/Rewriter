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

코드는 세 축으로 나뉜다.

| 경로 | 역할 |
| --- | --- |
| `src/common/rules` | 공유 rewrite 규칙. `get_all_specs()`로 42개 RuleSpec을 로드하며 baseline과 superopt 모두 이 함수를 단일 진실 공급원으로 사용한다. |
| `src/onnx_rewrite` | rule-based baseline. ONNX 레벨에서 ConstantFolding → RuleRunner → Cleanup 파이프라인을 적용한다. |
| `src/superopt` | e-graph 기반 superopt. ONNX를 IR로 낮추고, equality saturation으로 rewrite space를 탐색한 뒤 cost-aware extraction으로 candidate를 추출한다. |

Superopt의 큰 흐름은 다음과 같다.

```text
ONNX model
  -> pre-pass (ConstantFolding)
  -> ONNX -> IR
  -> e-graph equality saturation (shared RuleSpecs)
  -> cost-aware greedy extraction
  -> IR -> ONNX materialization
  -> post-pass (ConstantFolding, shape inference, Cleanup)
  -> contract check
```

## Shared Rewrite Rules

baseline과 superopt는 `src/common/rules/`에 정의된 동일한 42개 규칙을 공유한다.
전체 목록은 [docs/rules.md](docs/rules.md)를 본다.

| Category | Count | 예시 |
| --- | --- | --- |
| Legalization | 34 | `neg_to_mul`, `layernorm_decompose`, `gemm_decompose`, `matmul_to_conv` |
| Arithmetic | 4 | `add_comm`, `mul_comm`, `add_assoc_right`, `mul_assoc_right` |
| Layout | 3 | `reshape_reshape`, `transpose_cancel_perm_*` |
| Fusion | 1 | `bias_add_commute` |

## Target Contracts

지원 op contract는 `src/common/contracts.py`에 있다.

| Contract | 용도 |
| --- | --- |
| `VISION_SUPPORTED_OPS` | vision model target |
| `LLM_SUPPORTED_OPS` | decoder LLM target |
| `UNION_SUPPORTED_OPS` | scaffold / ORT CPU friendly union set |

## Benchmark Models

자세한 모델 정의는 [benchmark.md](benchmark.md)를 본다.

| # | Model | Domain | Target |
| --- | --- | --- | --- |
| 1 | MobileNetV2 | Vision / CNN | `VISION_SUPPORTED_OPS` |
| 2 | MobileViT-XXS | Vision / hybrid | `VISION_SUPPORTED_OPS` |
| 3 | YOLO26-Nano | Vision / detection | `VISION_SUPPORTED_OPS` |
| 4 | TinyLlama-15M | LLM / decoder | `LLM_SUPPORTED_OPS` |
| 5 | Pythia-70M | LLM / decoder | `LLM_SUPPORTED_OPS` |
| 6 | SmolLM-135M | LLM / decoder | `LLM_SUPPORTED_OPS` |

### Results

6개 모델 전부 superopt illegal=0 통과. 자세한 비교는 [report.md](report.md)를 본다.

| Model | Original | Baseline | BL illegal | Superopt | SO illegal |
|-------|----------|----------|------------|----------|------------|
| mobilenetv2 | 100 | 100 | 0 | 103 | 0 |
| yolo26_nano | 397 | 400 | 8 | 405 | 0 |
| mobilevit_xxs | 417 | 576 | 0 | 742 | 0 |
| tinyllama_15m | 1152 | 776 | 2 | 707 | 0 |
| pythia_70m | 589 | 619 | 9 | 704 | 0 |
| smollm_135m | 2844 | 3103 | 0 | 3379 | 0 |

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
| Contract result | unsupported op가 남았는지 |
| Correctness | output count, shape, dtype, value tolerance |
| Latency | original / rule baseline / ORT optimizer / superopt candidate 비교 |
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

Superopt:

```bash
python3.11 -m src.superopt.run \
  -i benchmarks/onnx/vision/mobilenetv2/onnx/model.onnx \
  -o artifacts/superopt/mobilenetv2.onnx \
  --contract vision
```

Full benchmark (baseline + superopt, all models):

```bash
python3.11 -m src.superopt.bench_all
```

## Current Status

이 프로젝트는 production optimizer가 아니라, 현업형 문제 정의를 가진 research / portfolio prototype이다.

잘 되어 있는 부분:

- baseline과 superopt가 동일한 42개 RuleSpec을 공유한다 (`get_all_specs()`).
- ONNX graph를 IR로 낮춘 뒤 다시 ONNX로 materialize하는 경로가 있다.
- hand-rolled e-graph 기반 equality saturation을 main path로 사용한다.
- 6개 benchmark 모델 전부 superopt에서 contract violation 없이 통과한다.
- candidate를 실제 ORT correctness와 latency로 평가하는 방향이 잡혀 있다.
- CPU budget은 single-thread / sequential ORT 실행을 기본으로 둔다.

아직 닫아야 할 부분:

- superopt이 baseline보다 노드 수가 많은 경우가 있다 (cost model / extraction 개선 필요).
- mobilevit_xxs, pythia_70m에서 saturation에 90~110초 소요 (산술 규칙 match 폭발).
- original / pre-pass를 candidate 0으로 포함하는 fallback.
- materialized graph hash dedup.
- 자동 regression test suite.

## Related Work

- **Tensat**: tensor graph superoptimization with equality saturation.
- **egg**: e-graph / equality saturation library (Willsey et al., 2020).
- **ONNX Runtime graph optimizer**: baseline optimizer and execution reference.

## Project Thesis

Graph rewrite는 단순히 op 수를 줄이는 작업이 아니다. 제한된 backend contract, numerical correctness, 실제 runtime latency 사이에서 실행 가능한 graph를 고르는 target adaptation 문제다.
