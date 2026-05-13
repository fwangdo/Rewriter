# Rewriter

ONNX model을 target backend가 실행할 수 있는 graph로 변환하고, cost model을 기준으로 후보를 고르는 superoptimization 프로젝트.

본 프로젝트는 아래와 같은 문제를 해결하고자 한다. 

> ONNX graph를 지원 가능한 operator로 변환할 때 변환 규칙을 어떤 순서로 적용하는지를 자동으로 알 수 있는 방법은 없는가?

일반적인 ONNX optimizer는 보통 실행 가능한 backend를 이미 가정하고 성능 최적화에 집중한다. NPU, edge runtime, vendor compiler처럼 지원 op set이 좁거나 계속 바뀌는 환경에서 graph를 target contract에 맞게 legalize하고, 가능한 후보 중 실제로 맞고 빠른 것을 선택한다.

## Architecture

Superopt의 큰 흐름은 다음과 같다.

```text
ONNX model
  -> pre-pass (ConstantFolding)
  -> ONNX -> IR
  -> e-graph equality saturation (shared RuleSpecs)
  -> ILP extraction (SciPy/HiGHS 기본, OR-Tools SCIP 선택)
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

현재 superopt는 greedy local-best extraction에서 ILP extraction으로 전환 중이다.
ILP는 global legality를 더 잘 다루지만, 큰 e-graph에서는 solver budget이 병목이 된다.
자세한 비교와 ILP 실험 결과는 [report.md](report.md)를 본다.

Greedy + `children_cost` 결과:

| Model | Original | Baseline | BL illegal | Superopt | SO illegal |
|-------|----------|----------|------------|----------|------------|
| mobilenetv2 | 100 | 100 | 0 | 103 | 0 |
| yolo26_nano | 397 | 400 | 8 | 402 | 5 |
| mobilevit_xxs | 417 | 576 | 0 | 507 | 17 |
| tinyllama_15m | 1152 | 776 | 2 | 702 | 8 |
| pythia_70m | 589 | 619 | 9 | 666 | 3 |
| smollm_135m | 2844 | 3103 | 0 | 3119 | 124 |

ILP soft extraction에서 확인한 legal 결과:

| Model | max_nodes | ILP Superopt | Illegal |
|-------|-----------|--------------|---------|
| mobilenetv2 | 50000 | 100 | 0 |
| yolo26_nano | 50000 | 397 | 0 |
| tinyllama_15m | 50000 | 701 | 0 |
| mobilevit_xxs | 5000 | 576 | 0 |
| pythia_70m | 2000 | 603 | 0 |
| smollm_135m | 5000 | 3092 | 0 |

## Evaluation

Correctness와 ONNX Runtime CPU 기준으로 측정하며, performance는 graph에 존재하는 노드 수를 기준으로 한다. 
지원되지 않는 op가 그래프에 존재하는 경우, 해당 op의 수들을 통해 cost를 계산한다. 

주요 metric:

| 항목 | 의미 |
| --- | --- |
| Contract result | unsupported op가 남았는지 |
| Correctness | output count, shape, dtype, value tolerance |
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

ILP solver / budget 선택:

```bash
python3.11 -m src.superopt.run \
  -i benchmarks/onnx/vision/mobilenetv2/onnx/model.onnx \
  -o artifacts/superopt/mobilenetv2.onnx \
  --contract vision \
  --ilp-solver scipy \
  --ilp-time-limit 60
```

`--ilp-solver ortools_scip`는 OR-Tools가 설치된 환경에서 SCIP backend를 사용한다.

Full benchmark (baseline + superopt, all models):

```bash
python3 -m src.superopt.bench_all
```

## Related Work

- **Tensat**: tensor graph superoptimization with equality saturation.
- **egg**: e-graph / equality saturation library (Willsey et al., 2020).
- **ONNX Runtime graph optimizer**: baseline optimizer and execution reference.