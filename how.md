# 실행과 결과 해석

이 문서는 Rewriter를 어떻게 실행하고, 내부에서 무엇이 일어나며, 결과가 맞는지 어떻게 확인하고, 그 결과를 어떻게 해석해야 하는지 설명한다.

Rewriter의 기준 실행은 **ONNX Runtime끼리 비교**한다. AOT runner나 별도 native backend를 실행하지 않는다.

## 무엇을 확인하는가

입력은 원본 ONNX 모델이고, 출력은 rewrite 또는 superopt가 만든 ONNX 모델이다.

확인해야 하는 조건은 두 가지다.

1. **Legality**: 선택된 ONNX graph가 target contract를 만족하는가?
2. **Correctness**: 선택된 ONNX graph가 원본 ONNX와 같은 입력에서 허용 오차 안의 출력을 내는가?

fallback 후보를 쓰더라도 기준은 같다. fallback은 “superopt 후보가 실패했을 때 아무거나 고르는 것”이 아니라, **legality와 correctness를 모두 만족하는 후보 중에서 고르는 것**이다.

## 전체 구조

```text
ONNX model
  |
  +-- Stage 1: rule-based baseline
  |     src/onnx_rewrite
  |     고정 순서 rewrite pass를 적용한다.
  |
  +-- Stage 2: superopt
        src/superopt
        ONNX -> IR -> e-graph/egglog -> candidate extraction -> ONNX
```

두 stage는 같은 원본 ONNX를 입력으로 받는다. Stage 2가 Stage 1 결과에 의존하지 않는 것이 기본 구조다. 비교를 공정하게 하기 위해 rule-based baseline, ORT optimizer, superopt candidate를 따로 만들고 ORT로 평가한다.

## Superopt가 동작하는 방식

### 1. ONNX를 IR로 변환

`src/superopt/ir/convert.py`의 `onnx_to_ir()`가 ONNX graph를 `IRGraph`로 바꾼다.

`IRGraph`는 다음 정보를 가진 단순 DAG다.

- node id
- op type
- input edges
- attributes
- shape / dtype metadata
- initializer payload

이 IR은 ONNX protobuf를 직접 e-graph에 넣지 않고, rewrite와 extraction에 필요한 정보만 다루기 위한 중간 표현이다.

### 2. callback rule bridge

일부 rewrite rule은 단순 pattern 치환으로 끝나지 않는다.

예를 들면 다음이 필요할 수 있다.

- input shape 확인
- scalar constant 값 확인
- synthetic constant 생성
- weight tensor payload 읽기

이런 rule은 `check` 또는 `apply_fn` callback을 사용한다. 현재 egglog Python API만으로는 이 처리가 어렵기 때문에, `src/superopt/egraph`의 legacy bridge에서 먼저 materialize한 뒤 egglog path로 넘긴다.

### 3. egglog equality saturation

`src/superopt/backends/egglog.py`의 `EgglogBackend`가 IR을 egglog term으로 encode한다.

그 뒤 rewrite rule을 두 계열로 적용한다.

- legalization rule: unsupported op를 target이 받을 수 있는 op 조합으로 낮춘다.
- optimization rule: arithmetic simplification, layout rewrite, fusion 후보를 만든다.

e-graph에서는 여러 동등 표현이 동시에 존재한다. 즉 rule을 순서대로 하나씩 확정 적용하는 것이 아니라, 가능한 후보 공간을 넓힌 뒤 extraction 단계에서 하나를 고른다.

### 4. cost-aware extraction

`src/superopt/extract/cost.py`의 `CostModel`이 e-node에 비용을 준다.

- input / weight / noop 같은 boundary op는 비용 0
- contract 밖 op는 큰 penalty
- contract 안 op는 FLOPs 추정 또는 profiling table 기반 비용

이 비용은 최종 성능값이 아니다. 후보를 고르기 위한 heuristic이다. 최종 판단은 ORT correctness와 measured latency를 봐야 한다.

## 실행 방법

### 단일 모델 superopt 실행

```bash
python -m src.superopt.run \
  -i benchmarks/onnx/vision/mobilenetv2/onnx/model.onnx \
  -o artifacts/superopt/mobilenetv2.onnx \
  --contract vision
```

LLM 모델은 `--contract llm`을 사용한다.

```bash
python -m src.superopt.run \
  -i benchmarks/onnx/nlp/pythia_70m/onnx/model.onnx \
  -o artifacts/superopt/pythia_70m.onnx \
  --contract llm
```

### correctness까지 포함한 superopt 평가

```bash
python -m src.superopt.eval_superopt \
  benchmarks/onnx/nlp/pythia_70m/onnx/model.onnx \
  --contract llm \
  --output artifacts/superopt/pythia_70m.onnx
```

이 명령은 superopt 결과를 만든 뒤 원본 ONNX와 결과 ONNX를 ORT로 비교한다.

### rule-based baseline 평가

```bash
python -m src.onnx_rewrite.eval_rewrite \
  --input benchmarks/onnx/vision/mobilenetv2/onnx/model.onnx \
  --output artifacts/rewrite/mobilenetv2.onnx \
  --report artifacts/rewrite/mobilenetv2_eval.json
```

이 명령은 rule-based rewrite 결과에 대해 correctness와 latency를 측정한다.

### 전체 latency benchmark

```bash
python -m src.superopt.bench_latency
```

비교 대상은 다음이다.

- original ONNX
- rule-based baseline
- ORT optimizer output
- superopt candidate

latency benchmark도 correctness를 먼저 본다. correctness가 실패한 candidate는 빨라도 선택하면 안 된다.

## 결과 확인 방법

### correctness만 빠르게 확인

이미 만들어진 candidate를 ORT-only로 확인하려면 `src.superopt.validation.validate_correctness()`를 사용한다.

```bash
python - <<'PY'
from pathlib import Path
from src.superopt.validation import validate_correctness

original = Path("benchmarks/onnx/nlp/pythia_70m/onnx/model.onnx")
candidate = Path("artifacts/superopt/five_min_check/pythia_70m/candidate_0.onnx")

result = validate_correctness(original, candidate, domain="llm")
print(result)
PY
```

결과에서 봐야 할 필드는 다음이다.

- `ok`: correctness 통과 여부
- `stage`: 실패했다면 어느 단계에서 실패했는지
- `reason`: 실패 이유
- `max_abs_diff`: 최대 절대 오차
- `atol`, `rtol`: 비교 tolerance

### legality 확인

선택된 candidate가 contract를 만족하는지는 `check_contract()`로 확인한다.

```bash
python - <<'PY'
from pathlib import Path
from src.superopt.contracts import get_contract, check_contract

candidate = Path("artifacts/superopt/five_min_check/pythia_70m/candidate_0.onnx")
contract = get_contract("portable_cpu", "llm")

result = check_contract(candidate, contract)
print(result)
PY
```

결과에서 봐야 할 필드는 다음이다.

- `ok`: contract 만족 여부
- `unsupported_ops`: contract 밖에 있는 op
- `must_remove_remaining`: 반드시 제거해야 하는데 남아 있는 op
- `op_histogram`: 전체 op histogram

## 결과의 의미

### correctness PASS

`correctness PASS`는 원본 ONNX와 candidate ONNX를 같은 입력으로 ORT에서 실행했을 때, 모든 output이 tolerance 안에 들어왔다는 뜻이다.

이는 graph rewrite가 현재 validation input set에서는 수치적으로 안전하다는 의미다. 모든 가능한 입력에 대한 수학적 증명은 아니다.

### legality PASS

`legality PASS`는 candidate graph의 op set이 target contract를 만족한다는 뜻이다.

예를 들어 LLM contract에서는 `Where` 같은 must-remove op가 남으면 실패다. 반대로 target이 정확도 유지를 위해 `Erf`를 지원 op로 인정한다면, exact GELU 경로의 `Erf`는 legal op로 남을 수 있다.

### latency 개선

latency 개선은 correctness와 legality를 모두 만족한 후보 사이에서만 의미가 있다.

틀린 graph가 빠른 것은 성과가 아니다. contract를 만족하지 않는 graph도 target backend 관점에서는 선택할 수 없다.

따라서 최종 후보 선택 순서는 다음이 되어야 한다.

```text
candidate 생성
  -> ONNX checker 통과
  -> contract 통과
  -> ORT load 통과
  -> correctness 통과
  -> latency가 가장 낮은 후보 선택
```

## pythia_70m 해석

`pythia_70m`은 fallback 정책을 설명하기 좋은 사례다.

원본 모델에는 strict LLM contract 기준으로 unsupported 또는 must-remove op가 남아 있다. 따라서 원본 그대로는 legal candidate가 아니다.

선택 후보는 다음을 만족해야 한다.

- `Where` 같은 must-remove op 제거
- contract 밖 op 제거
- ORT correctness 통과

superopt extraction 후보가 correctness를 깨면 폐기한다. 이 경우 fallback 후보를 사용할 수 있지만, fallback도 반드시 legality와 correctness를 모두 만족해야 한다.

## 주요 파일

| 파일 | 역할 |
| --- | --- |
| `src/superopt/run.py` | 단일 모델 superopt CLI |
| `src/superopt/eval_superopt.py` | superopt + correctness 평가 CLI |
| `src/superopt/pipeline.py` | ONNX load부터 candidate 저장까지의 main pipeline |
| `src/superopt/backends/egglog.py` | IR과 egglog term 사이의 adapter |
| `src/superopt/extract/cost.py` | contract-aware cost model |
| `src/superopt/contracts.py` | candidate contract 검사 |
| `src/superopt/validation.py` | ORT correctness / latency helper |
| `src/onnx_rewrite/eval_rewrite.py` | rule-based baseline 평가 CLI |
| `src/common/contracts.py` | 공통 supported-op contract |
| `src/common/rules/` | baseline과 superopt가 공유하는 rewrite rule spec |
