# ONNX Rewrite

이 문서는 `src/onnx_rewrite` 모듈의 목적, 구조, 사용법, 현재 구현 범위를 설명한다.

## Overview

`src/onnx_rewrite`는 Rewriter 프로젝트의 ONNX frontend rewrite 모듈이다.

> 입력 ONNX graph를 읽고, 최종 결과를 domain-specific `supported-op-only` graph로 내린다.

이 시스템은 일반적인 ONNX optimizer가 아니라, 특정 하드웨어/중간 표현이 받아들일 수 있는 연산 집합으로 graph를 강제로 낮추는 legality-first rewrite pipeline이다.

현재는 correctness / validation / latency 계측까지 포함한 scaffold를 갖추고 있고, benchmark 6종에 대해 unsupported op 제거를 목표로 둔다.

## Benchmark Models

- `mobilenetv2` — mobile CNN baseline
- `mobilevit_xxs` — hybrid vision + transformer
- `yolo26_nano` — detection topology
- `tinyllama_15m` — smallest decoder (RoPE + RMSNorm + SwiGLU)
- `pythia_70m` — GPT-NeoX style decoder (parallel attn + LayerNorm + GELU)
- `smollm_135m` — modern decoder (GQA)

모델 경로는 [specs/catalog.py](specs/catalog.py)에 정의되어 있다.

## Domain Contracts

실전 target contract는 두 개다.

- `VISION_SUPPORTED_OPS` (24 ops): vision 모델용
- `LLM_SUPPORTED_OPS` (16 ops): decoder LLM용

현재 `SUPPORTED_OPS`는 migration 중인 scaffold용 union set이다.
자세한 contract 정의는 [../common/contracts.py](../common/contracts.py) 참조.

## Directory Layout

- [run_onnx_rewrite.py](run_onnx_rewrite.py): rewrite / audit CLI
- [eval_rewrite.py](eval_rewrite.py): rewrite + correctness + latency 평가 CLI
- [core/optimizer.py](core/optimizer.py): top-level optimize entrypoint
- [passes/passer.py](passes/passer.py): pass 실행 순서
- [passes/folder.py](passes/folder.py): pass 공통 helper, shape inference, producer/consumer map
- [checker/op_checker.py](checker/op_checker.py): supported-op-only checker
- [analysis/audit.py](analysis/audit.py): ONNX histogram / unsupported audit
- [runtime/validation.py](runtime/validation.py): ORT correctness 비교
- [runtime/benchmark.py](runtime/benchmark.py): ORT latency 측정
- [report.md](report.md): benchmark / correctness 상태 정리

## Correctness / Validation

[runtime/validation.py](runtime/validation.py)는 ONNX Runtime 기반으로 원본/변환 모델을 비교한다.

현재 방식:

- deterministic seed
- 여러 dynamic size
- 여러 mask mode
- 여러 integer input mode

핵심 metric:

- `max_abs_diff`
- `worst_case`
- pass / fail

현재 default gate:

```text
max_abs_diff <= 1e-4
```

## Latency / Throughput Measurement

[runtime/benchmark.py](runtime/benchmark.py)는 ONNX Runtime CPU 기준 median / p95 latency를 측정한다.

현재 benchmark 조건:

- single-process
- `intra_op_num_threads=1`
- `inter_op_num_threads=1`
- sequential execution

## Current Status

Vision 3종(`mobilenetv2`, `mobilevit_xxs`, `yolo26_nano`)은 supported-op-only + correctness 달성.

LLM 3종(`tinyllama_15m`, `pythia_70m`, `smollm_135m`)은 union-contract correctness까지 확보. strict `LLM_SUPPORTED_OPS` legality는 진행 중.

자세한 수치는 [report.md](report.md)를 참조.

## Usage

### 1. audit only

```bash
python3.11 -m src.onnx_rewrite.run_onnx_rewrite \
  --input benchmarks/onnx/vision/mobilenetv2/onnx/model.onnx \
  --audit-only
```

### 2. rewrite

```bash
python3.11 -m src.onnx_rewrite.run_onnx_rewrite \
  --input benchmarks/onnx/vision/mobilenetv2/onnx/model.onnx \
  --output artifacts/rewrite/mobilenetv2.onnx
```

### 3. rewrite + correctness + latency

```bash
python3.11 -m src.onnx_rewrite.eval_rewrite \
  --input benchmarks/onnx/vision/mobilenetv2/onnx/model.onnx \
  --output artifacts/rewrite/mobilenetv2.onnx \
  --report artifacts/rewrite/mobilenetv2_eval.json
```
