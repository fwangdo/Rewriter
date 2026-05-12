# Baseline vs Superopt 최종 비교

이 문서는 연구 노트가 아니라 최종 end-to-end 결과 비교만 기록한다.

## 비교 조건

- Baseline: IR 기반 manual rewrite
- Superopt: e-graph saturation + ILP extraction
- ILP backend: OR-Tools SCIP
- Superopt limit: `max_nodes=50000`, `ilp_time_limit=600s`
- 측정값: 출력 ONNX 기준 node count
- Illegal: 대상 backend contract에서 지원하지 않는 op 수

## 결과

| Model | Baseline nodes | Baseline illegal | Superopt nodes | Superopt illegal | Node delta | ILP status |
|-------|----------------|------------------|----------------|------------------|------------|------------|
| mobilenetv2 | 103 | 0 | 100 | 0 | -3 | optimal |
| yolo26_nano | 400 | 2 (Unsqueeze 2) | 397 | 0 | -3 | optimal |
| tinyllama_15m | 781 | 0 | 703 | 0 | -78 | optimal |
| mobilevit_xxs | 600 | 0 | 731 | 0 | +131 | feasible |
| pythia_70m | 624 | 0 | 687 | 0 | +63 | feasible |
| smollm_135m | 3167 | 1 (Trilu 1) | 3250 | 0 | +83 | feasible |

## 해석

Superopt는 6개 모델 모두에서 illegal op 없이 출력 ONNX를 만든다. Baseline은 `yolo26_nano`와 `smollm_135m`에서 illegal op가 남는다.

Node count만 보면 Superopt는 `mobilenetv2`, `yolo26_nano`, `tinyllama_15m`에서 Baseline보다 작고, `mobilevit_xxs`, `pythia_70m`, `smollm_135m`에서는 더 크다. 즉 현재 결과는 "항상 더 작은 graph"라기보다, ILP extraction을 통해 legality를 전역적으로 만족시키는 쪽에 강점이 있다.

`feasible`은 제한 시간 안에 legal 해를 찾았지만 최적성 증명까지 끝나지는 않았다는 뜻이다. 따라서 해당 모델들의 node count는 더 개선될 여지가 있다.
