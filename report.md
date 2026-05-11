# Baseline vs Superopt 비교 결과

## Greedy extraction 결과

아래 결과는 greedy extractor에 `children_cost`를 더한 뒤의 결과다.
이 변경은 local node cost만 보던 기존 extractor보다 expansion을 덜 고르게 만들어
노드 수를 줄이는 효과가 있었다. 그러나 greedy는 e-class마다 local best 하나만
고정하기 때문에 global legality를 보장하지 못한다.

| Model | Original | Baseline | BL illegal | Superopt | SO illegal |
|-------|----------|----------|------------|----------|------------|
| mobilenetv2 | 100 | 100 | 0 | 103 | 0 |
| yolo26_nano | 397 | 400 | 8 (Neg 3, Unsqueeze 5) | 402 | 5 (Unsqueeze 5) |
| mobilevit_xxs | 417 | 576 | 0 | 507 | 17 (LayerNormalization 17) |
| tinyllama_15m | 1152 | 776 | 2 (Neg 2) | 702 | 8 (Pow 8) |
| pythia_70m | 589 | 619 | 9 (Neg 9) | 666 | 3 (Pow 3) |
| smollm_135m | 2844 | 3103 | 0 | 3119 | 124 (Neg 14, Pow 55, Where 55) |

## 소요 시간

| Model | Baseline | Superopt |
|-------|----------|----------|
| mobilenetv2 | 0.08s | 0.31s |
| yolo26_nano | 0.27s | 0.85s |
| mobilevit_xxs | 0.63s | 101.82s |
| tinyllama_15m | 2.02s | 2.19s |
| pythia_70m | 2.05s | 102.72s |
| smollm_135m | 114.47s | 111.54s |

해석:

- `children_cost` 도입 후 superopt 노드 수는 줄었지만, illegal op가 다시 남았다.
- 예를 들어 `mobilevit_xxs`는 742 노드에서 507 노드로 줄었지만 `LayerNormalization` 17개가 남았다.
- `tinyllama_15m`, `pythia_70m`, `smollm_135m`도 `Pow`, `Neg`, `Where` 같은 illegal op가 남는다.
- 이는 legality가 node-local property가 아니라 extracted DAG 전체의 property이기 때문이다.
- 따라서 e-class별 local best 하나만 고르는 greedy extraction으로는 "작고 legal한 graph"를 동시에 보장하기 어렵다.

## ILP soft extraction 실험

Greedy local-best extraction의 한계를 확인하기 위해 `extract_ilp`를 soft legalization
mode로 연결했다. ILP extraction은 추출 문제를 binary variable과 linear constraint로
표현한다.

- `t_c`: e-class `c`가 최종 DAG에 필요한지 나타내는 binary variable
- `x_n`: e-node `n`을 최종 DAG의 구현으로 선택했는지 나타내는 binary variable
- objective: `sum(cost(n) * x_n)` 최소화
- constraint: root e-class는 active
- constraint: active e-class에서는 e-node 하나를 선택
- constraint: e-node를 선택하면 child e-class도 active

Soft legalization mode에서는 illegal e-node를 금지하지 않고 `CostModel`의 큰 penalty로
회피한다. Blacklist된 e-node는 cycle handling 이후 구조적으로 위험하므로 여전히 금지한다.

| Model | max_nodes | ILP Superopt | Illegal | 비고 |
|-------|-----------|--------------|---------|------|
| mobilenetv2 | 50000 | 100 | 0 | 50k에서 즉시 완료 |
| yolo26_nano | 50000 | 397 | 0 | 50k에서 즉시 완료 |
| tinyllama_15m | 50000 | 701 | 0 | 50k에서 완료 |
| mobilevit_xxs | 50000 | timeout | - | ILP solver 30분 이상 미완료 |
| mobilevit_xxs | 5000 | 576 | 0 | node limit을 낮추면 완료 |
| pythia_70m | 50000 | timeout | - | ILP solver 장시간 미완료 |
| pythia_70m | 5000 | timeout | - | ILP solver 8분 이상 미완료 |
| pythia_70m | 2000 | 603 | 0 | 2k에서 legal 결과 확보 |
| pythia_70m | 1000 | 589 | 26 (Squeeze 2, Pow 7, Where 5, Neg 12) | 탐색 부족 |
| smollm_135m | 5000 | 3092 | 0 | 5k에서 legal 결과 확보 |
| smollm_135m | 1000 | 2843 | 185 (Pow 61, Where 64, Neg 60) | 탐색 부족 |

관찰:

- ILP extraction은 greedy local-best에서 발생하던 illegal 경로 선택 문제를 해결한다.
- 작은 모델에서는 baseline보다 작거나 같은 legal graph를 뽑는다.
  - `mobilenetv2`: baseline 100, ILP superopt 100
  - `yolo26_nano`: baseline 400 illegal 8, ILP superopt 397 illegal 0
  - `tinyllama_15m`: baseline 776 illegal 2, ILP superopt 701 illegal 0
- 큰 e-graph를 그대로 ILP로 풀면 solver 시간이 급격히 커진다.
- node limit을 낮추면 legal 결과를 확보할 수 있지만, saturation 공간이 줄어들어 최적성은 약해진다.
- 따라서 실용화를 위해서는 `max_nodes`/candidate pruning/strict legality bound/ILP time limit이 필요하다.

결론:

- Greedy + `children_cost`는 노드 수 감소에는 효과가 있지만 legality를 깨뜨릴 수 있다.
- ILP extraction은 global legality와 DAG cost를 더 정확히 다루지만, 큰 e-graph에서는 solver 비용이 병목이다.
- 현재 방향은 "greedy를 더 고치는 것"보다 "ILP를 실용 가능한 크기로 제한하는 것"이 더 타당하다.
