# Baseline vs Superopt 비교 결과

## 노드 수 비교

| Model | Original | Baseline | BL illegal | Superopt | SO illegal |
|-------|----------|----------|------------|----------|------------|
| mobilenetv2 | 100 | 100 | 0 | 103 | 0 |
| yolo26_nano | 397 | 400 | 8 (Neg 3, Unsqueeze 5) | 405 | 0 |
| mobilevit_xxs | 417 | 576 | 0 | 742 | 0 |
| tinyllama_15m | 1152 | 776 | 2 (Neg 2) | 707 | 0 |
| pythia_70m | 589 | 619 | 9 (Neg 9) | 704 | 0 |
| smollm_135m | 2844 | 3103 | 0 | 3379 | 0 |

## 소요 시간

| Model | Baseline | Superopt |
|-------|----------|----------|
| mobilenetv2 | 0.08s | 0.32s |
| yolo26_nano | 0.27s | 0.77s |
| mobilevit_xxs | 0.63s | 94.94s |
| tinyllama_15m | 2.02s | 2.16s |
| pythia_70m | 2.05s | 103.96s |
| smollm_135m | 114.47s | 100.86s |
