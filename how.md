# How It Works

## Architecture: Two-Stage Pipeline

```
ONNX model
  │
  ├─ Stage 1: Baseline (onnx_rewrite)
  │    Rule-based rewrites: BN folding, constant folding, LayerNorm decomposition, etc.
  │    Fast, deterministic. Serves as comparison baseline.
  │
  └─ Stage 2: Superopt (e-graph superoptimization)
       IR conversion → e-graph saturation → contract-aware extraction
       Explores equivalent graphs via equality saturation, then extracts
       the lowest-cost graph that satisfies the supported-op contract.
```

Both stages take the same ONNX input. Stage 2 does **not** depend on Stage 1 output — they run independently to enable fair comparison.

## How Superopt Works

### 1. ONNX → IR Conversion

`onnx_to_ir()` converts the ONNX graph into `IRGraph`, a simplified DAG of `IRNode`s with typed edges and shape metadata.

### 2. Legacy Callback Bridge

Some legalization rules require Python-side value inspection or synthetic constant generation (`check`/`apply_fn` callbacks). These run through the hand-rolled e-graph engine before handing off to egglog.

### 3. E-Graph Saturation (egglog)

The `EgglogBackend` loads the IR into an egglog `EGraph` and applies rewrite rules in two phases:

1. **Legalization rules** — convert unsupported ops into supported equivalents (e.g., `Gelu → Erf` decomposition)
2. **Optimization rules** — arithmetic simplifications, layout transforms, operator fusion

Rules are applied via equality saturation: all equivalent forms coexist in the e-graph.

### 4. Contract-Aware Extraction

The `CostModel` assigns costs to each e-node:
- **Free ops** (Input, Weight, Noop, Proj): cost 0
- **Unsupported ops** (not in `supported_ops`): cost 1e9 (penalty)
- **Supported ops**: FLOPs-based estimate, with profiled-latency fallback

Extraction picks the minimum-cost program from the e-graph, effectively steering the result toward supported ops.

## Running

### Single model
```bash
python -m src.superopt.run -i model.onnx -o output.onnx --contract vision
```

### Latency benchmark (all 6 models)
```bash
python -m src.superopt.bench_latency
```
Compares: original → baseline (onnx_rewrite) → ORT optimizer → superopt.

### Op-count benchmark
```bash
python -m src.superopt.bench_all
```
Compares baseline vs superopt op counts and contract violations.

## Adding Rules

### Common rules (`src/common/rules/`)
Shared legalization specs used by both the baseline and superopt. Define patterns as `(source, target)` pairs in spec files, then register in `__init__.py`.

### Superopt rules (`src/superopt/rules/`)
Wrappers that convert common specs into `RewriteRule` objects for the e-graph engine:

- `legalization.py` — op decomposition/lowering rules
- `arithmetic.py` — algebraic simplifications
- `layout.py` — reshape/transpose optimizations
- `fusion.py` — operator fusion patterns

To add a new rule:
1. Define the pattern in the appropriate rule file
2. Return it from the corresponding `get_*_rules()` function
3. The pipeline picks it up automatically

## Experimenting

### Toggling rule families
Comment out rule lists in `pipeline.py`'s `_run_egglog()` to disable families:
```python
# opt_stats = backend.run_rules(
#     get_fusion_rules(),
#     ...
# )
```

### Changing the contract
Contracts are defined in `src/common/contracts.py` as `frozenset[str]` of allowed op names. Edit `VISION_SUPPORTED_OPS` or `LLM_SUPPORTED_OPS`, or pass a custom set via the Python API.

### Cost model tuning
Adjust `CostModel` in `src/superopt/extract/cost.py`:
- Change the unsupported-op penalty (default: `1e9`)
- Modify FLOPs scaling factors
- Update the profiled cost table at `artifacts/superopt/op_cost_table.json`

## Key Files

| File | Role |
|------|------|
| `src/superopt/pipeline.py` | End-to-end pipeline: load → saturate → extract → save |
| `src/superopt/backends/egglog.py` | egglog adapter: IR ↔ egglog term translation, rule application, extraction |
| `src/superopt/extract/cost.py` | FLOPs-based cost model with contract-aware legality penalties |
| `src/superopt/ir/` | IR graph representation and ONNX ↔ IR conversion |
| `src/superopt/egraph/` | Hand-rolled e-graph (used for legacy callback rules) |
| `src/superopt/rules/` | Rewrite rule definitions (legalization, arithmetic, layout, fusion) |
| `src/superopt/contracts.py` | Contract checking: verify output satisfies supported-op constraints |
| `src/common/contracts.py` | Supported-op set definitions (VISION, LLM) |
| `src/common/rules/` | Shared legalization specs |
| `src/onnx_rewrite/passes/passer.py` | Rule-based baseline optimizer |
| `src/superopt/bench_latency.py` | Latency benchmark: original vs baseline vs ORT vs superopt |
| `src/superopt/bench_all.py` | Op-count benchmark: baseline vs superopt |
| `src/superopt/run.py` | CLI entry point |
