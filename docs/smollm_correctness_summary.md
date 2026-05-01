# SmolLM Correctness Summary

This change separates two concerns:

1. Whether the rewritten graph is legal for the target contract.
2. Whether the rewritten graph is numerically close enough to the original.

## What to understand

For large decoder graphs, correctness and legality do not always fail together.

- A graph can still run in ONNX Runtime even if it contains unsupported ops.
- A rewrite can be correct within a small numeric tolerance even if legality is not finished yet.
- The evaluation pipeline therefore needs two switches:
  - one to save a rewritten graph even when unsupported ops remain
  - one to control the numeric tolerance used for comparison

## How the code works

The optimizer now accepts a flag that decides whether unsupported ops are fatal.

- If the flag is on, the old behavior stays the same.
- If the flag is off, the graph is still saved so correctness can be measured.

The validation path also accepts per-run tolerances.

- `max_abs_tolerance` controls the absolute error threshold.
- `max_rel_tolerance` controls the relative error threshold.

## How to re-implement it

1. Let graph optimization return a rewritten model even when legality is incomplete, if the caller asked for that.
2. Save the model so ORT can still execute it.
3. Feed the saved model into the comparison step.
4. Pass a tolerance that matches the model’s observed numerical noise.

The key lesson is that legality and correctness should be evaluated independently when you are debugging a rewrite pipeline.
