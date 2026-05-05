# Superopt Latency Correctness Summary

## What changed

The current superopt direction is latency-first:

- All ONNX ops are allowed during extraction.
- Extraction chooses the graph with the lowest profiled ONNX Runtime op cost.
- Correctness is still a hard gate after extraction.

This means the cost model can prefer any op if its measured cost is low, but
the extracted graph must still load in ONNX Runtime and match the original
model output within tolerance.

## Why the bug happened

ONNX shape inference represents symbolic or unknown dimensions as unknowns.
In this project, unknown dimensions are stored in IR shapes as `-1`.

That is safe as metadata, but it is not always safe as an actual `Reshape`
shape initializer.

ONNX `Reshape` has a special rule:

- `-1` means "infer this one dimension from the input size".
- A `Reshape` shape tensor may contain at most one `-1`.

The previous `Squeeze/Unsqueeze -> Reshape` e-graph rule took the output
shape metadata and materialized it directly as a `Reshape` shape tensor.
For tensors with multiple symbolic dimensions, that produced invalid shapes
such as:

```text
[-1, 1, 1, -1]
[1, 1, -1, -1]
```

ONNX Runtime rejects those graphs before inference.

## Fix

The legalization rule now only emits a `Reshape` replacement when the target
shape has at most one `-1`.

If the target shape has multiple unknown dimensions, the rule does not fire.
The original `Squeeze` or `Unsqueeze` candidate remains in the e-graph, and
latency-cost extraction can still choose among valid alternatives.

## How to re-implement it

To re-implement this fix from scratch:

1. Find the rule that converts `Squeeze` or `Unsqueeze` into `Reshape`.
2. Identify where it reads the matched e-class output shape.
3. Before creating the shape initializer, count how many dimensions equal `-1`.
4. If more than one dimension is `-1`, return the original matched e-class.
5. Otherwise create the shape constant and the replacement `Reshape` node.
6. Regenerate a model and run ORT correctness against the original model.

The important idea is the distinction between shape metadata and executable
operator inputs. Metadata can be approximate; operator inputs must satisfy
the ONNX spec.
