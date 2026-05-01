# CPU Budget Summary

This change does two things:

1. It updates the agent instructions so validation and benchmark work should stay single-threaded by default.
2. It makes the ONNX Runtime correctness path use a constrained session configuration.

## What to understand

The important idea is that correctness checking is not just about comparing outputs. It is also a workload-management problem.

- `intra_op_num_threads=1` keeps a single operator from fanning out across many CPU threads.
- `inter_op_num_threads=1` prevents multiple graph ops from running in parallel inside the same session.
- `ORT_SEQUENTIAL` avoids extra execution parallelism.

For this project, that matters because the validation harness may run many cases, and large ONNX models can stress the machine quickly if each session uses the runtime defaults.

## How the code works

The helper that creates ORT sessions is reused by:

- input generation for validation
- before/after model comparison
- manual runtime smoke tests

That keeps the behavior consistent. If the correctness path needs a session, it gets the same low-pressure configuration every time.

## How to re-implement it

1. Create one helper that builds an `onnxruntime.InferenceSession` from a model path.
2. Set the session options to single-threaded execution.
3. Use that helper everywhere correctness or smoke tests create sessions.
4. Keep parallelism out of the validation path unless the user explicitly asks for it.

The design goal is not maximum throughput. The goal is predictable, low-CPU correctness runs.
