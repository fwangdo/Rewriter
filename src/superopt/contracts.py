"""Supported-op contract checks for superopt candidates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import onnx

from src.common.contracts import (
    LLM_MUST_REMOVE_OPS,
    LLM_SUPPORTED_OPS,
    SUPPORTED_OPS,
    VISION_SUPPORTED_OPS,
)


@dataclass(frozen=True)
class Contract:
    """Backend operator contract used as a candidate legality gate."""

    name: str
    supported_ops: frozenset[str]
    must_remove_ops: frozenset[str] = frozenset()
    preferred_ops: frozenset[str] = frozenset()


def op_histogram(model_or_path: str | Path | onnx.ModelProto) -> dict[str, int]:
    """Count ONNX node op types in a model."""
    model = (
        onnx.load(str(model_or_path))
        if isinstance(model_or_path, str | Path)
        else model_or_path
    )
    hist: dict[str, int] = {}
    for node in model.graph.node:
        hist[node.op_type] = hist.get(node.op_type, 0) + 1
    return hist


def get_contract(target: str, domain: str) -> Contract:
    """Return the initial supported-op contract for a target/domain pair."""
    normalized_domain = "llm" if domain in {"llm", "nlp"} else domain
    if target == "ort_cpu":
        return Contract(name=f"{target}:{normalized_domain}", supported_ops=SUPPORTED_OPS)
    if target != "portable_cpu":
        raise ValueError(f"unknown superopt contract target: {target}")
    if normalized_domain == "vision":
        return Contract(
            name=f"{target}:vision",
            supported_ops=VISION_SUPPORTED_OPS,
        )
    if normalized_domain == "llm":
        return Contract(
            name=f"{target}:llm",
            supported_ops=LLM_SUPPORTED_OPS,
            must_remove_ops=LLM_MUST_REMOVE_OPS,
        )
    raise ValueError(f"unknown superopt contract domain: {domain}")


def check_contract(
    model_or_path: str | Path | onnx.ModelProto,
    contract: Contract | Iterable[str],
) -> dict[str, object]:
    """Check whether a candidate model satisfies a supported-op contract."""
    if not isinstance(contract, Contract):
        contract = Contract(name="custom", supported_ops=frozenset(contract))
    hist = op_histogram(model_or_path)
    unsupported = {
        op: count
        for op, count in hist.items()
        if op not in contract.supported_ops
    }
    must_remove_remaining = {
        op: count
        for op, count in hist.items()
        if op in contract.must_remove_ops
    }
    return {
        "ok": not unsupported and not must_remove_remaining,
        "contract": contract.name,
        "op_histogram": hist,
        "unsupported_ops": unsupported,
        "must_remove_remaining": must_remove_remaining,
    }
