from __future__ import annotations

import onnx

from src.common.rules import get_all_specs

from .cleanup import Cleanup
from .constant_folding import ConstantFolding
from .rule_runner import RuleRunner


class Passer:
    """Run the frontend rewrite pipeline in a fixed order."""

    def __init__(self) -> None:
        specs = get_all_specs()
        assert [s.name for s in specs] == [s.name for s in get_all_specs()], \
            "baseline and superopt rule sets diverged"
        self.passes = [
            ConstantFolding(),
            ConstantFolding(),
            RuleRunner(specs),
            ConstantFolding(),
            Cleanup(),
        ]

    def optimize(self, model: onnx.ModelProto) -> tuple[onnx.ModelProto, list[str]]:
        all_logs: list[str] = []

        for rewrite_pass in self.passes:
            pass_name = rewrite_pass.__class__.__name__
            before_nodes = len(model.graph.node)
            before_initializers = len(model.graph.initializer)
            model, logs = rewrite_pass.run(model)
            after_nodes = len(model.graph.node)
            after_initializers = len(model.graph.initializer)
            all_logs.append(
                f"PASS {pass_name}: "
                f"nodes {before_nodes} -> {after_nodes}, "
                f"initializers {before_initializers} -> {after_initializers}, "
                f"removed {rewrite_pass.deleted_node}"
            )
            all_logs.extend(f"  {line}" for line in logs)

        return model, all_logs
