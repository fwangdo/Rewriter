from __future__ import annotations

import onnx

from .cleanup import Cleanup
from .constant_folding import ConstantFolding
from .eliminate_id import EliminateId
from .rewrite_bn import RewriteBN
from .rewrite_clip import RewriteClip
from .rewrite_compare import RewriteCompare
from .rewrite_gemm import RewriteGemm
from .rewrite_layernorm import RewriteLayerNorm
from .rewrite_neg import RewriteNeg
from .rewrite_pow import RewritePow
from .rewrite_reshape_shape import RewriteReshapeShape
from .rewrite_trilu import RewriteTrilu


class Passer:
    """Run the frontend rewrite pipeline in a fixed order."""

    def __init__(self) -> None:
        self.passes = [
            ConstantFolding(),
            EliminateId(),
            RewriteClip(),
            RewriteCompare(),
            RewriteReshapeShape(),
            RewriteLayerNorm(),
            RewriteBN(),
            RewriteNeg(),
            RewritePow(),
            RewriteGemm(),
            RewriteTrilu(),
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
