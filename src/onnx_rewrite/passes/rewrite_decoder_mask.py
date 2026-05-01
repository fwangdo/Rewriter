from __future__ import annotations

import onnx
from onnx import helper

from ..utils import cons
from .folder import Folder


class RewriteDecoderMask(Folder):
    """Rewrite a common decoder mask subgraph into broadcast arithmetic."""

    def _is_negative_infinity_like(self, value_name: str) -> bool:
        value = self.init_map.get(value_name)
        if value is None or value.size != 1:
            return False
        scalar = float(value.reshape(-1)[0])
        return scalar <= -1.0e30

    def _match_where3_chain(
        self,
        where4: onnx.NodeProto,
    ) -> tuple[str, str, str, str] | None:
        if len(where4.input) != 3:
            return None

        neginf_name = where4.input[1]
        if not self._is_negative_infinity_like(neginf_name):
            return None
        expand = self.get_producer(where4.input[2])
        if expand is None or expand.op_type != cons.OP_EXPAND or len(expand.input) != 2:
            return None

        causal_mask_name = expand.input[0]

        cast8 = self.get_producer(where4.input[0])
        if cast8 is None or cast8.op_type != cons.OP_CAST:
            return None
        cast7 = self.get_producer(cast8.input[0])
        if cast7 is None or cast7.op_type != cons.OP_CAST:
            return None
        cast6 = self.get_producer(cast7.input[0])
        if cast6 is None or cast6.op_type != cons.OP_CAST:
            return None
        where3 = self.get_producer(cast6.input[0])
        if where3 is None or where3.op_type != cons.OP_WHERE or len(where3.input) != 3:
            return None
        where3_neginf_name = where3.input[1]
        if not self._is_negative_infinity_like(where3_neginf_name):
            return None

        sub1 = self.get_producer(where3.input[2])
        if sub1 is None or sub1.op_type != cons.OP_SUB or len(sub1.input) != 2:
            return None

        cast3 = self.get_producer(sub1.input[1])
        if cast3 is None or cast3.op_type != cons.OP_CAST:
            return None

        expand1 = self.get_producer(cast3.input[0])
        if expand1 is None or expand1.op_type != cons.OP_EXPAND or len(expand1.input) != 2:
            return None

        mask_name = expand1.input[0]
        one_name = sub1.input[0]
        return causal_mask_name, mask_name, one_name, where3_neginf_name

    def _match_gpt_neox_where4(
        self,
        where4: onnx.NodeProto,
    ) -> tuple[str, str, str, str] | None:
        if len(where4.input) != 3:
            return None

        neginf_name = where4.input[1]
        if not self._is_negative_infinity_like(neginf_name):
            return None

        expand = self.get_producer(where4.input[2])
        if expand is None or expand.op_type != cons.OP_EXPAND or len(expand.input) != 2:
            return None
        causal_mask_name = expand.input[0]

        cast7 = self.get_producer(where4.input[0])
        if cast7 is None or cast7.op_type != cons.OP_CAST:
            return None
        where3 = self.get_producer(cast7.input[0])
        if where3 is None or where3.op_type != cons.OP_WHERE or len(where3.input) != 3:
            return None
        where3_neginf_name = where3.input[1]
        if not self._is_negative_infinity_like(where3_neginf_name):
            return None

        sub1 = self.get_producer(where3.input[2])
        if sub1 is None or sub1.op_type != cons.OP_SUB or len(sub1.input) != 2:
            return None

        cast3 = self.get_producer(sub1.input[1])
        if cast3 is None or cast3.op_type != cons.OP_CAST:
            return None

        expand1 = self.get_producer(cast3.input[0])
        if expand1 is None or expand1.op_type != cons.OP_EXPAND or len(expand1.input) != 2:
            return None

        mask_name = expand1.input[0]
        one_name = sub1.input[0]
        return causal_mask_name, mask_name, one_name, where3_neginf_name

    def _rewrite_where4(self, node: onnx.NodeProto) -> None:
        match = self._match_where3_chain(node)
        if match is None:
            match = self._match_gpt_neox_where4(node)
        if match is None:
            return

        causal_mask_name, mask_name, one_name, neginf_name = match
        prefix = self.get_prefix(node)

        mask_float_name = self.tensor_name(prefix, "mask_float")
        mask_inverse_name = self.tensor_name(prefix, "mask_inverse")
        causal_invalid_name = self.tensor_name(prefix, "causal_invalid")
        overlap_name = self.tensor_name(prefix, "mask_overlap")
        combined_sum_name = self.tensor_name(prefix, "mask_sum")
        combined_invalid_name = self.tensor_name(prefix, "combined_invalid")

        replacements = [
            helper.make_node(
                cons.OP_CAST,
                [mask_name],
                [mask_float_name],
                name=self.node_name(prefix, "mask_cast"),
                to=1,  # FLOAT
            ),
            helper.make_node(
                cons.OP_SUB,
                [one_name, mask_float_name],
                [mask_inverse_name],
                name=self.node_name(prefix, "mask_inverse"),
            ),
            helper.make_node(
                cons.OP_DIV,
                [causal_mask_name, neginf_name],
                [causal_invalid_name],
                name=self.node_name(prefix, "causal_invalid"),
            ),
            helper.make_node(
                cons.OP_MUL,
                [causal_invalid_name, mask_inverse_name],
                [overlap_name],
                name=self.node_name(prefix, "mask_overlap"),
            ),
            helper.make_node(
                cons.OP_ADD,
                [causal_invalid_name, mask_inverse_name],
                [combined_sum_name],
                name=self.node_name(prefix, "mask_sum"),
            ),
            helper.make_node(
                cons.OP_SUB,
                [combined_sum_name, overlap_name],
                [combined_invalid_name],
                name=self.node_name(prefix, "mask_or"),
            ),
            helper.make_node(
                cons.OP_MUL,
                [combined_invalid_name, neginf_name],
                [node.output[0]],
                name=self.node_name(prefix, "mask_scale"),
            ),
        ]

        self.replace_node(node, replacements)
        self.log.append(
            f" - DecoderMask({prefix}) is rewritten as arithmetic OR mask"
        )

    def run(self, model: onnx.ModelProto) -> tuple[onnx.ModelProto, list[str]]:
        self.prepare(model)

        for node in list(self.graph.node):
            if node.op_type == cons.OP_WHERE:
                self._rewrite_where4(node)

        self.remove_marked_nodes()
        return model, self.log
