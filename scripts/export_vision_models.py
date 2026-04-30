from __future__ import annotations

import argparse
from pathlib import Path

import timm
import torch
import torchvision.models as tv_models


DEFAULT_OPSET = 17
DEFAULT_INPUT_SHAPE = (1, 3, 224, 224)


def export_mobilenetv2(output_path: Path, opset: int) -> None:
    model = tv_models.mobilenet_v2(weights=None).eval()
    dummy = torch.randn(*DEFAULT_INPUT_SHAPE)
    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["input"],
        output_names=["logits"],
        opset_version=opset,
        do_constant_folding=True,
    )


def export_mobilevit_xxs(output_path: Path, opset: int) -> None:
    model = timm.create_model("mobilevit_xxs", pretrained=False).eval()
    dummy = torch.randn(*DEFAULT_INPUT_SHAPE)
    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["input"],
        output_names=["logits"],
        opset_version=opset,
        do_constant_folding=True,
    )


EXPORTERS = {
    "mobilenetv2": (
        Path("benchmarks/onnx/vision/mobilenetv2/onnx/model.onnx"),
        export_mobilenetv2,
    ),
    "mobilevit_xxs": (
        Path("benchmarks/onnx/vision/mobilevit_xxs/onnx/model.onnx"),
        export_mobilevit_xxs,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export vision benchmark models to ONNX.")
    parser.add_argument(
        "--only",
        nargs="+",
        choices=sorted(EXPORTERS),
        help="Export only the selected models.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=DEFAULT_OPSET,
        help="ONNX opset version to export.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    names = args.only or list(EXPORTERS)
    for name in names:
        output_path, exporter = EXPORTERS[name]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        exporter(output_path, args.opset)
        print(f"[export] {name} -> {output_path} (opset={args.opset})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
