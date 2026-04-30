from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path


def load_benchmark_download_specs() -> dict[str, dict[str, object]]:
    repo_root = Path(__file__).resolve().parents[1]
    catalog_path = repo_root / "src" / "onnx_rewrite" / "specs" / "catalog.py"
    spec = importlib.util.spec_from_file_location("benchmark_catalog", catalog_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load benchmark catalog from {catalog_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.BENCHMARK_DOWNLOAD_SPECS


BENCHMARK_DOWNLOAD_SPECS = load_benchmark_download_specs()


def build_command(name: str) -> list[str]:
    spec = BENCHMARK_DOWNLOAD_SPECS[name]
    command = [
        "hf",
        "download",
        str(spec["repo_id"]),
        "--repo-type",
        "model",
        "--local-dir",
        str(spec["local_dir"]),
    ]
    revision = spec.get("revision")
    if revision:
        command.extend(["--revision", str(revision)])
    for pattern in spec["include"]:
        command.extend(["--include", pattern])
    return command


def run_download(name: str) -> None:
    spec = BENCHMARK_DOWNLOAD_SPECS[name]
    local_dir = Path(spec["local_dir"])
    local_dir.mkdir(parents=True, exist_ok=True)
    command = build_command(name)
    print(f"[download] {name}: {' '.join(command)}")
    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download benchmark ONNX artifacts.")
    parser.add_argument(
        "--only",
        nargs="+",
        choices=sorted(BENCHMARK_DOWNLOAD_SPECS),
        help="Download only the named benchmark models.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_names = args.only or list(BENCHMARK_DOWNLOAD_SPECS)
    for name in model_names:
        run_download(name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
