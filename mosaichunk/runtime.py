"""Resolve explicit local dependencies before importing either video backbone."""

import atexit
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def bootstrap(split):
    name = "raven" if split == "t2v" else "lingbot"
    source = ROOT / "third_party" / name
    if not source.is_dir():
        raise FileNotFoundError(
            f"{source} is missing. Run python scripts/setup.py --split {split}."
        )
    sys.path.insert(0, str(source))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return source


def sha256(path):
    result = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def revision():
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except subprocess.CalledProcessError:
        return "unversioned"


def record_environment_on_exit():
    """Record the libraries actually loaded by rank zero, without credentials."""
    output = os.environ.get("MOSAICHUNK_OUTPUT")
    if not output or int(os.environ.get("RANK", "0")) != 0:
        return

    def write():
        import torch

        packages = {}
        for name in (
            "torch",
            "numpy",
            "transformers",
            "diffusers",
            "flash-attn",
            "flash-attn-3",
            "magi-attention",
            "nvidia-cudnn-cu12",
            "nvidia-cublas-cu12",
        ):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                pass
        maps = Path("/proc/self/maps")
        libraries = (
            sorted(
                {
                    line.split()[-1]
                    for line in maps.read_text().splitlines()
                    if any(name in line for name in ("libcudnn", "libcublas", "libcuda.so"))
                }
            )
            if maps.exists()
            else []
        )
        report = {
            "python": platform.python_version(),
            "packages": packages,
            "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_initialized() else None,
            "loaded_cuda_libraries": libraries,
        }
        (Path(output) / "environment.json").write_text(json.dumps(report, indent=2) + "\n")

    atexit.register(write)
