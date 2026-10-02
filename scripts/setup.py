"""Install one pinned backbone and its Python environment (requires uv and git)."""

import argparse
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args, **kwargs):
    subprocess.run([str(arg) for arg in args], check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["t2v", "i2v"], required=True)
    parser.add_argument(
        "--metrics",
        action="store_true",
        help="Install metric dependencies and pinned Pi3X code in the I2V environment.",
    )
    parser.add_argument(
        "--wheel-dir", type=Path, help="Optional cache of compiled T2V attention wheels."
    )
    args = parser.parse_args()
    if args.metrics and args.split != "i2v":
        parser.error("Use the I2V environment to score either split.")
    name = "raven" if args.split == "t2v" else "lingbot"
    spec = json.loads((ROOT / "third_party/manifest.json").read_text())[name]
    checkout = ROOT / "third_party" / name
    patch = ROOT / "third_party/patches" / f"{name}.patch"
    if not checkout.exists():
        run("git", "clone", "--filter=blob:none", spec["url"], checkout)
        run("git", "-C", checkout, "checkout", "--detach", spec["revision"])
        run("git", "-C", checkout, "apply", patch)
    else:
        head = subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
        ).strip()
        if head != spec["revision"]:
            raise RuntimeError(f"{checkout}: expected {spec['revision']}, found {head}")
        run("git", "-C", checkout, "apply", "--reverse", "--check", patch)
    venv = ROOT / f".venv-{args.split}"
    python = venv / "bin/python"
    if not python.exists():
        run("uv", "venv", venv, "--python", "3.10.20" if args.split == "t2v" else "3.11")
    run(
        "uv",
        "pip",
        "install",
        "--python",
        python,
        "-r",
        ROOT / f"requirements/{args.split}.txt",
        "--extra-index-url",
        "https://download.pytorch.org/whl/cu128",
        "--index-strategy",
        "unsafe-best-match",
    )
    if args.split == "i2v":
        # PyTorch supplies the cuDNN loader. Pin its dynamically loaded engines
        # separately: system cuDNN upgrades otherwise change VAE conditioning.
        run(
            "uv",
            "pip",
            "install",
            "--python",
            python,
            "--no-deps",
            "--target",
            ROOT / "runtime-libs/cudnn97",
            "nvidia-cudnn-cu12==9.7.0.66",
        )
        # Keep the evaluated NumPy version; the OpenCV wheel runs with this ABI.
        run(
            "uv",
            "pip",
            "install",
            "--python",
            python,
            "--no-deps",
            "opencv-python-headless==5.0.0.93",
        )
        run(
            "uv",
            "pip",
            "install",
            "--python",
            python,
            "--no-deps",
            "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/"
            "flash_attn-2.8.3.post1%2Bcu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl",
        )
    else:
        sources = {
            "flash_attn-2*.whl": "git+https://github.com/Dao-AILab/flash-attention.git@060c9188beec3a8b62b33a3bfa6d5d2d44975fab",
            "flash_attn_3*.whl": "git+https://github.com/Dao-AILab/flash-attention.git@e2743ab5b3803bb672b16437ba98a3b1d4576c50#subdirectory=hopper",
            "magi_attention-*.whl": "git+https://github.com/SandAI-org/MagiAttention.git@e08bea8a051031978dbfcd069e2d876b36559bb9",
        }
        env = dict(
            os.environ,
            TORCH_CUDA_ARCH_LIST="9.0;9.0a",
            MAGI_ATTENTION_BUILD_COMPUTE_CAPABILITY="90",
            FLASH_ATTENTION_FORCE_BUILD="TRUE",
            FLASH_ATTN_CUDA_ARCHS="90",
            FLASH_ATTENTION_DISABLE_SM80="TRUE",
        )
        env.setdefault("MAX_JOBS", "16")
        for pattern, source in sources.items():
            wheels = list(args.wheel_dir.glob(pattern)) if args.wheel_dir else []
            if len(wheels) > 1:
                raise ValueError(f"Ambiguous cached wheels: {wheels}")
            run(
                "uv",
                "pip",
                "install",
                "--python",
                python,
                "--no-build-isolation",
                "--no-deps",
                wheels[0] if wheels else source,
                env=env,
            )
    if args.metrics:
        run(
            "uv",
            "pip",
            "install",
            "--python",
            python,
            "--no-deps",
            "-r",
            ROOT / "requirements/metrics.txt",
        )
        pi3 = ROOT / "third_party/pi3"
        spec = json.loads((ROOT / "third_party/manifest.json").read_text())["pi3"]
        revision = spec["revision"]
        if not pi3.exists():
            run("git", "clone", "--filter=blob:none", spec["url"], pi3)
            run("git", "-C", pi3, "checkout", "--detach", revision)
        elif (
            subprocess.check_output(["git", "-C", str(pi3), "rev-parse", "HEAD"], text=True).strip()
            != revision
        ):
            raise RuntimeError("Pi3 checkout revision differs from the evaluation recipe.")
    print(f"Ready: source {venv}/bin/activate")


if __name__ == "__main__":
    main()
