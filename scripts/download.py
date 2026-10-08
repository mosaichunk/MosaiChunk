"""Download the fixed release revisions; existing HF login is used automatically."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

RELEASES = {
    "CLIP": (
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        "model",
        "1c2b8495b28150b8a4922ee1c8edee224c284c0c",
        ["*.json", "*.txt", "model.safetensors"],
    ),
    "Pi3X": (
        "yyfz233/Pi3X",
        "model",
        "bb1deea4d7423de5b30691739cb451a3f57dc1d5",
        ["*.json", "model.safetensors"],
    ),
    "RememBench": (
        "mosaichunk/RememBench",
        "dataset",
        "cdeabf0fce9c075876a903a1d5234483ad677fd5",
        None,
    ),
    "MosaiChunk": (
        "mosaichunk/MosaiChunk",
        "model",
        "03eb1531e2363cbee4fe27d773c4242a07eb1b32",
        None,
    ),
    "LingBot": (
        "robbyant/lingbot-world-v2-14b-causal-fast",
        "model",
        "5c33dd40b213598c418fd25bff30fdbd23fd38a7",
        None,
    ),
    "MiniMax-H3": (
        "MiniMaxAI/MiniMax-H3",
        "model",
        "42ed227ee7df40d41602854ae760620d6eb651fe",
        [
            "FL2VA/transformer/*",
            "FL2VA/tokenizer/*",
            "FL2VA/text_encoder/*",
            "FL2VA/video_vae/*",
            "audio_vae/*",
        ],
    ),
    "MiniMax-H3-RAVEN-Streaming-LoRA": (
        "mvp-lab/MiniMax-H3-RAVEN-Streaming-LoRA",
        "model",
        "90f30b11639e73ce0e3f2ab6ed70d1a133a66caf",
        ["*.safetensors"],
    ),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=["t2v", "i2v"])
    parser.add_argument("--metrics", action="store_true", help="Also download CLIP and Pi3X.")
    parser.add_argument("--assets", type=Path, default=Path("assets"))
    parser.add_argument(
        "--image-limit", type=int, help="Prepare only the first N I2V conditioning images."
    )
    args = parser.parse_args()
    names = ["RememBench", "MosaiChunk"] + (
        ["LingBot"] if args.split == "i2v" else ["MiniMax-H3", "MiniMax-H3-RAVEN-Streaming-LoRA"]
    )
    if args.metrics:
        names += ["CLIP", "Pi3X"]
    for name in names:
        repo, kind, revision, patterns = RELEASES[name]
        snapshot_download(
            repo,
            repo_type=kind,
            revision=revision,
            local_dir=args.assets / name,
            allow_patterns=patterns,
        )
    model = args.assets / "MosaiChunk"
    for line in (model / "checksums.sha256").read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        digest = hashlib.sha256((model / name.strip()).read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f"Checkpoint checksum mismatch: {name}")
    if args.split == "i2v":
        command = [
            sys.executable,
            str(Path(__file__).with_name("prepare_images.py")),
            "--data",
            str(args.assets / "RememBench"),
        ]
        if args.image_limit:
            command += ["--limit", str(args.image_limit)]
        subprocess.run(command, check=True)
    (args.assets / f"{args.split}-revisions.json").write_text(
        json.dumps({name: RELEASES[name][:3] for name in names}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
