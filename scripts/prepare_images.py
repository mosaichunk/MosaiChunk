"""Retrieve only the conditioning frames selected for the I2V test run."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import fsspec
from huggingface_hub import get_token, hf_hub_url
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("assets/RememBench"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location(
        "benchmark_images", args.data / "scripts/prepare_i2v_images.py"
    )
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    samples = [json.loads(s) for s in (args.data / "data/i2v/test.jsonl").read_text().splitlines()]
    ids = list(dict.fromkeys(s["scene_id"] for s in samples))
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        ids = ids[: args.limit]
    sources = {
        s["scene_id"]: s
        for s in map(json.loads, (args.data / "sources/i2v_images.jsonl").read_text().splitlines())
    }
    token = get_token()
    for i, scene in enumerate(ids, 1):
        row = sources[scene]
        target = args.data / row["image_path"]
        if target.exists():
            image = Image.open(target).convert("RGB")
            if hashlib.sha256(image.tobytes()).hexdigest() != row["rgb_sha256"]:
                raise ValueError(f"Existing image does not match RememBench: {scene}")
        else:
            url = hf_hub_url(
                row["source_repo"],
                row["source_archive"],
                repo_type="dataset",
                revision=row["source_revision"],
            )
            headers = {"Authorization": "Bearer " + token} if token else {}
            with fsspec.open(url, mode="rb", block_size=65536, headers=headers) as stream:
                image = helper.prepare_image(helper.archive_image(stream), row)
            target.parent.mkdir(parents=True, exist_ok=True)
            image.save(target)
        print(f"{i}/{len(ids)} verified {scene}", flush=True)


if __name__ == "__main__":
    main()
