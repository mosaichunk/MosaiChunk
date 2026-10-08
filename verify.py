"""Compare a generated run with known videos and optional saved latents."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch


def compare_video(actual, reference):
    counts, changed, first_difference, max_error, total_error, elements = 0, 0, None, 0, 0, 0
    hashes = [hashlib.sha256(), hashlib.sha256()]
    with imageio.get_reader(str(actual)) as a, imageio.get_reader(str(reference)) as b:
        for frame, pair in enumerate(itertools.zip_longest(a, b)):
            x, y = pair
            if x is None or y is None or x.shape != y.shape:
                return {
                    "identical": False,
                    "reason": "frame count or dimensions differ",
                    "frame": frame,
                }
            hashes[0].update(x.tobytes())
            hashes[1].update(y.tobytes())
            error = np.abs(x.astype(np.int16) - y.astype(np.int16))
            maximum = int(error.max())
            if maximum:
                changed += 1
                if first_difference is None:
                    first_difference = frame
            max_error = max(max_error, maximum)
            total_error += int(error.sum())
            elements += error.size
            counts += 1
    return {
        "identical": counts > 0 and changed == 0,
        "frames": counts,
        "changed_frames": changed,
        "first_difference": first_difference,
        "max_pixel_error": max_error,
        "mean_pixel_error": total_error / max(1, elements),
        "rgb_sha256": hashes[0].hexdigest(),
        "reference_rgb_sha256": hashes[1].hexdigest(),
    }


def compare_latents(actual, reference):
    a = torch.load(actual, map_location="cpu", weights_only=True)
    b = torch.load(reference, map_location="cpu", weights_only=True)
    if "x0" in a:
        if "x0" not in b or set(a["x0"]) != set(b["x0"]):
            return {"identical": False, "reason": "latent chunk keys differ"}
        keys = sorted(a["x0"])
        left, right = [a["x0"][key] for key in keys], [b["x0"][key] for key in keys]
    else:
        left, right = a["video"] + a["audio"], b["video"] + b["audio"]
    if not left or len(left) != len(right) or any(x.shape != y.shape for x, y in zip(left, right)):
        return {"identical": False, "reason": "latent shapes differ"}
    errors = [float((x.float() - y.float()).abs().max()) for x, y in zip(left, right)]
    return {
        "identical": all(torch.equal(x, y) for x, y in zip(left, right)),
        "max_error": max(errors),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument(
        "--reference",
        required=True,
        type=Path,
        help="JSON: scene ID -> {video, latents (optional)}.",
    )
    args = parser.parse_args()
    split = json.loads((args.run / "run.json").read_text())["arguments"]["split"]
    samples = json.loads((args.run / "samples.json").read_text())
    references = json.loads(args.reference.read_text())
    report = {}
    for index, row in enumerate(samples):
        if split == "i2v":
            folder = args.run / "videos" / row["sample_id"]
            video, latent = folder / "rollout.mp4", folder / "x0.pt"
        else:
            videos = list(args.run.glob(f"mosaichunk/run/media/**/prompt{index:04d}.mp4"))
            if len(videos) != 1:
                raise ValueError(f"Expected one video for {row['scene_id']}: {videos}")
            video, latent = videos[0], args.run / "latents" / f"{index:04d}.pt"
        reference = references[row["scene_id"]]
        result = {"video": compare_video(video, reference["video"])}
        if reference.get("latents"):
            result["latents"] = compare_latents(latent, reference["latents"])
        report[row["scene_id"]] = result
        print(row["scene_id"], result, flush=True)
    (args.run / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    if not all(item["identical"] for result in report.values() for item in result.values()):
        raise SystemExit("Reproduction mismatch; see verification.json.")


if __name__ == "__main__":
    main()
