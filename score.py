"""Score generated rollouts with the paper's departure/revisit protocol."""

import argparse
import json
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import lpips
import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from transformers import CLIPImageProcessor, CLIPModel

from mosaichunk.metrics import _win, chunks_of, temp_ssim
from mosaichunk.runtime import ROOT, sha256


class ClipFeatures:
    def __init__(self, assets):
        self.model = CLIPModel.from_pretrained(assets / "CLIP").cuda().eval()
        self.processor = CLIPImageProcessor.from_pretrained(assets / "CLIP")

    @torch.no_grad()
    def __call__(self, images):
        features = []
        for start in range(0, len(images), 32):
            pixels = self.processor(images=images[start : start + 32], return_tensors="pt")[
                "pixel_values"
            ].cuda()
            embedding = self.model.get_image_features(pixel_values=pixels)
            features.append(torch.nn.functional.normalize(embedding.float(), dim=-1).cpu())
        return torch.cat(features).numpy()


def read_bgr(path):
    reader = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = reader.read()
        if not ok:
            break
        frames.append(frame)
    reader.release()
    if not frames:
        raise ValueError(f"No video frames in {path}")
    return frames


@torch.no_grad()
def reconstruct(model, frames):
    height, width = frames[0].shape[:2]
    scale = (255000 / float(width * height)) ** 0.5
    size = (
        max(14, int(round(width * scale / 14)) * 14),
        max(14, int(round(height * scale / 14)) * 14),
    )
    resized = np.stack(
        [
            np.asarray(Image.fromarray(frame).resize(size, Image.Resampling.LANCZOS))
            for frame in frames
        ]
    )
    inputs = torch.from_numpy(resized).permute(0, 3, 1, 2).float().div_(255.0)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        result = model(inputs[None].cuda())
    poses = result["camera_poses"][0].float().cpu().numpy()
    del result
    torch.cuda.empty_cache()
    return poses


def revisit(poses, translation):
    poses = np.asarray(poses, dtype=np.float64)
    forward = poses[:, :3, 2]
    cosine = (forward @ forward[0]) / (
        np.linalg.norm(forward, axis=1) * np.linalg.norm(forward[0]) + 1e-9
    )
    heading = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    departure = (
        np.linalg.norm(poses[:, :3, 3] - poses[0, :3, 3], axis=1) if translation else heading
    )
    turn = int(np.argmax(departure))
    return turn + int(np.argmin(heading[turn:])), turn


@torch.no_grad()
def pair_metrics(clip, perceptual, frames, source, target, perceptual_frames=None):
    images = [frames[source], frames[target]]
    features = clip(images)
    images = (
        [perceptual_frames[source], perceptual_frames[target]]
        if perceptual_frames is not None
        else images
    )
    inputs = [
        torch.from_numpy(im.copy()).permute(2, 0, 1)[None].float().div_(127.5).sub_(1.0).cuda()
        for im in images
    ]
    return {
        "clip": float(features[0] @ features[1]),
        "lpips_alex": float(perceptual(*inputs).flatten()[0]),
    }


@torch.no_grad()
def quality(clip, bgr, split):
    # The published Drift script passed OpenCV BGR arrays to CLIP. Keep that
    # convention here explicitly; RGB would define a different numerical series.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    gray = (
        torch.from_numpy(np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in bgr]))
        .float()
        .cuda()
    )
    ssim = temp_ssim(gray, _win())
    indices, owner = [], []
    for chunk, (lo, hi) in enumerate(chunks_of(len(bgr), split)):
        selected = np.linspace(lo, hi, min(4, hi - lo + 1)).round().astype(int)
        indices += list(selected)
        owner += [chunk] * len(selected)
    features = torch.as_tensor(clip([bgr[i] for i in indices])).float()
    pooled = []
    for chunk in sorted(set(owner)):
        value = features[[i for i, c in enumerate(owner) if c == chunk]].mean(0)
        pooled.append(value / value.norm())
    pooled = torch.stack(pooled)
    drift = 1 - (pooled[1:] * pooled[:-1]).sum(-1)
    return {"tempssim": float(ssim.mean()), "drift": float(drift.mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--assets", type=Path, default=ROOT / "assets")
    parser.add_argument(
        "--pairs", type=Path, help="T2V JSON: scene ID -> {departure_frame, revisit_frame}."
    )
    parser.add_argument(
        "--paper-pairs",
        action="store_true",
        help="Use published annotations only after verifying rollout equality.",
    )
    parser.add_argument(
        "--quality",
        action="store_true",
        help="Also compute Temporal SSIM and the published Drift convention.",
    )
    args = parser.parse_args()
    run = json.loads((args.run / "run.json").read_text())["arguments"]
    samples = json.loads((args.run / "samples.json").read_text())
    split = run["split"]
    if split == "t2v" and bool(args.pairs) == args.paper_pairs:
        parser.error("T2V requires exactly one of --pairs or --paper-pairs.")
    pairs = {}
    if args.pairs:
        pairs = json.loads(args.pairs.read_text())
    elif args.paper_pairs:
        method = "Base" if run["method"] == "base" else "MosaiChunk"
        path = args.assets / "RememBench/evaluation/t2v_reference_pairs.jsonl"
        pairs = {
            r["scene_id"]: r
            for r in map(json.loads, path.read_text().splitlines())
            if r["method"] == method and r["comparison_budget_chunks"] == run["budget"]
        }
    reconstruction = None
    if split == "i2v":
        sys.path.insert(0, str(ROOT / "third_party/pi3"))
        from pi3.models.pi3x import Pi3X

        reconstruction = Pi3X().cuda().eval()
        reconstruction.load_state_dict(
            load_file(str(args.assets / "Pi3X/model.safetensors")), strict=True
        )
    clip = ClipFeatures(args.assets)
    perceptual = lpips.LPIPS(net="alex", version="0.1", verbose=False).cuda().eval()
    results = {}
    for i, row in enumerate(samples):
        if split == "i2v":
            path = args.run / "videos" / row["sample_id"] / "rollout.mp4"
        else:
            matches = list(args.run.glob(f"mosaichunk/run/media/**/prompt{i:04d}.mp4"))
            if len(matches) != 1:
                raise ValueError(f"Expected one video for {row['scene_id']}, found {matches}")
            path = matches[0]
        bgr = read_bgr(path)
        if len(bgr) != row["num_frames"]:
            raise ValueError(f"{path}: expected {row['num_frames']} frames, found {len(bgr)}")
        if reconstruction is not None:
            rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in bgr]
            poses = reconstruct(reconstruction, rgb)
            target, turn = revisit(poses, row["translation"])
            source = 0
            np.savez_compressed(
                path.with_name("pi3_poses.npz"), poses=poses, idx=np.arange(len(bgr))
            )
        else:
            source, target = (
                pairs[row["scene_id"]]["departure_frame"],
                pairs[row["scene_id"]]["revisit_frame"],
            )
            turn = None
        if not 0 <= source <= target < len(bgr):
            raise ValueError(f"Invalid frame pair: {source}, {target}")
        # The pair CLIP implementation decodes with imageio/FFmpeg; keep its pixels.
        with imageio.get_reader(str(path)) as reader:
            frames = {j: frame for j, frame in enumerate(reader) if j in (source, target)}
        cudnn = torch.backends.cudnn.allow_tf32
        torch.backends.cudnn.allow_tf32 = True
        values = pair_metrics(
            clip, perceptual, frames, source, target, rgb if split == "i2v" else None
        )
        torch.backends.cudnn.allow_tf32 = cudnn
        if args.quality:
            values.update(quality(clip, bgr, split))
        results[row["scene_id"]] = dict(
            values,
            departure_frame=int(source),
            revisit_frame=int(target),
            turnaround_frame=turn,
            video_sha256=sha256(path),
        )
        print(row["scene_id"], values, flush=True)
        (args.run / "scores.json").write_text(
            json.dumps({"split": split, "scenes": results}, indent=2) + "\n"
        )
    names = ["clip", "lpips_alex"] + (["tempssim", "drift"] if args.quality else [])
    summary = {name: float(np.median([row[name] for row in results.values()])) for name in names}
    (args.run / "scores.json").write_text(
        json.dumps(
            {"split": split, "n_scenes": len(results), "median": summary, "scenes": results},
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
