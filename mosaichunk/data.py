"""Read RememBench without changing scene identity, prompt text, or noise seeds."""

import json
import shutil
from pathlib import Path


def read_samples(root, split, scene_ids=None, setting="rotation_180", limit=None):
    root = Path(root)
    rows = [
        json.loads(line) for line in (root / "data" / split / "test.jsonl").read_text().splitlines()
    ]
    if split == "i2v":
        rows = [row for row in rows if row["setting"] == setting]
    if scene_ids:
        requested = list(scene_ids)
        by_id = {row["scene_id"]: row for row in rows}
        missing = set(requested) - by_id.keys()
        if missing:
            raise ValueError(f"Unknown scenes for {split}/{setting}: {sorted(missing)}")
        rows = [by_id[name] for name in requested]
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive.")
        rows = rows[:limit]
    if not rows:
        raise ValueError("No samples selected.")
    for row in rows:
        # JSON seeds are decimal strings: float conversion would lose 63-bit precision.
        if not isinstance(row["seed"], str) or not row["seed"].isdigit():
            raise ValueError(f"Invalid seed for {row['scene_id']}")
    return rows


def prepare_i2v(root, rows, output):
    import numpy as np

    root, output = Path(root), Path(output)
    clips = []
    for row in rows:
        image = root / row["image_path"]
        if not image.is_file():
            raise FileNotFoundError(
                f"{image} is missing. Run the RememBench image preparation script."
            )
        folder = output / row["sample_id"]
        (folder / "gt_frames").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(image, folder / "gt_frames/00000.png")
        (folder / "prompt.txt").write_text(row["prompt"] + "\n")
        with np.load(root / row["trajectory_path"], allow_pickle=False) as trajectory:
            np.save(folder / "poses.npy", trajectory["poses"])
            np.save(folder / "intrinsics.npy", trajectory["intrinsics"])
        clips.append(str(folder.resolve()))
    path = output / "clips.txt"
    path.write_text("\n".join(clips) + "\n")
    return path


def prepare_t2v(rows, output):

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    prompts = [row["prompts"][0].strip() for row in rows]
    if len(set(prompts)) != len(prompts):
        raise ValueError("Duplicate first-segment prompts cannot identify seeds unambiguously.")
    scenes = output / "prompts.txt"
    scenes.write_text("\n".join(prompts) + "\n")
    segments = output / "segments.json"
    segments.write_text(
        json.dumps({p: row["prompts"] for p, row in zip(prompts, rows)}, indent=2) + "\n"
    )
    seeds = output / "seeds.json"
    from .t2v.seed_pin import key_of

    table = {key_of(p): int(row["seed"]) for p, row in zip(prompts, rows)}
    if len(table) != len(rows):
        raise ValueError("Normalized prompt identities must be unique.")
    seeds.write_text(json.dumps(table, indent=2) + "\n")
    return scenes, segments, seeds
