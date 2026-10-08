"""Export a trained router, excluding the backbone, optimizer, and training state."""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=["t2v", "i2v"])
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    t2v = args.split == "t2v"
    if t2v:
        import torch.distributed.checkpoint as dcp

        reader = dcp.FileSystemReader(args.source)
        metadata = reader.read_metadata()
        prefix = "models.selector."
        weights = {
            name[len(prefix) :]: torch.empty(tuple(item.size), dtype=item.properties.dtype)
            for name, item in metadata.state_dict_metadata.items()
            if name.startswith(prefix)
        }
        dcp.load(state_dict={"models": {"selector": weights}}, storage_reader=reader, no_dist=True)
        step = int(args.source.name)
        settings = {}
    else:
        source = torch.load(args.source, map_location="cpu", weights_only=True)
        weights, step, settings = source["sel"], source["gstep"], source["args"]
    if not weights or not all(torch.isfinite(value).all() for value in weights.values()):
        raise ValueError("Checkpoint has missing or non-finite router weights.")
    args.output.mkdir(parents=True, exist_ok=True)
    save_file(
        {name: value.detach().cpu().contiguous().clone() for name, value in weights.items()},
        str(args.output / "model.safetensors"),
    )
    config = {
        "format_version": 1,
        "task": "text-to-video" if t2v else "image-to-video",
        "source_checkpoint": args.source.name,
        "training_step": step,
        "component": "selector",
        "selector": {
            "d_in": 7168 if t2v else 5120,
            "n_views": 9,
            "d": settings.get("d_desc", 1024),
            "centre": True,
            "train_alpha": False,
        },
        "sections": {
            "partition": "kmeans",
            "n_sections": settings.get("n_sections", 40 if t2v else 48),
            "descriptor_layers": [12, 25, 38] if t2v else [10, 20, 30],
            "pooling_moments": ["mean", "max", "min"],
        },
        "inference": {
            "far_budget_chunks": settings.get("budget_chunks", 2),
            "n_local": 2,
            "mmr_lam": settings.get("mmr_lam", 1.0),
        },
    }
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()
