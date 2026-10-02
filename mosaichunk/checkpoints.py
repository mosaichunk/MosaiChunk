"""Load released router weights with their architecture and retrieval settings."""

import json
from pathlib import Path


def load_i2v_router(path):
    from safetensors.torch import load_file

    path = Path(path)
    config = json.loads(path.with_name("config.json").read_text())
    if config["task"] != "image-to-video" or config["format_version"] != 1:
        raise ValueError("Expected a version-1 MosaiChunk I2V checkpoint.")
    return {
        "sel": load_file(str(path)),
        "gstep": config.get("training_step", config["source_checkpoint"]),
        "args": {
            "n_sections": config["sections"]["n_sections"],
            "d_desc": config["selector"]["d"],
            "train_alpha": config["selector"]["train_alpha"],
            "budget_chunks": config["inference"]["far_budget_chunks"],
            "mmr_lam": config["inference"]["mmr_lam"],
        },
    }
