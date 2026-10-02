"""I2V geometry, context layout, and scene-disjoint training split."""

import os
import random
import re

DIT_DIM = 5120
N_LAYERS = 40
N_HEADS = 40
HEAD_DIM = DIT_DIM // N_HEADS
N_SLABS = N_LAYERS * 2
LAT_H, LAT_W = (60, 104)
GRID_H, GRID_W = (30, 52)
FRAME_TOKENS = GRID_H * GRID_W
CHUNK = 4
CHUNK_TOKENS = CHUNK * FRAME_TOKENS
FPC = 4 * CHUNK


def chunk_slice(c, n=1):
    """Pixel-frame slice of chunk c (or n consecutive chunks starting at c)."""
    return slice(c * FPC, (c + n) * FPC)


def chunk_of(frame):
    """Which chunk a pixel frame index belongs to."""
    return frame // FPC


N_FAR = int(os.environ.get("PTR_N_FAR", "3"))
F_SINK = 0
F_FAR = tuple((4 + 4 * i for i in range(N_FAR)))
F_PAD = int(os.environ.get("PTR_F_PAD", "0"))
N_LOCAL_SLOTS = int(os.environ.get("PTR_N_LOCAL_SLOTS", "2"))
F_LOCAL = tuple((F_FAR[-1] + 8 + F_PAD - 4 * i for i in reversed(range(N_LOCAL_SLOTS))))
F_CUR = F_LOCAL[-1] + 4
ORACLE_TOKENS = len(F_FAR) * CHUNK_TOKENS
FAR_FROM_CHUNK = int(os.environ.get("PTR_FAR_FROM", "0"))
FAR_ANCHORS = list(range(F_FAR[0], F_FAR[-1] + 4 - CHUNK + 1))
BUDGET_CHUNKS = 2.0
MEM_TOKENS = int(BUDGET_CHUNKS * CHUNK_TOKENS)
N_STEPS = 4
STEP_IDX = (0, 250, 500, 750)
THETA_DEG = 120.0
FRAMES = 321
N_CH = ((FRAMES - 1) // 4 + 1) // CHUNK * CHUNK // CHUNK
CKPT_SRC = os.environ.get("MOSAICHUNK_BACKBONE", "assets/LingBot")
CLIP_LIST = os.environ.get("MOSAICHUNK_TRAIN_LIST", "data/train/i2v/clips.txt")
RUNS = os.environ.get("MOSAICHUNK_OUTPUT", "outputs/train/i2v")
VAL_FRAC, TEST_FRAC, SPLIT_SEED, CLIPS_PER_SCENE = (0.05, 0.05, 0, 3)


def scene_of(clip):
    """sekai clip dirname -> scene id. '0WFAbXjpC1s_s02_c00__room_00' -> '0WFAbXjpC1s'."""
    return re.sub("_s\\d+_c\\d+$", "", os.path.basename(clip).split("__")[0])


def split(clip_list=CLIP_LIST, which="train"):
    """Scene-level split, no leakage. which in {all, train, val, test}."""
    with open(clip_list) as stream:
        clips = [line.strip() for line in stream if line.strip()]
    missing = [clip for clip in clips if not os.path.isdir(clip)]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} input clips; first: {missing[0]}")
    if which == "all":
        return sorted(clips)
    by = {}
    for c in clips:
        by.setdefault(scene_of(c), []).append(c)
    scenes = sorted(by)
    random.Random(SPLIT_SEED).shuffle(scenes)
    n_test = max(1, int(len(scenes) * TEST_FRAC))
    n_val = max(1, int(len(scenes) * VAL_FRAC))
    test_s, val_s = (set(scenes[:n_test]), set(scenes[n_test : n_test + n_val]))
    if which == "test":
        return sorted((c for s in scenes if s in test_s for c in by[s][:CLIPS_PER_SCENE]))
    if which == "val":
        return sorted((c for s in scenes if s in val_s for c in by[s][:CLIPS_PER_SCENE]))
    return sorted((c for s in scenes if s not in test_s and s not in val_s for c in by[s]))
