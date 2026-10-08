# Working on MosaiChunk

Start with `README.md`, `docs/evaluation.md`, and `configs/t2v.yaml`. There are two isolated runtime environments; do not combine their Torch or attention packages.

- `train.py`, `test.py`, `mosaichunk/cli.py`: public commands and fixed recipes.
- `mosaichunk/i2v/`: LingBot conditioning, KV capture, balanced sections, selection, composition, and training.
- `mosaichunk/t2v/`: RAVEN segmented prompts, sequence-parallel section features, CPU history, selection, and self-distillation. Train and test share `ptr_rollout.py` and `ptr_common.py`.
- `scripts/setup.py`, `scripts/download.py`: pinned code, data, and weights. Upstream checkouts are ignored, with small tracked patches under `third_party/patches/`.
- `tests/test_protocol.py`: CPU tests for seeds, budgets, partition coverage, gradients, and training trajectories.

Preserve these invariants:

1. RememBench seeds are decimal strings. Parse directly to integers; prompt reordering must never change a scene's noise.
2. T2V requires the frozen RAVEN streaming LoRA in addition to MiniMax-H3 and the router checkpoint.
3. Match Base's total active context to MosaiChunk, including each backbone's sink. I2V's positional local slots must expand with the Base window.
4. Store unrotated keys; re-anchor temporal RoPE only, retaining spatial coordinates. Pool full-head features under T2V sequence parallelism.
5. Keep the full-candidate softmax and detached mean normalization of selected value gates. Released checkpoints use redundancy coefficient 1.0.
6. T2V test boundaries are `5:9:16`; training uses `3:6:11`. Prompt padding and blend order affect outputs.
7. I2V revisit selection uses each output video's reconstructed camera poses, never the commanded poses. Original T2V annotations apply only to matching rollouts.
8. Never commit tokens, downloaded weights/data, generated videos, caches, or machine-specific paths. Never treat visual similarity as bit-exact reproduction.
9. Preserve pinned CPU threading and both splits' cuDNN engines. CPU consumers must wait for GPU-to-CPU KV copies before reading them; async conversion followed by CPU assignment can corrupt history.

Run `python -m unittest discover -s tests` in the I2V environment with metric dependencies installed. For runtime changes, use `verify.py` to compare fixed-scene latents and decoded frames against a known reference, retaining the configuration and environment. Do not replace the paper recipe with a faster approximate implementation.
