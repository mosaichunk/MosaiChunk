# Training

Only the descriptor encoder and its query/key projections are trained. Teacher and student share the frozen generator, noise, denoising step, and local context. The teacher has richer historical memory. The loss is prediction MSE, scaled by 100.

## T2V

`data/train/t2v/` contains 2,000 four-segment training prompts. The recipe uses 260 frames (16 chunks), segment boundaries `3:6:11`, four denoising steps, and AdamW at `1e-4`. The paper run used 32 H200 GPUs, sequence parallel size 4, and FSDP shard size 8. The released router is step 416. The default single-node command is functional but changes the effective batch size; it is not an exact replay of the 32-GPU training run.

For the 32-GPU recipe, run the following on each of four nodes, changing `RANK` to 0, 1, 2, or 3 and `HOST` to node 0's address. All nodes must share data and output paths.

```bash
python train.py --split t2v --nodes 4 --node-rank RANK --master-addr HOST --nproc 8 --output outputs/train/t2v
```

`--dry-run` writes the resolved configuration and command without loading models.

## I2V

The manifest `data/train/i2v/clips.txt` fixes the original Sekai clip identities. Under `--train-data`, each named directory must contain:

```
CLIP_NAME/
  gt_frames/00000.png
  prompt.txt
  intrinsics.npy
```

The image is the prepared 832 × 480 initial frame. Intrinsics contain `[fx, fy, cx, cy]` in that image's pixel coordinates. Preserve clip basenames: they determine scene grouping and training seeds. Source footage is not redistributed with this code. RememBench's test inputs must not be used for training.

Scene splitting uses seed 0, 5% validation and 5% test scenes, and at most three clips per held-out scene. The original prepared corpus yields 2,388 training clips. The canonical training camera turns 120° and returns. The recipe uses 20 chunks, AdamW at `2e-4`, 200 warmup steps, two epochs, and manual gradient averaging of the router across ranks; the frozen backbone uses FSDP. The paper run used 16 H200 GPUs; the released checkpoint is step 500. Use the same multi-node options with `--nodes 2 --nproc 8` to match the GPU count.

For a small training check, add `--max-clips 1 --chunks 5 --epochs 1`. This deliberately changes training length and is only a smoke test. Resume an I2V run with `--resume outputs/train/i2v/run/last.pt`.

I2V resumes the saved optimizer and restarts the recorded clip; it does not serialize an in-progress video's KV bank. A checkpoint taken midway through a clip therefore does not promise a bit-exact continuation of interrupted training.

## Evaluate your router

```bash
python export.py --split i2v --source outputs/train/i2v/run/last.pt --output outputs/router/i2v
python test.py --split i2v --method mc --checkpoint outputs/router/i2v/model.safetensors --limit 3 --output outputs/my-router
```

For T2V, point `--source` to a numbered distributed checkpoint directory under `outputs/train/t2v/mosaichunk/run/checkpoints/`. Exporting copies only the learned router tensors.
