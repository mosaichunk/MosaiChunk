# Evaluation protocol

The launchers pin dataset, router, and backbone revisions. Each scene retains its prompt and seed across methods; I2V also retains its initial frame and complete input camera path. T2V runs 379 frames at 24 fps and 1376 × 768; I2V runs 253 frames at 16 fps and 832 × 480. Both use four denoising steps per chunk.

| Far budget | MosaiChunk | Base |
|---|---|---|
| 1 chunk | 2 local + 1 selected | 3 local |
| 2 chunks | 2 local + 2 selected | 4 local |

The backbone's attention sink is additional and identical across methods. I2V uses a visual sink; H3-AR's sink holds text. T2V has 40 sections per chunk and I2V has 48. The released routers use the same full-candidate scoring and redundancy coefficient (1.0) in train and test.

CLIP uses ViT-H/14 LAION2B; LPIPS uses AlexNet v0.1. Report medians over scenes. Use the original generated MP4s, before website packaging or resizing.

- **T2V:** use each method's manually marked departure/revisit pair. The RememBench annotations describe the original published rollouts; annotate new outputs if their actions occur at different frames.
- **I2V:** reconstruct all 253 frames with Pi3X in one pass, resizing only reconstruction inputs to approximately 255,000 pixels with dimensions divisible by 14. Identify the turnaround from maximum heading departure (rotation) or maximum position departure (translation), then select the smallest unsigned heading difference to the initial frame on the return leg. Score that frame against frame 0 at the original video resolution.
- **Temporal SSIM:** mean grayscale SSIM between adjacent frames, using an 11 × 11 Gaussian window, sigma 1.5.
- **Drift:** mean adjacent-chunk CLIP cosine distance. Each chunk descriptor averages four evenly spaced frame embeddings and is normalized. Decoded chunk lengths are 5 then 17 for T2V, and 13 then 16 for I2V.

The original Drift implementation passes OpenCV BGR arrays directly to CLIP. `--quality` retains this convention to reproduce the reported series; converting those inputs to RGB changes the values. Departure/revisit CLIP and LPIPS use RGB. Pair CLIP uses imageio/FFmpeg decoding, as in the original evaluation.

Exact numerical reproduction additionally depends on GPU, Torch, attention kernels, CUDA libraries, and video decoding. Record these alongside the code revision. A shared seed alone does not establish equality. Compare saved latents first, then decoded RGB frames and scores; container-level MP4 hashes may differ without any pixel difference.

The I2V launcher uses cuDNN 9.7.0.66 convolution engines installed by `scripts/setup.py`, alongside PyTorch's own cuDNN loader. Loading newer system engines changes the initial VAE conditioning and therefore the entire autoregressive rollout. Each completed run writes `environment.json` with package versions and loaded CUDA libraries.

The launchers also pin one OpenMP thread per process, matching the reference runs. This matters for T2V's CPU section partitioning: different reduction orders can change ties during k-means initialization.
