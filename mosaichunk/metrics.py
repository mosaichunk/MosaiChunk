"""Paper metrics, preserving frame selection, precision, and decoding conventions."""

import torch
import torch.nn.functional as F

C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2


def _win(sigma=1.5, size=11, device="cuda"):
    c = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2
    g = torch.exp(-(c**2) / (2 * sigma**2))
    g = (g / g.sum())[:, None]
    return (g @ g.T)[None, None]


@torch.no_grad()
def temp_ssim(gray, win):
    """Mean SSIM over consecutive frame pairs. `gray` is [T,H,W] float32 on GPU, 0..255."""
    a, b = (gray[:-1][:, None], gray[1:][:, None])
    mu_a = F.conv2d(a, win)
    mu_b = F.conv2d(b, win)
    aa, bb, ab = (mu_a * mu_a, mu_b * mu_b, mu_a * mu_b)
    va = F.conv2d(a * a, win) - aa
    vb = F.conv2d(b * b, win) - bb
    vab = F.conv2d(a * b, win) - ab
    s = (2 * ab + C1) * (2 * vab + C2) / ((aa + bb + C1) * (va + vb + C2))
    return s.mean(dim=(1, 2, 3))


def chunks_of(n_frames, kind):
    """Pixel-frame ranges of the backbone's own chunks."""
    if kind == "t2v":
        out = [(0, 4)] + [(5 + 17 * (k - 1), 5 + 17 * k - 1) for k in range(1, 23)]
    else:
        out = [(0, 12)] + [(13 + 16 * (k - 1), 13 + 16 * k - 1) for k in range(1, 16)]
    assert out[-1][1] == n_frames - 1, (
        f"{kind}: chunks end at {out[-1][1]}, clip has {n_frames} frames"
    )
    return out
