"""Chunk geometry and temporal positions for retrieved H3-AR KV."""

from __future__ import annotations

import torch

from .rope_reanchor import rope_freqs, video_t_grid


def chunk_frames(layout, vk: int) -> int:
    _, gh, gw = layout.video_patch_grid
    return layout.chunks[vk].video_rows // (gh * gw)


def frame_offset(layout, vk: int) -> int:
    """Global latent-frame index of video chunk vk's first frame."""
    return sum((chunk_frames(layout, k) for k in range(vk)))


def total_frames(layout) -> int:
    return sum((chunk_frames(layout, k) for k in range(len(layout.chunks))))


def row_thw(layout, vk: int, rows: torch.Tensor, t_grid: torch.Tensor) -> torch.Tensor:
    """Row indices WITHIN cache chunk vk+1 -> [N, 3] (t, h, w)."""
    _, gh, gw = layout.video_patch_grid
    ch = layout.chunks[vk]
    cells = gh * gw
    r = rows.to(torch.long) - ch.audio_rows
    assert int(r.min()) >= 0, "a section row landed in the audio block"
    f = r // cells
    rem = r % cells
    g = frame_offset(layout, vk)
    return torch.stack(
        [
            t_grid[(g + f).clamp_max(t_grid.numel() - 1)].float(),
            (rem // gw).float(),
            (rem % gw).float(),
        ],
        dim=-1,
    )


def far_band(layout, c: int, n_local: int, n_slots: int) -> tuple[int, int]:
    """The FRAME range a re-anchored row may land in: [lo, hi) in global latent frames."""
    hi_chunk = max(0, c - n_local)
    hi = frame_offset(layout, hi_chunk)
    width = sum((chunk_frames(layout, k) for k in range(max(0, hi_chunk - n_slots), hi_chunk)))
    return (max(0, hi - width), hi)


def anchors_for(n_src: int, band: tuple[int, int], span: int) -> list[int]:
    """n contributing source chunks -> one START FRAME each, spread over the band."""
    lo, hi = band
    top = max(lo, hi - span)
    if n_src <= 1:
        return [top]
    return [lo + round(r * (top - lo) / (n_src - 1)) for r in range(n_src)]


class Geometry:
    """Per-rollout cache of the time grid and the model's inv_freq."""

    def __init__(self, layout, text_len: int, inv_freq: torch.Tensor):
        self.layout = layout
        self.inv_freq = inv_freq.detach().float().cpu()
        self.t_grid = video_t_grid(total_frames(layout), float(text_len))

    def freqs_at(self, vk: int, rows: torch.Tensor, dest_start: int | None = None):
        """freqs for `rows` of chunk vk as cached, and as they would be at `dest_start`."""
        pos = row_thw(self.layout, vk, rows, self.t_grid)
        f_old = rope_freqs(pos, self.inv_freq)
        if dest_start is None:
            return (f_old, f_old)
        _, gh, gw = self.layout.video_patch_grid
        cells = gh * gw
        r = rows.to(torch.long) - self.layout.chunks[vk].audio_rows
        f_in = r // cells
        t_new = self.t_grid[(int(dest_start) + f_in).clamp_(0, self.t_grid.numel() - 1)].float()
        pos_new = torch.stack([t_new, pos[:, 1], pos[:, 2]], dim=-1)
        return (f_old, rope_freqs(pos_new, self.inv_freq))


def find_inv_freq(backbone) -> torch.Tensor:
    """The rope buffer, wherever FSDP and peft have put it."""
    for m in backbone.modules():
        if type(m).__name__ == "MiniMaxH3Rope":
            return m.inv_freq
    raise RuntimeError("MiniMaxH3Rope not found in the backbone: cannot re-anchor")


__all__ = [
    "Geometry",
    "far_band",
    "anchors_for",
    "find_inv_freq",
    "chunk_frames",
    "frame_offset",
    "total_frames",
    "row_thw",
]
