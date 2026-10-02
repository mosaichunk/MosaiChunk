"""The retrieval and composition rule shared by T2V training and evaluation."""

from __future__ import annotations

import math
import os

import torch

from . import ptr_geom
from .rope_reanchor import unapply_rope
from .sections import allocate_global, kmeans_sections, maxsim
from .selector import pool_views
from .sp_exact import gather_features

DESC_LAYERS = tuple((int(x) for x in os.environ.get("PTR_DESC_LAYERS", "12,25,38").split(",")))
N_SECTIONS = int(os.environ.get("PTR_N_SECTIONS", "40"))
N_LOCAL = int(os.environ.get("PTR_N_LOCAL", "2"))
TEACHER_SINK = int(os.environ.get("PTR_TEACHER_SINK", "6"))
STUDENT_SINK = int(os.environ.get("PTR_STUDENT_SINK", "1"))
EXPLORE = float(os.environ.get("PTR_EXPLORE", "0.15"))
MMR_LAM = float(os.environ.get("PTR_MMR_LAM", "0.0"))
FAR_BUDGET_CHUNKS = float(os.environ.get("PTR_FAR_BUDGET", "2.0"))
EVAL_FAR_BUDGET_CHUNKS = float(
    os.environ.get(
        "PTR_EVAL_FAR_BUDGET", os.environ.get("PTR_EVAL_FAR_CHUNKS", str(FAR_BUDGET_CHUNKS))
    )
)


def budget_sections(far_chunks: float) -> int:
    return int(round(far_chunks * N_SECTIONS))


def far_video_chunks(c: int) -> list[int]:
    """Video chunks the graded/current chunk c may NOT see locally."""
    return list(range(0, max(0, c - N_LOCAL)))


def redundancy_chunks(c: int) -> list[int]:
    """Video chunks the MMR term measures redundancy against: the local window, minus
    the query chunk c-1."""
    return [k - 1 for k in local_store_idx(c) if k - 1 != c - 1 and k - 1 >= 0]


def local_store_idx(c: int) -> list[int]:
    """Cache-chunk indices of the local window for video chunk c."""
    return list(range(max(1, c - N_LOCAL + 1), c + 1))


def first_trainable_chunk() -> int:
    """Lowest video chunk with strictly more candidate sections than the budget."""
    n_far_needed = -(-budget_sections(FAR_BUDGET_CHUNKS) // N_SECTIONS) + 1
    return max(STUDENT_SINK + N_LOCAL, N_LOCAL + n_far_needed)


class Selection:
    """Per-rollout memo of one clip's sections and pooled descriptor inputs."""

    def __init__(self, store, layout, device, geom=None):
        self.store, self.layout, self.device = (store, layout, device)
        self.geom = geom
        _, self.gh, self.gw = layout.video_patch_grid
        self._secs: dict[int, list[torch.Tensor]] = {}
        self._views: dict[int, torch.Tensor] = {}
        self._prek: dict[tuple[int, int], torch.Tensor] = {}

    def pre_rope_k(self, vk: int, layer: int) -> torch.Tensor:
        """Chunk vk's keys at `layer`, with the cached rotation removed."""
        key = (vk, layer)
        if key in self._prek:
            return self._prek[key]
        k, _ = self.store.kv_at(vk + 1, layer)
        if self.geom is None:
            self._prek[key] = k
            return k
        ch = self.layout.chunks[vk]
        frames = ch.video_rows // (self.gh * self.gw)
        rows = torch.arange(
            ch.audio_rows, ch.audio_rows + frames * self.gh * self.gw, dtype=torch.long
        )
        f_old, _ = self.geom.freqs_at(vk, rows)
        out = k.clone()
        r = rows.to(k.device)
        out[r] = unapply_rope(k[r].float(), f_old.to(k.device)).to(k.dtype)
        self._prek[key] = out
        return out

    def sections(self, vk: int) -> list[torch.Tensor]:
        if vk not in self._secs:
            ch = self.layout.chunks[vk]
            frames = ch.video_rows // (self.gh * self.gw)
            mid = DESC_LAYERS[len(DESC_LAYERS) // 2]
            self._secs[vk] = kmeans_sections(
                self.pre_rope_k(vk, mid), ch.audio_rows, self.gh, self.gw, frames, N_SECTIONS
            )
        return self._secs[vk]

    def views(self, vk: int) -> torch.Tensor:
        """[N_SECTIONS, n_views, d_in] on the selector's device, full model width."""
        if vk not in self._views:
            secs = self.sections(vk)
            krows = {l: self.pre_rope_k(vk, l) for l in DESC_LAYERS}
            local = torch.stack(
                [
                    pool_views(
                        {l: krows[l][s.to(krows[l].device)] for l in DESC_LAYERS}, DESC_LAYERS
                    )
                    for s in secs
                ]
            ).to(self.device)
            self._views[vk] = gather_features(local)
        return self._views[vk]

    def freqs(self, vk: int, rows: torch.Tensor, dest_vk: int | None):
        if self.geom is None:
            return (None, None)
        return self.geom.freqs_at(vk, rows, dest_vk)


def _sp_mean(logits: torch.Tensor) -> torch.Tensor:
    """Average the scores over the sequence-parallel group."""
    from common.distributed.unified_parallel import (
        get_unified_parallel_group,
        get_unified_parallel_world_size,
        is_unified_parallel_initialized,
    )

    if is_unified_parallel_initialized() and get_unified_parallel_world_size() > 1:
        torch.distributed.all_reduce(
            logits, op=torch.distributed.ReduceOp.SUM, group=get_unified_parallel_group()
        )
        logits = logits / get_unified_parallel_world_size()
    return logits


def choose(selector, sel: Selection, c: int, budget: int, gen=None, explore: float = 0.0):
    """ONE global top-N over every (far chunk, section) pair for video chunk c."""
    far = far_video_chunks(c)
    if not far or budget <= 0:
        return ([], torch.empty(0, dtype=torch.long), torch.empty(0), None)
    cand = [(k, s) for k in far for s in range(N_SECTIONS)]
    k_emb = selector(torch.cat([sel.views(k) for k in far], 0), "k")
    logits = maxsim(selector(sel.views(c - 1), "q"), k_emb)
    if MMR_LAM > 0:
        red = redundancy_chunks(c)
        if red:
            red_emb = selector(torch.cat([sel.views(k) for k in red], 0), "k")
            logits = logits - MMR_LAM * maxsim(red_emb, k_emb).max(0).values[None, :]
    logits = _sp_mean(logits)
    picks, gate = allocate_global(logits, budget, float(selector.log_tau.exp()), explore, gen)
    return (cand, picks, gate, logits)


def _all_rows(store, cache_chunk: int):
    return torch.arange(store.lens[cache_chunk], dtype=torch.long)


def student_parts(store, sel: Selection, c: int, cand, picks, gate):
    """text | gated, RE-ANCHORED far sections | the local window, whole."""
    parts = [(0, _all_rows(store, 0), None, None, None)]
    idx = picks.tolist() if torch.is_tensor(picks) else list(picks)
    srcs = sorted({cand[i][0] for i in idx}, reverse=True)
    dest = {}
    if sel.geom is not None and srcs:
        band = ptr_geom.far_band(sel.layout, c, N_LOCAL, max(1, int(math.ceil(FAR_BUDGET_CHUNKS))))
        span = max((ptr_geom.chunk_frames(sel.layout, k) for k in srcs))
        dest = dict(zip(srcs, ptr_geom.anchors_for(len(srcs), band, span)))
    sel.last_dest = dest
    for n, i in enumerate(idx):
        k, sname = cand[i]
        rows = sel.sections(k)[sname]
        f_old, f_new = sel.freqs(k, rows, dest.get(k))
        parts.append((k + 1, rows, gate[n : n + 1], f_old, f_new))
    parts += [(i, _all_rows(store, i), None, None, None) for i in local_store_idx(c)]
    return parts


def teacher_parts(store, c: int):
    """text | video chunks 0..TEACHER_SINK-2 whole | the local window, whole."""
    local = set(local_store_idx(c))
    parts = [(0, _all_rows(store, 0), None, None, None)]
    parts += [
        (k, _all_rows(store, k), None, None, None)
        for k in range(1, TEACHER_SINK)
        if k not in local and k <= c
    ]
    parts += [(i, _all_rows(store, i), None, None, None) for i in sorted(local)]
    return parts


__all__ = [
    "DESC_LAYERS",
    "N_SECTIONS",
    "N_LOCAL",
    "TEACHER_SINK",
    "STUDENT_SINK",
    "EXPLORE",
    "FAR_BUDGET_CHUNKS",
    "EVAL_FAR_BUDGET_CHUNKS",
    "MMR_LAM",
    "budget_sections",
    "far_video_chunks",
    "local_store_idx",
    "redundancy_chunks",
    "first_trainable_chunk",
    "Selection",
    "choose",
    "student_parts",
    "teacher_parts",
]
