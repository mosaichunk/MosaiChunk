"""T2V self-distillation against richer memory in the same frozen H3-AR model."""

from __future__ import annotations

import json as _json
import os
import pathlib
import random
import time
from typing import Any, Iterator

import projects.minimax_h3.meta_models.causal_minimax_h3_base as _base
import torch
import torch.nn.functional as F
from common.distributed.ops import get_device
from common.distributed.unified_parallel import (
    SPDistForward,
    get_unified_parallel_world_size,
    is_unified_parallel_initialized,
)

from . import ptr_common as P
from .h3_cache import GatedCache, SplitStore
from .ptr_rollout import PtrRolloutMixin
from .seg_meta import SegmentedCache, SegmentedPromptMetaModel

TOKENIZER = os.environ.get("PTR_TOKENIZER", "assets/MiniMax-H3/FL2VA/tokenizer")
DESC_LAYERS = P.DESC_LAYERS
N_SECTIONS = P.N_SECTIONS
N_LOCAL = P.N_LOCAL
TEACHER_SINK = P.TEACHER_SINK
STUDENT_SINK = P.STUDENT_SINK
FAR_BUDGET_CHUNKS = P.FAR_BUDGET_CHUNKS


def _probe(tag, all_ranks=False):
    """Stage marker on rank 0. Three hangs in a row left no trace of WHICH stage the
    step reached -- NCCL's own desync debug printed nothing and the faulthandler dump
    only showed threading internals -- so the step reports its own progress."""
    if os.environ.get("PTR_PROBE") != "1":
        return
    r = -1
    try:
        if torch.distributed.is_initialized():
            r = torch.distributed.get_rank()
            if r != 0 and (not all_ranks):
                return
    except Exception:
        pass
    print(f"[probe] rank{r} {tag}", flush=True)


SEG_TABLE = os.environ.get("PTR_SEG_TABLE", "")
EXPLORE = P.EXPLORE
W_DIST = float(os.environ.get("PTR_W_DIST", "100.0"))


class _StashingCache(SegmentedCache):
    """Records the instance the rollout built, AND installs the per-chunk segment text."""

    last = None

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        type(self).last = self


class PtrTrainMetaModel(PtrRolloutMixin, SegmentedPromptMetaModel):
    """Selector training on top of the H3 causal rollout, with segmented prompting."""

    _seg_table = None
    _tokenizer = None

    def _segments_for(self, ctx, text_ids, text_len):
        """The sample's four segment embeddings, looked up by its own TOKEN IDS."""
        if type(self)._seg_table is None:
            assert SEG_TABLE, "PTR_SEG_TABLE is unset: training would run unsegmented"
            from projects.minimax_h3.modeling.tokenizer import MiniMaxH3Tokenizer

            tok = MiniMaxH3Tokenizer(TOKENIZER)
            table = {}
            for key, segs in _json.loads(pathlib.Path(SEG_TABLE).read_text()).items():
                enc = [tok.encode(t) for t in segs]
                widths = {int(n) for _, n in enc}
                assert len(widths) == 1, (
                    f"segments are not equal length under MiniMaxH3Tokenizer: {widths}. Re-run mk_segfiles.py -- it must pad with the same tokenizer."
                )
                table[tuple((int(v) for v in enc[0][0].flatten().tolist()))] = [
                    ids for ids, _ in enc
                ]
            type(self)._seg_table, type(self)._tokenizer = (table, tok)
        first = (
            text_ids[0]
            if isinstance(text_ids, (list, tuple))
            else text_ids[0]
            if text_ids.dim() > 1
            else text_ids
        )
        key = tuple((int(v) for v in first.flatten().tolist()[:text_len]))
        segs = type(self)._seg_table.get(key)
        assert segs is not None, (
            f"no segment entry for this sample's {len(key)} ids. The prompt file and the segment table must be the pair mk_segfiles.py wrote."
        )
        out = []
        for ids in segs:
            n = int(ids.numel())
            assert n == text_len, (
                f"segment is {n} tokens but the packed sample is {text_len}; the table and the prompt file are out of step"
            )
            out.append(self._encode_prompts(ctx["models"], [ids.flatten().to(first.device)], [n]))
        return out

    _tapped = False

    def _tap_backbone(self, models):
        """Give each FSDP-wrapped block ONE trainable parameter, so backward is scheduled."""
        n = int(os.environ.get("PTR_TAP", "1"))
        if not n or type(self)._tapped:
            return
        bb = models["backbone"]
        tapped = 0
        for m in bb.modules():
            if not type(m).__name__.endswith("DiTBlock"):
                continue
            ps = [q for q in m.parameters(recurse=True) if q.numel() > 0]
            if ps:
                min(ps, key=lambda q: q.numel()).requires_grad_(True)
                tapped += 1
        type(self)._tapped = True
        _probe(f"tapped {tapped} blocks")

    def prepare_inputs(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Reads: batch, models. Writes: inputs. (No clean_latents: text-only data.)"""
        self._tap_backbone(ctx["models"])
        batch = ctx["batch"]
        prompt_embeds = self._encode_prompts(
            ctx["models"], batch["text_input_ids"], [int(v) for v in batch["text_lens"]]
        )
        ctx["inputs"] = self._build_inputs(batch, prompt_embeds)
        self._seg_embeds = self._segments_for(
            ctx, batch["text_input_ids"], int(batch["text_lens"][0])
        )
        return ctx

    def sync_inputs(self, ctx: dict[str, Any]) -> Iterator[dict[str, Any]]:
        if not is_unified_parallel_initialized() or get_unified_parallel_world_size() <= 1:
            yield ctx
            return
        segs = torch.stack([e[0] for e in self._seg_embeds], 0)
        payload = (self._to_device(ctx["batch"]), ctx["inputs"].prompt_embeds, segs)
        sync = SPDistForward(name="ptr_train_inputs", comm_shape=True, device=get_device())
        for batch, prompt_embeds, seg_stack in sync(payload):
            sub = dict(ctx)
            sub["inputs"] = self._build_inputs(batch, prompt_embeds)
            self._seg_embeds = [[seg_stack[i]] for i in range(seg_stack.shape[0])]
            yield sub

    def sample_timesteps(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """No-op: the rollout draws its own schedule, exactly as inference does."""
        return ctx

    def add_noise(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """No-op: the rollout starts each chunk from its own noise slice."""
        return ctx

    def forward(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Reads: models, inputs, rng, iter. Writes: pred = (student, teacher)."""
        backbone = ctx["models"]["backbone"]
        selector = ctx["models"]["selector"]
        ctx["pred"] = self._ptr_step(
            backbone, selector, ctx["inputs"], ctx["rng"], int(ctx["iter"])
        )
        return ctx

    def compute_loss(self, ctx: dict[str, Any]) -> dict[str, Any]:
        _probe("loss-begin", all_ranks=True)
        student, teacher, metrics = ctx["pred"]
        mse = F.mse_loss(student.float(), teacher.float())
        ctx["loss"] = W_DIST * mse
        self._log_step(ctx, float(mse), metrics)
        _probe("loss-done", all_ranks=True)
        ctx["metrics"] = metrics
        return ctx

    def _log_step(self, ctx, mse, metrics):
        """One line per ITERATION, so the loss can be read per graded step."""
        path = os.environ.get("PTR_STEPLOG")
        if not path:
            return
        try:
            if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
                return
            rec = {
                "iter": int(ctx.get("iter", -1)),
                "mse": mse,
                "step": int(metrics.get("ptr/graded_step", -1)),
                "chunk": int(metrics.get("ptr/graded_chunk", -1)),
                "n_cand": int(metrics.get("ptr/n_candidates", 0)),
                "n_picked": int(metrics.get("ptr/n_picked", 0)),
                "src": int(metrics.get("ptr/distinct_far_chunks", 0)),
            }
            pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a") as f:
                f.write(_json.dumps(rec) + "\n")
        except Exception:
            pass

    @torch.no_grad()
    def _ptr_capture(self, backbone, inputs, rng, selector, gstep):
        """Roll out ONE clip as the student and hand back the rollout's side store."""
        saved_cls = _base.NaiveCache
        _StashingCache.last = None
        self._ptr_salt = int(gstep)
        try:
            _base.NaiveCache = _StashingCache
            with self._ptr_arm(selector, P.FAR_BUDGET_CHUNKS, EXPLORE):
                _, traj = self._rollout_latents(backbone, inputs, rng, keep_trajectory=True)
                side, geom = self._ptr_side_result()
        finally:
            _base.NaiveCache = saved_cls
        _probe("capture-done", all_ranks=True)
        cache = _StashingCache.last
        assert cache is not None, "the rollout built no cache -- the intercept missed"
        n_ch = len(inputs.layouts[0].chunks)
        assert side is not None and len(side.lens) == n_ch, (
            f"the side store filed {(0 if side is None else len(side.lens))} entries for a {n_ch}-chunk clip; expected {n_ch} (text + video 0..{n_ch - 2})"
        )
        return (cache, side, geom, traj)

    def _ptr_step(self, backbone, selector, inputs, rng, gstep):
        assert len(inputs.layouts) == 1, (
            f"the batch packs {len(inputs.layouts)} prompts, but the ChunkStore and the GatedCache are built for one clip. sample_lens would then have more entries than the cache has samples and the backbone rejects it. Lower data.args.max_seqlen so exactly one sample fits (59464 for 768x1376 / 192)."
        )
        cache, side, geom, traj = self._ptr_capture(backbone, inputs, rng, selector, gstep)
        layers_all = sorted((l for l, k in cache.key_cache.items() if k is not None))
        layout = inputs.layouts[0]
        _, gh, gw = layout.video_patch_grid
        dev = get_device()
        n_video = len(layout.chunks)
        lo = P.first_trainable_chunk()
        assert n_video > lo, (
            f"clip too short: {n_video} video chunks, need > {lo} (window {N_LOCAL} + far budget {FAR_BUDGET_CHUNKS} + 1)"
        )
        pick_rng = random.Random(gstep * 2654435761 ^ 40503)
        c = pick_rng.randrange(lo, n_video)
        store = SplitStore(cache, side, sample=0, text_at=c)
        sel = P.Selection(store, layout, dev, geom=geom)
        far = P.far_video_chunks(c)
        assert far, "no far chunks"
        budget = P.budget_sections(FAR_BUDGET_CHUNKS)
        n_cand = len(far) * N_SECTIONS
        assert budget < n_cand, (
            f"budget {budget} >= candidates {n_cand}: the student could hold every far row and the loss would not require a choice"
        )
        d_seen = int(sel.views(far[0]).shape[-1])
        d_cfg = selector.trunk.d_in
        assert d_seen == d_cfg, (
            f"selector d_in={d_cfg} but the cache gives {d_seen} features per view. K/V are sharded by head across sequence parallel: d_in must be num_attention_heads / sp_size * attention_head_dim (= 56 / {56 * 128 // d_seen} * 128 = {d_seen} here). Fix models.selector.args.d_in."
        )
        gen = torch.Generator(device="cpu")
        gen.manual_seed(2654435769 ^ int(gstep))
        cand, picks, gate, logits = P.choose(selector, sel, c, budget, gen, EXPLORE)
        _probe("picked", all_ranks=True)
        ch = layout.chunks[c]
        _probe("xt", all_ranks=True)
        v_xt, a_xt, vt, at, graded_step = self._ptr_xt_at(inputs, ch, traj, c, gstep, dev)
        t_parts = P.teacher_parts(store, c)
        s_parts = P.student_parts(store, sel, c, cand, picks, gate)
        n_pad = 1 if self._sp_size() > 1 else 0
        with torch.no_grad():
            t_cache = GatedCache.assemble(store, layers_all, t_parts, dev, n_pad)
            _probe("teacher-cache", all_ranks=True)
            _probe("teacher-fwd-begin", all_ranks=True)
            _t0 = time.time()
            v_t, _ = self._chunk_forward(
                backbone,
                inputs,
                chunk_index=c,
                role="noise",
                video_rows_source=v_xt,
                audio_rows_source=a_xt,
                video_timesteps=vt,
                audio_timesteps=at,
                cache=t_cache,
                update_cache=False,
            )
        _probe(f"teacher-fwd-done {time.time() - _t0:.1f}s", all_ranks=True)
        s_cache = GatedCache.assemble(store, layers_all, s_parts, dev, n_pad)
        _probe(
            f"student-cache rows={sum((int(p[1].numel()) for p in s_parts))} picks={int(picks.numel())}",
            all_ranks=True,
        )
        _probe("student-fwd-begin", all_ranks=True)
        _s0 = time.time()
        v_s, _ = self._chunk_forward(
            backbone,
            inputs,
            chunk_index=c,
            role="noise",
            video_rows_source=v_xt,
            audio_rows_source=a_xt,
            video_timesteps=vt,
            audio_timesteps=at,
            cache=s_cache,
            update_cache=False,
        )
        _probe(f"student-fwd-done {time.time() - _s0:.1f}s", all_ranks=True)
        picked_chunks = sorted({cand[i][0] for i in picks.tolist()})
        metrics = {
            "ptr/graded_chunk": float(c),
            "ptr/graded_step": float(graded_step),
            "ptr/n_candidates": float(len(cand)),
            "ptr/n_picked": float(budget),
            "ptr/distinct_far_chunks": float(len(picked_chunks)),
            "ptr/gate_mean": float(gate.mean()),
        }
        return (v_s[0], v_t[0].detach(), metrics)

    def _ptr_xt_at(self, inputs, ch, traj, c, gstep, dev):
        """x_t and timesteps for ONE RANDOMLY CHOSEN denoise step of chunk c."""
        n_steps = int(self.sampling_timesteps.timesteps.numel())
        i = random.Random(gstep * 2654435761 ^ (c + 1) * 40503).randrange(n_steps)
        v_all, a_all = traj[i]
        v_xt = [v_all[b][:, ch.video_start : ch.video_stop] for b in range(inputs.batch_size)]
        a_xt = [a_all[b][:, :, ch.audio_start : ch.audio_stop] for b in range(inputs.batch_size)]
        vt = self.sampling_timesteps.timesteps[i].expand(inputs.batch_size).to(dev)
        at = self.audio_sampling_timesteps.timesteps[i].expand(inputs.batch_size).to(dev)
        return (v_xt, a_xt, vt, at, i)


__all__ = ["PtrTrainMetaModel"]
