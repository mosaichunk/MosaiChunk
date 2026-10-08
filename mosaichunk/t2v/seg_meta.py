"""Four-segment prompting with equal token lengths and smooth transitions."""

from __future__ import annotations

import dataclasses
import os
import pathlib

import projects.minimax_h3.meta_models.causal_minimax_h3_base as _base
import torch
from common.logging import get_logger
from projects.minimax_h3.meta_models.causal_minimax_h3_base import CausalMiniMaxH3Base
from utils.naive_cache import NaiveCache

logger = get_logger()
BLEND = int(os.environ.get("PTR_SEG_BLEND", "1"))
SEG_SPLIT = os.environ.get("PTR_SEG_SPLIT", "2:10")


class SegmentedCache(NaiveCache):
    """NaiveCache that rewrites the text rows of chunk 0 before each chunk's forward."""

    plan = None
    text_kv = None
    _armed = False

    def _install_text(self):
        cls = type(self)
        if cls.text_kv is None or not cls._armed:
            return
        if self.key_cache.get(0) is None:
            return
        c = len(self.chunk_lens) - 1
        if c < 0:
            return
        w = cls.plan(c)
        if w is None or w == self._last_w:
            return
        self._last_w = w
        logger.info("[seg] chunk %d: installing text %s", c, w)
        off = 0
        for b in range(self.batch_size):
            n = self.kvlens[b]
            if n <= 0:
                continue
            L = self.chunk_lens[0][b]
            for layer in list(self.key_cache):
                kc, vc = (self.key_cache[layer], self.value_cache[layer])
                if kc is None:
                    continue
                k = None
                v = None
                for s, weight in w:
                    ks, vs = cls.text_kv[s][layer]
                    k = ks * weight if k is None else k + ks * weight
                    v = vs * weight if v is None else v + vs * weight
                assert k.shape[0] == L, (
                    f"segment text has {k.shape[0]} rows but chunk 0 of sample {b} holds {L}; every segment must pad to the same token length"
                )
                kc[off : off + L] = k.to(kc.dtype)
                vc[off : off + L] = v.to(vc.dtype)
            off += n

    _last_w = None

    @property
    def seq_len(self):
        self._install_text()
        return super().seq_len


class SegmentedPromptMetaModel(CausalMiniMaxH3Base):
    """Inference meta model whose text conditioning changes at chunk boundaries."""

    _seed_pinned = False

    def _validation_inputs(self, config, models, prompts):
        if not SegmentedPromptMetaModel._seed_pinned:
            SegmentedPromptMetaModel._seed_pinned = True
            try:
                from . import seed_pin

                n = seed_pin.install(_base, logger=logger)
                if n == 0 and os.environ.get("PTR_SEED_TABLE"):
                    raise RuntimeError(
                        "PTR_SEED_TABLE is set but no prompt matched it; the arm would run on line-number seeds and silently not be comparable"
                    )
            except Exception:
                if os.environ.get("PTR_SEED_TABLE"):
                    raise
        return self._segmented_validation_inputs(config, models, prompts)

    def _segmented_validation_inputs(self, config, models, prompts):
        """Build inputs from segment 0 and stash every segment's prompt embeds."""
        if "segment_file" in config.validation:
            import json as _json

            table = _json.loads(pathlib.Path(config.validation.segment_file).read_text())
            key = prompts[0].strip()
            hit = table.get(key) if isinstance(table, dict) else None
            if hit is None and isinstance(table, list):
                lines = [
                    l.strip()
                    for l in pathlib.Path(config.validation.prompt_path).read_text().splitlines()
                    if l.strip()
                ]
                hit = table[lines.index(key)]
            assert hit, f"no segments for prompt: {key[:60]}"
            segs = list(hit)
        else:
            segs = list(config.validation.segment_prompts)
        packer = self._validation_packer(config)
        tok = packer.tokenizer

        def n_tok(text):
            return int(tok.encode(text)[1])

        assert len(prompts) == 1, (
            f"segmented prompting renders one clip per rollout, got {len(prompts)} prompts; set validation.batch_size to 1"
        )
        use_scene = bool(config.validation.get("segment_scene", True))
        scene = prompts[0].rstrip()
        texts = [f"{scene} {s.strip()}" for s in segs] if use_scene else [s.strip() for s in segs]
        target = max((n_tok(t) for t in texts))
        padded = []
        for t in texts:
            while n_tok(t) < target:
                t = t + " ."
            assert n_tok(t) == target, (
                f"could not pad a segment to {target} tokens (landed on {n_tok(t)}); a filler token that is not exactly one token breaks the padding loop"
            )
            padded.append(t)
        lines = pathlib.Path(config.validation.prompt_path).read_text().splitlines()
        self._prompt_index = lines.index(prompts[0].strip())
        inputs = super()._validation_inputs(config, models, [padded[0]])
        self._seg_embeds = []
        for t in padded:
            samples = [packer._pack_sample(prompt=t)]
            batch = {k: [s[k] for s in samples] for k in samples[0]}
            self._seg_embeds.append(
                self._encode_prompts(
                    models, batch["text_input_ids"], [int(v) for v in batch["text_lens"]]
                )
            )
        self._seg_texts = padded
        return inputs

    def _rollout_latents(self, backbone, inputs, rng, **kw):
        segs = getattr(self, "_seg_embeds", None)
        if not segs or len(segs) < 2:
            return super()._rollout_latents(backbone, inputs, rng, **kw)
        text_len = int(inputs.text_lens[0])
        n_chunks = len(inputs.layouts[0].chunks)
        bounds = [int(x) for x in SEG_SPLIT.replace(":", ",").split(",")]
        assert len(bounds) == len(segs) - 1, (
            f"PTR_SEG_SPLIT={SEG_SPLIT} gives {len(bounds)} boundaries for {len(segs)} segments; it needs {len(segs) - 1}"
        )
        assert all((0 < b < n_chunks for b in bounds)) and bounds == sorted(bounds), (
            f"segment boundaries {bounds} must be increasing and inside (0, {n_chunks})"
        )
        text_kv = []
        for embeds in segs:
            scratch = NaiveCache(
                batch_size=inputs.batch_size + (1 if self._sp_size() > 1 else 0),
                num_layers=self._num_layers(backbone),
                sink=list(inputs.sinks) + ([0] if self._sp_size() > 1 else []),
                window_size=list(inputs.window_sizes) + ([0] if self._sp_size() > 1 else []),
            )
            with torch.no_grad():
                self._text_cache_fill(
                    backbone, dataclasses.replace(inputs, prompt_embeds=embeds), scratch
                )
            per_layer = []
            for layer in sorted(scratch.key_cache):
                k, v = (scratch.key_cache[layer], scratch.value_cache[layer])
                assert k is not None and k.shape[0] >= text_len, (
                    f"text fill produced {(None if k is None else k.shape[0])} rows for layer {layer}, expected at least text_len={text_len}"
                )
                per_layer.append((k[:text_len].clone().float(), v[:text_len].clone().float()))
            text_kv.append(per_layer)
        starts = [0] + bounds
        ends = bounds + [n_chunks]

        def plan(c):
            """Weights over segments for video chunk c, ramped over BLEND chunks."""
            w = [0.0] * len(segs)
            seg = sum((1 for b in bounds if c >= b))
            w[seg] = 1.0
            if BLEND > 0 and seg > 0:
                b = bounds[seg - 1]
                span = ends[seg] - starts[seg]
                blend = min(BLEND, max(0, span - 1))
                if blend > 0 and c < b + blend:
                    a = (c - b + 1) / float(blend + 1)
                    w = [0.0] * len(segs)
                    w[seg] = a
                    w[seg - 1] = 1.0 - a
            return [(i, x) for i, x in enumerate(w) if x > 0.0]

        logger.info(
            "[seg] %d segments, text_len=%d, %d chunks, boundaries=%s, blend=%d",
            len(segs),
            text_len,
            n_chunks,
            bounds,
            BLEND,
        )
        for c in range(n_chunks):
            logger.info("[seg] chunk %2d -> %s", c, plan(c))
        saved = _base.NaiveCache
        SegmentedCache.plan = staticmethod(plan)
        SegmentedCache.text_kv = text_kv
        SegmentedCache._armed = True
        try:
            _base.NaiveCache = (
                saved
                if isinstance(saved, type) and issubclass(saved, SegmentedCache)
                else SegmentedCache
            )
            result = super()._rollout_latents(backbone, inputs, rng, **kw)
            output = os.environ.get("MOSAICHUNK_LATENTS")
            if output and torch.distributed.get_rank() % self._sp_size() == 0:
                path = pathlib.Path(output)
                path.mkdir(parents=True, exist_ok=True)
                latents = result[0]
                torch.save(
                    {
                        "video": [x.detach().cpu() for x in latents.video],
                        "audio": [x.detach().cpu() for x in latents.audio],
                    },
                    path / f"{self._prompt_index:04d}.pt",
                )
            return result
        finally:
            _base.NaiveCache = saved
            SegmentedCache.plan = None
            SegmentedCache.text_kv = None
            SegmentedCache._armed = False


__all__ = ["SegmentedPromptMetaModel"]
