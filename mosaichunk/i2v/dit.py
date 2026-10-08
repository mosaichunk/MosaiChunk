"""Frozen LingBot backbone shared by router training and evaluation."""

import os

import torch
from wan.modules.model_fast import causal_rope_apply

from . import cfg
from .extract_kv import KVExtractor, build_pipeline


class FrozenDiT:
    def __init__(self, ckpt_dir=None, device_id=0, layers=None):
        self.dev = torch.device("cuda", device_id)
        self.pipe = build_pipeline(
            ckpt_dir or cfg.CKPT_SRC, local_attn_size=-1, sink_size=0, device_id=device_id
        )
        self.layers = list(range(cfg.N_LAYERS) if layers is None else layers)
        self.ex = KVExtractor(self.pipe, self.layers)
        self.ex.capture = False
        m = self.pipe.model.config
        assert (m.dim, m.num_heads, m.num_layers) == (cfg.DIT_DIM, cfg.N_HEADS, cfg.N_LAYERS), (
            f"model is {(m.dim, m.num_heads, m.num_layers)}, cfg says {(cfg.DIT_DIM, cfg.N_HEADS, cfg.N_LAYERS)}"
        )
        self.freqs = self.pipe.model.freqs.to(self.dev)
        self.grid = torch.tensor([[cfg.CHUNK, cfg.GRID_H, cfg.GRID_W]], device=self.dev)
        self.pdtype = self.pipe.param_dtype
        shift = __import__("wan").configs.WAN_CONFIGS["i2v-A14B"].sample_shift
        self.pipe.scheduler.set_timesteps(
            self.pipe.num_train_timesteps, shift=shift, device=self.dev
        )
        self.ts = self.pipe.scheduler.timesteps[list(cfg.STEP_IDX)]

    def self_cache(self, n_tok=cfg.CHUNK_TOKENS):
        """A fresh per-chunk self-attention cache. 4.76 GiB at n_tok=6240 across 40 layers, so it is
        allocated per forward and never held."""
        return self.pipe._initialize_self_kv_cache(
            num_layers=cfg.N_LAYERS,
            shape=[1, n_tok, cfg.N_HEADS, cfg.HEAD_DIM],
            dtype=self.pipe.pipe_dtype,
            device=self.dev,
        )

    def cross_cache(self):
        return self.pipe._initialize_crossattn_cache(
            num_layers=cfg.N_LAYERS,
            shape=[1, 512, cfg.N_HEADS, cfg.HEAD_DIM],
            dtype=self.pipe.pipe_dtype,
            device=self.dev,
        )

    def cond(self, clip, action_dir, lat_f):
        """Everything a clip's rollout needs besides noise. y/plucker are split per chunk."""
        import glob

        ff = sorted(glob.glob(os.path.join(clip, "gt_frames", "*.png")))[0]
        y = self.ex._build_y(ff, lat_f)
        ctx = self.ex._build_context(self.ex._resolve_prompt(clip, None))
        pl = self.ex._build_plucker(action_dir, lat_f)
        return dict(y=y.split(cfg.CHUNK, 1), ctx=ctx, pl=pl.split(cfg.CHUNK, 2))

    def rope_k(self, k, start_frame, n_frames=cfg.CHUNK):
        """k [1, n_frames*1560, 40, 128] -> RoPE'd at latent frames [start_frame, start_frame+n_frames)."""
        g = torch.tensor([[n_frames, cfg.GRID_H, cfg.GRID_W]], device=k.device)
        return causal_rope_apply(k, g, self.freqs, start_frame=start_frame)

    def mem_kv(self, slots):
        out = [None] * cfg.N_LAYERS
        for L in self.layers:
            ks, vs = ([], [])
            for kv, sf in slots:
                e = kv[L]
                n_fr = None
                for name, dst in (("k", ks), ("v", vs)):
                    t = e[name]
                    if t.dim() != 4:
                        t = (
                            t.to(self.dev, non_blocking=True)
                            .view(1, -1, cfg.N_HEADS, cfg.HEAD_DIM)
                            .to(self.pdtype)
                        )
                    if n_fr is None:
                        n_fr = t.shape[1] // cfg.FRAME_TOKENS
                    dst.append(self.rope_k(t, sf, n_fr) if name == "k" else t)
            out[L] = {"k": torch.cat(ks, 1), "v": torch.cat(vs, 1)}
        return out

    def forward(self, x, t, c, chunk_i, mem_kv, cache=None, first_cross=False, mem_fns=None):
        """One frozen-DiT velocity prediction. rope_frame_offset is ALWAYS F_CUR: the current chunk's
        own queries must sit at the same positions in every pass, or teacher and student are not
        comparable and the loss simply will not fall."""
        with torch.amp.autocast("cuda", dtype=self.pdtype):
            return self.pipe.model(
                x=[x],
                t=torch.as_tensor([float(t)], device=self.dev),
                context=[c["ctx"][0]],
                seq_len=cfg.CHUNK_TOKENS,
                y=[c["y"][chunk_i]],
                dit_cond_dict={"c2ws_plucker_emb": c["pl"][chunk_i].chunk(1, dim=0)},
                kv_cache=cache if cache is not None else self.self_cache(),
                crossattn_cache=c["ckv"],
                current_start=0,
                max_attention_size=cfg.CHUNK_TOKENS,
                frame_seqlen=cfg.FRAME_TOKENS,
                cross_attn_first_call=first_cross,
                mem_kv=mem_kv,
                mem_fns=mem_fns,
                rope_frame_offset=cfg.F_CUR,
            )[0]

    def step(self, cur, x0, i, gen):
        """Flow-matching renoise between the 4 sampler steps. Last step returns x0."""
        if i >= len(self.ts) - 1:
            return x0
        s = (self.ts[i + 1] / 1000.0).float()
        return (1 - s) * x0 + s * torch.randn(x0.shape, generator=gen, device=self.dev)

    @torch.no_grad()
    def capture(self, x0, t_zero_c, chunk_i, mem_kv, first_cross=False):
        """Run the t=0 pass that writes this chunk's own KV, and return it as {layer:{k,v}} with
        [6240, 5120] tensors. This is the ONLY producer of stored memory: the KV is taken pre-RoPE
        at self_attn.norm_k / self_attn.v, which is what makes it re-positionable later."""
        for L in self.layers:
            self.ex._cap[L] = {}
        self.ex.capture = True
        try:
            self.forward(x0, 0.0, t_zero_c, chunk_i, mem_kv, first_cross=first_cross)
        finally:
            self.ex.capture = False
        cap = {
            L: {kv: self.ex._cap[L][kv].reshape(cfg.CHUNK_TOKENS, cfg.DIT_DIM) for kv in ("k", "v")}
            for L in self.layers
        }
        for L in self.layers:
            self.ex._cap[L] = {}
        return cap

    @torch.no_grad()
    def decode(self, x0_by_chunk):
        """{c: [16,4,60,104]} -> uint8 video [F, H, W, 3] for an mp4."""
        lat = torch.cat([x0_by_chunk[c] for c in sorted(x0_by_chunk)], dim=1)
        vid = self.pipe.vae.decode([lat.to(self.pdtype)])[0].float()
        g = ((vid.clamp(-1, 1) + 1) / 2).permute(1, 2, 3, 0)
        return (g * 255).clamp(0, 255).byte().cpu().numpy()

    def close(self):
        self.ex.close()
