"""Compose selected KV rows while preserving spatial RoPE coordinates."""

import torch

from . import cfg


def row_grid_index(rows):
    """flat row index within a chunk -> (f, h, w). Layout is f-major then h then w, matching
    causal_rope_apply's reshape(seq_len, ...) over grid (f, h, w)."""
    hw = cfg.GRID_H * cfg.GRID_W
    f = rows // hw
    r = rows % hw
    return (f, r // cfg.GRID_W, r % cfg.GRID_W)


def rope_rows(x, f_idx, h_idx, w_idx, freqs, start_frame):
    """x [1, N, n_heads, head_dim] -> RoPE'd in place of a grid call, one position per row."""
    n, c = (x.shape[2], x.shape[3] // 2)
    f0, f1, f2 = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    fr = torch.cat([f0[start_frame + f_idx], f1[h_idx], f2[w_idx]], dim=-1)
    xc = torch.view_as_complex(x[0].to(torch.float64).reshape(-1, n, c, 2))
    out = torch.view_as_real(xc * fr[:, None, :]).flatten(2)
    return out[None].type_as(x)


def self_test(freqs, device="cpu", atol=0.0):
    """rope_rows on a FULL grid must equal causal_rope_apply exactly. Run this once per process; if it
    ever fails, every position in the memory is wrong and no metric downstream means anything."""
    from wan.modules.model_fast import causal_rope_apply

    T = cfg.CHUNK_TOKENS
    x = torch.randn(1, T, cfg.N_HEADS, cfg.HEAD_DIM, device=device)
    g = torch.tensor([[cfg.CHUNK, cfg.GRID_H, cfg.GRID_W]], device=device)
    for sf in (0, 4, 8, 12, 24):
        a = causal_rope_apply(x, g, freqs, start_frame=sf)
        rows = torch.arange(T, device=device)
        f, h, w = row_grid_index(rows)
        b = rope_rows(x, f, h, w, freqs, sf)
        d = float((a - b).abs().max())
        assert d <= atol, f"rope_rows != causal_rope_apply at start_frame={sf}: max|diff|={d}"
    return True


class Composer:
    """Assembles the compositional chunk. Holds no parameters."""

    def __init__(self, dit, store):
        self.dit, self.store = (dit, store)

    @staticmethod
    def anchors_for(n_src):
        """n_src contributing source chunks -> one frame anchor each, spread over the far region."""
        lo, hi = (cfg.F_FAR[0], cfg.F_LOCAL[0] - cfg.CHUNK)
        if n_src <= 1:
            return [lo]
        return [lo + round(r * (hi - lo) / (n_src - 1)) for r in range(n_src)]

    @torch.no_grad()
    def gather_rows(self, picks, ranks):
        """picks [(j, s)], ranks [r] -> ([80, N, 5120] on GPU, f/h/w [N], frame anchor [N])."""
        dev = self.dit.dev
        anch = self.anchors_for(max(ranks) + 1 if len(ranks) else 1)
        kv, fs, hs, ws, sf = ([], [], [], [], [])
        for (j, s), r in zip(picks, ranks):
            idx = self.store.sections[j][s].long()
            kv.append(self.store.kv[j][:, idx, :].to(dev, non_blocking=True))
            f, h, w = row_grid_index(idx.to(dev))
            fs.append(f)
            hs.append(h)
            ws.append(w)
            sf.append(torch.full_like(f, anch[min(r, len(anch) - 1)]))
        return (torch.cat(kv, dim=1), torch.cat(fs), torch.cat(hs), torch.cat(ws), torch.cat(sf))

    def build(self, picks, ranks, gates=None, extra_slots=()):
        """-> mem_kv, the per-block [{"k","v"}] list the DiT wants."""
        dev = self.dit.dev
        K, f, h, w, sf = self.gather_rows(picks, ranks)
        N = K.shape[1]
        R = self.store.R
        if gates is not None:
            g = gates.repeat_interleave(R).to(dev)
        out = [None] * cfg.N_LAYERS
        anchors = sorted(set((int(v) for v in sf.tolist())))
        for L in self.dit.layers:
            k_rows = K[L * 2].view(1, N, cfg.N_HEADS, cfg.HEAD_DIM)
            v_rows = K[L * 2 + 1].view(1, N, cfg.N_HEADS, cfg.HEAD_DIM)
            kr = torch.empty_like(k_rows)
            for a in anchors:
                m = sf == a
                kr[:, m] = rope_rows(k_rows[:, m], f[m], h[m], w[m], self.dit.freqs, a)
            vr = v_rows if gates is None else v_rows * g[None, :, None, None].to(v_rows.dtype)
            ks, vs = ([kr], [vr])
            for kv, start in extra_slots:
                e = kv[L]
                t_k = (
                    e["k"]
                    if e["k"].dim() == 4
                    else e["k"]
                    .to(dev, non_blocking=True)
                    .view(1, -1, cfg.N_HEADS, cfg.HEAD_DIM)
                    .to(self.dit.pdtype)
                )
                t_v = (
                    e["v"]
                    if e["v"].dim() == 4
                    else e["v"]
                    .to(dev, non_blocking=True)
                    .view(1, -1, cfg.N_HEADS, cfg.HEAD_DIM)
                    .to(self.dit.pdtype)
                )
                ks.append(self.dit.rope_k(t_k, start, t_k.shape[1] // cfg.FRAME_TOKENS))
                vs.append(t_v)
            out[L] = {
                "k": torch.cat(ks, 1).to(self.dit.pdtype),
                "v": torch.cat(vs, 1).to(self.dit.pdtype),
            }
        return out

    def build_whole_chunks(self, slots):
        """The oracle / base path: whole chunks at whole slots, no picks, no gates."""
        return self.dit.mem_kv(slots)
