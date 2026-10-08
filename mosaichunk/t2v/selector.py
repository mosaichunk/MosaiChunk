"""Shared descriptor trunk and query/key projections for H3-AR."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def pool_views(kv_rows, layers, moments=("mean", "max", "min")):
    """[N_layers][rows, heads, dim] -> [n_views, d_in]. One view per (layer, moment)."""
    out = []
    for l in layers:
        x = kv_rows[l].reshape(kv_rows[l].shape[0], -1).float()
        for m in moments:
            out.append(
                x.mean(0) if m == "mean" else x.max(0).values if m == "max" else x.min(0).values
            )
    return torch.stack(out, 0)


class Trunk(nn.Module):
    """Shared encoder for pooled-KV features -> [N, n_views, d]."""

    def __init__(self, d_in, n_views, d=1024, hidden=None):
        super().__init__()
        self.d_in, self.n_views, self.d = (d_in, n_views, d)
        hidden = hidden or 2 * d
        self.inp = nn.Linear(d_in, d)
        self.view_emb = nn.Parameter(torch.randn(n_views, d) * 0.02)
        self.ln = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d))
        self.out = nn.LayerNorm(d)

    def forward(self, feats):
        """feats [N, n_views, d_in] (or flat [N, n_views*d_in]) -> [N, n_views, d]."""
        if feats.dim() == 2:
            feats = feats.view(feats.shape[0], self.n_views, self.d_in)
        x = F.layer_norm(feats.float(), (self.d_in,)).to(self.inp.weight.dtype)
        h = self.ln(self.inp(x) + self.view_emb[None])
        return self.out(h + self.mlp(h))


class Selector(nn.Module):
    def __init__(self, d_in, n_views=9, d=1024, tau=8.0, centre=True, train_alpha=False):
        super().__init__()
        self.trunk = Trunk(d_in, n_views, d)
        self.n_views = n_views
        self.d_proj = nn.Linear(d, d, bias=False)
        self.q_proj = nn.Linear(d, d, bias=False)
        self.register_buffer("log_tau", torch.tensor([math.log(tau)]))
        self.centre = centre
        if train_alpha:
            self.alpha = nn.Parameter(torch.tensor(1.0))
        else:
            self.register_buffer("alpha", torch.tensor(1.0), persistent=False)

    def forward(self, views, mode, pool=None):
        assert mode in ("q", "k"), mode
        return self.embed_q(views) if mode == "q" else self.embed_k(views, pool)

    def embed_q(self, views):
        return F.normalize(self.q_proj(self.trunk(views)), dim=-1)

    def embed_k(self, views, pool=None):
        k = self.d_proj(self.trunk(views))
        if self.centre:
            ref = k if pool is None else self.d_proj(self.trunk(pool))
            k = k - ref.mean(0, keepdim=True)
        return F.normalize(k, dim=-1)

    def scores(self, q, k):
        """MaxSim: [Sq,V,d] x [K,V,d] -> [Sq,K], max over the CANDIDATE's views,
        mean over the QUERY's, then scaled by tau."""
        pair = torch.einsum("iad,jbd->ijab", q, k)
        return pair.amax(-1).mean(-1) * self.log_tau.exp()

    def n_params(self):
        return sum((p.numel() for p in self.parameters()))
