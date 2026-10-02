"""Learned section descriptors and global budgeted retrieval."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import cfg
from . import store as store_mod


class _Trunk(nn.Module):
    """Shared encoder for pooled-KV features -> a MULTI-VECTOR descriptor, [N, n_views, d]."""

    def __init__(self, d=1024, n_terms=None, hidden=None):
        super().__init__()
        self.n_views = n_terms or store_mod.N_TERMS
        self.d = d
        hidden = hidden or 2 * d
        self.inp = nn.Linear(cfg.DIT_DIM, d)
        self.view_emb = nn.Parameter(torch.randn(self.n_views, d) * 0.02)
        self.ln = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d))
        self.out = nn.LayerNorm(d)

    def forward(self, feats):
        """feats [N, n_views*5120] -> [N, n_views, d]."""
        N = feats.shape[0]
        x = feats.view(N, self.n_views, cfg.DIT_DIM)
        x = F.layer_norm(x.float(), (cfg.DIT_DIM,)).to(self.inp.weight.dtype)
        h = self.ln(self.inp(x) + self.view_emb[None])
        return self.out(h + self.mlp(h))


class Selector(nn.Module):
    """Descriptor + query + the top-k read. Holds no memory; it only produces indices and gates."""

    def __init__(
        self,
        d=1024,
        tau=8.0,
        topk=3,
        explore=0.0,
        centre=True,
        budget_sections=0,
        train_alpha=False,
    ):
        super().__init__()
        self.trunk = _Trunk(d=d)
        self.n_views = self.trunk.n_views
        self.d_proj = nn.Linear(d, d, bias=False)
        self.q_proj = nn.Linear(d, d, bias=False)
        self.log_tau = nn.Parameter(torch.tensor(math.log(tau)))
        self.topk, self.explore, self.centre = (topk, explore, centre)
        self.budget_sections = int(budget_sections)
        self.train_alpha = bool(train_alpha)
        if train_alpha:
            self.alpha = torch.nn.Parameter(torch.tensor(1.0))
        else:
            self.register_buffer("alpha", torch.tensor(1.0), persistent=False)
        self.lam = 0.0

    def keys(self, cand_feats):
        """[n_cand*S, F] -> [n_cand*S, n_views, d], each view L2-normalised."""
        k = self.d_proj(self.trunk(cand_feats))
        if self.centre:
            k = k - k.mean(0, keepdim=True)
        return F.normalize(k, dim=-1)

    def queries(self, q_feats):
        return F.normalize(self.q_proj(self.trunk(q_feats)), dim=-1)

    def scores(self, q, k):
        """MaxSim, in ColBERT's actual form: [Sq, n_views, d] x [K, n_views, d] -> [Sq, K]."""
        pair = torch.einsum("iad,jbd->ijab", q, k)
        return pair.amax(-1).mean(-1) * self.log_tau.exp()

    def forward(self, q_feats, cand_feats, cand_index, gen=None, red_feats=None, **kw):
        """q_feats [S_cur, F], cand_feats [n_cand*S, F], cand_index [(j, s)] aligned to cand_feats."""
        q = self.queries(q_feats)
        k = self.keys(cand_feats)
        logits = self.scores(q, k)
        if red_feats is not None and self.lam > 0:
            red = self.keys(red_feats)
            logits = (
                logits - self.lam * self.scores(red, self.keys(cand_feats)).max(0).values[None, :]
            )
        return self._allocate_global(logits, cand_index, gen)

    def _allocate_global(self, logits, cand_index, gen=None):
        """ONE ranking over every candidate, N slots handed out globally."""
        s = logits.max(0).values
        p = s.softmax(-1)
        n = min(self.budget_sections, p.shape[0])
        sel = p.topk(n).indices
        if self.training and self.explore > 0 and (gen is not None):
            frac = float(torch.rand(1, generator=gen, device=p.device) * self.explore)
            n_rand = int(frac * n)
            if n_rand > 0:
                r = torch.randint(0, p.shape[0], (n_rand,), generator=gen, device=p.device)
                sel = torch.cat([sel[: n - n_rand], r])
        gs = p[sel]
        g = gs / gs.mean().detach().clamp_min(1e-12) * self.alpha
        picks = [cand_index[i] for i in sel.tolist()]
        order = {j: r for r, j in enumerate(sorted({j for j, _ in picks}, reverse=True))}
        ranks = [order[j] for j, _ in picks]
        return (picks, ranks, g, logits)

    def n_params(self):
        return sum((p.numel() for p in self.parameters()))


def make_selector(mode, n_sections, topk, budget_sections, **kw):
    """One place that maps a --pick mode to a policy, so train and eval cannot disagree."""
    _sel_kw = {
        k: v for k, v in kw.items() if k not in ("zgap", "zn", "moments", "pick_views", "sec_views")
    }
    if mode == "model":
        return Selector(topk=topk, budget_sections=budget_sections, **_sel_kw)
    raise ValueError(mode)
