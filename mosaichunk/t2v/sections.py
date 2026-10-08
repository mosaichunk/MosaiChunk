"""Balanced visual-token sections and global top-N selection."""

from __future__ import annotations

import torch


def video_row_index(audio_rows, gh, gw, frames):
    """Row indices of the video block, in packed order."""
    return torch.arange(audio_rows, audio_rows + frames * gh * gw, dtype=torch.long)


def kmeans_sections(
    feat_shard, audio_rows, gh, gw, frames, n_sections, iters=12, seed=0, proj_dim=128
):
    """Balanced k-means over the chunk's own keys -- lingbot's partition(), ported."""
    import numpy as np

    from .sp_exact import shard_slice, sum_reduce

    idx = video_row_index(audio_rows, gh, gw, frames)
    T = idx.numel()
    assert T % n_sections == 0, f"{n_sections} sections must divide {T} video rows exactly"
    R = T // n_sections
    dev = feat_shard.device
    X = feat_shard[idx.to(dev)].reshape(T, -1).float()
    g = torch.Generator(device="cpu").manual_seed(seed)
    P_full = torch.randn(X.shape[1] * _sp_size(), proj_dim, generator=g)
    P = shard_slice(P_full, dim=0).to(dev)
    Z = torch.nn.functional.normalize(sum_reduce(X @ P), dim=-1)
    first = int(Z.norm(dim=-1).argmax())
    idxs = [first]
    d2 = ((Z - Z[first]) ** 2).sum(-1)
    for _ in range(n_sections - 1):
        idxs.append(int(d2.argmax()))
        d2 = torch.minimum(d2, ((Z - Z[idxs[-1]]) ** 2).sum(-1))
    C = Z[idxs].clone()
    for _ in range(iters):
        a = (Z @ C.T).argmax(-1)
        for s_i in range(n_sections):
            m = a == s_i
            if m.any():
                C[s_i] = torch.nn.functional.normalize(Z[m].mean(0), dim=-1)
    aff = (Z @ C.T).flatten().cpu().numpy()
    order = np.argsort(-aff, kind="stable")
    tok = (order // n_sections).astype(np.int64)
    sec = (order % n_sections).astype(np.int64)
    taken = np.zeros(T, dtype=bool)
    room = np.full(n_sections, R, dtype=np.int64)
    out = [[] for _ in range(n_sections)]
    for t_i, s_i in zip(tok, sec):
        if not taken[t_i] and room[s_i] > 0:
            taken[t_i] = True
            room[s_i] -= 1
            out[s_i].append(int(t_i))
    assert all((len(o) == R for o in out)), [len(o) for o in out]
    return [idx[torch.tensor(o, dtype=torch.long)] for o in out]


def _sp_size():
    from .sp_exact import sp_group

    return sp_group()[1]


def maxsim(q_views, k_views):
    """ColBERT form. q_views [Q, V, D], k_views [K, V, D], both L2-normalised.
    Returns [Q, K]: max over the CANDIDATE's views, mean over the QUERY's."""
    sim = torch.einsum("qvd,kwd->qvkw", q_views, k_views)
    return sim.max(dim=3).values.mean(dim=1)


def allocate_global(logits, budget, tau, explore=0.0, gen=None):
    """One ranking over every candidate; top-N; differentiable mean-1 gate."""
    s = logits.max(0).values
    p = torch.softmax(s * tau, dim=-1)
    n = min(int(budget), s.numel())
    n_ex = int(n * explore)
    top = torch.topk(p, n - n_ex).indices if n_ex < n else torch.empty(0, dtype=torch.long)
    if n_ex > 0:
        rest = torch.tensor([i for i in range(s.numel()) if i not in set(top.tolist())])
        pick_ex = rest[torch.randperm(rest.numel(), generator=gen)[:n_ex]] if rest.numel() else rest
        picks = torch.cat([top, pick_ex.to(top.device)])
    else:
        picks = top
    g = p[picks]
    gate = g / g.mean().detach().clamp_min(1e-12)
    return (picks, gate)
