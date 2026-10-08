"""CPU section bank with balanced clustering and pooled key features."""

import os

import torch

from . import cfg

SECTION_LAYER = 20
DESC_LAYERS = tuple((int(x) for x in os.environ.get("PTR_DESC_LAYERS", "10,20,30").split(",")))
MOMENTS = tuple((x for x in os.environ.get("PTR_MOMENTS", "mean,max,min").split(",") if x))
if not MOMENTS or not set(MOMENTS) <= {"mean", "max", "min"}:
    raise ValueError(f"PTR_MOMENTS must be a non-empty subset of mean,max,min -- got {MOMENTS}")
N_TERMS = len(DESC_LAYERS) * len(MOMENTS)


def _slab(kv_pair, layer):
    """{layer: {"k":[T,C], "v":[T,C]}} -> that layer's K, [T, C]."""
    return kv_pair[layer]["k"]


@torch.no_grad()
def partition(cap, n_sections, mode="kmeans", iters=12, seed=0, proj_dim=128):
    """One chunk's captured KV -> [S, R] row indices, balanced so every section holds exactly
    R = 6240 // S rows."""
    if mode != "kmeans":
        raise ValueError(f"partition mode must be 'kmeans' -- got {mode!r}")
    T = cfg.CHUNK_TOKENS
    assert T % n_sections == 0, f"{n_sections} sections must divide {T} rows exactly"
    R = T // n_sections
    dev = _slab(cap, SECTION_LAYER).device
    X = _slab(cap, SECTION_LAYER).float()
    g = torch.Generator(device="cpu").manual_seed(seed)
    P = torch.randn(X.shape[1], proj_dim, generator=g).to(dev)
    Z = torch.nn.functional.normalize(X @ P, dim=-1)
    idx = [int(Z.norm(dim=-1).argmax())]
    d2 = ((Z - Z[idx[0]]) ** 2).sum(-1)
    for _ in range(n_sections - 1):
        idx.append(int(d2.argmax()))
        d2 = torch.minimum(d2, ((Z - Z[idx[-1]]) ** 2).sum(-1))
    Cn = Z[idx].clone()
    for _ in range(iters):
        a = (Z @ Cn.T).argmax(-1)
        for s in range(n_sections):
            m = a == s
            if m.any():
                Cn[s] = torch.nn.functional.normalize(Z[m].mean(0), dim=-1)
    import numpy as np

    aff = (Z @ Cn.T).flatten().cpu().numpy()
    order = np.argsort(-aff, kind="stable")
    tok = (order // n_sections).astype(np.int32)
    sec = (order % n_sections).astype(np.int32)
    taken = np.zeros(T, dtype=bool)
    room = np.full(n_sections, R, dtype=np.int32)
    out = [[] for _ in range(n_sections)]
    for t_i, s_i in zip(tok, sec):
        if not taken[t_i] and room[s_i] > 0:
            taken[t_i] = True
            room[s_i] -= 1
            out[s_i].append(int(t_i))
    assert all((len(o) == R for o in out)), [len(o) for o in out]
    return torch.tensor(out, dtype=torch.int32)


@torch.no_grad()
def section_feats(cap, sections):
    """Descriptor INPUT (not the descriptor): per section, mean/max/min of its rows' K at DESC_LAYERS."""
    idx = sections.long().to(_slab(cap, DESC_LAYERS[0]).device)
    outs = []
    for L in DESC_LAYERS:
        g = _slab(cap, L).float()[idx]
        outs += [{"mean": g.mean, "max": g.amax, "min": g.amin}[m](1) for m in MOMENTS]
    return torch.cat(outs, dim=-1)


class KVStore:
    """One clip's memory. Chunks are added as they are generated; nothing is ever evicted or encoded."""

    def __init__(
        self, n_sections=16, mode="kmeans", dtype=torch.bfloat16, keep_last=None, pin=False
    ):
        self.n_sections, self.mode, self.dtype, self.keep_last = (
            n_sections,
            mode,
            dtype,
            keep_last,
        )
        self.pin = pin
        self.kv, self.sections, self.feats = ({}, {}, {})
        self.R = cfg.CHUNK_TOKENS // n_sections

    @torch.no_grad()
    def add(self, c, cap, seed=0):
        """cap: {layer: {"k": [6240,5120], "v": [6240,5120]}} on GPU, straight from FrozenDiT.capture."""
        self.sections[c] = partition(cap, self.n_sections, self.mode, seed=seed)
        self.feats[c] = section_feats(cap, self.sections[c])
        buf = torch.empty(
            cfg.N_SLABS,
            cfg.CHUNK_TOKENS,
            cfg.DIT_DIM,
            dtype=self.dtype,
            device="cpu",
            pin_memory=self.pin,
        )
        for L in range(cfg.N_LAYERS):
            for j, w in enumerate(("k", "v")):
                # The CPU assignment reads immediately. An asynchronous D2H
                # conversion can expose unfinished data with multiple CPU threads.
                buf[L * 2 + j] = cap[L][w].to("cpu", self.dtype)
        self.kv[c] = buf
        if self.keep_last is not None:
            for k in [k for k in self.kv if k != 0 and k < c - self.keep_last]:
                del self.kv[k], self.sections[k], self.feats[k]

    def whole_of(self, c):
        """chunk c as the {layer: {"k","v"}} dict the whole-chunk slot path wants -- a VIEW, no copy."""
        b = self.kv[c]
        return {L: {"k": b[2 * L], "v": b[2 * L + 1]} for L in range(cfg.N_LAYERS)}

    def candidates(self, c, n_local, keep_sink=True):
        """Every retained past chunk that is neither the sink nor inside the local window. Those two
        are already in the context as real KV, so re-retrieving them would spend the row budget on
        rows the DiT can already see."""
        excl = ({0} if keep_sink else set()) | set(range(max(0, c - n_local), c))
        return [j for j in sorted(self.kv) if j < c and j not in excl]

    def all_feats(self, cand):
        """[len(cand)*S, F] on GPU, in the order the selector will index."""
        return torch.cat([self.feats[j] for j in cand], dim=0)

    @torch.no_grad()
    def gather(self, picks, dev):
        """picks: [(chunk j, section s), ...] -> [80, n*R, 5120] on `dev`, rows in pick order."""
        out = []
        for j, s in picks:
            idx = self.sections[j][s].long()
            out.append(self.kv[j][:, idx, :].to(dev, non_blocking=True))
        return torch.cat(out, dim=1)

    def bytes_resident(self):
        return sum((t.numel() * t.element_size() for t in self.kv.values()))
