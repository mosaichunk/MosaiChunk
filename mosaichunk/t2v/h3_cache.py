"""CPU history and read-only differentiable KV composition for H3-AR."""

from __future__ import annotations

import torch

from .rope_reanchor import reanchor


class ChunkStore:
    """Per-chunk K/V for one clip, kept on CPU, sliced out of a no-eviction cache."""

    def __init__(self, chunk_lens_per_sample, device="cpu"):
        self.lens = list(chunk_lens_per_sample)
        self.device = device
        self.kv = [dict() for _ in self.lens]
        self.offsets = []
        off = 0
        for n in self.lens:
            self.offsets.append(off)
            off += n
        self.total = off

    @torch.no_grad()
    def fill_from_cache(self, cache, sample=0):
        """Slice a finished no-eviction NaiveCache into per-chunk blocks."""
        for layer, k in cache.key_cache.items():
            if k is None:
                continue
            v = cache.value_cache[layer]
            assert k.shape[0] >= self.total, (
                f"cache layer {layer} has {k.shape[0]} rows, chunk_lens sum to {self.total}: the capture pass must run with window_size=None so nothing was evicted"
            )
            for c, (off, n) in enumerate(zip(self.offsets, self.lens)):
                self.kv[c][layer] = (
                    k[off : off + n].detach().to(self.device, copy=True),
                    v[off : off + n].detach().to(self.device, copy=True),
                )

    def kv_at(self, c, layer):
        """(k, v) of cache chunk c at one layer, as stored. The one accessor
        ptr_common reads through, so a ChunkStore and a LiveStore are
        interchangeable there."""
        return self.kv[c][layer]

    def rows(self, c, layer, idx, device, gate=None):
        """Gather rows `idx` of chunk `c`, optionally scaling V by `gate`."""
        k, v = self.kv[c][layer]
        k = k.to(device, non_blocking=True).index_select(0, idx)
        v = v.to(device, non_blocking=True).index_select(0, idx)
        if gate is not None:
            dtype = v.dtype
            v = (v * gate.view(-1, *[1] * (v.dim() - 1))).to(dtype)
        return (k, v)

    def nbytes(self):
        return sum(
            (
                a.numel() * a.element_size() + b.numel() * b.element_size()
                for d in self.kv
                for a, b in d.values()
            )
        )


class GatedCache:
    """A NaiveCache-shaped view over hand-assembled K/V, for ONE forward."""

    def __init__(self, key_cache, value_cache, kvlens):
        self.key_cache = key_cache
        self.value_cache = value_cache
        self.kvlens = list(kvlens)
        self.chunk_lens = []
        self.sink = [0] * len(self.kvlens)
        self.window_size = [None] * len(self.kvlens)
        self.batch_size = len(self.kvlens)

    @property
    def num_layers(self):
        return len(self.key_cache)

    @property
    def seq_len(self):
        return self.seq_lens(min(self.key_cache))

    def seq_lens(self, idx):
        """Rows this layer holds. RAISES on a missing layer rather than returning 0."""
        k = self.key_cache.get(idx)
        assert k is not None, (
            f"layer {idx} is missing from the assembled cache ({sorted(self.key_cache)} present): the forward would run with no memory and no text at all"
        )
        return k.shape[0]

    def update_kvcache(self, *a, **k):
        raise RuntimeError(
            "GatedCache is read-only: it holds the exact rows the loss is computed against, and absorbing a write would corrupt them. Call _chunk_forward with update_cache=False."
        )

    def update_kvlens(self, *a, **k):
        raise RuntimeError("GatedCache is read-only (see update_kvcache).")

    @staticmethod
    def assemble(store, layers, parts, device, pad_samples=0):
        """parts = ordered list of (chunk, rows, gate_or_None, freqs_old, freqs_new)."""
        by_chunk, order = ({}, [])
        for pos, part in enumerate(parts):
            c, idx, gate = (part[0], part[1], part[2])
            f_old = part[3] if len(part) > 3 else None
            f_new = part[4] if len(part) > 4 else None
            if c not in by_chunk:
                by_chunk[c] = []
                order.append(c)
            by_chunk[c].append((pos, idx, gate, f_old, f_new))
        merged = {}
        for c in order:
            items = by_chunk[c]
            idx_cat = torch.cat([i.to(device) for _, i, _, _, _ in items], 0)
            if any((g is not None for _, _, g, _, _ in items)):
                gates = []
                for _, i, g, _, _ in items:
                    n = int(i.numel())
                    gates.append(
                        g.reshape(-1).expand(n)
                        if g is not None
                        else torch.ones(n, device=device, dtype=torch.float32)
                    )
                gate_cat = torch.cat(gates, 0)
            else:
                gate_cat = None
            has = [fo is not None and fn is not None for _, _, _, fo, fn in items]
            if any(has):
                assert all(has), f"chunk {c} mixes re-anchored and un-anchored parts"
                fo_cat = torch.cat([fo.to(device) for _, _, _, fo, _ in items], 0)
                fn_cat = torch.cat([fn.to(device) for _, _, _, _, fn in items], 0)
            else:
                fo_cat = fn_cat = None
            merged[c] = (
                idx_cat,
                gate_cat,
                fo_cat,
                fn_cat,
                [pos for pos, _, _, _, _ in items],
                [int(i.numel()) for _, i, _, _, _ in items],
            )
        kc, vc = ({}, {})
        total = 0
        for layer in layers:
            pieces_k = [None] * len(parts)
            pieces_v = [None] * len(parts)
            n = 0
            for c in order:
                idx_cat, gate_cat, fo, fn, positions, sizes = merged[c]
                k, v = store.rows(c, layer, idx_cat, device, gate_cat)
                if fn is not None:
                    k = reanchor(k, fo, fn)
                ks = torch.split(k, sizes, dim=0)
                vs = torch.split(v, sizes, dim=0)
                for pos, kk, vv in zip(positions, ks, vs):
                    pieces_k[pos] = kk
                    pieces_v[pos] = vv
                n += int(k.shape[0])
            kc[layer] = torch.cat(pieces_k, dim=0)
            vc[layer] = torch.cat(pieces_v, dim=0)
            total = n
        return GatedCache(kc, vc, [total] + [0] * int(pad_samples))


class LiveStore:
    """A ChunkStore-shaped VIEW over a cache that is still being written."""

    def __init__(self, cache, sample=0):
        self.cache, self.sample = (cache, sample)
        self.lens = [row[sample] for row in cache.chunk_lens]
        self.offsets, off = ([], 0)
        for n in self.lens:
            self.offsets.append(off)
            off += n
        self.total = off
        self.base = sum(cache.kvlens[:sample])

    def kv_at(self, c, layer):
        k, v = (self.cache.key_cache[layer], self.cache.value_cache[layer])
        lo = self.base + self.offsets[c]
        hi = lo + self.lens[c]
        assert k is not None and k.shape[0] >= hi, (
            f"layer {layer} has {(None if k is None else k.shape[0])} rows but chunk {c} ends at {hi}: the rollout must run with window_size=None so nothing was evicted"
        )
        return (k[lo:hi], v[lo:hi])

    def rows(self, c, layer, idx, device, gate=None):
        k, v = self.kv_at(c, layer)
        idx = idx.to(k.device)
        k = k.index_select(0, idx).to(device, non_blocking=True)
        v = v.index_select(0, idx).to(device, non_blocking=True)
        if gate is not None:
            dtype = v.dtype
            v = (v * gate.to(v.device).view(-1, *[1] * (v.dim() - 1))).to(dtype)
        return (k, v)

    def layers(self):
        return sorted((l for l, k in self.cache.key_cache.items() if k is not None))


class SideStore:
    """Every chunk's K/V, kept on the host, filled as the rollout produces them."""

    def __init__(self, text_len: int):
        self.lens = [int(text_len)]
        self.kv: list[dict] = [dict()]
        self.text: dict[int, dict] = {}

    @torch.no_grad()
    def add(self, cache, sample: int = 0):
        """File the chunk the rollout has just written -- the LAST one in the cache."""
        n = int(cache.chunk_lens[-1][sample])
        base = sum(cache.kvlens[:sample])
        hi = base + int(cache.kvlens[sample])
        rec = {}
        for layer, k in cache.key_cache.items():
            if k is None:
                continue
            v = cache.value_cache[layer]
            rec[layer] = (
                k[hi - n : hi].detach().to("cpu", copy=True),
                v[hi - n : hi].detach().to("cpu", copy=True),
            )
        assert rec, "the cache had no populated layers to file"
        self.kv.append(rec)
        self.lens.append(n)

    @torch.no_grad()
    def snap_text(self, cache, c: int, sample: int = 0):
        """Record chunk 0's rows as they stand for video chunk c. Idempotent."""
        if c in self.text:
            return
        n = int(cache.chunk_lens[0][sample])
        base = sum(cache.kvlens[:sample])
        self.text[c] = {
            l: (
                k[base : base + n].detach().to("cpu", copy=True),
                cache.value_cache[l][base : base + n].detach().to("cpu", copy=True),
            )
            for l, k in cache.key_cache.items()
            if k is not None
        }

    def kv_at(self, c, layer):
        return self.kv[c][layer]

    def rows(self, c, layer, idx, device, gate=None):
        k, v = self.kv[c][layer]
        idx = idx.to(k.device)
        k = k.index_select(0, idx).to(device, non_blocking=True)
        v = v.index_select(0, idx).to(device, non_blocking=True)
        if gate is not None:
            dtype = v.dtype
            v = (v * gate.to(v.device).view(-1, *[1] * (v.dim() - 1))).to(dtype)
        return (k, v)

    def nbytes(self):
        return sum(
            (
                a.numel() * a.element_size() + b.numel() * b.element_size()
                for d in self.kv
                for a, b in d.values()
            )
        )


class SplitStore:
    """Route chunk 0 to the LIVE cache and every video chunk to the side store."""

    def __init__(self, live, side: SideStore, sample: int = 0, text_at: int | None = None):
        self.live, self.side, self.sample = (live, side, sample)
        self.text_at = text_at
        self.lens = list(side.lens)
        self.lens[0] = int(live.chunk_lens[0][sample])

    def _text(self, layer):
        if self.text_at is not None:
            return self.side.text[self.text_at][layer]
        n = self.lens[0]
        base = sum(self.live.kvlens[: self.sample])
        k, v = (self.live.key_cache[layer], self.live.value_cache[layer])
        return (k[base : base + n], v[base : base + n])

    def kv_at(self, c, layer):
        return self._text(layer) if c == 0 else self.side.kv_at(c, layer)

    def rows(self, c, layer, idx, device, gate=None):
        if c != 0:
            return self.side.rows(c, layer, idx, device, gate)
        k, v = self._text(layer)
        idx = idx.to(k.device)
        k = k.index_select(0, idx).to(device, non_blocking=True)
        v = v.index_select(0, idx).to(device, non_blocking=True)
        if gate is not None:
            dtype = v.dtype
            v = (v * gate.to(v.device).view(-1, *[1] * (v.dim() - 1))).to(dtype)
        return (k, v)

    def layers(self):
        return sorted((l for l, k in self.live.key_cache.items() if k is not None))
