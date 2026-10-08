"""Sequence-parallel reductions preserve the full-head descriptor features."""

from __future__ import annotations

import torch


def sp_group():
    """The sequence-parallel group, or None when there is no parallelism."""
    try:
        from common.distributed.unified_parallel import (
            get_unified_parallel_group,
            get_unified_parallel_world_size,
            is_unified_parallel_initialized,
        )

        if is_unified_parallel_initialized() and get_unified_parallel_world_size() > 1:
            return (get_unified_parallel_group(), get_unified_parallel_world_size())
    except Exception:
        pass
    return (None, 1)


def gather_features(x: torch.Tensor) -> torch.Tensor:
    """[..., d_shard] -> [..., d_shard * sp], concatenated in rank order."""
    grp, n = sp_group()
    if grp is None:
        return x
    xs = [torch.empty_like(x) for _ in range(n)]
    torch.distributed.all_gather(xs, x.contiguous(), group=grp)
    return torch.cat(xs, dim=-1)


def sum_reduce(x: torch.Tensor) -> torch.Tensor:
    """Sum a per-shard partial product across the group, in place-safe fashion."""
    grp, n = sp_group()
    if grp is None:
        return x
    y = x.contiguous().clone()
    torch.distributed.all_reduce(y, op=torch.distributed.ReduceOp.SUM, group=grp)
    return y


def shard_slice(full: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """This rank's contiguous slice of a commonly-seeded full-width tensor."""
    grp, n = sp_group()
    if grp is None:
        return full
    r = torch.distributed.get_rank(group=grp)
    step = full.shape[dim] // n
    return full.narrow(dim, r * step, step)


__all__ = ["sp_group", "gather_features", "sum_reduce", "shard_slice"]
