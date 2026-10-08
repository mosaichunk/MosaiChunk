"""Invert and reapply temporal RoPE without changing spatial coordinates."""

from __future__ import annotations

import torch


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """model.py:361-363, verbatim."""
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _cos_sin(freqs: torch.Tensor, dtype: torch.dtype):
    """freqs [N, F] -> (cos, sin), each [N, 1, rot_dim], matching model.py:499-509 and
    the unsqueeze(1) the non-fused branch applies at :537-538."""
    half = freqs.shape[-1] // 2
    cos_half = torch.cos(freqs[:, :half])
    sin_half = torch.sin(freqs[:, :half])
    cos = torch.cat((cos_half, cos_half), dim=-1).unsqueeze(1).to(dtype)
    sin = torch.cat((sin_half, sin_half), dim=-1).unsqueeze(1).to(dtype)
    return (cos, sin)


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """The model's own forward rotation. x [N, heads, dim]."""
    cos, sin = _cos_sin(freqs, x.dtype)
    rot = cos.shape[-1]
    x_rot, x_pass = (x[..., :rot], x[..., rot:])
    return torch.cat((x_rot * cos + _rotate_half(x_rot) * sin, x_pass), dim=-1)


def unapply_rope(y: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """The inverse: same cos, negated sin."""
    cos, sin = _cos_sin(freqs, y.dtype)
    rot = cos.shape[-1]
    y_rot, y_pass = (y[..., :rot], y[..., rot:])
    return torch.cat((y_rot * cos - _rotate_half(y_rot) * sin, y_pass), dim=-1)


def _check_shapes(x, freqs):
    rot = freqs.shape[-1]
    assert x.shape[-1] >= rot, (
        f"RoPE covers {rot} dims but the head is only {x.shape[-1]} wide -- the freqs and the cache disagree about the model (inv_freq len {rot // 6} implies 6*{rot // 6} = {rot} rotated dims)"
    )
    assert x.shape[0] == freqs.shape[0], (
        f"{x.shape[0]} rows but {freqs.shape[0]} positions: every row must carry its own"
    )


def reanchor(k_cached: torch.Tensor, freqs_old: torch.Tensor, freqs_new: torch.Tensor):
    """A cached (rotated) key moved from freqs_old to freqs_new."""
    _check_shapes(k_cached, freqs_old)
    _check_shapes(k_cached, freqs_new)
    dt = k_cached.dtype
    x = unapply_rope(k_cached.float(), freqs_old.float())
    return apply_rope(x, freqs_new.float()).to(dt)


_T_GROUP = 5
_FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
_FRAME_RESCALE = 5.0 / 3.0


def video_t_grid(n: int, origin: float) -> torch.Tensor:
    """Temporal position of each video latent frame. packing.py:258-265, verbatim."""
    spans = torch.tensor(
        [_FRAME_RESCALE * _FRAME_PER_TOKEN[k % _T_GROUP] for k in range(n)], dtype=torch.float64
    )
    return origin + torch.cat([torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])


def rope_freqs(pos: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
    """(t,h,w) positions [S,3] -> freqs [S, 6*len(inv_freq)]. model.py:485-497."""
    per_axis = pos.to(torch.float32).unsqueeze(-1) * inv_freq.view(1, 1, -1).to(torch.float32)
    t_f, h_f, w_f = per_axis.unbind(dim=1)
    half = torch.cat((t_f, h_f, w_f), dim=-1)
    return torch.cat((half, half), dim=-1)


def self_test(n=512, heads=14, dim=128, rot_dim=64, device="cpu"):
    """reanchor(rope(x, p_old), p_old, p_new) must equal rope(x, p_new)."""
    g = torch.Generator(device="cpu").manual_seed(0)
    x = torch.randn(n, heads, dim, generator=g).to(device)
    f_old = torch.randn(n, rot_dim * 2, generator=g).to(device) * 3.0
    f_new = torch.randn(n, rot_dim * 2, generator=g).to(device) * 3.0
    y_old = apply_rope(x, f_old)
    e_inv = float((unapply_rope(y_old, f_old) - x).abs().max())
    assert e_inv < 0.0001, f"RoPE inverse is not an inverse: max|diff|={e_inv}"
    want = apply_rope(x, f_new)
    got = reanchor(y_old, f_old, f_new)
    e_move = float((got - want).abs().max())
    assert e_move < 0.0001, f"re-anchor != rotating the original to the new position: {e_move}"
    tail = got[..., rot_dim * 2 :]
    e_pass = 0.0 if tail.numel() == 0 else float((tail - x[..., rot_dim * 2 :]).abs().max())
    assert e_pass == 0.0, f"non-rotated head dims were modified: {e_pass}"
    xb = x.to(torch.bfloat16)
    y_b = apply_rope(xb, f_old)
    want_b = apply_rope(xb, f_new)
    got_b = reanchor(y_b, f_old, f_new)
    e_bf = float((got_b.float() - want_b.float()).abs().max())
    scale = float(want_b.float().abs().max())
    assert e_bf < scale * 0.05, (
        f"re-anchoring costs {e_bf:.3g} on a {scale:.3g} tensor -- far more than bf16 rounding; the fp32 intermediate is not doing its job"
    )
    return {
        "inverse": e_inv,
        "move": e_move,
        "passthrough": e_pass,
        "bf16_abs": e_bf,
        "bf16_rel": e_bf / scale,
        "scale": scale,
    }


def self_test_real(
    n_frames=112, gh=12, gw=22, text_len=61, inv_len=16, heads=14, head_dim=128, device="cpu"
):
    """The same property, on the real 3D layout and the real time grid."""
    g = torch.Generator(device="cpu").manual_seed(1)
    inv_freq = (10000.0 ** (-torch.arange(0, 32, 2, dtype=torch.float32) / 32))[:inv_len]
    t_grid = video_t_grid(n_frames, float(text_len))
    src_f, dst_f = (40, 8)
    hs = torch.arange(gh).repeat_interleave(gw)
    ws = torch.arange(gw).repeat(gh)
    n = hs.numel()
    pos_old = torch.stack([torch.full((n,), float(t_grid[src_f])), hs.float(), ws.float()], -1)
    pos_new = torch.stack([torch.full((n,), float(t_grid[dst_f])), hs.float(), ws.float()], -1)
    f_old = rope_freqs(pos_old, inv_freq).to(device)
    f_new = rope_freqs(pos_new, inv_freq).to(device)
    assert f_old.shape[-1] == 6 * inv_len, f_old.shape
    x = torch.randn(n, heads, head_dim, generator=g).to(device)
    cached = apply_rope(x, f_old)
    got = reanchor(cached, f_old, f_new)
    want = apply_rope(x, f_new)
    e = float((got - want).abs().max())
    assert e < 0.0001, f"real-geometry re-anchor is wrong: max|diff|={e}"
    e_id = float((reanchor(cached, f_old, f_old) - cached).abs().max())
    assert e_id < 0.0001, f"re-anchoring to the same position is not a no-op: {e_id}"
    n_pass = head_dim - 6 * inv_len
    return {
        "rows": n,
        "rot_dim": 6 * inv_len,
        "passthrough_dims": n_pass,
        "move": e,
        "identity": e_id,
        "t_src": float(t_grid[src_f]),
        "t_dst": float(t_grid[dst_f]),
    }


if __name__ == "__main__":
    for rot in (32, 64):
        r = self_test(rot_dim=rot)
        print(
            f"rot_dim {rot:>3} / head 128 | fp32 inverse {r['inverse']:.2e} move {r['move']:.2e} passthrough {r['passthrough']:.0e} | bf16 {r['bf16_abs']:.3g} on scale {r['scale']:.3g} = {r['bf16_rel'] * 100:.2f}%"
        )
    r = self_test_real()
    print(
        f"real geometry | {r['rows']} rows, rotate {r['rot_dim']}/128 dims ({r['passthrough_dims']} pass through) | t {r['t_src']:.2f} -> {r['t_dst']:.2f} | move {r['move']:.2e} identity {r['identity']:.2e}"
    )
    print("ROPE RE-ANCHOR OK")
