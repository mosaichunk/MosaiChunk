"""The canonical 120-degree out-and-back training trajectory."""

import math
import os
from pathlib import Path

import numpy as np

CANON_R0 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=np.float64)
T0 = np.array([0.0, 0.0, 1.5], dtype=np.float64)


def chunk_poses(c2w, n_ch, chunk_frames):
    """Per-chunk (mean position, mean rotation-as-forward) for pose-distance retrieval. c2w [N,4,4]."""
    pos = np.zeros((n_ch, 3))
    fwd = np.zeros((n_ch, 3))
    for c in range(n_ch):
        seg = c2w[c * chunk_frames : (c + 1) * chunk_frames]
        pos[c] = seg[:, :3, 3].mean(0)
        fwd[c] = seg[:, :3, 2].mean(0)
    return (pos, fwd)


def pose_dist(pi, fi, pj, fj, w_pos=1.0, w_rot=1.5):
    """Distance between two chunk poses: position L2 + forward-direction angle (rad)."""
    dp = float(np.linalg.norm(pi - pj))
    cosang = float(
        np.clip(np.dot(fi, fj) / (np.linalg.norm(fi) * np.linalg.norm(fj) + 1e-09), -1, 1)
    )
    dr = math.acos(cosang)
    return w_pos * dp + w_rot * dr


def build_rotation_hssd_style(clip, out_dir, frames=321, theta_deg=120.0):
    """HSSD-CONFORMING trajectory on a Sekai first frame: fixed +theta yaw sweep, eased, turnaround at mid."""
    span = math.radians(theta_deg)
    N = int(frames)
    poses = np.zeros((N, 4, 4), dtype=np.float32)
    for i in range(N):
        p = _triangle_tau(i / (N - 1), 0.5)
        poses[i, :3, :3] = _Rz(span * p) @ CANON_R0
        poses[i, :3, 3] = T0
        poses[i, 3, 3] = 1.0
    poses[-1] = poses[0]
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "poses.npy"), poses)
    intr = np.load(os.path.join(clip, "intrinsics.npy"))[:1].repeat(N, 0)
    np.save(os.path.join(out_dir, "intrinsics.npy"), intr)
    with open(os.path.join(out_dir, "prompt.txt"), "w") as fh:
        pt = os.path.join(clip, "prompt.txt")
        fh.write(Path(pt).read_text() if os.path.exists(pt) else "")
    return N


def _Rz(th):
    c, s = (math.cos(th), math.sin(th))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _smoother(u):
    return u * u * u * (u * (6 * u - 15) + 10)


def _triangle_tau(s, tau):
    """Palindrome with turnaround point tau. tau=0.5 = symmetric triangle (going & return same pace)."""
    return _smoother(s / max(tau, 1e-06)) if s <= tau else _smoother((1 - s) / max(1 - tau, 1e-06))
