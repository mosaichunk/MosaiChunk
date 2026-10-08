"""Pose-nearest historical chunks for I2V self-distillation."""

from . import cfg
from .traj_palindrome import pose_dist


def _oracle_far(c, cand, yaw, n_far=None):
    """The teacher's set: the n_far pose-nearest past chunks. NOT a label given to the student -- it
    only defines the target."""
    n_far = n_far or len(cfg.F_FAR)
    cpos, cfwd = yaw
    return sorted(sorted(cand, key=lambda j: pose_dist(cpos[c], cfwd[c], cpos[j], cfwd[j]))[:n_far])
