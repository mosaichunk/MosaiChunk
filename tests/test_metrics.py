"""Regression checks for independent revisit selection and temporal SSIM."""

import unittest

import numpy as np
import torch
from skimage.metrics import structural_similarity

from mosaichunk.metrics import _win, chunks_of, temp_ssim
from score import revisit


class MetricTests(unittest.TestCase):
    def test_full_turn_uses_unsigned_heading(self):
        angles = np.deg2rad([0, 90, 175, 260, 355, 385])
        poses = np.repeat(np.eye(4)[None], len(angles), axis=0)
        poses[:, :3, 2] = np.stack([np.sin(angles), np.zeros_like(angles), np.cos(angles)], axis=1)
        self.assertEqual(revisit(poses, translation=False), (4, 2))

    def test_translation_turnaround_uses_position(self):
        poses = np.repeat(np.eye(4)[None], 5, axis=0)
        angles = np.deg2rad([0, 150, 100, 2, 20])
        poses[:, :3, 2] = np.stack([np.sin(angles), np.zeros_like(angles), np.cos(angles)], axis=1)
        poses[:, 0, 3] = [0, 1, 2, 1, 0]
        self.assertEqual(revisit(poses, translation=True), (3, 2))

    def test_ssim_matches_independent_implementation(self):
        generator = np.random.default_rng(0)
        first = generator.integers(0, 256, (96, 128), dtype=np.uint8)
        second = np.clip(
            first.astype(int) + generator.integers(-12, 13, first.shape), 0, 255
        ).astype(np.uint8)
        expected = structural_similarity(
            first,
            second,
            data_range=255,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
        )
        images = torch.from_numpy(np.stack([first, second])).float()
        actual = float(temp_ssim(images, _win(device="cpu"))[0])
        self.assertLess(abs(actual - expected), 2e-4)

    def test_chunks_cover_decoded_video_without_overlap(self):
        for split, count in [("t2v", 379), ("i2v", 253)]:
            chunks = chunks_of(count, split)
            covered = [frame for lo, hi in chunks for frame in range(lo, hi + 1)]
            self.assertEqual(covered, list(range(count)))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
