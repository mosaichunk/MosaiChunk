"""Small checks for failures that would invalidate a benchmark comparison."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from mosaichunk.cli import environment
from mosaichunk.data import prepare_t2v
from mosaichunk.i2v import cfg
from mosaichunk.i2v.heads import Selector
from mosaichunk.i2v.store import partition
from mosaichunk.i2v.traj_palindrome import build_rotation_hssd_style
from mosaichunk.t2v.seed_pin import key_of


class ProtocolTests(unittest.TestCase):
    def test_seed_follows_scene_not_order(self):
        rows = [
            {"prompts": [p] * 4, "seed": str(seed)}
            for p, seed in [("Open the tin.", 4328929311085371135), ("Open the door.", 1903)]
        ]
        with tempfile.TemporaryDirectory() as directory:
            _, _, first = prepare_t2v(rows, Path(directory) / "first")
            _, _, second = prepare_t2v(rows[::-1], Path(directory) / "second")
            a, b = json.loads(first.read_text()), json.loads(second.read_text())
        self.assertEqual(a, b)
        self.assertEqual(a[key_of("Open the tin. . .")], 4328929311085371135)

    def test_base_matches_active_budget(self):
        for budget in (1, 2):
            self.assertEqual(
                environment("i2v", budget, "base")["PTR_N_LOCAL_SLOTS"], str(2 + budget)
            )
            self.assertEqual(environment("i2v", budget, "mc")["PTR_N_LOCAL_SLOTS"], "2")

    def test_experimental_environment_does_not_leak(self):
        with patch.dict(
            "os.environ",
            {"PTR_MMR_LAM": "999", "PTR_PART_MODE": "grid", "RAVEN_PIN_LATENT": "wrong.pt"},
        ):
            env = environment("t2v", 2, "mc")
        self.assertEqual(env["PTR_MMR_LAM"], "1.0")
        self.assertEqual(env["PTR_PART_MODE"], "kmeans")
        self.assertNotIn("RAVEN_PIN_LATENT", env)

    def test_balanced_partition_covers_each_row_once(self):
        with patch.object(cfg, "CHUNK_TOKENS", 24):
            cap = {20: {"k": torch.randn(24, 16)}}
            state = torch.random.get_rng_state()
            sections = partition(cap, 4, seed=7, proj_dim=8)
            self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
            self.assertEqual(tuple(sections.shape), (4, 6))
            self.assertEqual(sorted(sections.flatten().tolist()), list(range(24)))
            self.assertTrue(torch.equal(sections, partition(cap, 4, seed=7, proj_dim=8)))

    def test_global_budget_and_unselected_gradient(self):
        selector = Selector(d=8, budget_sections=3).eval()
        logits = torch.tensor(
            [[0.0, 1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0, 5.0]], requires_grad=True
        )
        candidates = [(c, 0) for c in range(5)]
        picked, _, gates, _ = selector._allocate_global(logits, candidates)
        self.assertEqual(picked, [(4, 0), (3, 0), (2, 0)])
        self.assertAlmostEqual(float(gates.mean().detach()), 1.0, places=6)
        gates.sum().backward()
        # The full-pool softmax and detached normalization supervise omitted sections too.
        self.assertGreater(float(logits.grad[:, :2].abs().sum()), 0.0)

    def test_training_trajectory_is_closed_and_uses_intrinsics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(
                root / "intrinsics.npy", np.array([[400.0, 400.0, 416.0, 240.0]], dtype=np.float32)
            )
            (root / "prompt.txt").write_text("A room.")
            build_rotation_hssd_style(str(root), str(root / "action"))
            poses = np.load(root / "action/poses.npy")
            intrinsics = np.load(root / "action/intrinsics.npy")
        self.assertEqual(poses.shape, (321, 4, 4))
        np.testing.assert_array_equal(poses[0], poses[-1])
        self.assertFalse(np.allclose(poses[0], poses[160]))
        np.testing.assert_array_equal(intrinsics[0], [400.0, 400.0, 416.0, 240.0])


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
