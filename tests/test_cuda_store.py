"""GPU regression for storing float32 keys in the CPU bfloat16 history bank."""

import unittest
from unittest.mock import patch

import torch

from mosaichunk.i2v import store


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class CUDAStoreTests(unittest.TestCase):
    def test_cpu_bank_contains_completed_transfers(self):
        previous = torch.get_num_threads()
        torch.set_num_threads(4)
        self.addCleanup(torch.set_num_threads, previous)
        # Large enough to expose a CPU read racing an unfinished D2H conversion.
        rows, width = 6240, 5120
        cap = {0: {name: torch.randn(rows, width, device="cuda") for name in ("k", "v")}}
        with (
            patch.object(store.cfg, "CHUNK_TOKENS", rows),
            patch.object(store.cfg, "DIT_DIM", width),
            patch.object(store.cfg, "N_LAYERS", 1),
            patch.object(store.cfg, "N_SLABS", 2),
            patch.object(store, "partition", return_value=torch.arange(rows)[None]),
            patch.object(store, "section_feats", return_value=torch.zeros(1, 1)),
        ):
            bank = store.KVStore(n_sections=1)
            bank.add(0, cap)
            actual = bank.whole_of(0)[0]
        for name in ("k", "v"):
            expected = cap[0][name].to(torch.bfloat16).cpu()
            self.assertTrue(torch.equal(actual[name], expected), name)


if __name__ == "__main__":
    unittest.main()
