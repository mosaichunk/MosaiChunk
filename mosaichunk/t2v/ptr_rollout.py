"""Shared student rollout with a bounded active cache and CPU history."""

from __future__ import annotations

import logging
import os
import pathlib

import torch
from common.distributed import ops
from common.distributed.ops import get_device
from utils.naive_cache import NaiveCache

from . import ptr_common as P
from . import ptr_geom
from .h3_cache import GatedCache, SideStore, SplitStore

ROLLOUT_MODE = os.environ.get("PTR_ROLLOUT", "student")
logger = logging.getLogger(__name__)


def _rank0():
    try:
        return ops.get_rank() == 0
    except Exception:
        return True


class PtrRolloutMixin:
    """Composes the student memory for every noise forward of a rollout."""

    _ptr_selector = None
    _ptr_budget_chunks = None
    _ptr_explore = 0.0
    _ptr_sel = None
    _ptr_cache_id = None
    _ptr_stats = None
    _ptr_memo = None
    _ptr_side = None
    _ptr_geom = None
    _ptr_last = (None, None)

    def _validation_inputs(self, config, models, prompts):
        """The one validation hook handed `models`, so the selector is stashed here."""
        sel = models.get("selector")
        assert sel is not None, (
            "no selector in models: this config must declare models.selector and resume its weights, or the arm silently degrades to the baseline"
        )
        self._ptr_selector = sel
        self._ptr_budget_chunks = P.EVAL_FAR_BUDGET_CHUNKS
        self._ptr_explore = 0.0
        return super()._validation_inputs(config, models, prompts)

    def _ptr_arm(self, selector, far_budget_chunks, explore=0.0):
        """Context manager: run a rollout as the ptr arm at this budget."""
        return _Arm(self, selector, far_budget_chunks, explore)

    def _rollout_latents(self, backbone, inputs, rng, **kw):
        """Eviction stays ON. The far chunks live in a SIDE STORE."""
        if ROLLOUT_MODE != "student" or self._ptr_selector is None:
            return super()._rollout_latents(backbone, inputs, rng, **kw)
        self._ptr_reset()
        self._ptr_side = SideStore(int(inputs.text_lens[0]))
        try:
            self._ptr_geom = ptr_geom.Geometry(
                inputs.layouts[0], int(inputs.text_lens[0]), ptr_geom.find_inv_freq(backbone)
            )
        except Exception as e:
            raise RuntimeError(
                f"cannot build the re-anchoring geometry ({type(e).__name__}: {e}); refusing to run the arm with rows left at their original frames, which is a different method"
            ) from e
        try:
            out = super()._rollout_latents(backbone, inputs, rng, **kw)
        finally:
            stats = self._ptr_stats or []
            self._ptr_dump_picks(stats)
            self._ptr_last = (self._ptr_side, self._ptr_geom)
            self._ptr_reset()
        live = [d for d in stats if d["n_picked"] > 0]
        floor_arm = P.budget_sections(self._ptr_budget_chunks) <= 0
        assert stats, "the ptr arm composed no memory at all: the hook never fired"
        assert live or floor_arm or len(inputs.layouts[0].chunks) <= P.N_LOCAL + 1, (
            f"the ptr arm picked nothing on any of {len(stats)} chunks, yet the clip has {len(inputs.layouts[0].chunks)} > {P.N_LOCAL + 1}"
        )
        if floor_arm and _rank0():
            logger.info(
                "[ptr] arm=FLOOR (text + local only, through the assembly path) | %d chunks composed",
                len(stats),
            )
        if live and _rank0():
            src = sorted({k for d in live for k in d["src"]})
            logger.info(
                "[ptr] arm=student budget=%.1f chunks | %d/%d chunks retrieved | picks %d..%d of %d..%d candidates | source chunks %s | anchored into %s",
                self._ptr_budget_chunks,
                len(live),
                len(stats),
                min((d["n_picked"] for d in live)),
                max((d["n_picked"] for d in live)),
                min((d["n_cand"] for d in live)),
                max((d["n_cand"] for d in live)),
                src,
                sorted({a for d in live for a in d.get("anchors", [])}),
            )
        return out

    def _ptr_dump_picks(self, stats):
        """One picks JSON per rollout, named by the prompt index the media uses."""
        d = os.environ.get("PTR_PICKS_DIR")
        if not d or not stats:
            return
        try:
            if ops.get_rank() % self._sp_size() != 0:
                return
            import json as _j

            i = self._prompt_index
            p = pathlib.Path(d)
            p.mkdir(parents=True, exist_ok=True)
            (p / f"picks{i:04d}.json").write_text(
                _j.dumps(
                    {
                        "prompt_idx": i,
                        "budget_chunks": self._ptr_budget_chunks,
                        "mmr_lam": P.MMR_LAM,
                        "n_sections": P.N_SECTIONS,
                        "n_local": P.N_LOCAL,
                        "chunks": stats,
                    }
                )
            )
        except Exception as e:
            logger.error("[ptr] FAILED to write picks to %s: %s: %s", d, type(e).__name__, e)

    def _ptr_side_result(self):
        """The store and geometry the LAST rollout built, for a caller that grades a
        chunk afterwards."""
        return getattr(self, "_ptr_last", (None, None))

    def _ptr_reset(self):
        self._ptr_cache_id = self._ptr_sel = self._ptr_memo = None
        self._ptr_side = self._ptr_geom = None
        self._ptr_stats = []

    def _chunk_forward(self, model, inputs, *, chunk_index, role, cache, update_cache, **kw):
        armed = (
            ROLLOUT_MODE == "student"
            and self._ptr_selector is not None
            and isinstance(cache, NaiveCache)
        )
        if armed and role == "noise":
            sub = self._ptr_memory(cache, inputs, int(chunk_index))
            if sub is not None:
                cache = sub
        out = super()._chunk_forward(
            model,
            inputs,
            chunk_index=chunk_index,
            role=role,
            cache=cache,
            update_cache=update_cache,
            **kw,
        )
        if armed and role == "clean" and update_cache and (self._ptr_side is not None):
            self._ptr_side.add(cache, sample=0)
            self._ptr_memo = None
        return out

    def _ptr_memory(self, cache, inputs, c):
        _ = cache.seq_len
        memo = getattr(self, "_ptr_memo", None)
        if memo is not None and memo[0] == id(cache) and (memo[1] == c):
            return memo[2]
        if len(self._ptr_side.lens) != c + 1:
            raise AssertionError(
                f"chunk {c}: the side store holds {len(self._ptr_side.lens)} chunks, expected {c + 1} (text + video 0..{c - 1})"
            )
        self._ptr_side.snap_text(cache, c)
        store = SplitStore(cache, self._ptr_side, sample=0)
        if self._ptr_sel is None or self._ptr_cache_id != id(cache):
            self._ptr_cache_id = id(cache)
            self._ptr_sel = P.Selection(store, inputs.layouts[0], get_device(), geom=self._ptr_geom)
            self._ptr_memo = None
        sel = self._ptr_sel
        sel.store = store
        budget = P.budget_sections(self._ptr_budget_chunks)
        explore = float(self._ptr_explore)
        gen = None
        if explore > 0:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(2654435769 ^ int(self._ptr_step_salt()) * 1009 + c)
        cand, picks, gate, _ = P.choose(self._ptr_selector, sel, c, budget, gen, explore)
        parts = P.student_parts(store, sel, c, cand, picks, gate)
        self._ptr_stats.append(
            {
                "picked": [
                    [int(cand[i][0]), int(cand[i][1])]
                    for i in (picks.tolist() if torch.is_tensor(picks) else picks)
                ],
                "dest": {str(k): int(v) for k, v in getattr(sel, "last_dest", {}).items()},
                "gate": [round(float(g), 5) for g in gate] if len(gate) else [],
                "chunk": c,
                "n_cand": len(cand),
                "n_picked": int(picks.numel()) if torch.is_tensor(picks) else len(picks),
                "src": sorted(
                    {cand[i][0] for i in (picks.tolist() if torch.is_tensor(picks) else picks)}
                ),
                "anchors": sorted(set(getattr(sel, "last_dest", {}).values())),
            }
        )
        n_pad = 1 if self._sp_size() > 1 else 0
        gated = GatedCache.assemble(store, store.layers(), parts, get_device(), n_pad)
        self._ptr_sel = sel
        self._ptr_memo = (id(cache), c, gated)
        return gated

    def _ptr_step_salt(self):
        return getattr(self, "_ptr_salt", 0)


class _Arm:
    def __init__(self, meta, selector, far_budget_chunks, explore):
        self.meta, self.selector = (meta, selector)
        self.budget, self.explore = (far_budget_chunks, explore)

    def __enter__(self):
        m = self.meta
        self._saved = (m._ptr_selector, m._ptr_budget_chunks, m._ptr_explore)
        m._ptr_selector = self.selector
        m._ptr_budget_chunks = self.budget
        m._ptr_explore = self.explore
        m._ptr_cache_id, m._ptr_sel, m._ptr_memo = (None, None, None)
        return m

    def __exit__(self, *exc):
        self.meta._ptr_selector, self.meta._ptr_budget_chunks, self.meta._ptr_explore = self._saved
        return False


__all__ = ["PtrRolloutMixin", "ROLLOUT_MODE"]
