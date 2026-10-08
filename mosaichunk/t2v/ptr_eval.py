"""Evaluate the same T2V memory router used during training."""

from __future__ import annotations

import logging

from .ptr_rollout import PtrRolloutMixin
from .seg_meta import SegmentedPromptMetaModel

_LOG = logging.getLogger(__name__)


class PtrEvalMetaModel(PtrRolloutMixin, SegmentedPromptMetaModel):
    """Segmented prompting + the ptr memory. Validation only; nothing is trained."""


__all__ = ["PtrEvalMetaModel"]
