"""Bind validation noise to the scene rather than its line number."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re

_TABLE: dict[str, int] | None = None
_IDX: dict[int, int] | None = None
_MISSES: set = set()
_LOGGED: set = set()


def key_of(prompt: str) -> str:
    """Stable identity of a clip: its prompt with the padding stripped."""
    return hashlib.sha1(re.sub("(\\s*\\.)+\\s*$", "", prompt).strip().encode()).hexdigest()[:16]


def _load():
    global _TABLE, _IDX
    if _TABLE is not None:
        return
    t = os.environ.get("PTR_SEED_TABLE", "")
    p = os.environ.get("PTR_EVAL_PROMPTS") or os.environ.get("PTR_SEED_PROMPTS", "")
    _TABLE, _IDX = ({}, {})
    if not t or not os.path.exists(t):
        return
    _TABLE = {k: int(v) for k, v in json.loads(pathlib.Path(t).read_text()).items()}
    if p and os.path.exists(p):
        lines = [l for l in pathlib.Path(p).read_text().splitlines() if l.strip()]
        _IDX = {i: _TABLE[key_of(l)] for i, l in enumerate(lines) if key_of(l) in _TABLE}


def install(base_module, logger=None) -> int:
    """Rebind base_module.combine_seed. Returns how many prompts got a pinned seed."""
    _load()
    if not _IDX:
        return 0
    orig = base_module.combine_seed

    def wrapper(*args):
        if len(args) == 4 and args[1] == "validation":
            s = _IDX.get(int(args[2]))
            if s is not None:
                if logger and int(args[2]) not in _LOGGED:
                    _LOGGED.add(int(args[2]))
                    logger.info("[seed] prompt idx %d -> %d", int(args[2]), s)
                return s
            _MISSES.add(int(args[2]))
        return orig(*args)

    wrapper.__wrapped__ = orig
    base_module.combine_seed = wrapper
    if logger:
        logger.info("[seed] pinned %d prompts from %s", len(_IDX), os.environ.get("PTR_SEED_TABLE"))
    return len(_IDX)


def misses():
    return sorted(_MISSES)


__all__ = ["install", "key_of", "misses"]
