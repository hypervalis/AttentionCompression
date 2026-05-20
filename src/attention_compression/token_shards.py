from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class TokenShard:
    shard_id: int
    path: Path


def discover_token_shards(tokens_dir: str | Path) -> list[TokenShard]:
    """Return token shards in deterministic path order."""
    root = Path(tokens_dir)
    paths = sorted(root.glob("*.npy"))
    if not paths:
        raise FileNotFoundError(f"No .npy token shards found under {root}")
    return [TokenShard(shard_id=i, path=p) for i, p in enumerate(paths)]


def load_tokens(path: str | Path, *, mmap: bool = True) -> np.ndarray:
    """Load a flat token-id shard."""
    mmap_mode = "r" if mmap else None
    arr = np.load(path, mmap_mode=mmap_mode)
    if arr.ndim != 1:
        raise ValueError(f"Token shard must be flat/1-D: {path} has shape {arr.shape}")
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"Token shard must contain integer IDs: {path} has dtype {arr.dtype}")
    return arr


def iter_token_shards(tokens_dir: str | Path, *, mmap: bool = True) -> Iterable[tuple[TokenShard, np.ndarray]]:
    for shard in discover_token_shards(tokens_dir):
        yield shard, load_tokens(shard.path, mmap=mmap)
