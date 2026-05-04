from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON experiment config."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def require_int(config: dict[str, Any], key: str) -> int:
    value = config.get(key)
    if not isinstance(value, int):
        raise ValueError(f"Config key {key!r} must be an integer, got {value!r}")
    return value
