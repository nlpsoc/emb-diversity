"""Global cache root resolution. Not part of the public API."""

import os
from pathlib import Path

_ENV_VAR = "EMB_DIVERSITY_CACHE"


def _resolve_cache_root() -> Path:
    env = os.environ.get(_ENV_VAR)
    if env:
        return Path(env).expanduser().resolve()
    return Path.home() / ".cache" / "emb-diversity"


CACHE_ROOT: Path = _resolve_cache_root()
