"""Text detection and the ``@accepts_text`` decorator."""

from __future__ import annotations

import functools
import inspect
from typing import Sequence

from .embed import embed_texts


# ── Input detection ──────────────────────────────────────────────────


def _is_text_input(data: object) -> bool:
    """Return True if *data* looks like a list of strings."""
    if not isinstance(data, (list, tuple)):
        return False
    if len(data) == 0:
        return False
    return isinstance(data[0], str)


# ── Decorator ────────────────────────────────────────────────────────


def accepts_text(fn):
    """Wrap a measure function so it can receive raw text.

    The decorated function gains two optional keyword arguments:

    * ``diversity_axis``  – registered axis name (default ``"semantic"``)
    * ``embedding_model`` – explicit model id (overrides axis)

    When the first positional argument (*data*) is a list of strings the
    decorator embeds them before calling the original measure.  When *data*
    is already numeric the decorator passes it through unchanged.
    """

    @functools.wraps(fn)
    def wrapper(data, *args, diversity_axis="semantic", embedding_model=None, **kwargs):
        if _is_text_input(data):
            data = embed_texts(
                data,
                diversity_axis=diversity_axis,
                embedding_model=embedding_model,
            )
        return fn(data, *args, **kwargs)

    # Preserve the original signature but add the new params for docs/IDE
    _patch_signature(wrapper, fn)
    return wrapper


def _patch_signature(wrapper, fn):
    """Add diversity_axis / embedding_model to the wrapper's signature."""
    sig = inspect.signature(fn)
    extra = [
        inspect.Parameter(
            "diversity_axis",
            inspect.Parameter.KEYWORD_ONLY,
            default="semantic",
        ),
        inspect.Parameter(
            "embedding_model",
            inspect.Parameter.KEYWORD_ONLY,
            default=None,
        ),
    ]
    params = list(sig.parameters.values())
    # Insert before **kwargs if present, else append
    kw_var = [p for p in params if p.kind == inspect.Parameter.VAR_KEYWORD]
    if kw_var:
        idx = params.index(kw_var[0])
        params = params[:idx] + extra + params[idx:]
    else:
        params = params + extra
    wrapper.__signature__ = sig.replace(parameters=params)
