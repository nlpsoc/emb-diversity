"""Decorator that lets measure functions accept raw text.

When a measure function is decorated with ``@accepts_text``, it gains
the ability to receive a list of strings instead of embeddings.  The
decorator detects text input, embeds it via :func:`embediver.embed.embed_texts`,
and passes the resulting vectors to the original measure function.

Two keyword arguments are added to the decorated function:

- ``diversity_axis`` (default ``"semantic"``): which axis to use for embedding
- ``embedding_model``: explicit model name, overrides the axis

If the input is already numeric (e.g. a numpy array), the decorator
passes it through unchanged.
"""

from __future__ import annotations

import functools
import inspect
from typing import Sequence

from .embed import embed_texts


# ── Input detection ──────────────────────────────────────────────────


def _is_text_input(data):
    """Return True if *data* looks like a list of strings."""
    return len(data) > 0 and isinstance(data[0], str)


# ── Decorator ────────────────────────────────────────────────────────


def accepts_text(func):
    """Wrap a measure function so it can receive raw text.

    The decorated function gains two optional keyword arguments:

    * ``diversity_axis``  – registered axis name (default ``"semantic"``)
    * ``embedding_model`` – explicit model id (overrides axis)

    When the first positional argument (*data*) is a list of strings the
    decorator embeds them before calling the original measure.  When *data*
    is already numeric the decorator passes it through unchanged.
    """

    @functools.wraps(func)
    def wrapper(data, *args, diversity_axis="semantic", embedding_model=None, **kwargs):
        if _is_text_input(data):
            data = embed_texts(
                data,
                diversity_axis=diversity_axis,
                embedding_model=embedding_model,
            )
        return func(data, *args, **kwargs)

    # Preserve the original signature but add the new params for docs/IDE
    _patch_signature(wrapper, func)
    return wrapper


def _patch_signature(wrapper, func):
    """Add diversity_axis / embedding_model to the wrapper's signature."""
    sig = inspect.signature(func)
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
