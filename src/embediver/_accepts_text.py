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
    """Fix the wrapper's signature so help() and IDEs show useful parameter names.

    This is purely cosmetic — it only affects what help() and IDEs display,
    not how the function actually runs.

    Example for mean_pw_dist:
        Before: mean_pw_dist(data, *args, diversity_axis="semantic", embedding_model=None, **kwargs)
        After:  mean_pw_dist(data, metric="cosine", diversity_axis="semantic", embedding_model=None, **metric_kwargs)
    """
    # The two parameters that the decorator adds
    diversity_axis_param = inspect.Parameter(
        "diversity_axis", inspect.Parameter.KEYWORD_ONLY, default="semantic",
    )
    embedding_model_param = inspect.Parameter(
        "embedding_model", inspect.Parameter.KEYWORD_ONLY, default=None,
    )
    new_params = [diversity_axis_param, embedding_model_param]

    # Read the original function's parameters
    # e.g. for mean_pw_dist: [data, metric="cosine", **metric_kwargs]
    original_sig = inspect.signature(func)
    original_params = list(original_sig.parameters.values())

    # Find the **kwargs parameter if it exists (e.g. **metric_kwargs)
    kwargs_params = [p for p in original_params if p.kind == inspect.Parameter.VAR_KEYWORD]
    has_kwargs = len(kwargs_params) > 0

    # Build the combined parameter list:
    #   original params + new params + **kwargs at the end
    # We insert before **kwargs so the signature reads naturally:
    #   (data, metric, diversity_axis, embedding_model, **metric_kwargs)
    if has_kwargs:
        kwargs_position = original_params.index(kwargs_params[0])
        params_before_kwargs = original_params[:kwargs_position]
        combined = params_before_kwargs + new_params + [kwargs_params[0]]
    else:
        combined = original_params + new_params

    wrapper.__signature__ = original_sig.replace(parameters=combined)
