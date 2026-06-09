"""Interactive progress feedback for embedding-model loading.

Loading an embedding model for the first time downloads weights and prints a
lot of HuggingFace chatter (download bars, an ``HF_TOKEN`` notice, a model
load report). In an interactive session that wall of text is replaced here by a
single spinner; the underlying noise is suppressed only while the spinner is
shown.

The feedback is shown only when the session is interactive (a real terminal or
a notebook), so scripts, pipes, and CI logs stay clean. Set the environment
variable ``EMB_DIVERSITY_PROGRESS`` to ``1``/``0`` (or ``true``/``false``) to
force it on or off.

Display problems never block model loading: if anything in the spinner machinery
fails, the model is loaded exactly as it would be without it.
"""

from __future__ import annotations

import contextlib
import logging
import os
import warnings
from typing import Callable, TypeVar

T = TypeVar("T")

_ENV_VAR = "EMB_DIVERSITY_PROGRESS"
_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _env_override() -> bool | None:
    """Return the forced on/off state from the environment, or None if unset."""
    raw = os.environ.get(_ENV_VAR)
    if raw is None:
        return None
    value = raw.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    return None


def _is_interactive() -> bool:
    """True when stderr is a terminal or the session is a notebook."""
    try:
        from rich.console import Console

        console = Console(stderr=True)
        return console.is_terminal or console.is_jupyter
    except Exception:
        return False


def progress_enabled() -> bool:
    """Whether to show the loading spinner in the current session."""
    override = _env_override()
    if override is not None:
        return override
    return _is_interactive()


@contextlib.contextmanager
def _quiet_huggingface():
    """Suppress HuggingFace download bars, the token notice, and the model load
    report for the duration of the block, restoring prior state afterwards."""
    http_logger = logging.getLogger("huggingface_hub.utils._http")
    prev_http_level = http_logger.level

    disabled_bars_here = False
    prev_transformers_verbosity = None

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")
        warnings.filterwarnings("ignore", message=".*unauthenticated.*")
        http_logger.setLevel(logging.ERROR)

        try:
            from huggingface_hub.utils import (
                are_progress_bars_disabled,
                disable_progress_bars,
            )

            if not are_progress_bars_disabled():
                disable_progress_bars()
                disabled_bars_here = True
        except Exception:
            pass

        try:
            from transformers.utils import logging as transformers_logging

            prev_transformers_verbosity = transformers_logging.get_verbosity()
            transformers_logging.set_verbosity_error()
        except Exception:
            pass

        try:
            yield
        finally:
            http_logger.setLevel(prev_http_level)
            if disabled_bars_here:
                try:
                    from huggingface_hub.utils import enable_progress_bars

                    enable_progress_bars()
                except Exception:
                    pass
            if prev_transformers_verbosity is not None:
                try:
                    from transformers.utils import logging as transformers_logging

                    transformers_logging.set_verbosity(prev_transformers_verbosity)
                except Exception:
                    pass


def load_with_spinner(model_name: str, load_fn: Callable[[], T]) -> T:
    """Run *load_fn* while showing a spinner, in interactive sessions only.

    Args:
        model_name: Model id, shown in the spinner text.
        load_fn: Zero-argument callable that downloads and/or loads the model.

    Returns:
        Whatever *load_fn* returns.
    """
    if not progress_enabled():
        return load_fn()

    # Set up the display defensively; if it cannot be created, load plainly so a
    # cosmetic failure never blocks real work. Errors raised by load_fn itself
    # are allowed to propagate.
    try:
        from rich.console import Console

        console = Console(stderr=True)
        status = console.status(
            f"[bold cyan]Downloading & preparing model[/] '{model_name}' "
            f"[dim](first run may take a while)[/]",
            spinner="dots",
        )
    except Exception:
        return load_fn()

    with _quiet_huggingface(), status:
        model = load_fn()

    try:
        console.print(f"[green]✓[/] Model '{model_name}' ready")
    except Exception:
        pass
    return model
