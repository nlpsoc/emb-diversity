"""Diversity axes registry.

A diversity axis maps a concept (e.g. "semantic", "style") to a default
embedding model and optional alternatives.  Users and developers can
register custom axes via :func:`register_axis`.

Uses the registry pattern for O(1) lookup
(see https://dev.to/dentedlogic/stop-writing-giant-if-else-chains-master-the-python-registry-pattern-ldm).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DiversityAxis:
    """Configuration for a diversity axis.

    Attributes:
        name: Short identifier (e.g. ``"semantic"``).
        default_model: HuggingFace model id used by default for this axis.
        alternative_models: Other models that work well for this axis.
        description: Human-readable explanation shown in docs and CLI.
    """

    name: str
    default_model: str
    alternative_models: list[str] = field(default_factory=list)
    description: str = ""


# Registry: dict mapping axis name -> DiversityAxis
_axes: dict[str, DiversityAxis] = {}


def register_axis(
    name: str,
    default_model: str,
    alternative_models: list[str] | None = None,
    description: str = "",
) -> None:
    """Register a new diversity axis (or overwrite an existing one).

    Args:
        name: Short identifier for the axis.
        default_model: HuggingFace model id to use by default.
        alternative_models: Other models that work for this axis.
        description: One-line explanation.

    Example:
        >>> from embediver import register_axis
        >>> register_axis(
        ...     "multilingual",
        ...     default_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        ...     description="Cross-lingual semantic diversity",
        ... )
    """
    _axes[name] = DiversityAxis(
        name=name,
        default_model=default_model,
        alternative_models=alternative_models or [],
        description=description,
    )


def get_axis(name: str) -> DiversityAxis:
    """Look up an axis by name.

    Raises:
        KeyError: If the axis has not been registered.
    """
    if name not in _axes:
        registered = ", ".join(sorted(_axes)) or "(none)"
        raise KeyError(
            f"Unknown diversity axis {name!r}. Registered axes: {registered}"
        )
    return _axes[name]


def list_axes() -> list[DiversityAxis]:
    """Return all registered axes (sorted by name)."""
    return [_axes[k] for k in sorted(_axes)]


# ── Built-in axes ────────────────────────────────────────────────────

register_axis(
    "semantic",
    default_model="all-mpnet-base-v2",
    alternative_models=["all-MiniLM-L6-v2"],
    description="Meaning-based diversity using semantic similarity",
)

register_axis(
    "style",
    default_model="AnnaWegmann/Style-Embedding",
    alternative_models=["rrivera1849/LUAR-MUD"],
    description="Writing style diversity",
)
