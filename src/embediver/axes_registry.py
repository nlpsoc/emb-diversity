"""Diversity axes registry.

A diversity axis maps a concept (e.g. "semantic", "style") to a default
embedding model and optional alternatives.  Users and developers can
register custom axes via the module-level functions.

Uses the Registry pattern (see https://dev.to/dentedlogic/stop-writing-giant-if-else-chains-master-the-python-registry-pattern-ldm).
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


class Registry:
    """Generic key-value registry with O(1) lookup.

    Provides register, get, list, and contains operations on a
    dictionary-backed store.
    """

    def __init__(self):
        self._store: dict[str, DiversityAxis] = {}

    def register(self, key: str, value: DiversityAxis) -> None:
        """Add an entry. Overwrites if key already exists."""
        self._store[key] = value

    def get(self, key: str) -> DiversityAxis:
        """Look up by key. Raises KeyError if not found."""
        if key not in self._store:
            registered = ", ".join(sorted(self._store)) or "(none)"
            raise KeyError(
                f"Unknown diversity axis {key!r}. Registered axes: {registered}"
            )
        return self._store[key]

    def list_all(self) -> list[DiversityAxis]:
        """Return all entries sorted by key."""
        return [self._store[k] for k in sorted(self._store)]

    def __contains__(self, key: str) -> bool:
        return key in self._store


# ── Module-level registry instance ───────────────────────────────

_registry = Registry()


# ── Public API (convenience functions that delegate to the registry) ──

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
    _registry.register(
        name,
        DiversityAxis(
            name=name,
            default_model=default_model,
            alternative_models=alternative_models or [],
            description=description,
        ),
    )


def get_axis(name: str) -> DiversityAxis:
    """Look up an axis by name.

    Raises:
        KeyError: If the axis has not been registered.
    """
    return _registry.get(name)


def list_axes() -> list[DiversityAxis]:
    """Return all registered axes (sorted by name)."""
    return _registry.list_all()


# ── Built-in axes ────────────────────────────────────────────────

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
