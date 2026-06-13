"""Diversity axes registry.

A diversity axis maps a concept (e.g. "semantic", "style") to a default
embedding model and optional alternatives.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .utility._registry import Registry


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


# Module-level registry instance
axes = Registry()

# ── Built-in axes ────────────────────────────────────────────────────

axes.register(
    "semantic",
    DiversityAxis(
        name="semantic",
        default_model="all-mpnet-base-v2",
        alternative_models=["all-MiniLM-L6-v2"],
        description="Meaning-based diversity using semantic similarity",
    ),
)

axes.register(
    "style",
    DiversityAxis(
        name="style",
        default_model="AnnaWegmann/Style-Embedding",
        alternative_models=["StyleDistance/styledistance", "rrivera1849/LUAR-MUD", "AIDA-UPM/star"],
        description="Writing style diversity",
    ),
)
