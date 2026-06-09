"""Generic registry pattern for O(1) key-value lookup.

Based on https://dev.to/dentedlogic/stop-writing-giant-if-else-chains-master-the-python-registry-pattern-ldm
"""

from __future__ import annotations

from typing import Callable


class _Lazy:
    """A registry value produced by *loader* the first time it is looked up."""

    __slots__ = ("loader",)

    def __init__(self, loader: Callable[[], object]):
        self.loader = loader


class Registry:
    """A simple key-value store with register, get, and list operations.

    Used by both the axes registry and the measures registry to avoid
    repeated dictionary boilerplate. Values may be registered eagerly or
    lazily; a lazy value is produced (and cached) on first lookup.
    """

    def __init__(self):
        self._store: dict[str, object] = {}

    def register(self, key: str, value: object) -> None:
        """Add an entry. Overwrites if key already exists."""
        self._store[key] = value

    def register_lazy(self, key: str, loader: Callable[[], object]) -> None:
        """Add an entry whose value is produced by *loader* on first get."""
        self._store[key] = _Lazy(loader)

    def get(self, key: str) -> object:
        """Look up by key, resolving (and caching) a lazy value on first use.

        Raises:
            KeyError: If the key has not been registered.
        """
        if key not in self._store:
            registered = ", ".join(sorted(self._store)) or "(none)"
            raise KeyError(f"Unknown key {key!r}. Registered: {registered}")
        value = self._store[key]
        if isinstance(value, _Lazy):
            value = value.loader()
            self._store[key] = value  # cache the resolved value
        return value

    def list_all(self) -> list[object]:
        """Return all values sorted by key (resolves any lazy entries)."""
        return [self.get(k) for k in sorted(self._store)]

    def keys(self) -> list[str]:
        """Return all registered keys."""
        return list(self._store.keys())

    def __contains__(self, key: str) -> bool:
        return key in self._store

    def __len__(self) -> int:
        return len(self._store)

    def __getitem__(self, key: str) -> object:
        return self.get(key)

    def __iter__(self):
        return iter(self._store)
