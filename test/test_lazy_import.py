"""The package imports cheaply, without pulling in heavy optional dependencies."""

import subprocess
import sys

from emb_diversity._registry import Registry


def test_heavy_deps_not_imported_at_package_import():
    """`import emb_diversity` must not import torch/sklearn/umap/transformers.

    Run in a fresh interpreter so the check is not polluted by modules other
    tests have already imported.
    """
    code = (
        "import sys, emb_diversity; "
        "heavy = ('torch', 'sklearn', 'umap', 'transformers', "
        "'sentence_transformers', 'pandas'); "
        "print([m for m in heavy if m in sys.modules])"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]", f"heavy imports leaked: {result.stdout}"


def test_registry_and_measures_subpackage_do_not_collide():
    """The registry and the `measures` code subpackage have distinct names.

    The registry is ``measure_registry``; ``emb_diversity.measures`` is the
    subpackage. Accessing a measure imports that subpackage, which must remain a
    plain module (not get confused with the registry).
    """
    import types

    import emb_diversity
    from emb_diversity.measures_registry import measure_registry

    assert isinstance(measure_registry, Registry)
    assert callable(emb_diversity.vendi_score)  # lazy top-level access works
    assert isinstance(emb_diversity.measures, types.ModuleType)
