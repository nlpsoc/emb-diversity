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


def test_measures_registry_survives_lazy_measure_import():
    """Accessing a measure must not replace `measures` with the subpackage."""
    import emb_diversity

    _ = emb_diversity.vendi_score  # triggers a lazy measure-module import
    assert isinstance(emb_diversity.measures, Registry)
