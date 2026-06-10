"""Tests for the package's lazy public attributes.

``import emb_diversity`` must stay cheap: heavy dependencies (torch,
scikit-learn, umap, …) load only when a measure or the embedding pipeline is
actually used. Import-time behaviour is checked in a fresh interpreter so the
test is independent of what the current process already imported.
"""

import subprocess
import sys

import pytest

import emb_diversity
from emb_diversity.measures_registry import MEASURE_NAMES, get_measure


class TestLazyImport:

    def test_import_does_not_load_heavy_dependencies(self):
        """A fresh ``import emb_diversity`` pulls in none of the heavy deps."""
        code = (
            "import sys\n"
            "import emb_diversity\n"
            "heavy = ['torch', 'sklearn', 'umap', 'scipy', 'pandas',\n"
            "         'transformers', 'sentence_transformers', 'numba']\n"
            "loaded = [m for m in heavy if m in sys.modules]\n"
            "assert not loaded, f'heavy modules loaded at import: {loaded}'\n"
        )
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_measure_attribute_resolves_to_callable(self):
        """Accessing a measure on the package returns the measure function."""
        from emb_diversity.measures.mean_pw_dist import mean_pw_dist

        assert emb_diversity.mean_pw_dist is mean_pw_dist

    def test_unknown_attribute_raises_attribute_error(self):
        """An unknown name raises AttributeError, like a regular module."""
        with pytest.raises(AttributeError, match="no_such_attr"):
            emb_diversity.no_such_attr

    def test_dir_lists_all_public_names(self):
        """``dir()`` includes the lazy names without importing them."""
        listing = dir(emb_diversity)
        for name in ("measure_diversity", "embed_texts", "axes", *MEASURE_NAMES):
            assert name in listing

    def test_all_names_are_resolvable(self):
        """Every name in ``__all__`` resolves via attribute access."""
        for name in emb_diversity.__all__:
            assert getattr(emb_diversity, name) is not None

    def test_get_measure_unknown_name_raises_key_error(self):
        """get_measure rejects unregistered names with a helpful KeyError."""
        with pytest.raises(KeyError, match="not_a_measure"):
            get_measure("not_a_measure")
