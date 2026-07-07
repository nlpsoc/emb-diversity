# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
import tomllib
from pathlib import Path
sys.path.insert(0, os.path.abspath('../../src'))

# Mock imports for packages that might not be available during docs build
autodoc_mock_imports = [
    'numpy',
    'scipy',
    'sklearn',
    'sentence_transformers',
    'vendi_score',
    'matplotlib',
    'pandas',
    'transformers',
    'torch',
    'torchaudio',
    'speechbrain',
    'typer',
    'networkx',
    'umap',
]

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'emb-diversity'
copyright = '2026, Cantao Su, Menan Velayuthan, Esther Ploeger, Dong Nguyen, Anna Wegmann'
author = 'Cantao Su, Menan Velayuthan, Esther Ploeger, Dong Nguyen, Anna Wegmann'

# Read the version straight from pyproject.toml so the docs never drift from the
# package — and so it still works in the GitHub Pages build, where only the docs
# dependency group is installed (the package itself is not).
_pyproject = Path(__file__).resolve().parents[2] / 'pyproject.toml'
release = tomllib.loads(_pyproject.read_text(encoding='utf-8'))['project']['version']
version = release

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx_autodoc_typehints',
    'myst_parser',
]

templates_path = ['_templates']
exclude_patterns = []

language = 'en'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = ['custom.css']
