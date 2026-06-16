# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from importlib.metadata import PackageNotFoundError, version as _get_version
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
    'typer',
    'networkx',
    'umap',
]

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'emb-diversity'
copyright = '2026, Cantao Su, Menan Velayuthan, Esther Ploeger, Dong Nguyen, Anna Wegmann'
author = 'Cantao Su, Menan Velayuthan, Esther Ploeger, Dong Nguyen, Anna Wegmann'

# Read the version from the installed package metadata so the docs never drift
# from pyproject.toml.
try:
    release = _get_version('emb-diversity')
except PackageNotFoundError:
    release = '0.0.0'
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
