# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))

# Mock imports for packages that might not be available during docs build
autodoc_mock_imports = [
    'numpy',
    'scipy',
    'sklearn',
    'sentence_transformers',
    'vendi_score',
    'datasets',
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

project = 'embediver'
copyright = '2026, Anna Wegmann, Cantao Su, Menan Velayuthan, Dong Nguyen, Esther Ploeger'
author = 'Anna Wegmann, Cantao Su, Menan Velayuthan, Dong Nguyen, Esther Ploeger'

version = '0.1.0'
release = '0.1.0'

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
