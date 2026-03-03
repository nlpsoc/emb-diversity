.. Diversity Measurement documentation master file, created by
   sphinx-quickstart on Mon Dec  8 14:29:05 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Diversity Measurement Documentation
====================================

A minimal Python package with a simple diversity function for measuring diversity in vector representations of data.

Installation
------------

To use the library directly, follow these steps:

1. Clone the repository
2. Install dependencies using ``uv sync --no-group dev`` or ``uv sync --group dev`` for development mode
3. Activate the environment using ``source .venv/bin/activate``

Usage
-----

Basic example:

.. code-block:: python

   from measure_diversity import dummy_diversity

   result = dummy_diversity([[0, 1], [0, 0]])
   print(result)  # Output: 0.5

API Reference
-------------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   measure_diversity

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
