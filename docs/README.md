# Documentation

This folder contains the Sphinx documentation for the Diversity Measurement package.

## Building the Documentation

### Prerequisites

Make sure you have installed the development dependencies:

```bash
uv sync --group dev
```

### Building HTML Documentation

To build the HTML documentation, run:

```bash
cd docs
make html
```

The generated HTML files will be in `docs/build/html/`. Open `docs/build/html/index.html` in your browser to view the documentation.

### Cleaning Build Files

To remove the generated documentation files:

```bash
cd docs
make clean
```

## Documentation Structure

- `source/conf.py` - Sphinx configuration file
- `source/index.rst` - Main documentation page
- `source/modules.rst` - API reference for all modules
- `build/` - Generated documentation (gitignored)

## Updating Documentation

The documentation is automatically generated from docstrings in the source code. To update the documentation:

1. Update the docstrings in the source code
2. Rebuild the documentation using `make html`

## Viewing the Documentation

After building, you can view the documentation by opening:

```
docs/build/html/index.html
```

in your web browser.
