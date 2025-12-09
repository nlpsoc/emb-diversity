# Documentation

This folder contains the Sphinx documentation for the Diversity Measurement package.

## Building the Documentation

### Prerequisites

Make sure you have installed the development dependencies:

```bash
uv sync --group dev
```

### Building HTML Documentation

To build the HTML documentation, follow these two steps:

```bash
cd docs

# Step 1: Generate API documentation RST files from source code
make apidoc

# Step 2: Build HTML from the RST files
make html
```

The generated HTML files will be in `docs/build/html/`. Open `docs/build/html/index.html` in your browser to view the documentation.

### Available Make Commands

- `make apidoc` - Generate API documentation RST files from source code (discovers all modules including utility, eval, etc.)
- `make html` - Build HTML documentation from RST files
- `make clean` - Remove built documentation files
- `make cleanall` - Remove both built documentation AND generated API RST files

## Workflow: Documentation Generation from Code

This project uses `sphinx-apidoc` to generate documentation from source code. You should NOT manually edit `modules.rst` or any generated `measure_diversity*.rst` files - they are auto-generated.

### How It Works

1. **Run `make apidoc`** to scan your source code in `../src/measure_diversity/`
2. `sphinx-apidoc` discovers all Python modules and submodules (measure, two_d, utility, eval, embeddings, etc.)
3. It generates RST files with a **flat structure** using the `--no-headings` flag
   - All modules appear as `measure_diversity.module_name`
   - No hierarchical "Submodules" or "Subpackages" sections
4. **Run `make html`** to build the HTML documentation
5. Sphinx reads the RST files and extracts docstrings from your Python code

### When to Run Each Command

- **Run `make apidoc`** when:
  - You add new Python modules or packages
  - You restructure your code (move/rename modules)
  - Starting fresh or the RST files are missing

- **Run `make html`** when:
  - You update docstrings in existing code
  - After running `make apidoc`
  - You modify `source/index.rst` or other manual RST files

### Why This Approach?

- **No manual maintenance** - New modules are automatically discovered
- **Always accurate** - Documentation reflects actual code structure
- **Clean structure** - Flat module listing without confusing hierarchies
- **Explicit control** - Separate `apidoc` and `html` steps
- **DRY principle** - Documentation lives in code as docstrings

## Documentation Structure

- `source/conf.py` - Sphinx configuration file
- `source/index.rst` - Main documentation page (manually maintained)
- `source/modules.rst` - Auto-generated API reference (DO NOT EDIT MANUALLY)
- `source/measure_diversity*.rst` - Auto-generated module docs (DO NOT EDIT MANUALLY)
- `build/` - Generated documentation (gitignored)

## Viewing the Documentation

After building, you can view the documentation by opening:

```
docs/build/html/index.html
```

in your web browser.
