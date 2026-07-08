# Documentation

This folder holds the [Sphinx](https://www.sphinx-doc.org/) documentation for
emb-diversity. The rendered site is published at
<https://nlpsoc.github.io/emb-diversity/>.

## Publishing is automatic

You do **not** need to build or commit anything for the live site to update.
The [`docs.yml`](../.github/workflows/docs.yml) workflow ("Deploy Docs to
GitHub Pages") runs on every push to `main` and:

1. installs `uv` and Python, then `uv sync --only-group docs` (just the Sphinx
   toolchain — the `emb-diversity` package itself is **not** installed);
2. runs `make apidoc` then `make html` inside `docs/`;
3. uploads `docs/build/html/` and deploys it to GitHub Pages.

So merging a PR into `main` is what updates the published docs.

## Building locally (to preview before merging)

```bash
uv sync --group dev      # the dev group includes the docs toolchain
                         # (or: uv sync --only-group docs, exactly like CI)

cd docs
make apidoc              # regenerate the API-reference .rst from the source
make html                # build the HTML into build/html/

open build/html/index.html   # macOS — or open that file in any browser
```

### Make targets

- `make apidoc` — run `sphinx-apidoc` over `../src/emb_diversity/` and write the
  API-reference `.rst` files.
- `make html` — build the HTML site into `build/html/`.
- `make clean` — remove `build/`.
- `make cleanall` — remove `build/` **and** the generated API `.rst` files.

Run `make apidoc` whenever you add, rename, or move a module; run `make html`
after editing any docstring or page.

## What is hand-written vs generated

**Hand-written — committed, edit these:**

- `source/index.md` — the landing page and the `toctree` that wires the site
  together. It pulls the intro, install, quickstart, configuration, and citation
  text out of the **root [`README.md`](../README.md)** via MyST `{include}`
  markers (e.g. `docs-quickstart-start`/`-end`), so that content lives in one
  place and is not duplicated here.
- `source/user-guide/*.md` — the user guide pages (`measures`,
  `adding-a-measure`, `vectors`, `axes`, `cli`, `cache`), written in MyST
  Markdown.
- `source/development.md` — the contributor-facing Development guide (dev
  setup, workflow, adding measures/axes, docstring style, releasing).
- `source/conf.py` — the Sphinx configuration.
- `Makefile` — the `apidoc`/`html`/`clean` targets above.

**Generated — git-ignored, never edit or commit (see [`.gitignore`](../.gitignore)):**

- `source/emb_diversity*.rst` and `source/modules.rst` — the API reference,
  produced by `make apidoc` from the docstrings in `src/emb_diversity/`. The
  "API Reference" `toctree` in `index.md` points at the generated
  `emb_diversity` page. Because these are regenerated on every build (including
  in CI), there is no need to keep them in git.
- `build/` — the rendered HTML output.

## How the build fits together

`make apidoc` runs `sphinx-apidoc --no-headings` over `../src/emb_diversity/`,
producing one flat `emb_diversity.<subpackage>.rst` file per package plus a
`modules.rst` index.

`make html` then runs Sphinx, which reads `conf.py`, the Markdown pages, and the
generated `.rst`, and renders HTML with the `sphinx_rtd_theme`. A few details
worth knowing, all configured in `conf.py`:

- **Docstrings** are extracted by the `autodoc` extension and parsed in
  Google style by `napoleon` (with `sphinx_autodoc_typehints` for type hints
  and `myst_parser` so the Markdown pages are understood).
- **Heavy runtime dependencies** (`numpy`, `scipy`, `torch`, `sentence_transformers`,
  …) are listed in `autodoc_mock_imports`, so the build can import the package
  for its docstrings without those packages installed. `sys.path` is pointed at
  `../../src`, so `autodoc` imports `emb_diversity` straight from the source tree
  — the package does not need to be installed.
- **The version** shown in the docs is read directly from `pyproject.toml`, so it
  always matches the package (and works in the Pages build, where the package
  itself is not installed).

For docstring conventions (the Google-style layout the measures use), see the
**Docstring Style Guide** in the root [`README.md`](../README.md#docstring-style-guide).
