# CLAUDE.md

Guidance for AI agents and contributors working in this repository.

## Project

**emb-diversity** — embedding-based diversity measures for text and vector data.
Library code lives in `src/emb_diversity/`. User docs are in `docs/source/`
(Sphinx + MyST) and `README.md`. Tests are in `test/` (pytest).

## Keep docs and comments general

Write code comments and documentation to describe how the code works **now** —
not how it changed. Do not reference change history, migrations, or specific
pull requests (avoid phrasings like "an earlier version used X", "previously",
"we removed Y", "this used to be Z"). Such notes go stale and confuse future
readers. State the current behaviour and, where useful, the rationale for the
current design — without contrasting it against past implementations.

## Pull requests: leave the branch free for review

When you open a PR, **leave its branch free for the user to check out** — do not
leave the PR branch occupied by a long-lived worktree. Git refuses
`git checkout <branch>` while that branch is checked out anywhere else (e.g. a
`.claude/worktrees/…` worktree), which blocks the user from reviewing or pulling
it into their main checkout.

After pushing the PR branch, prefer to switch the working worktree off it (e.g.
back to a scratch/detached state) so the branch name is unoccupied. At minimum,
tell the user the branch is held by a worktree and give them the unblock path —
review on GitHub, `cd` into the worktree, or
`git checkout -b <newname> origin/<branch>`.

## ⚠️ Keep the measure sets and the default measure in sync

The default measure and the named measure sets are defined in **one** place —
`src/emb_diversity/measures_registry.py` (`DEFAULT_MEASURE` and `MEASURE_SETS`).

The *runtime logic is generic* and reads those at runtime, so it needs **no**
change when you edit a set:

- `convenience._resolve_measure_names` resolves `measure="<set>"`,
- the CLI `-m <set>` shortcut, and
- `list-measures` tagging

all derive from `MEASURE_SETS` / `DEFAULT_MEASURE` automatically.

**But the human-facing text is hardcoded.** Whenever you add, rename, or change a
set — or change `DEFAULT_MEASURE` — you MUST also update every place below:

- [ ] `src/emb_diversity/cli.py` — the `-m` / `--measure` help string (it lists
      the set names and the default). ⚠️ It is duplicated in **two** commands.
- [ ] `src/emb_diversity/convenience.py` — the `measure_diversity` docstring
      (the `measure` options list and the `Example`).
- [ ] `docs/source/user-guide/cli.md` — the Options table (default value + the
      "Named sets" list) **and** the example commands.
- [ ] `README.md` — the Quickstart example.
- [ ] `docs/source/user-guide/vectors.md` — the example.
- [ ] `docs/source/user-guide/measures.md` — the `**default measure**` label
      (only if `DEFAULT_MEASURE` changed).
- [ ] `test/test_measure_sets.py` — the assertions pin the exact set contents and
      the default; update them to match.

Then verify:

```bash
uv run pytest test/test_measure_sets.py
uv run emb-diversity list-measures      # tags should reflect the new sets
```

`test/test_measure_sets.py` is the hard guard: it fails if any set references a
measure that is not registered, which catches typos.

## Adding a new measure

1. Create `src/emb_diversity/measures/<name>.py`. A measure is a plain function
   (no decorator) with this shape:

   ```python
   def <name>(
       data,
       <measure-specific params>,
       *,
       diversity_axis: str = "semantic",
       embedding_model: str | None = None,
   ) -> MeasureResult:
       data, embedding_model = resolve_embeddings(data, diversity_axis, embedding_model)
       ...
       return {
           "value": float(score),
           "parameters": {<measure-specific params>, "embedding_model": embedding_model},
       }
   ```

   `resolve_embeddings` (from `..embed`) embeds text input and returns the
   resolved model id (`None` for vector input); `MeasureResult` is from
   `._types`. Embedding args are keyword-only (after `*`). Add a Google-style
   docstring. The file and the function must share the measure's name — the
   registry and the package's lazy attribute access rely on that convention.
2. Add the name to `MEASURE_NAMES` in `src/emb_diversity/measures_registry.py`.
   The public API (`emb_diversity.<name>`), the CLI, and `measure_diversity`
   all derive from it.
3. Add the matching import to the `TYPE_CHECKING` block in
   `src/emb_diversity/__init__.py` (static-only, for IDEs and type checkers).
4. Add a row in `docs/source/user-guide/measures.md`.

## ⚠️ Keep `import emb_diversity` fast

Public names are resolved lazily (PEP 562 `__getattr__` in `__init__.py`), so
importing the package does not load the heavy dependencies (torch,
scikit-learn, scipy, umap, pandas, …). Don't break this: nothing that runs at
package import time (`__init__.py`, `measures_registry.py`, `axes_registry.py`,
`convenience.py`, `cli.py` at module level) may import a heavy dependency.
Inside a measure module or behind a function call is fine — that's the point.
`test/test_lazy_import.py` guards this in a fresh interpreter.

## Adding a new diversity axis

1. Register it in `src/emb_diversity/axes_registry.py`.
2. Document it in `docs/source/user-guide/axes.md`.

(The `README.md` "Development" section also covers these two workflows.)

## Writing tests

Match the existing style in `test/` (see `test_diversity.py` and
`test_measure_sets.py`) so the suite stays consistent:

- **Group related tests in a class** named `Test<Thing>`, with `test_*`
  methods that each have a one-line docstring describing the behaviour.
- **Construct inputs inline** — small Python lists, or
  `np.random.RandomState(seed)` for arrays (seed for reproducibility). A
  `@staticmethod` helper for shared fixtures is fine (e.g. `_toy`, `_vectors`).
- **Use `pytest.mark.parametrize`** when running the same assertion across many
  cases (e.g. every registered measure) — it reports each case as its own
  pass/fail and lets you re-run one with `pytest -k <case>`. See
  `test_convenience.py`. (It composes with the `Test…` class; put it on the
  method.)

## Build, test, lock

```bash
uv sync --group dev                   # set up the dev environment
uv run pytest -m "not integration"    # fast tests (skips model downloads / network)
uv build                              # build sdist + wheel
```

After changing dependencies in `pyproject.toml`, run `uv lock` and commit the
updated `uv.lock`.
