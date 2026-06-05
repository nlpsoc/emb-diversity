# CLAUDE.md

Guidance for AI agents and contributors working in this repository.

## Project

**emb-diversity** — embedding-based diversity measures for text and vector data.
Library code lives in `src/emb_diversity/`. User docs are in `docs/source/`
(Sphinx + MyST) and `README.md`. Tests are in `test/` (pytest).

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

1. Create `src/emb_diversity/measures/<name>.py`, decorated with `@accepts_text`
   and a Google-style docstring.
2. Register it in `src/emb_diversity/measures_registry.py`.
3. Export it from `src/emb_diversity/__init__.py` (the import **and** `__all__`).
4. Add a row in `docs/source/user-guide/measures.md`.

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
