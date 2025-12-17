# Diversity Measurement Package

A minimal Python package for measuring data diversity. Although the package focuses on text (NLP) datasets, many  of its metrics are broadly applicable and can be applied to other types of data as well.
This library is developed as part of the [DataDivers](https://datadivers-erc.github.io/) project.

## Install
> [!NOTE]
> You must have **uv** installed before running `uv sync`.
> Full installation guide: <https://docs.astral.sh/uv/getting-started/installation/>

After installing `uv` on your system, you can now follow either **development mode** or **standard installation mode** depending on your use case.

### Development mode

Follow these steps to set up the project for development.
- Clone the repo
- Install all dependencies required for development mode: mode
   ```bash
   uv sync --group dev
   ```
- Activate the Python environment created by `uv`
   ```bash
   source .venv/bin/activate
   ```

### standard installation

To use the library directly do the following,

- Clone the repo
- Install all dependencies required for standard mode
   ```bash
   uv sync --no-group dev
   ```
-  Activate the Python environment created by `uv`
   ```bash
   source .venv/bin/activate
   ```
There are few more items you need to be aware of uv before working on it. Please
[Jump to the working with uv section](#working-with-uv)


## Usage

```python
from measure_diversity import dummy_diversity

result = dummy_diversity([[0, 1], [0, 0]])
print(result)  # Output: 5
```

## Suggested Workflow for Collaboration

1. **Create a new branch** for your feature or bug fix:
   ```bash
   git checkout -b feature/my-feature
   ```
   
2. **Make your changes** in the codebase.
3. **Run tests** to ensure everything works as expected:
   ```bash
   pytest
   ```
4. **Commit your changes** with a descriptive message:
   ```bash
    git add .
    git commit -m "Add feature X"
    ```
5. **Push your branch** to the remote repository:
6. ```bash
   git push origin feature/my-feature
   ```
7. **Create a pull request** on GitHub to merge your changes into the main branch
   and request a review from your team members.
8. **Address any feedback** from the review process.
9. Once approved, **merge your pull request** into the main branch.
10. **Delete your branch** after merging to keep the repository clean:
    ```bash
    git branch -d feature/my-feature
    git push origin --delete feature/my-feature
    ```
    

## Working with uv

### Adding Packages with `uv add`

To add packages to your project, always use `uv add` rather than `uv pip install`. This ensures that your dependencies are properly managed and recorded in your `pyproject.toml`. For example:

```bash
uv add <package-name>
```

### Adding Packages to a Dev Group

If you need to add a package specifically to your development environment, you can add it to the `dev` group like this:

```bash
uv add --group dev <package-name>
```

### Switching Between Dev and Standard Mode

After you are done with the testing and want to go back to standard mode, you can do this so that uv removes packages utilized in the development mode

```bash
uv sync --no-group dev
```

This will disable all additional groups and just load your main project dependencies.

### Best Practice: Run `uv lock -U`

Whenever you upgrade, downgrade, or change versions of packages, it’s a good practice to run:

```bash
uv lock -U
```

This updates your pyproject.toml file to ensure all versions are consistent and everything is in sync.

# Funding
This work is supported by the ERC Starting Grant **DataDivers** (101162980).
