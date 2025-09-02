import os
from pathlib import Path


def find(marker_files=None):
    """Find project root by looking for marker files."""
    if marker_files is None:
        marker_files = ['.project_root', 'pyproject.toml', 'setup.py', '.git', 'requirements.txt']

    current_dir = Path(__file__).resolve().parent

    for parent in [current_dir] + list(current_dir.parents):
        if any((parent / marker).exists() for marker in marker_files):
            return parent

    raise FileNotFoundError("Project root not found")

