"""Make the GEDE essays dataset available locally.

First step of a small example that compares the diversity of human- vs
LLM-written essays on the GEDE dataset (from "Assessing LLM Text Detection in
Educational Contexts", https://arxiv.org/abs/2508.08096). This step only ensures
the dataset is present; later steps add loading, the diversity measures, the
results table, and a plot.

The download needs the ``examples`` extra (``gdown``); install it with
``pip install emb-diversity[examples]`` or, from a checkout,
``uv sync --extra examples``. Run it with::

    uv run --extra examples python examples/gede_diversity.py

It downloads the GEDE dataset (via ``gdown``, into ``gede_essay_detection/`` in
the current directory) if it is not already present, then prints where
``gede_essays.json`` is.
"""

from __future__ import annotations

import sys
import tarfile
from pathlib import Path

# GEDE dataset on Google Drive — the file id from the share URL
# https://drive.google.com/file/d/1c3x_CR44ZCUqHf1dHVPm7K04ZIbTSYoD/view
GEDE_FILE_ID = "1c3x_CR44ZCUqHf1dHVPm7K04ZIbTSYoD"
DATA_DIR = Path("gede_essay_detection")
TARBALL = Path("gede_essay_detection.tar.gz")


def ensure_dataset() -> Path:
    """Download and extract the GEDE dataset if missing; return the JSON path.

    If ``gede_essay_detection/`` is absent, fetch the tarball from Google Drive
    with ``gdown``, extract it, and remove the tarball. Returns the path to
    ``gede_essays.json`` inside the dataset.
    """
    if DATA_DIR.exists():
        print(f"Skipping download ({DATA_DIR}/ already exists)")
    else:
        try:
            import gdown
        except ImportError:
            sys.exit(
                "Downloading the dataset needs gdown (the 'examples' extra). "
                "Install it with `pip install emb-diversity[examples]`, or run "
                "`uv run --extra examples python examples/gede_diversity.py`."
            )
        print("Downloading gede_essay_detection...")
        gdown.download(id=GEDE_FILE_ID, output=str(TARBALL))
        with tarfile.open(TARBALL, "r:gz") as tar:
            tar.extractall(filter="data")
        TARBALL.unlink()

    matches = sorted(DATA_DIR.rglob("gede_essays.json"))
    if not matches:
        sys.exit(f"Could not find gede_essays.json under {DATA_DIR}/ after download.")
    return matches[0]


def main() -> None:
    path = ensure_dataset()
    print(f"GEDE dataset ready: {path}")


if __name__ == "__main__":
    main()
