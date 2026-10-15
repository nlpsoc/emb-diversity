#!/usr/bin/env bash
# Clone the reference implementations used by test_reference_equivalence.py.
#
# These are third-party repositories under their own licences and are not
# vendored into this repository. Pinned to the commits the equivalence results
# were produced with; drop the `git checkout` lines to test against their HEAD.
#
# Usage:  bash test/reference/fetch.sh
set -euo pipefail

cd "$(dirname "$0")"

clone() {
    local url=$1 dir=$2 commit=$3
    if [ -d "$dir" ]; then
        echo "skip $dir (already present)"
        return
    fi
    git clone --quiet "$url" "$dir"
    if [ -n "$commit" ]; then
        git -C "$dir" checkout --quiet "$commit"
    fi
    echo "$dir -> $(git -C "$dir" rev-parse --short HEAD)"
}

clone https://github.com/Abusagit/Challenges-on-generating-structurally-diverse-graphs energy  baa672b2
clone https://github.com/BlueWhaleLab/DCScore                                          dcscore eab8bf4a
clone https://github.com/mjalali/renyi-kernel-entropy                                  rke     ""
clone https://github.com/HXYfighter/HamDiv                                             hamdiv  ""

cat <<'EOF'

Reference code fetched. The tests also need dependencies that emb-diversity
itself does not require:

    uv pip install torch rdkit python_tsp tqdm

    torch       DCScore's softmax
    rdkit       HamDiv's module-level imports
    python_tsp  HamDiv's exact solver import
    tqdm        progress bars in both reference modules

Then:

    uv run pytest test/test_reference_equivalence.py -v
EOF