"""Numerical equivalence against the authors' reference implementations.

Five of the 22 measures have a publicly released reference implementation.
This module checks that ours produce the same numbers, at default parameters
on both sides.

    Energy   github.com/Abusagit/Challenges-on-generating-structurally-diverse-graphs
    DCScore  github.com/BlueWhaleLab/DCScore
    RKE      github.com/mjalali/renyi-kernel-entropy
    HamDiv   github.com/HXYfighter/HamDiv
    Vendi    github.com/vertaix/Vendi-Score  (the vendi-score PyPI package)

These tests are skipped unless the reference code is present. To run them:

    bash test/reference/fetch.sh          # clones the four repos
    uv pip install torch rdkit python_tsp tqdm  # deps the reference code needs
    uv run pytest test/test_reference_equivalence.py -v

Vendi Score is included for completeness only: emb_diversity delegates to the
official package, so the comparison is a regression check rather than a
validation of a reimplementation. What actually needs pinning there is the
dual/Gram agreement, covered in TestVendiScoreDualFormulation.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from emb_diversity import dcscore, energy, hamdiv, renyi_entropy, vendi_score

REF_DIR = Path(__file__).parent / "reference"

# --------------------------------------------------------------------------
# Reference implementations, imported only if the clones are present
# --------------------------------------------------------------------------


def _stub_unused_imports() -> None:
    """Stub the heavy module-level imports in the DCScore reference.

    dcscore_function.py imports transformers, sentence_transformers and
    torchvision for its text- and image-embedding paths. The function we
    exercise, calculate_dcscore_by_embedding, never touches them.
    """
    for name, attrs in [
        ("transformers", ["AutoTokenizer", "AutoModel", "AutoModelForCausalLM",
                          "LlamaForCausalLM"]),
        ("sentence_transformers", ["SentenceTransformer"]),
        ("torchvision", []),
        ("torchvision.models", ["inception_v3"]),
        ("torchvision.transforms", ["Compose", "Resize", "CenterCrop",
                                    "ToTensor", "Normalize",
                                    "InterpolationMode"]),
    ]:
        if name in sys.modules:
            continue
        module = types.ModuleType(name)
        for attr in attrs:
            setattr(module, attr, object)
        sys.modules[name] = module


def _load_references():
    """Return (callables, None), or (None, reason) explaining what is missing."""
    for sub in ("rke", "dcscore", "hamdiv"):
        path = REF_DIR / sub
        if not path.exists():
            return None, f"{path} not found; run test/reference/fetch.sh"
        sys.path.insert(0, str(path))

    _stub_unused_imports()
    try:
        from dcscore_function import DCScore
        from diversity import HamDiv
        from rke_score.rke import RKE
        from vendi_score import vendi
    except ImportError as exc:
        return None, (
            f"reference code present but not importable ({exc}); "
            "see test/reference/fetch.sh for the extra dependencies"
        )

    def ref_dcscore(X, tau=1.0):
        # Unbound call: __init__ would load an embedding model we do not need.
        return DCScore.calculate_dcscore_by_embedding(None, X, kernel_type="cs",
                                                      tau=tau)

    def ref_rke2(X):
        # compute_rke_mc_frobenius_norm returns ||S||_F^2 with S = K / n;
        # renyi_entropy(alpha=2) returns -log of that quantity.
        def cosine(a, b):
            return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

        fnorm = RKE(kernel_function=cosine).compute_rke_mc_frobenius_norm(X)
        return -np.log(fnorm)

    def ref_vendi(X):
        Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
        return vendi.score_X(Xn)

    return {
        "dcscore": ref_dcscore,
        "rke": ref_rke2,
        "vendi": ref_vendi,
        "hamdiv": lambda D: HamDiv(dists=D),
    }, None


REFS, SKIP_REASON = _load_references()

requires_reference_code = pytest.mark.skipif(REFS is None, reason=SKIP_REASON or "")


def ref_energy(D, eps=1e-6):
    """Energy, transcribed from the reference.

    The reference operates on graphs through a distance callable, so the
    measure is split across two places in base.py:

        line  55   distance_function = lambda G1, G2: 1 / (d(G1, G2) + 1e-6)
        line 255   volume = -1.0 * sum(fitness) / (N * (N - 1))

    where each element's fitness is that callable summed over all others.
    Transcribed here over a precomputed distance matrix, because instantiating
    their framework requires graph objects rather than vectors. Note that gamma
    is hard-coded to 1 in the reference; there is no exponent in their code.
    """
    n = D.shape[0]
    inv = 1.0 / (D + eps)
    np.fill_diagonal(inv, 0.0)
    return -1.0 * inv.sum() / (n * (n - 1))


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
SIZES = [5, 50, 500]
DIMS = [2, 8, 384]
SEEDS = [0, 1, 2, 3, 4]

# float64 rounding error accumulated over O(n^2) sums, plus cancellation in
# the distance computation for near-degenerate pairs at d=2.
TOLERANCE = 1e-9


def unit_rows(n: int, d: int, seed: int) -> np.ndarray:
    X = np.random.default_rng(seed).standard_normal((n, d))
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def cosine_matrix(X: np.ndarray) -> np.ndarray:
    D = 1.0 - np.clip(X @ X.T, -1.0, 1.0)
    np.fill_diagonal(D, 0.0)
    return D


def assert_equivalent(ours: float, reference: float) -> None:
    deviation = abs(ours - reference) / max(abs(reference), 1e-30)
    assert deviation < TOLERANCE, (
        f"ours={ours!r} reference={reference!r} rel. dev.={deviation:.3e}"
    )


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------
@requires_reference_code
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("d", DIMS)
@pytest.mark.parametrize("n", SIZES)
class TestReferenceEquivalence:
    """Default parameters on both sides; no alignment step."""

    def test_energy(self, n, d, seed):
        X = unit_rows(n, d, seed)
        assert_equivalent(energy(X)["value"], ref_energy(cosine_matrix(X)))

    def test_dcscore(self, n, d, seed):
        X = unit_rows(n, d, seed)
        assert_equivalent(dcscore(X)["value"], REFS["dcscore"](X))

    def test_vendi(self, n, d, seed):
        X = unit_rows(n, d, seed)
        assert_equivalent(vendi_score(X)["value"], REFS["vendi"](X))


@requires_reference_code
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("d", DIMS)
@pytest.mark.parametrize("n", [5, 50])  # the reference is an O(n^2) Python loop
def test_renyi_kernel_entropy(n, d, seed):
    X = unit_rows(n, d, seed)
    assert_equivalent(renyi_entropy(X)["value"], REFS["rke"](X))


@requires_reference_code
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("n,d", [(5, 2), (5, 8), (20, 8), (50, 384)])
def test_hamdiv(n, d, seed):
    """Both sides default to the greedy solver.

    Equivalence holds because both wrap the same networkx routine. With
    solver="christofides" the values differ by ~4%, which is a property of the
    approximation algorithms rather than of either implementation.
    """
    X = unit_rows(n, d, seed)
    assert_equivalent(hamdiv(X)["value"], REFS["hamdiv"](cosine_matrix(X)))


@requires_reference_code
def test_hamdiv_solvers_differ():
    """Guard the documented scope note: the two solvers are not interchangeable."""
    X = unit_rows(20, 8, 0)
    greedy = hamdiv(X, solver="greedy")["value"]
    christofides = hamdiv(X, solver="christofides")["value"]
    assert abs(greedy - christofides) / greedy > 0.01