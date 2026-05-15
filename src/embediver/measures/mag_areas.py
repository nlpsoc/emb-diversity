from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

### Geometry-Based Diversity Measure (multi-dataset)

# magnipy still imports ``scipy.integrate.trapz`` which was removed in
# scipy 1.14. We rebind it to ``trapezoid`` so ``from magnipy import ...``
# succeeds in environments with scipy >= 1.14.
import scipy.integrate as _scipy_integrate
if not hasattr(_scipy_integrate, "trapz"):
    _scipy_integrate.trapz = _scipy_integrate.trapezoid  # type: ignore[attr-defined]

from magnipy import Diversipy  # noqa: E402


def mag_areas(
    data_list: Sequence[Sequence[Sequence[float]]],
    metric: str = "cosine",
    n_ts: int = 30,
    names: Optional[Sequence[str]] = None,
) -> List[float]:
    """
    Multi-dataset Metric Space Magnitude Area (MagArea).

    Returns one MagArea per dataset, all evaluated on a SHARED set of
    distance scales chosen jointly across the inputs. As a result the
    returned values are directly comparable across datasets — a larger
    MagArea means higher intrinsic diversity.

    Why this requires ``>= 2`` datasets:

      The magnitude function is integrated over a scale axis ``ts``.
      ``Diversipy`` builds a common ``ts`` by looking at all datasets at
      once (it takes a quantile of their individual convergence scales).
      Computing it on a single dataset would fall back to per-dataset
      auto-scale, in which case the returned MagAreas are NOT comparable
      across calls — that is exactly what this function is designed to
      prevent. Use ``magnipy.Magnipy`` directly if you want the single-
      dataset, auto-scale behaviour, but be aware the numbers are not
      diversity-rankable across datasets.

    Args:
        data_list:
            A sequence of datasets. Each entry is itself a 2D array-like
            of shape ``(n_i, d)`` containing embedding vectors. All
            datasets must share the same embedding dimension ``d``;
            ``n_i`` may differ between datasets.
        metric:
            Distance metric forwarded to magnitude theory. ``"cosine"``
            (default) matches the convention used elsewhere in
            ``embediver``; ``"euclidean"`` is the theoretically clean
            choice for magnitude (negative-type metric) and is what the
            magnipy tutorials use. Any metric accepted by
            ``scipy.spatial.distance.pdist`` is allowed.
        n_ts:
            Number of common evaluation scales for the magnitude
            functions. Higher = finer integration of the area.
        names:
            Optional dataset names. Length must equal ``len(data_list)``
            if provided.

    Returns:
        A list of MagArea values, one per dataset, in the same order as
        ``data_list``. Larger means more diverse.

    Raises:
        ValueError:
            If ``data_list`` has fewer than 2 entries, or any entry is
            not 2D, or shapes / ``names`` length disagree.

    References:
        Limbeck, K., Andreeva, R., Sarkar, R. and Rieck, B., 2024.
        Metric Space Magnitude for Evaluating the Diversity of Latent
        Representations. arXiv:2311.16054.
    """
    # ---- Validate inputs ----
    if not isinstance(data_list, (list, tuple)):
        raise ValueError(
            "mag_areas expects a list/tuple of datasets, not a single dataset."
        )
    n_datasets = len(data_list)
    if n_datasets < 2:
        raise ValueError(
            "mag_areas requires at least 2 datasets to produce comparable "
            f"MagAreas; got {n_datasets}. See the docstring for why."
        )
    if names is not None and len(names) != n_datasets:
        raise ValueError(
            f"len(names)={len(names)} does not match len(data_list)={n_datasets}"
        )
    if n_ts < 2:
        raise ValueError(f"n_ts must be >= 2, got {n_ts}")

    Xs: List[np.ndarray] = []
    d_ref: Optional[int] = None
    for i, X in enumerate(data_list):
        Xi = np.asarray(X, dtype=float)
        if Xi.ndim != 2:
            raise ValueError(
                f"data_list[{i}]: expected 2D array (n, d), got shape {Xi.shape}"
            )
        if Xi.shape[0] < 2:
            raise ValueError(
                f"data_list[{i}]: needs at least 2 rows, got {Xi.shape[0]}"
            )
        if d_ref is None:
            d_ref = Xi.shape[1]
        elif Xi.shape[1] != d_ref:
            raise ValueError(
                f"data_list[{i}]: embedding dim {Xi.shape[1]} does not match "
                f"first dataset's dim {d_ref}"
            )
        Xs.append(Xi)

    # ---- Run Diversipy on the common scale axis ----
    dp_kwargs = {"metric": metric, "n_ts": int(n_ts)}
    if names is not None:
        dp_kwargs["names"] = list(names)
    dp = Diversipy(Xs=Xs, **dp_kwargs)
    areas = dp.MagAreas()
    return [float(v) for v in areas]
