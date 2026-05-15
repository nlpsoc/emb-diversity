from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

### Geometry-Based Diversity Measure (multi-dataset)


def _load_magnipy():
    """Import ``magnipy.Diversipy`` lazily.

    The import is deferred so that ``import embediver`` (and any measure
    that doesn't use ``mag_areas``) doesn't pull in magnipy, doesn't trip
    the scipy 1.16 incompatibility, and doesn't apply a global monkey-
    patch as a side effect of importing the package.

    Also rebinds ``scipy.integrate.trapz`` to ``trapezoid`` first;
    magnipy still imports the former, which scipy 1.14+ removed. Doing
    the rebind inside this helper keeps the patch local to the moment
    ``mag_areas`` is actually called.
    """
    import scipy.integrate as _si
    if not hasattr(_si, "trapz"):
        _si.trapz = _si.trapezoid  # type: ignore[attr-defined]
    try:
        from magnipy import Diversipy
    except ImportError as e:
        raise ImportError(
            "mag_areas needs the 'magarea' optional dependency. "
            "Install it with: pip install -e \".[magarea]\" --no-deps  "
            "(--no-deps is required because magnipy's own pyproject pins "
            "scipy==1.13.0, which conflicts with embediver's scipy~=1.16.0)."
        ) from e
    return Diversipy


def mag_areas(
    data_list: Union[List[Sequence[Sequence[float]]], Tuple[Sequence[Sequence[float]], ...]],
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
            A **list or tuple** of datasets. Each entry is itself a 2D
            array-like of shape ``(n_i, d)`` containing embedding
            vectors. All datasets must share the same embedding
            dimension ``d``; ``n_i`` may differ between datasets. A
            bare 2D ndarray is rejected at runtime to prevent the
            common misuse ``mag_areas(X_2d)`` where ``X_2d`` is one
            dataset rather than a list of datasets.
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
        ImportError:
            If the ``magarea`` extra is not installed (see install
            instructions in the error message).
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
    Diversipy = _load_magnipy()
    dp_kwargs = {"metric": metric, "n_ts": int(n_ts)}
    if names is not None:
        dp_kwargs["names"] = list(names)
    dp = Diversipy(Xs=Xs, **dp_kwargs)
    areas = dp.MagAreas()
    return [float(v) for v in areas]
