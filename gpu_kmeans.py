"""Strict GPU-backed KMeans adapter for the DU-FMVC experiment pipeline."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


def _load_cuml_kmeans():
    try:
        from cuml.cluster import KMeans as CuMLKMeans
    except ImportError as exc:
        raise ImportError(
            "GPU KMeans requires RAPIDS cuML. Install the CUDA-matched "
            "cuml package in the AutoDL environment before running main.py. "
            "This project intentionally does not fall back to CPU KMeans."
        ) from exc
    return CuMLKMeans


def _to_numpy(value: Any) -> np.ndarray:
    """Normalize cuML/CuPy/cuDF outputs for the existing NumPy server code."""
    if hasattr(value, "get"):
        value = value.get()
    elif hasattr(value, "to_numpy"):
        value = value.to_numpy()
    return np.asarray(value)


class GPUKMeans:
    """Small sklearn-like wrapper around ``cuml.cluster.KMeans``.

    KMeans fitting always runs through cuML on the GPU. Outputs are converted to
    NumPy because the existing label-alignment and global-target code is NumPy
    based. No CPU KMeans fallback is provided.
    """

    def __init__(
        self,
        n_clusters: int,
        n_init: int,
        random_state: int = 1018,
        max_iter: int = 300,
        tol: float = 1e-4,
    ) -> None:
        cuml_kmeans = _load_cuml_kmeans()
        self._model = cuml_kmeans(
            n_clusters=int(n_clusters),
            n_init=int(n_init),
            random_state=int(random_state),
            max_iter=int(max_iter),
            tol=float(tol),
            init="k-means++",
            output_type="numpy",
        )
        self.labels_: Optional[np.ndarray] = None
        self.cluster_centers_: Optional[np.ndarray] = None

    @staticmethod
    def _prepare_input(values: Any) -> np.ndarray:
        matrix = np.asarray(values)
        if matrix.ndim != 2:
            raise ValueError("GPUKMeans input must be a two-dimensional matrix.")
        target_dtype = np.float64 if matrix.dtype == np.float64 else np.float32
        return np.ascontiguousarray(matrix, dtype=target_dtype)

    def _sync_outputs(self, labels: Any = None) -> None:
        if labels is None:
            labels = self._model.labels_
        self.labels_ = _to_numpy(labels).reshape(-1)
        self.cluster_centers_ = _to_numpy(self._model.cluster_centers_)

    def fit(self, values: Any) -> "GPUKMeans":
        self._model.fit(self._prepare_input(values))
        self._sync_outputs()
        return self

    def fit_predict(self, values: Any) -> np.ndarray:
        labels = self._model.fit_predict(self._prepare_input(values))
        self._sync_outputs(labels)
        return self.labels_.copy()
