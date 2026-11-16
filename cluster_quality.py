"""Label-free cluster quality statistics used by reliability-aware QAA."""

from __future__ import annotations

import numpy as np


def compute_cluster_quality_score(
    embeddings: np.ndarray,
    assignments: np.ndarray,
    cluster_centers: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """Return a scale-invariant, label-free cluster quality score in [0, 1].

    The score is the share of total scatter explained by separation between the
    KMeans clusters::

        R = between_cluster_scatter / (between_cluster_scatter + within_cluster_scatter)

    Unlike raw centroid variance, this statistic rewards both separated cluster
    centers and compact samples, and it is unchanged when an embedding is
    translated or multiplied by a common scale factor.
    """
    if eps <= 0:
        raise ValueError("eps must be positive.")

    values = np.asarray(embeddings, dtype=np.float64)
    centers = np.asarray(cluster_centers, dtype=np.float64)
    raw_assignments = np.asarray(assignments).reshape(-1)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("embeddings must be a non-empty two-dimensional matrix.")
    if centers.ndim != 2 or centers.shape[1] != values.shape[1]:
        raise ValueError("cluster_centers must match the embedding dimension.")
    if raw_assignments.shape[0] != values.shape[0]:
        raise ValueError("assignments must contain one cluster id per sample.")
    if not (
        np.all(np.isfinite(values))
        and np.all(np.isfinite(centers))
        and np.all(np.isfinite(raw_assignments))
    ):
        raise ValueError("cluster quality inputs must contain only finite values.")

    labels = raw_assignments.astype(np.int64)
    if not np.array_equal(raw_assignments, labels):
        raise ValueError("assignments must contain integer cluster ids.")
    if labels.min() < 0 or labels.max() >= centers.shape[0]:
        raise ValueError("assignments contain a cluster id outside cluster_centers.")

    global_center = np.mean(values, axis=0)
    counts = np.bincount(labels, minlength=centers.shape[0]).astype(np.float64)
    center_offsets = centers - global_center
    between_scatter = float(
        np.sum(counts * np.sum(center_offsets * center_offsets, axis=1))
    )
    residuals = values - centers[labels]
    within_scatter = float(np.sum(residuals * residuals))
    total_scatter = between_scatter + within_scatter
    if not np.isfinite(total_scatter) or total_scatter <= eps:
        return 0.0
    return float(np.clip(between_scatter / total_scatter, 0.0, 1.0))
