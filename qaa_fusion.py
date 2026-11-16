"""Numerically safe direct weights for QAA representation fusion."""

from __future__ import annotations

import numpy as np


def compute_qaa_feature_scales(
    weights: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """Return the normalized omega_v used directly in concat(omega_v Z_v).

    The paper specifies direct weighted concatenation. No square-root feature
    rescaling or additional temporal smoothing is applied.
    """
    if eps <= 0:
        raise ValueError("eps must be positive.")
    values = np.asarray(weights, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("weights must be non-empty.")
    values = np.maximum(
        np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0),
        0.0,
    )
    total = float(np.sum(values))
    if total <= eps:
        normalized = np.full(values.shape, 1.0 / values.size, dtype=np.float64)
    else:
        normalized = values / total
    return normalized
