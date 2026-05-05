from __future__ import annotations

import numpy as np


def _row_present_from_mask_and_data(X: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """Derive a row-level presence mask from feature mask and/or data values."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array with shape [N, D].")
    n_rows = X.shape[0]

    row_all_nan = np.all(np.isnan(X), axis=1)
    row_all_zero = np.all(np.nan_to_num(X, nan=0.0) == 0.0, axis=1)

    if mask is None:
        row_present = ~(row_all_nan | row_all_zero)
        return row_present.astype(bool)

    mask = np.asarray(mask)
    if mask.ndim == 1:
        if mask.shape[0] != n_rows:
            raise ValueError("mask_view must have shape [N].")
        row_present = mask.astype(bool)
    elif mask.ndim == 2:
        if mask.shape[0] != n_rows:
            raise ValueError("mask_feat must have shape [N, D].")
        row_present = ~np.all(mask.astype(bool), axis=1)
    else:
        raise ValueError("mask must be 1D (view mask) or 2D (feature mask).")

    row_present = row_present & ~(row_all_nan | row_all_zero)
    return row_present.astype(bool)


def _build_concat_reference(
    X_full_list: list[np.ndarray],
    mask_view_list: list[np.ndarray],
) -> np.ndarray:
    """Build concatenated reference representation with missing rows zeroed."""
    if len(X_full_list) != len(mask_view_list):
        raise ValueError("X_full_list and mask_view_list must have the same length.")

    blocks = []
    for X, mask in zip(X_full_list, mask_view_list):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("Each X must be 2D.")
        mask = np.asarray(mask, dtype=bool)
        if mask.shape[0] != X.shape[0]:
            raise ValueError("mask_view length must match X rows.")
        X_block = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).copy()
        X_block[~mask] = 0.0
        blocks.append(X_block)
    return np.concatenate(blocks, axis=1)


def make_feature_missing(
    X: np.ndarray, rate: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly mark a fraction of entries as missing.

    Args:
        X: Feature matrix with shape [N, D].
        rate: Fraction of entries to mark missing.
        seed: Random seed for reproducibility.

    Returns:
        A tuple (X_miss, mask) where mask is True for missing entries.
    """
    if X.ndim != 2:
        raise ValueError("X must be a 2D array with shape [N, D].")

    rate = float(rate)
    rate = min(max(rate, 0.0), 1.0)

    X_miss = X.astype(float, copy=True)
    mask = np.zeros_like(X_miss, dtype=bool)

    total = X_miss.size
    miss_count = int(rate * total)
    if miss_count <= 0:
        return X_miss, mask

    rng = np.random.default_rng(seed)
    miss_indices = rng.choice(total, size=miss_count, replace=False)
    mask.flat[miss_indices] = True
    X_miss[mask] = np.nan
    return X_miss, mask


def impute_knn(
    X_miss: np.ndarray,
    mask: np.ndarray,
    top_ratio: float = 0.1,
    eps: float = 1e-8,
    sigma: float = 0.01,
    seed: int | None = None,
) -> np.ndarray:
    """Impute missing values using distance-weighted KNN.

    Distances are computed only on shared non-missing dimensions.

    Args:
        X_miss: Feature matrix with NaNs for missing entries.
        mask: Boolean mask, True for missing entries.
        top_ratio: Ratio of neighbors to use (based on N-1).
        eps: Small constant to avoid division by zero.
        sigma: Standard deviation of Gaussian noise added after imputation.

    Returns:
        Imputed feature matrix with the same shape as X_miss.
    """
    if X_miss.ndim != 2:
        raise ValueError("X_miss must be a 2D array with shape [N, D].")
    if mask.shape != X_miss.shape:
        raise ValueError("mask must have the same shape as X_miss.")

    X_base = X_miss.astype(float, copy=True)
    mask = mask.astype(bool, copy=False)
    X_filled = X_base.copy()
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

    N, D = X_base.shape
    if N == 0 or D == 0:
        return X_filled

    global_mean = np.nanmean(X_base, axis=0)
    global_mean = np.where(np.isnan(global_mean), 0.0, global_mean)

    if N <= 1:
        missing_rows, missing_cols = np.where(mask)
        if missing_rows.size > 0:
            noise = rng.normal(0.0, sigma, size=missing_rows.size)
            X_filled[missing_rows, missing_cols] = global_mean[missing_cols] + noise
        return X_filled

    top_ratio = float(top_ratio)
    top_ratio = min(max(top_ratio, 0.0), 1.0)
    P = int(np.ceil((N - 1) * top_ratio))
    if P < 1:
        P = 1
    if P > (N - 1):
        P = N - 1

    not_missing = ~mask

    for i in range(N):
        if not mask[i].any():
            continue

        dist = np.full(N, np.inf, dtype=float)
        for j in range(N):
            if j == i:
                continue
            common = not_missing[i] & not_missing[j]
            if not np.any(common):
                continue
            diff = X_base[i, common] - X_base[j, common]
            dist[j] = np.sqrt(np.dot(diff, diff))

        valid_idx = np.where(np.isfinite(dist))[0]
        if valid_idx.size == 0:
            missing_dims = np.where(mask[i])[0]
            if missing_dims.size > 0:
                noise = rng.normal(0.0, sigma, size=missing_dims.size)
                X_filled[i, missing_dims] = global_mean[missing_dims] + noise
            continue

        order = np.argsort(dist[valid_idx])
        neighbors = valid_idx[order][:P]
        neighbor_dist = dist[neighbors]
        finite = np.isfinite(neighbor_dist)

        missing_dims = np.where(mask[i])[0]
        for d in missing_dims:
            avail = (~mask[neighbors, d]) & finite
            if not np.any(avail):
                fill_value = global_mean[d]
                X_filled[i, d] = fill_value + rng.normal(0.0, sigma)
                continue
            vals = X_base[neighbors[avail], d]
            w = 1.0 / (neighbor_dist[avail] + eps)
            w_sum = np.sum(w)
            if w_sum <= 0:
                fill_value = global_mean[d]
            else:
                fill_value = float(np.sum(vals * w) / w_sum)
            X_filled[i, d] = fill_value + rng.normal(0.0, sigma)

    return X_filled


def sanity_check_impute(
    X_before: np.ndarray,
    X_miss: np.ndarray,
    mask: np.ndarray,
    X_after: np.ndarray,
) -> dict:
    """Compute simple diagnostics for feature-missing + imputation."""
    X_before = np.asarray(X_before)
    X_miss = np.asarray(X_miss)
    mask = np.asarray(mask, dtype=bool)
    X_after = np.asarray(X_after)

    if X_before.shape != X_after.shape or X_miss.shape != mask.shape or X_after.shape != mask.shape:
        raise ValueError("X_before, X_miss, mask, and X_after must have matching shapes.")

    miss_rate_actual = float(mask.mean()) if mask.size else 0.0
    nan_count_before = int(np.isnan(X_miss).sum())
    nan_count_after = int(np.isnan(X_after).sum())

    nonmissing = ~mask
    changed_nonmissing_count = int(
        np.count_nonzero(nonmissing & ~np.isclose(X_before, X_after, rtol=0.0, atol=0.0))
    )
    per_col_all_missing_count = int(np.sum(np.all(mask, axis=0))) if mask.size else 0

    return {
        "miss_rate_actual": miss_rate_actual,
        "nan_count_before": nan_count_before,
        "nan_count_after": nan_count_after,
        "changed_nonmissing_count": changed_nonmissing_count,
        "per_col_all_missing_count": per_col_all_missing_count,
    }


def refine_impute_by_cluster(
    X: np.ndarray,
    mask: np.ndarray,
    p: np.ndarray,
    return_stats: bool = False,
    alpha: float = 0.3,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Refine imputation using cluster-wise means from soft assignments."""
    X = np.asarray(X, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    p = np.asarray(p)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array with shape [N, D].")
    if mask.shape != X.shape:
        raise ValueError("mask must have the same shape as X.")
    if p.ndim != 2 or p.shape[0] != X.shape[0]:
        raise ValueError("p must have shape [N, K] matching X rows.")

    X_refined = X.copy()
    N, D = X.shape
    if N == 0 or D == 0:
        return X_refined

    global_mean = np.nanmean(X, axis=0)
    global_mean = np.where(np.isnan(global_mean), 0.0, global_mean)

    cluster_assign = np.argmax(p, axis=1)
    fallback_to_global_mean = 0

    for i in range(N):
        if not mask[i].any():
            continue

        u = cluster_assign[i]
        cluster_idx = np.where(cluster_assign == u)[0]
        if cluster_idx.size > 0:
            cluster_idx = cluster_idx[cluster_idx != i]

        missing_dims = np.where(mask[i])[0]
        if cluster_idx.size == 0:
            X_refined[i, missing_dims] = global_mean[missing_dims]
            fallback_to_global_mean += missing_dims.size
            continue

        cluster_vals = X[cluster_idx]
        cluster_mask = mask[cluster_idx]
        for d in missing_dims:
            avail = ~cluster_mask[:, d]
            if not np.any(avail):
                mean_val = global_mean[d]
                fallback_to_global_mean += 1
            else:
                vals = cluster_vals[avail, d]
                mean_val = np.nanmean(vals)
                if np.isnan(mean_val):
                    mean_val = global_mean[d]
                    fallback_to_global_mean += 1
            if np.isnan(X[i, d]):
                X_refined[i, d] = mean_val
            else:
                X_refined[i, d] = (1.0 - alpha) * X[i, d] + alpha * mean_val

    if return_stats:
        return X_refined, {
            "updated_entries": int(mask.sum()),
            "fallback_to_global_mean": int(fallback_to_global_mean),
        }
    return X_refined


def impute_sample_view_missing_knn(
    X_full_list: list[np.ndarray],
    mask_view_list: list[np.ndarray],
    k: int,
    ref_mode: str = "concat",
    tau: float = 1.0,
    eps: float = 1e-8,
) -> list[np.ndarray]:
    """Impute row-level (sample-view) missing entries using KNN in a reference space."""
    if len(X_full_list) != len(mask_view_list):
        raise ValueError("X_full_list and mask_view_list must have the same length.")
    if k <= 0:
        raise ValueError("k must be a positive integer.")

    num_views = len(X_full_list)
    X_out = [np.asarray(X, dtype=float).copy() for X in X_full_list]

    # Normalize/derive row-level masks.
    row_masks = []
    for X, mask in zip(X_out, mask_view_list):
        row_mask = _row_present_from_mask_and_data(X, mask)
        row_masks.append(row_mask)

    if ref_mode not in {"concat", "masked"}:
        raise ValueError(
            f"Unsupported ref_mode: {ref_mode}. Use 'concat' or 'masked'."
        )

    if ref_mode == "concat":
        R = _build_concat_reference(X_out, row_masks)
        R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)

        N = R.shape[0]
        if N == 0:
            return X_out

        n_candidates = min(k, N)
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=n_candidates, algorithm="auto")
        nn.fit(R)

        for v in range(num_views):
            mask_present = row_masks[v]
            missing_idx = np.where(~mask_present)[0]
            if missing_idx.size == 0:
                continue

            X_v = X_out[v]
            if np.any(mask_present):
                present_mean = np.mean(X_v[mask_present], axis=0)
            else:
                present_mean = np.zeros(X_v.shape[1], dtype=float)

            distances, indices = nn.kneighbors(R[missing_idx], n_neighbors=n_candidates)

            for local_i, i in enumerate(missing_idx):
                cand_idx = indices[local_i]
                cand_dist = distances[local_i]
                keep = mask_present[cand_idx]
                cand_idx = cand_idx[keep]
                cand_dist = cand_dist[keep]

                if cand_idx.size == 0:
                    X_v[i] = present_mean
                    continue

                d = np.asarray(cand_dist, dtype=float)
                d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
                sim = np.exp(-d)
                sim_sum = float(np.sum(sim))
                if sim_sum <= 0.0:
                    sim = np.ones_like(sim) / sim.size
                else:
                    sim = sim / sim_sum
                order = np.argsort(-sim)
                sim = sim[order]
                cand_idx = cand_idx[order]
                cand_dist = d[order]
                cum = np.cumsum(sim)
                k_i = int(np.searchsorted(cum, 0.8, side="left") + 1)
                if k_i < 1:
                    k_i = 1
                if k_i > cand_idx.size:
                    k_i = cand_idx.size
                cand_idx = cand_idx[:k_i]
                cand_dist = cand_dist[:k_i]
                scale = max(float(tau), eps)
                scaled = -cand_dist / scale
                scaled -= np.max(scaled)
                w = np.exp(scaled)
                w_sum = np.sum(w)
                if w_sum <= 0:
                    w = np.ones_like(w) / w.size
                else:
                    w = w / w_sum

                X_v[i] = np.sum(X_v[cand_idx] * w[:, None], axis=0)

            X_out[v] = X_v

        return X_out

    # masked distance mode: compute distances only over shared present views
    X_views = [np.asarray(X, dtype=float) for X in X_out]
    N = X_views[0].shape[0] if X_views else 0
    if N == 0:
        return X_out

    norm2_list = [np.sum(X_v ** 2, axis=1) for X_v in X_views]

    for v in range(num_views):
        mask_present = row_masks[v]
        missing_idx = np.where(~mask_present)[0]
        if missing_idx.size == 0:
            continue

        X_v = X_views[v]
        if np.any(mask_present):
            present_mean = np.mean(X_v[mask_present], axis=0)
        else:
            present_mean = np.zeros(X_v.shape[1], dtype=float)

        for i in missing_idx:
            dist = np.zeros(N, dtype=float)
            counts = np.zeros(N, dtype=float)

            for u in range(num_views):
                if not row_masks[u][i]:
                    continue
                X_u = X_views[u]
                xi = X_u[i]
                xi_norm2 = float(np.dot(xi, xi))
                dist_u = norm2_list[u] + xi_norm2 - 2.0 * (X_u @ xi)
                avail = row_masks[u]
                dist[avail] += dist_u[avail]
                counts[avail] += 1.0

            valid = counts > 0
            dist[~valid] = np.inf
            dist = np.maximum(dist, 0.0)
            dist[i] = np.inf

            dist[~mask_present] = np.inf
            valid = np.isfinite(dist)
            if not np.any(valid):
                X_v[i] = present_mean
                continue

            dist[valid] = dist[valid] / counts[valid]

            k_eff = int(min(k, np.sum(valid)))
            if k_eff <= 0:
                X_v[i] = present_mean
                continue

            idx = np.argpartition(dist, k_eff - 1)[:k_eff]
            idx = idx[np.argsort(dist[idx])]
            d = dist[idx]
            sim = np.exp(-d)
            sim_sum = float(np.sum(sim))
            if sim_sum <= 0.0:
                sim = np.ones_like(sim) / sim.size
            else:
                sim = sim / sim_sum
            order = np.argsort(-sim)
            sim = sim[order]
            idx = idx[order]
            d = d[order]
            cum = np.cumsum(sim)
            k_i = int(np.searchsorted(cum, 0.8, side="left") + 1)
            if k_i < 1:
                k_i = 1
            if k_i > idx.size:
                k_i = idx.size
            idx = idx[:k_i]
            d = d[:k_i]

            scale = max(float(tau), eps)
            scaled = -d / scale
            scaled -= np.max(scaled)
            w = np.exp(scaled)
            w_sum = np.sum(w)
            if w_sum <= 0:
                w = np.ones_like(w) / w.size
            else:
                w = w / w_sum

            X_v[i] = np.sum(X_v[idx] * w[:, None], axis=0)

        X_out[v] = X_v

    return X_out
