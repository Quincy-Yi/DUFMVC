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
        # A supplied sample-view mask is the paper's observability indicator
        # M_iv. Trust it directly: an observed all-zero vector is still an
        # observed view.
        return mask.astype(bool)
    elif mask.ndim == 2:
        if mask.shape[0] != n_rows:
            raise ValueError("mask_feat must have shape [N, D].")
        row_present = ~np.all(mask.astype(bool), axis=1)
    else:
        raise ValueError("mask must be 1D (view mask) or 2D (feature mask).")

    row_present = row_present & ~(row_all_nan | row_all_zero)
    return row_present.astype(bool)


def _build_proxy_context(
    candidate_indices: np.ndarray,
    candidate_distances: np.ndarray,
    tau: float,
    coverage_threshold: float,
    eps: float,
) -> dict[str, np.ndarray | int]:
    """Build the fixed proxy neighborhood and weights in Eqs. (4)--(8)."""
    indices = np.asarray(candidate_indices, dtype=np.int64).reshape(-1)
    distances = np.asarray(candidate_distances, dtype=float).reshape(-1)
    if indices.size != distances.size:
        raise ValueError("candidate indices and distances must have equal length.")
    if indices.size == 0:
        return {
            "candidate_indices": np.empty(0, dtype=np.int64),
            "candidate_distances": np.empty(0, dtype=float),
            "adaptive_k": 0,
            "neighbor_indices": np.empty(0, dtype=np.int64),
            "neighbor_weights": np.empty(0, dtype=float),
        }

    order = np.argsort(distances, kind="stable")
    indices = indices[order]
    distances = np.maximum(
        np.nan_to_num(distances[order], nan=0.0, posinf=np.inf, neginf=0.0),
        0.0,
    )

    # Adaptive coverage uses a_ij=exp(-d_ij), independently of tau_1.
    coverage_logits = -distances
    coverage_logits -= np.max(coverage_logits)
    coverage_similarity = np.exp(coverage_logits)
    coverage_total = float(np.sum(coverage_similarity))
    if coverage_total <= eps:
        coverage_similarity = np.ones_like(coverage_similarity) / indices.size
    else:
        coverage_similarity /= coverage_total
    adaptive_k = int(
        np.searchsorted(
            np.cumsum(coverage_similarity), coverage_threshold, side="left"
        )
        + 1
    )
    adaptive_k = min(max(adaptive_k, 1), indices.size)

    neighbor_indices = indices[:adaptive_k]
    neighbor_distances = distances[:adaptive_k]
    weight_logits = -neighbor_distances / max(float(tau), eps)
    weight_logits -= np.max(weight_logits)
    neighbor_weights = np.exp(weight_logits)
    weight_total = float(np.sum(neighbor_weights))
    if weight_total <= eps:
        neighbor_weights = np.ones_like(neighbor_weights) / adaptive_k
    else:
        neighbor_weights /= weight_total

    return {
        "candidate_indices": indices.copy(),
        "candidate_distances": distances.copy(),
        "adaptive_k": adaptive_k,
        "neighbor_indices": neighbor_indices.copy(),
        "neighbor_weights": neighbor_weights.copy(),
    }


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
    coverage_threshold: float = 0.8,
    eps: float = 1e-8,
    return_context: bool = False,
) -> list[np.ndarray] | tuple[list[np.ndarray], list[dict[int, dict[str, np.ndarray | int]]]]:
    """Impute row-level (sample-view) missing entries using KNN in a reference space."""
    if len(X_full_list) != len(mask_view_list):
        raise ValueError("X_full_list and mask_view_list must have the same length.")
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    if not 0.0 < coverage_threshold <= 1.0:
        raise ValueError("coverage_threshold must be in (0, 1].")

    num_views = len(X_full_list)
    X_out = [np.asarray(X, dtype=float).copy() for X in X_full_list]
    imputation_context: list[dict[int, dict[str, np.ndarray | int]]] = [
        {} for _ in range(num_views)
    ]

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
            return (X_out, imputation_context) if return_context else X_out

        n_query = min(k + 1, N)
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=n_query, algorithm="auto")
        nn.fit(R)

        for v in range(num_views):
            mask_present = row_masks[v]
            X_v = X_out[v]
            if np.any(mask_present):
                present_mean = np.mean(X_v[mask_present], axis=0)
            else:
                present_mean = np.zeros(X_v.shape[1], dtype=float)

            distances, indices = nn.kneighbors(R, n_neighbors=n_query)

            for i in range(N):
                cand_idx = indices[i]
                cand_dist = distances[i]
                keep = mask_present[cand_idx] & (cand_idx != i)
                cand_idx = cand_idx[keep]
                cand_dist = cand_dist[keep]
                if cand_idx.size > k:
                    cand_idx = cand_idx[:k]
                    cand_dist = cand_dist[:k]

                context = _build_proxy_context(
                    cand_idx, cand_dist, tau, coverage_threshold, eps
                )
                imputation_context[v][int(i)] = context
                if not mask_present[i] and context["adaptive_k"] == 0:
                    X_v[i] = present_mean
                    continue
                if not mask_present[i]:
                    X_v[i] = np.sum(
                        X_v[context["neighbor_indices"]]
                        * context["neighbor_weights"][:, None],
                        axis=0,
                    )

            X_out[v] = X_v

        return (X_out, imputation_context) if return_context else X_out

    # masked distance mode: compute distances only over shared present views
    X_views = [np.asarray(X, dtype=float) for X in X_out]
    N = X_views[0].shape[0] if X_views else 0
    if N == 0:
        return (X_out, imputation_context) if return_context else X_out

    norm2_list = [np.sum(X_v ** 2, axis=1) for X_v in X_views]

    for v in range(num_views):
        mask_present = row_masks[v]
        X_v = X_views[v]
        if np.any(mask_present):
            present_mean = np.mean(X_v[mask_present], axis=0)
        else:
            present_mean = np.zeros(X_v.shape[1], dtype=float)

        # Fixed proxy neighborhoods are needed for every sample: missing rows
        # use them for initialization/U_d and all rows use them for U_s.
        for i in range(N):
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
                if not mask_present[i]:
                    X_v[i] = present_mean
                imputation_context[v][int(i)] = _build_proxy_context(
                    np.empty(0, dtype=np.int64),
                    np.empty(0, dtype=float),
                    tau,
                    coverage_threshold,
                    eps,
                )
                continue

            dist[valid] = dist[valid] / counts[valid]

            k_eff = int(min(k, np.sum(valid)))
            if k_eff <= 0:
                if not mask_present[i]:
                    X_v[i] = present_mean
                imputation_context[v][int(i)] = _build_proxy_context(
                    np.empty(0, dtype=np.int64),
                    np.empty(0, dtype=float),
                    tau,
                    coverage_threshold,
                    eps,
                )
                continue

            idx = np.argpartition(dist, k_eff - 1)[:k_eff]
            context = _build_proxy_context(
                idx, dist[idx], tau, coverage_threshold, eps
            )
            imputation_context[v][int(i)] = context
            if not mask_present[i]:
                X_v[i] = np.sum(
                    X_v[context["neighbor_indices"]]
                    * context["neighbor_weights"][:, None],
                    axis=0,
                )

        X_out[v] = X_v

    return (X_out, imputation_context) if return_context else X_out


def impute_sample_view_missing_mean(
    X_full_list: list[np.ndarray],
    mask_view_list: list[np.ndarray],
) -> list[np.ndarray]:
    """Initialize every missing sample-view row with its observed-view mean.

    This deliberately simple initializer is used only by the paper ablation
    ``w/o Proxy Initialization``.  It never reads the hidden missing-view
    values: the mean is computed exclusively from rows marked as observed in
    the same view.  Observed rows are copied without modification.
    """
    if len(X_full_list) != len(mask_view_list):
        raise ValueError("X_full_list and mask_view_list must have the same length.")

    initialized: list[np.ndarray] = []
    for X, mask in zip(X_full_list, mask_view_list):
        X_view = np.asarray(X, dtype=float).copy()
        present = _row_present_from_mask_and_data(X_view, mask)
        missing = ~present
        if np.any(present):
            observed_mean = np.mean(X_view[present], axis=0)
        else:
            observed_mean = np.zeros(X_view.shape[1], dtype=float)
        if np.any(missing):
            X_view[missing] = observed_mean
        initialized.append(X_view)
    return initialized


def generate_perturbed_recoveries(
    X_observed_zero_list: list[np.ndarray],
    normal_recovery_list: list[np.ndarray],
    mask_view_list: list[np.ndarray],
    imputation_context: list[dict[int, dict[str, np.ndarray | int]]],
    samples_B: int,
    seed: int,
    tau: float = 1.0,
    eps: float = 1e-12,
    perturbation_scale: float = 0.05,
) -> list[list[np.ndarray]]:
    """Sample B alternate recoveries without reading any missing-view value.

    Candidate pools and adaptive neighbor counts come directly from the normal
    proxy-neighbor imputation.  The normal recovery itself is never modified.
    """
    if samples_B < 2:
        raise ValueError("samples_B must be at least 2.")
    if not (
        len(X_observed_zero_list)
        == len(normal_recovery_list)
        == len(mask_view_list)
        == len(imputation_context)
    ):
        raise ValueError("All imputation lists must have the same number of views.")

    rng = np.random.default_rng(seed)
    recoveries_by_view: list[list[np.ndarray]] = [
        [] for _ in range(len(normal_recovery_list))
    ]
    scale = max(float(tau), eps)
    for _ in range(samples_B):
        for view, normal_recovery in enumerate(normal_recovery_list):
            observed_source = np.asarray(X_observed_zero_list[view], dtype=float)
            present_mask = np.asarray(mask_view_list[view], dtype=bool).reshape(-1)
            perturbed = np.asarray(normal_recovery, dtype=float).copy()
            for sample_index, context in imputation_context[view].items():
                # Contexts are also stored for observed rows so U_s can use
                # their fixed proxy neighborhoods. Only missing rows are
                # recovery perturbation targets.
                if present_mask[int(sample_index)]:
                    continue
                candidates = np.asarray(context["candidate_indices"], dtype=np.int64)
                distances = np.asarray(context["candidate_distances"], dtype=float)
                adaptive_k = int(context["adaptive_k"])
                if adaptive_k <= 0 or candidates.size == 0:
                    continue
                # The paper perturbs only the original adaptive neighborhood
                # N_{i,k_i}; candidates beyond k_i belong merely to the
                # pre-retained top-K0 pool and must never enter U_d sampling.
                pool_size = min(adaptive_k, candidates.size, distances.size)
                candidates = candidates[:pool_size]
                distances = distances[:pool_size]
                if not np.all(present_mask[candidates]):
                    raise RuntimeError("A perturbation candidate belongs to a missing view.")

                base_logits = -distances / scale
                base_logits -= np.max(base_logits)
                selection_prob = np.exp(base_logits)
                selection_prob /= np.sum(selection_prob)
                # A lightweight bootstrap produces plausible alternatives
                # while keeping every selected reference inside N_{i,k_i}.
                sample_size = candidates.size
                chosen_pos = rng.choice(
                    candidates.size,
                    size=sample_size,
                    replace=candidates.size > 1,
                    p=selection_prob,
                )
                chosen = candidates[chosen_pos]
                chosen_distances = distances[chosen_pos]
                weight_logits = -chosen_distances / scale
                if perturbation_scale > 0:
                    weight_logits += rng.normal(
                        loc=0.0, scale=perturbation_scale, size=weight_logits.shape
                    )
                weight_logits -= np.max(weight_logits)
                weights = np.exp(weight_logits)
                weights /= np.sum(weights)
                perturbed[int(sample_index)] = np.sum(
                    observed_source[chosen] * weights[:, None], axis=0
                )
            recoveries_by_view[view].append(perturbed)
    return recoveries_by_view


def semantic_refine_missing_views(
    X_current_list: list[np.ndarray],
    mask_view_list: list[np.ndarray],
    local_q_list: list[np.ndarray],
    local_z_list: list[np.ndarray],
    gate_list: list[np.ndarray] | None = None,
    eps: float = 1e-12,
) -> list[np.ndarray]:
    """Apply the paper's recursive cluster-aware semantic refinement."""
    view_count = len(X_current_list)
    if not (
        len(mask_view_list) == len(local_q_list) == len(local_z_list) == view_count
    ):
        raise ValueError("CAR inputs must have one entry per view.")
    if gate_list is not None and len(gate_list) != view_count:
        raise ValueError("gate_list must have one entry per view.")

    q_stack = np.stack([np.asarray(q) for q in local_q_list], axis=0)
    global_soft_assignment = np.mean(q_stack, axis=0)
    cluster_assign = np.argmax(global_soft_assignment, axis=1)
    refined_views = []
    for view in range(view_count):
        X_previous = np.asarray(X_current_list[view], dtype=float)
        X_refined = X_previous.copy()
        mask_present = np.asarray(mask_view_list[view], dtype=bool).reshape(-1)
        missing_rows = np.where(~mask_present)[0]
        if missing_rows.size == 0:
            refined_views.append(X_refined)
            continue

        present_idx = np.where(mask_present)[0]
        if present_idx.size == 0:
            refined_views.append(X_refined)
            continue
        present_mean = np.mean(X_previous[present_idx], axis=0)
        Z_view = np.asarray(local_z_list[view], dtype=float)
        Z_view = Z_view / (np.linalg.norm(Z_view, axis=1, keepdims=True) + eps)
        gates = None if gate_list is None else np.asarray(gate_list[view], dtype=float)

        for sample_index in missing_rows:
            same_cluster = cluster_assign == cluster_assign[sample_index]
            candidates = present_idx[same_cluster[present_idx]]
            if candidates.size == 0:
                x_semantic = present_mean
            else:
                similarity = np.dot(Z_view[candidates], Z_view[sample_index])
                similarity -= np.max(similarity)
                weights = np.exp(similarity)
                weight_sum = np.sum(weights)
                if weight_sum <= 0:
                    weights = np.ones_like(weights) / weights.size
                else:
                    weights /= weight_sum
                x_semantic = np.sum(X_previous[candidates] * weights[:, None], axis=0)

            gate = 1.0 if gates is None else float(np.clip(gates[sample_index], 0.0, 1.0))
            X_refined[sample_index] = (
                (1.0 - gate) * X_previous[sample_index] + gate * x_semantic
            )
        refined_views.append(X_refined)
    return refined_views
