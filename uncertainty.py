"""Uncertainty utilities used by the original DU-FMVC training path.

The local model already returns Student-t soft assignments.  Consequently the
functions below normalize those probabilities for numerical safety, but never
apply an additional softmax.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


EXPERIMENT_MODES = {
    "original": (False, False, False, False, False),
    # Fair ablation baseline: measure both uncertainties but do not use either
    # one for recovery, alignment, or aggregation.
    "no_uncertainty": (True, True, False, False, False),
    # The single-uncertainty modes still measure both uncertainties.  Their
    # difference is which uncertainty is allowed to influence optimization.
    "ud_only": (True, True, True, False, True),
    "um_only": (True, True, False, True, True),
    "car_gate": (True, True, True, False, False),
    "weighted_alignment": (True, True, True, True, False),
    "full": (True, True, True, True, True),
}

EXPERIMENT_REDUCTION_FLAGS = {
    "original": (False, False),
    "no_uncertainty": (False, False),
    "ud_only": (True, False),
    "um_only": (False, True),
    "car_gate": (True, True),
    "weighted_alignment": (True, True),
    "full": (True, True),
}

UNCERTAINTY_SWITCHES = (
    "use_data_uncertainty",
    "use_model_uncertainty",
    "use_car_uncertainty_gate",
    "use_uncertainty_weighted_alignment",
    "use_uncertainty_qaa",
)


def configure_uncertainty_mode(args: Any) -> Any:
    """Apply a named ablation mode while allowing explicit switch overrides."""
    if args.experiment_mode not in EXPERIMENT_MODES:
        raise ValueError(f"Unknown experiment mode: {args.experiment_mode}")
    defaults = EXPERIMENT_MODES[args.experiment_mode]
    for name, enabled in zip(UNCERTAINTY_SWITCHES, defaults):
        if getattr(args, name, None) is None:
            setattr(args, name, enabled)
    reduce_data, reduce_model = EXPERIMENT_REDUCTION_FLAGS[args.experiment_mode]
    args.use_data_uncertainty_reduction = reduce_data
    args.use_model_uncertainty_reduction = reduce_model
    return args


def validate_uncertainty_config(args: Any) -> None:
    if args.uncertainty_samples_B < 2:
        raise ValueError("uncertainty_samples_B must be at least 2.")
    if args.uncertainty_eps <= 0:
        raise ValueError("uncertainty_eps must be positive.")
    if getattr(args, "qaa_eps", 1e-8) <= 0:
        raise ValueError("qaa_eps must be positive.")
    if args.qaa_temperature <= 0:
        raise ValueError("qaa_temperature must be positive.")
    qaa_weight_ema = float(getattr(args, "qaa_weight_ema", 0.0))
    if not 0.0 <= qaa_weight_ema < 1.0:
        raise ValueError("qaa_weight_ema must be in [0, 1).")
    if args.lambda_r < 0 or args.lambda_d < 0 or args.lambda_m < 0:
        raise ValueError("QAA lambda values must be non-negative.")
    if args.eta_1 < 0 or args.eta_2 < 0 or args.eta_3 < 0:
        raise ValueError("eta_1, eta_2, and eta_3 must be non-negative.")
    if not math.isclose(
        args.eta_1 + args.eta_2 + args.eta_3,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-8,
    ):
        raise ValueError("eta_1 + eta_2 + eta_3 must equal 1.")
    if args.uncertainty_perturbation_scale < 0:
        raise ValueError("uncertainty_perturbation_scale must be non-negative.")
    if args.use_car_uncertainty_gate:
        if args.experiment_mode == "ud_only" and not args.use_data_uncertainty:
            raise ValueError("Ud-only CAR gating requires U_d measurement.")
        if args.experiment_mode != "ud_only" and not (
            args.use_data_uncertainty and args.use_model_uncertainty
        ):
            raise ValueError("Dual-uncertainty CAR gating requires both U_d and U_m.")
    if args.use_uncertainty_weighted_alignment and not args.use_model_uncertainty:
        raise ValueError("Uncertainty-weighted alignment requires U_m.")
    if args.use_data_uncertainty_reduction and not args.use_data_uncertainty:
        raise ValueError("Data-uncertainty reduction requires U_d measurement.")
    if args.use_model_uncertainty_reduction and not args.use_model_uncertainty:
        raise ValueError("Model-uncertainty reduction requires U_m measurement.")


def normalize_probabilities(probabilities: torch.Tensor, eps: float) -> torch.Tensor:
    """Clamp and row-normalize an already probabilistic soft assignment."""
    p = torch.nan_to_num(probabilities.float(), nan=0.0, posinf=0.0, neginf=0.0)
    p = torch.clamp(p, min=eps)
    return p / torch.clamp(p.sum(dim=1, keepdim=True), min=eps)


def build_balanced_sharpened_target(
    probabilities: torch.Tensor,
    power: float = 2.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Build a confident, cluster-frequency-balanced pseudo target.

    ``power > 1`` sharpens each sample distribution, while the frequency
    correction prevents large clusters from absorbing every sample.  At
    ``power == 1`` the input distribution is returned unchanged.
    """
    if power < 1.0:
        raise ValueError("power must be at least 1.")
    q = normalize_probabilities(probabilities, eps)
    if math.isclose(power, 1.0, rel_tol=0.0, abs_tol=1e-12):
        return q
    cluster_frequency = torch.clamp(q.sum(dim=0), min=eps)
    weight = torch.pow(q, power) / torch.pow(
        cluster_frequency, power - 1.0
    )
    return normalize_probabilities(weight, eps)


def align_local_distribution(
    local_q: torch.Tensor,
    previous_global_target: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Align local cluster columns to global columns with the existing update_A rule."""
    if local_q.shape != previous_global_target.shape:
        raise ValueError("local_q and previous_global_target must have the same shape.")
    q = normalize_probabilities(local_q.detach(), eps)
    permutation = compute_alignment_permutation(
        q, previous_global_target, eps
    )
    return apply_alignment_permutation(q, permutation, eps)


def compute_alignment_permutation(
    local_q: torch.Tensor,
    previous_global_target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Estimate the once-per-round local-to-global Hungarian permutation."""
    if local_q.shape != previous_global_target.shape:
        raise ValueError("local_q and previous_global_target must have the same shape.")
    q = normalize_probabilities(local_q.detach(), eps)
    p = normalize_probabilities(previous_global_target.detach().to(q.device), eps)
    cluster_count = q.shape[1]
    # Soft overlap matches the paper's assignment-distribution alignment.
    # Rows are local clusters and columns are global clusters.
    overlap = q.transpose(0, 1) @ p
    overlap_np = overlap.detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(
        np.max(overlap_np) - overlap_np
    )
    permutation = np.zeros((cluster_count, cluster_count), dtype=np.float32)
    permutation[row_ind, col_ind] = 1.0
    return torch.as_tensor(permutation, dtype=q.dtype, device=q.device)


def apply_alignment_permutation(
    local_q: torch.Tensor,
    permutation: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Apply a fixed round-level permutation while preserving live gradients."""
    q = normalize_probabilities(local_q, eps)
    permutation = permutation.to(device=q.device, dtype=q.dtype)
    if permutation.shape != (q.shape[1], q.shape[1]):
        raise ValueError("permutation must have shape [K, K].")
    return normalize_probabilities(q @ permutation, eps)


def align_local_distribution_differentiable(
    local_q: torch.Tensor,
    previous_global_target: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Align cluster columns while preserving gradients with respect to ``local_q``.

    The discrete Hungarian permutation is estimated from detached predictions,
    but the selected permutation is applied to the live probability tensor.
    """
    if local_q.shape != previous_global_target.shape:
        raise ValueError("local_q and previous_global_target must have the same shape.")
    q_live = normalize_probabilities(local_q, eps)
    permutation = compute_alignment_permutation(
        q_live.detach(), previous_global_target, eps
    )
    return apply_alignment_permutation(q_live, permutation, eps)


def compute_data_uncertainty(
    perturbed_recoveries: list[torch.Tensor | np.ndarray],
    model: torch.nn.Module,
    graph_or_forward_context: dict[str, torch.Tensor],
    missing_mask: torch.Tensor | np.ndarray,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute sample-level generalized JSD across perturbed recoveries."""
    if not perturbed_recoveries:
        raise ValueError("perturbed_recoveries must not be empty.")

    first = torch.as_tensor(perturbed_recoveries[0])
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = first.device
    sample_count = first.shape[0]
    missing = torch.as_tensor(missing_mask, dtype=torch.bool, device=device).reshape(-1)
    if missing.numel() != sample_count:
        raise ValueError("missing_mask length must match the recovery row count.")
    if len(perturbed_recoveries) < 2:
        return torch.zeros(sample_count, dtype=torch.float32, device=device)

    was_training = model.training
    predictions = []
    try:
        model.eval()
        with torch.no_grad():
            for recovery in perturbed_recoveries:
                x = torch.as_tensor(recovery, dtype=torch.float32, device=device)
                if x.shape[0] != sample_count:
                    raise ValueError("All perturbed recoveries must have the same row count.")
                adj = graph_or_forward_context["adj"]
                if "M" in graph_or_forward_context:
                    output = model(x, adj, graph_or_forward_context["M"])
                else:
                    output = model(x, adj)
                q = output[-1]
                predictions.append(normalize_probabilities(q, eps))
    finally:
        model.train(was_training)

    q_samples = torch.stack(predictions, dim=0)
    q_bar = normalize_probabilities(q_samples.mean(dim=0), eps)
    entropy_bar = -(q_bar * torch.log(q_bar)).sum(dim=1)
    sample_entropies = -(q_samples * torch.log(q_samples)).sum(dim=2)
    generalized_jsd = (entropy_bar - sample_entropies.mean(dim=0)) / math.log(
        float(len(predictions))
    )
    sample_ud = torch.clamp(generalized_jsd, min=0.0, max=1.0)
    sample_ud = torch.where(missing, sample_ud, torch.zeros_like(sample_ud))
    return sample_ud.detach()


def compute_structural_inconsistency(
    previous_global_target: torch.Tensor,
    proxy_context: dict[int, dict[str, np.ndarray | int]],
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute U_s from fixed original-feature proxy neighborhoods.

    Each context entry contains the adaptive neighbor indices and normalized
    proxy weights created before graph construction. Samples without a valid
    proxy neighbor receive the conservative paper value U_s=1.
    """
    p = normalize_probabilities(previous_global_target.detach(), eps)
    sample_count = p.shape[0]
    structural = torch.ones(
        sample_count, dtype=p.dtype, device=p.device
    )
    log_two = math.log(2.0)

    for sample_index in range(sample_count):
        context = proxy_context.get(sample_index)
        if context is None:
            continue
        neighbor_indices = np.asarray(
            context.get("neighbor_indices", np.empty(0, dtype=np.int64)),
            dtype=np.int64,
        ).reshape(-1)
        neighbor_weights = np.asarray(
            context.get("neighbor_weights", np.empty(0, dtype=float)),
            dtype=float,
        ).reshape(-1)

        # Backward-compatible reconstruction for old cached/test contexts.
        if neighbor_indices.size == 0:
            adaptive_k = int(context.get("adaptive_k", 0))
            candidates = np.asarray(
                context.get("candidate_indices", np.empty(0, dtype=np.int64)),
                dtype=np.int64,
            ).reshape(-1)
            distances = np.asarray(
                context.get("candidate_distances", np.empty(0, dtype=float)),
                dtype=float,
            ).reshape(-1)
            usable = min(adaptive_k, candidates.size, distances.size)
            if usable > 0:
                neighbor_indices = candidates[:usable]
                logits = -distances[:usable]
                logits -= np.max(logits)
                neighbor_weights = np.exp(logits)

        valid = (
            (neighbor_indices >= 0)
            & (neighbor_indices < sample_count)
            & np.isfinite(neighbor_weights)
            & (neighbor_weights >= 0)
        )
        neighbor_indices = neighbor_indices[valid]
        neighbor_weights = neighbor_weights[valid]
        weight_sum = float(np.sum(neighbor_weights))
        if neighbor_indices.size == 0 or weight_sum <= eps:
            continue
        neighbor_weights = neighbor_weights / weight_sum

        neighbor_tensor = torch.as_tensor(
            neighbor_indices, dtype=torch.long, device=p.device
        )
        weights_tensor = torch.as_tensor(
            neighbor_weights, dtype=p.dtype, device=p.device
        )
        p_i = p[sample_index].unsqueeze(0).expand(neighbor_tensor.numel(), -1)
        p_j = p[neighbor_tensor]
        mixture = normalize_probabilities(0.5 * (p_i + p_j), eps)
        kl_i = (p_i * (torch.log(p_i) - torch.log(mixture))).sum(dim=1)
        kl_j = (p_j * (torch.log(p_j) - torch.log(mixture))).sum(dim=1)
        jsd = torch.clamp(
            0.5 * (kl_i + kl_j) / log_two, min=0.0, max=1.0
        )
        structural[sample_index] = torch.sum(weights_tensor * jsd)

    return torch.clamp(structural, min=0.0, max=1.0).detach()


def compute_model_uncertainty(
    local_q: torch.Tensor,
    previous_global_target: torch.Tensor,
    aligned_local_q: torch.Tensor | None,
    eta_1: float,
    eta_2: float,
    eps: float = 1e-8,
    structural_inconsistency: torch.Tensor | np.ndarray | None = None,
    eta_3: float = 0.0,
) -> torch.Tensor:
    """Measure target ambiguity, aligned disagreement, and structure conflict.

    This is the model uncertainty defined in the paper: the entropy term is
    computed from the *previous global target* P^(t-1), while the conflict term
    is the Jensen--Shannon divergence between the aligned local assignment and
    that target.
    """
    if not math.isclose(
        eta_1 + eta_2 + eta_3, 1.0, rel_tol=0.0, abs_tol=1e-8
    ):
        raise ValueError("eta_1 + eta_2 + eta_3 must equal 1.")
    p_prev = normalize_probabilities(previous_global_target.detach(), eps)
    if aligned_local_q is None:
        aligned_q = align_local_distribution(local_q, p_prev, eps)
    else:
        aligned_q = normalize_probabilities(aligned_local_q.detach().to(p_prev.device), eps)
    if aligned_q.shape != p_prev.shape:
        raise ValueError("aligned_local_q and previous_global_target must have the same shape.")

    cluster_count = p_prev.shape[1]
    if cluster_count < 2:
        return torch.zeros(p_prev.shape[0], dtype=p_prev.dtype, device=p_prev.device)
    target_entropy = -(
        p_prev * torch.log(p_prev)
    ).sum(dim=1) / math.log(
        float(cluster_count)
    )
    mixture = normalize_probabilities(0.5 * (aligned_q + p_prev), eps)
    kl_local = (aligned_q * (torch.log(aligned_q) - torch.log(mixture))).sum(dim=1)
    kl_global = (p_prev * (torch.log(p_prev) - torch.log(mixture))).sum(dim=1)
    local_global_conflict = 0.5 * (kl_local + kl_global) / math.log(2.0)
    if structural_inconsistency is None:
        if eta_3 > 0:
            raise ValueError("structural_inconsistency is required when eta_3 > 0.")
        structural = torch.zeros_like(target_entropy)
    else:
        structural = torch.as_tensor(
            structural_inconsistency, dtype=p_prev.dtype, device=p_prev.device
        ).reshape(-1)
        if structural.numel() != p_prev.shape[0]:
            raise ValueError("structural_inconsistency must contain N values.")
        structural = torch.clamp(
            torch.nan_to_num(structural, nan=1.0, posinf=1.0, neginf=0.0),
            min=0.0,
            max=1.0,
        )
    sample_um = (
        eta_1 * target_entropy
        + eta_2 * local_global_conflict
        + eta_3 * structural
    )
    return torch.clamp(sample_um, min=0.0, max=1.0).detach()


def compute_differentiable_model_uncertainty(
    local_q: torch.Tensor,
    previous_global_target: torch.Tensor,
    eta_1: float,
    eta_2: float,
    eps: float = 1e-8,
    structural_inconsistency: torch.Tensor | np.ndarray | None = None,
    eta_3: float = 0.0,
) -> torch.Tensor:
    """Return the differentiable model-uncertainty proxy used by local training."""
    if not math.isclose(
        eta_1 + eta_2 + eta_3, 1.0, rel_tol=0.0, abs_tol=1e-8
    ):
        raise ValueError("eta_1 + eta_2 + eta_3 must equal 1.")
    q = normalize_probabilities(local_q, eps)
    p = normalize_probabilities(previous_global_target.detach().to(q.device), eps)
    if q.shape != p.shape:
        raise ValueError("local_q and previous_global_target must have the same shape.")
    cluster_count = q.shape[1]
    if cluster_count < 2:
        return torch.zeros(q.shape[0], dtype=q.dtype, device=q.device)

    aligned_q = align_local_distribution_differentiable(q, p, eps)
    target_entropy = -(p * torch.log(p)).sum(dim=1) / math.log(
        float(cluster_count)
    )
    mixture = normalize_probabilities(0.5 * (aligned_q + p), eps)
    kl_local = (aligned_q * (torch.log(aligned_q) - torch.log(mixture))).sum(dim=1)
    kl_global = (p * (torch.log(p) - torch.log(mixture))).sum(dim=1)
    local_global_conflict = 0.5 * (kl_local + kl_global) / math.log(2.0)
    if structural_inconsistency is None:
        if eta_3 > 0:
            raise ValueError("structural_inconsistency is required when eta_3 > 0.")
        structural = torch.zeros_like(target_entropy)
    else:
        structural = torch.as_tensor(
            structural_inconsistency, dtype=q.dtype, device=q.device
        ).reshape(-1)
        if structural.numel() != q.shape[0]:
            raise ValueError("structural_inconsistency must contain N values.")
    return torch.clamp(
        eta_1 * target_entropy
        + eta_2 * local_global_conflict
        + eta_3 * structural,
        min=0.0,
        max=1.0,
    )


def compute_assignment_margin(
    local_q: torch.Tensor | np.ndarray,
) -> float:
    """Return the paper's client discriminability score R.

    R is the mean gap between the largest and second-largest cluster
    probabilities for each sample.
    """
    q = (
        local_q.detach().cpu().numpy()
        if torch.is_tensor(local_q)
        else np.asarray(local_q)
    )
    if q.ndim != 2 or q.shape[1] < 2:
        raise ValueError("local_q must have shape [N, K] with K >= 2.")
    q = np.nan_to_num(q.astype(np.float64, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    q = np.maximum(q, 0.0)
    row_sum = q.sum(axis=1, keepdims=True)
    q = q / np.maximum(row_sum, 1e-12)
    top_two = np.partition(q, kth=q.shape[1] - 2, axis=1)[:, -2:]
    return float(np.mean(np.max(top_two, axis=1) - np.min(top_two, axis=1)))


def compute_consensus_reliability(
    aligned_local_q_list: list[torch.Tensor],
    previous_global_target: torch.Tensor,
    observed_mask_list: list[torch.Tensor | np.ndarray] | None = None,
    confidence_threshold: float = 0.25,
    agreement_threshold: float = 0.5,
    reliability_floor: float = 0.05,
    eps: float = 1e-12,
) -> dict[str, torch.Tensor]:
    """Estimate label-free sample reliability from view and temporal agreement.

    Every local distribution must already be aligned to the cluster columns of
    ``previous_global_target``. Reliability is high only when the available
    views form a confident consensus and that consensus agrees with the
    previous global target. No ground-truth label is used.
    """
    if not aligned_local_q_list:
        raise ValueError("aligned_local_q_list must not be empty.")
    if not 0.0 <= confidence_threshold < 1.0:
        raise ValueError("confidence_threshold must be in [0, 1).")
    if not 0.0 <= agreement_threshold < 1.0:
        raise ValueError("agreement_threshold must be in [0, 1).")
    if not 0.0 <= reliability_floor <= 1.0:
        raise ValueError("reliability_floor must be in [0, 1].")

    first = aligned_local_q_list[0]
    if first.ndim != 2:
        raise ValueError("Each aligned local distribution must have shape [N, K].")
    device = first.device
    shape = first.shape
    normalized_views = []
    for local_q in aligned_local_q_list:
        if local_q.shape != shape:
            raise ValueError("All aligned local distributions must share shape [N, K].")
        normalized_views.append(
            normalize_probabilities(local_q.detach().to(device), eps)
        )
    q_stack = torch.stack(normalized_views, dim=0)
    view_count, sample_count, cluster_count = q_stack.shape
    if cluster_count < 2:
        ones = torch.ones(sample_count, dtype=q_stack.dtype, device=device)
        return {
            "reliability": ones,
            "consensus": q_stack.mean(dim=0),
            "confidence": ones,
            "agreement": ones,
            "temporal_similarity": ones,
        }

    if observed_mask_list is None:
        observed = torch.ones(
            (view_count, sample_count), dtype=torch.bool, device=device
        )
    else:
        if len(observed_mask_list) != view_count:
            raise ValueError("observed_mask_list must contain one mask per view.")
        masks = []
        for mask in observed_mask_list:
            mask_tensor = torch.as_tensor(
                mask, dtype=torch.bool, device=device
            ).reshape(-1)
            if mask_tensor.numel() != sample_count:
                raise ValueError("Every observed mask must contain N entries.")
            masks.append(mask_tensor)
        observed = torch.stack(masks, dim=0)

    observed_float = observed.to(q_stack.dtype)
    observed_count = observed_float.sum(dim=0)
    consensus_sum = torch.sum(q_stack * observed_float[:, :, None], dim=0)
    consensus = consensus_sum / torch.clamp(
        observed_count[:, None], min=1.0
    )
    no_observed_view = observed_count <= 0
    if torch.any(no_observed_view):
        consensus[no_observed_view] = q_stack[:, no_observed_view].mean(dim=0)
    consensus = normalize_probabilities(consensus, eps)

    max_probability = torch.max(consensus, dim=1).values
    uniform_probability = 1.0 / float(cluster_count)
    confidence = torch.clamp(
        (max_probability - uniform_probability)
        / max(1.0 - uniform_probability, eps),
        min=0.0,
        max=1.0,
    )

    consensus_label = torch.argmax(consensus, dim=1)
    view_labels = torch.argmax(q_stack, dim=2)
    agreement_count = torch.sum(
        (view_labels == consensus_label[None, :]).to(q_stack.dtype)
        * observed_float,
        dim=0,
    )
    agreement = agreement_count / torch.clamp(observed_count, min=1.0)
    if torch.any(no_observed_view):
        agreement[no_observed_view] = (
            view_labels[:, no_observed_view]
            == consensus_label[None, no_observed_view]
        ).to(q_stack.dtype).mean(dim=0)

    previous = normalize_probabilities(
        previous_global_target.detach().to(device), eps
    )
    if previous.shape != consensus.shape:
        raise ValueError("previous_global_target must have shape [N, K].")
    mixture = normalize_probabilities(0.5 * (consensus + previous), eps)
    kl_consensus = (
        consensus * (torch.log(consensus) - torch.log(mixture))
    ).sum(dim=1)
    kl_previous = (
        previous * (torch.log(previous) - torch.log(mixture))
    ).sum(dim=1)
    temporal_jsd = torch.clamp(
        0.5 * (kl_consensus + kl_previous) / math.log(2.0),
        min=0.0,
        max=1.0,
    )
    temporal_similarity = 1.0 - temporal_jsd

    confidence_gate = torch.clamp(
        (confidence - confidence_threshold)
        / max(1.0 - confidence_threshold, eps),
        min=0.0,
        max=1.0,
    )
    agreement_gate = torch.clamp(
        (agreement - agreement_threshold)
        / max(1.0 - agreement_threshold, eps),
        min=0.0,
        max=1.0,
    )
    evidence = torch.sqrt(
        torch.clamp(confidence_gate * agreement_gate, min=0.0, max=1.0)
    )
    reliability = reliability_floor + (1.0 - reliability_floor) * (
        evidence * temporal_similarity
    )
    reliability = torch.clamp(reliability, min=0.0, max=1.0).detach()
    return {
        "reliability": reliability,
        "consensus": consensus.detach(),
        "confidence": confidence.detach(),
        "agreement": agreement.detach(),
        "temporal_similarity": temporal_similarity.detach(),
    }


def data_uncertainty_reference_scale(
    sample_ud: torch.Tensor | np.ndarray,
    missing_mask: torch.Tensor | np.ndarray,
    quantile: float = 0.95,
    eps: float = 1e-12,
) -> float:
    """Calibrate one view's raw data uncertainty from warm-up missing samples."""
    if not 0.0 < quantile <= 1.0:
        raise ValueError("quantile must be in (0, 1].")
    values = (
        sample_ud.detach().cpu().numpy()
        if torch.is_tensor(sample_ud)
        else np.asarray(sample_ud)
    )
    missing = (
        missing_mask.detach().cpu().numpy()
        if torch.is_tensor(missing_mask)
        else np.asarray(missing_mask)
    )
    values = np.asarray(values, dtype=float).reshape(-1)
    missing = np.asarray(missing, dtype=bool).reshape(-1)
    if values.size != missing.size:
        raise ValueError("sample_ud and missing_mask must have the same length.")
    selected = values[missing]
    selected = selected[np.isfinite(selected)]
    if selected.size == 0:
        return 1.0
    scale = float(np.quantile(np.maximum(selected, 0.0), quantile))
    if not np.isfinite(scale) or scale <= eps:
        scale = float(np.max(np.maximum(selected, 0.0)))
    return max(scale, eps)


def build_balanced_recovery_gate(
    sample_ud: torch.Tensor,
    sample_um: torch.Tensor | None,
    missing_mask: torch.Tensor | np.ndarray,
    reference_scale: float,
    base_strength: float,
    gate_min: float,
    gate_max: float,
    use_model_confidence: bool,
    eps: float = 1e-12,
    sample_reliability: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a scale-safe CAR gate with matched mean recovery strength.

    The raw ``U_d`` is first divided by its warm-up reference.  The resulting
    priority is mean-normalized over missing rows so uncertainty-aware modes
    redistribute, rather than silently remove, the recovery update.
    """
    if not 0.0 <= gate_min <= gate_max <= 1.0:
        raise ValueError("gate bounds must satisfy 0 <= min <= max <= 1.")
    if not gate_min <= base_strength <= gate_max:
        raise ValueError(
            "base_strength must lie inside [gate_min, gate_max] so the "
            "requested mean recovery strength is feasible."
        )
    if reference_scale <= 0 or not np.isfinite(reference_scale):
        raise ValueError("reference_scale must be positive and finite.")

    ud = torch.nan_to_num(
        sample_ud.detach().float(), nan=0.0, posinf=reference_scale, neginf=0.0
    )
    missing = torch.as_tensor(
        missing_mask, dtype=torch.bool, device=ud.device
    ).reshape(-1)
    if ud.numel() != missing.numel():
        raise ValueError("sample_ud and missing_mask must have the same length.")

    normalized_ud = torch.clamp(ud / float(reference_scale), 0.0, 1.0)
    priority = eps + normalized_ud
    if sample_reliability is not None:
        reliability = torch.nan_to_num(
            sample_reliability.detach().to(ud.device).float(),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )
        if reliability.numel() != ud.numel():
            raise ValueError("sample_reliability and sample_ud must have the same length.")
        priority = priority * torch.clamp(reliability, min=eps, max=1.0)
    if use_model_confidence:
        if sample_um is None:
            raise ValueError("sample_um is required for dual-uncertainty gating.")
        um = torch.nan_to_num(
            sample_um.detach().to(ud.device).float(), nan=1.0, posinf=1.0, neginf=0.0
        )
        if um.numel() != ud.numel():
            raise ValueError("sample_um and sample_ud must have the same length.")
        priority = priority * torch.clamp(1.0 - um, min=eps, max=1.0)

    gate = torch.zeros_like(ud)
    if torch.any(missing):
        missing_priority = priority[missing]
        mean_priority = torch.clamp(missing_priority.mean(), min=eps)
        balanced = base_strength * missing_priority / mean_priority
        gate[missing] = _project_to_bounded_mean(
            balanced,
            target_mean=base_strength,
            lower=gate_min,
            upper=gate_max,
        )
    return gate.detach(), normalized_ud.detach()


def _project_to_bounded_mean(
    values: torch.Tensor,
    target_mean: float,
    lower: float,
    upper: float,
    iterations: int = 80,
) -> torch.Tensor:
    """Preserve ranking while enforcing bounds and an exact target mean.

    A direct clamp changes the average intervention strength and makes an
    uncertainty-aware ablation unfair to the constant-gate baseline.  This
    bounded shift projection keeps the requested mean after clipping.
    """
    if values.numel() == 0:
        return values
    if not lower <= target_mean <= upper:
        raise ValueError("target_mean must lie inside [lower, upper].")

    work = torch.nan_to_num(
        values.detach().double(),
        nan=target_mean,
        posinf=upper,
        neginf=lower,
    )
    shift_low = float(lower) - float(torch.max(work).item())
    shift_high = float(upper) - float(torch.min(work).item())
    for _ in range(iterations):
        shift_mid = 0.5 * (shift_low + shift_high)
        candidate_mean = torch.clamp(
            work + shift_mid, min=lower, max=upper
        ).mean()
        if float(candidate_mean.item()) < target_mean:
            shift_low = shift_mid
        else:
            shift_high = shift_mid
    projected = torch.clamp(
        work + 0.5 * (shift_low + shift_high), min=lower, max=upper
    )
    return projected.to(dtype=values.dtype, device=values.device)


def build_constant_recovery_gate(
    sample_count: int,
    missing_mask: torch.Tensor | np.ndarray,
    base_strength: float,
    device: torch.device | str,
) -> torch.Tensor:
    """Return the common non-uncertainty recovery coefficient for one view."""
    if not 0.0 < base_strength <= 1.0:
        raise ValueError("base_strength must be in (0, 1].")
    missing = torch.as_tensor(missing_mask, dtype=torch.bool, device=device).reshape(-1)
    if missing.numel() != int(sample_count):
        raise ValueError("missing_mask length must match sample_count.")
    gate = torch.zeros(sample_count, dtype=torch.float32, device=device)
    gate[missing] = float(base_strength)
    return gate


def _safe_minmax(
    values: np.ndarray,
    eps: float,
    metric_name: str,
    logger: logging.Logger | None,
) -> tuple[np.ndarray, bool]:
    """Return the exact cross-client min-max normalization in the paper."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    invalid = not np.all(np.isfinite(values))
    if invalid:
        message = f"[QAA warning] {metric_name} contains NaN/Inf; replacing with zero."
        (logger.warning if logger is not None else logging.warning)(message)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    value_range = float(values.max() - values.min()) if values.size else 0.0
    if value_range < eps:
        return np.zeros_like(values), invalid
    normalized = (values - values.min()) / (value_range + eps)
    return np.clip(normalized, 0.0, 1.0), invalid


def smooth_qaa_weights(
    candidate_weights: np.ndarray,
    previous_weights: np.ndarray,
    ema: float,
    eps: float = 1e-12,
) -> np.ndarray:
    """Smooth QAA weights across rounds and keep them on the probability simplex."""
    if not 0.0 <= ema < 1.0:
        raise ValueError("ema must be in [0, 1).")
    candidate = np.asarray(candidate_weights, dtype=np.float64).reshape(-1)
    previous = np.asarray(previous_weights, dtype=np.float64).reshape(-1)
    if candidate.shape != previous.shape or candidate.size == 0:
        raise ValueError("candidate_weights and previous_weights must match and be non-empty.")
    candidate = np.maximum(
        np.nan_to_num(candidate, nan=0.0, posinf=0.0, neginf=0.0), 0.0
    )
    previous = np.maximum(
        np.nan_to_num(previous, nan=0.0, posinf=0.0, neginf=0.0), 0.0
    )
    candidate_total = float(candidate.sum())
    previous_total = float(previous.sum())
    if candidate_total <= eps:
        candidate = np.full(candidate.shape, 1.0 / candidate.size)
    else:
        candidate /= candidate_total
    if previous_total <= eps:
        previous = np.full(previous.shape, 1.0 / previous.size)
    else:
        previous /= previous_total
    smoothed = ema * previous + (1.0 - ema) * candidate
    return smoothed / float(smoothed.sum())


def compute_uncertainty_qaa_weights(
    r_raw: np.ndarray,
    u_d_raw: np.ndarray,
    u_m_raw: np.ndarray,
    lambda_r: float,
    lambda_d: float,
    lambda_m: float,
    temperature: float,
    eps: float = 1e-12,
    logger: logging.Logger | None = None,
) -> dict[str, np.ndarray | float | bool]:
    """Compute stable additive QAA scores and softmax client weights."""
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    r_norm, bad_r = _safe_minmax(r_raw, eps, "R_raw", logger)
    ud_norm, bad_ud = _safe_minmax(u_d_raw, eps, "U_d_raw", logger)
    um_norm, bad_um = _safe_minmax(u_m_raw, eps, "U_m_raw", logger)
    if not (r_norm.shape == ud_norm.shape == um_norm.shape):
        raise ValueError("R, U_d, and U_m must contain one value per client.")

    score = lambda_r * r_norm - lambda_d * ud_norm - lambda_m * um_norm
    score = np.nan_to_num(score, nan=0.0, posinf=lambda_r, neginf=-(lambda_d + lambda_m))
    scaled_score = score / float(temperature)
    scaled_score = scaled_score - np.max(scaled_score)
    exp_score = np.exp(scaled_score)
    denominator = float(np.sum(exp_score))
    if not np.isfinite(denominator) or denominator <= 0:
        if logger is not None:
            logger.warning("[QAA warning] invalid softmax denominator; using uniform weights.")
        omega = np.full(score.shape, 1.0 / max(score.size, 1), dtype=np.float64)
        invalid = True
    else:
        omega = exp_score / denominator
        invalid = bad_r or bad_ud or bad_um or not np.all(np.isfinite(score))
    return {
        "R_norm": r_norm,
        "U_d_norm": ud_norm,
        "U_m_norm": um_norm,
        "score": score,
        "omega": omega,
        "omega_sum": float(np.sum(omega)),
        "has_invalid": bool(invalid or not np.all(np.isfinite(omega))),
    }


def tensor_stats(values: torch.Tensor | np.ndarray) -> tuple[float, float, float]:
    array = values.detach().cpu().numpy() if torch.is_tensor(values) else np.asarray(values)
    if array.size == 0:
        return 0.0, 0.0, 0.0
    return float(np.mean(array)), float(np.min(array)), float(np.max(array))
