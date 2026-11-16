"""Central configuration for the paper's reliability and structure ablations."""

from __future__ import annotations

from typing import Any


PAPER_ABLATION_CHOICES = (
    "none",
    "qaa_wo_ud",
    "qaa_wo_um",
    "qaa_wo_r",
    "average_aggregation",
    "wo_proxy_initialization",
    "wo_semantic_refinement",
)


def configure_paper_ablation(args: Any) -> Any:
    """Configure one paper-table ablation without duplicating training code."""
    ablation = str(getattr(args, "paper_ablation", "none"))
    if ablation not in PAPER_ABLATION_CHOICES:
        raise ValueError(f"Unknown paper ablation: {ablation}")
    if ablation != "none" and args.experiment_mode != "full":
        raise ValueError("--paper-ablation requires --experiment-mode full.")

    args.paper_qaa_use_r = ablation != "qaa_wo_r"
    args.paper_qaa_use_ud = ablation != "qaa_wo_ud"
    args.paper_qaa_use_um = ablation != "qaa_wo_um"
    args.force_uniform_aggregation = ablation == "average_aggregation"
    args.proxy_initialization = (
        "mean" if ablation == "wo_proxy_initialization" else "knn"
    )
    if ablation == "wo_semantic_refinement":
        args.iterative_impute = False

    # This label is retained only for compatibility with older run folders;
    # the paper method itself now starts in communication round 1.
    if ablation in {"wo_proxy_initialization", "wo_semantic_refinement"}:
        args.paper_ablation_warmup_group = ablation
    else:
        args.paper_ablation_warmup_group = "common"
    args.run_label = (
        args.experiment_mode
        if ablation == "none"
        else f"{args.experiment_mode}_{ablation}"
    )
    return args


def effective_qaa_coefficients(args: Any) -> tuple[float, float, float]:
    """Return the three active coefficients for the new reliability score."""
    lambda_r = args.lambda_r if args.paper_qaa_use_r else 0.0
    lambda_d = (
        args.lambda_d
        if args.use_data_uncertainty_reduction and args.paper_qaa_use_ud
        else 0.0
    )
    lambda_m = (
        args.lambda_m
        if args.use_model_uncertainty_reduction and args.paper_qaa_use_um
        else 0.0
    )
    return float(lambda_r), float(lambda_d), float(lambda_m)
