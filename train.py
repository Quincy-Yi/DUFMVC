import torch
import torch.nn.functional as F
from torch.optim import Adam

from uncertainty import (
    apply_alignment_permutation,
    compute_alignment_permutation,
    compute_model_uncertainty,
    normalize_probabilities,
)

import warnings
warnings.filterwarnings("ignore")

def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def _uncertainty_focus_weights(sample_um, minimum_weight, eps):
    """Compatibility helper returning the paper's reliability 1-U_m."""
    if not 0.0 <= minimum_weight <= 1.0:
        raise ValueError("minimum_weight must be in [0, 1].")
    with torch.no_grad():
        clean = torch.clamp(
            torch.nan_to_num(sample_um.detach(), nan=1.0, posinf=1.0, neginf=0.0),
            min=0.0,
            max=1.0,
        )
        return 1.0 - clean


def _model_alignment_loss(
    q,
    p,
    args,
    alignment_permutation=None,
    structural_inconsistency=None,
    uncertainty_target=None,
):
    """Paper model loss: sum_i stopgrad(1-U_m,i) KL(P_i || Q_i)."""
    weighted_alignment_active = (
        getattr(args, "use_uncertainty_weighted_alignment", False)
        and getattr(args, "uncertainty_intervention_active", True)
    )
    if not weighted_alignment_active:
        return F.kl_div(q.log(), p, reduction='batchmean')
    if uncertainty_target is None:
        uncertainty_target = p
    if alignment_permutation is None:
        alignment_permutation = compute_alignment_permutation(
            q.detach(), uncertainty_target, args.uncertainty_eps
        )
    q_safe = apply_alignment_permutation(
        q, alignment_permutation, args.uncertainty_eps
    )
    sample_um = compute_model_uncertainty(
        local_q=q,
        previous_global_target=uncertainty_target,
        aligned_local_q=q_safe,
        eta_1=args.eta_1,
        eta_2=args.eta_2,
        eps=args.uncertainty_eps,
        structural_inconsistency=structural_inconsistency,
        eta_3=getattr(args, "eta_3", 0.0),
    )
    alignment_weight = torch.clamp(1.0 - sample_um.detach(), min=0.0, max=1.0)
    p_safe = normalize_probabilities(p, args.uncertainty_eps)
    per_sample_kl = F.kl_div(q_safe.log(), p_safe, reduction='none').sum(dim=1)
    return torch.sum(alignment_weight * per_sample_kl)


def _model_uncertainty_loss(q, p, args):
    """Deprecated compatibility shim; the paper has no separate U_m loss."""
    return q.sum() * 0.0

def train_gat(
    model,
    adj,
    adj_label,
    M,
    data,
    p,
    args,
    alignment_permutation=None,
    structural_inconsistency=None,
    uncertainty_target=None,
    return_diagnostics=False,
):
    # print(model)
    optimizer = Adam(model.parameters(), lr=args.lr_gat, weight_decay=args.weight_decay_gat)

    alignment_loss_history = []

    for epoch in range(args.local_epoch):
        model.train()
        A_pred, z, q = model(data, adj, M)
        kl_loss = _model_alignment_loss(
            q,
            p,
            args,
            alignment_permutation=alignment_permutation,
            structural_inconsistency=structural_inconsistency,
            uncertainty_target=uncertainty_target,
        )
        re_loss = F.binary_cross_entropy(A_pred.view(-1), adj_label.view(-1))
        loss = args.beta_gat * kl_loss + re_loss
        alignment_loss_history.append(float(kl_loss.detach().item()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if getattr(args, "use_model_uncertainty", False):
        model.eval()
        with torch.no_grad():
            _, z, q = model(data, adj, M)
        model.train()
    if return_diagnostics:
        diagnostics = {
            "model_alignment_loss": (
                sum(alignment_loss_history) / len(alignment_loss_history)
                if alignment_loss_history
                else 0.0
            )
        }
        return z, q, diagnostics
    return z, q

def train_gcn(
    model,
    adj,
    adj_label,
    data,
    p,
    args,
    alignment_permutation=None,
    structural_inconsistency=None,
    uncertainty_target=None,
    return_diagnostics=False,
):
    # print(model)
    optimizer = Adam(model.parameters(), lr=args.lr_gcn, weight_decay=args.weight_decay_gcn)

    alignment_loss_history = []

    for epoch in range(args.local_epoch):
        model.train()
        A_pred, z, q = model(data, adj)
        kl_loss = _model_alignment_loss(
            q,
            p,
            args,
            alignment_permutation=alignment_permutation,
            structural_inconsistency=structural_inconsistency,
            uncertainty_target=uncertainty_target,
        )
        re_loss = F.binary_cross_entropy(A_pred.view(-1), adj_label.view(-1))


        loss = args.beta_gcn * kl_loss + re_loss
        alignment_loss_history.append(float(kl_loss.detach().item()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if getattr(args, "use_model_uncertainty", False):
        model.eval()
        with torch.no_grad():
            _, z, q = model(data, adj)
        model.train()
    if return_diagnostics:
        diagnostics = {
            "model_alignment_loss": (
                sum(alignment_loss_history) / len(alignment_loss_history)
                if alignment_loss_history
                else 0.0
            )
        }
        return z, q, diagnostics
    return z, q
