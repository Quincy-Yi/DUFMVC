import argparse
import torch
import numpy as np
from gpu_kmeans import GPUKMeans
from evaluation import eva
from model import GAT, DAEGC, GCN, GCNED
from pretrain import pretrain_gcn, pretrain_gat
from train import train_gat, train_gcn
from utils import update_A, update_P
from generate_incomplete_data import generate_data
from experiment_recorder import UncertaintyExperimentRecorder
from paper_ablations import (
    PAPER_ABLATION_CHOICES,
    configure_paper_ablation,
    effective_qaa_coefficients,
)
from utils_impute import generate_perturbed_recoveries, semantic_refine_missing_views
from uncertainty import (
    apply_alignment_permutation,
    compute_alignment_permutation,
    compute_assignment_margin,
    compute_data_uncertainty,
    compute_model_uncertainty,
    compute_structural_inconsistency,
    compute_uncertainty_qaa_weights,
    configure_uncertainty_mode,
    validate_uncertainty_config,
)
from sklearn.preprocessing import normalize
import random
import logging
import os
from datetime import datetime
import shutil
import json
import math

def setup_seed(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

def get_M(adj):
    adj_numpy = np.array(adj.cpu())
    # t_order
    t=2
    tran_prob = normalize(adj_numpy, norm="l1", axis=0)
    M_numpy = sum([np.linalg.matrix_power(tran_prob, i) for i in range(1, t + 1)]) / t
    return torch.Tensor(M_numpy)


def average_predictions(q, incomplete_indices, num_views, k, n):
    averaged_result = np.zeros((n, k))
    count = np.zeros(n)
    for v in range(num_views):
        qv = q[v]
        indices = incomplete_indices[v]
        for i, idx in enumerate(indices):
            averaged_result[idx] += qv[i]
            count[idx] += 1
    for i in range(n):
        if count[i] > 0:
            averaged_result[i] /= count[i]
    predicted_classes = np.argmax(averaged_result, axis=1)
    return predicted_classes

def compute_zeta_from_entropy(p_global, num_clusters, zeta_min, zeta_max, eps=1e-8):
    if p_global.ndim != 2:
        raise ValueError("p_global must be a 2D tensor with shape [N, K].")
    p = p_global.float()
    p = torch.clamp(p, min=0.0)
    row_sum = torch.sum(p, dim=1, keepdim=True)
    p = p / (row_sum + eps)
    p = torch.clamp(p, min=eps, max=1.0)
    log_p = torch.log(p + eps)
    entropy = -torch.sum(p * log_p, dim=1)
    log_k = math.log(float(num_clusters))
    h_norm = entropy / log_k
    h_norm = torch.clamp(h_norm, 0.0, 1.0)
    conf = 1.0 - h_norm
    mean_conf = float(torch.mean(conf).item()) if conf.numel() > 0 else 0.0
    zeta = float(zeta_min + (zeta_max - zeta_min) * mean_conf)
    if zeta < zeta_min:
        zeta = float(zeta_min)
    elif zeta > zeta_max:
        zeta = float(zeta_max)
    return zeta

def get_weights_safe(center, view):
    weights = get_discriminative_scores(center, view)
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0:
        return np.ones(view, dtype=float)
    weights = 1 + np.log2(1 + weights / total)
    return weights


def get_discriminative_scores(center, view):
    """Original positive QAA separability statistic: centroid variance per client."""
    return np.asarray([np.var(center[i]) for i in range(view)], dtype=float)


def forward_client(model, data, adj, M=None):
    """Run the existing client model and return its embedding and soft assignment."""
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            if isinstance(model, DAEGC):
                _, z, q = model(data, adj, M)
            elif isinstance(model, GCNED):
                _, z, q = model(data, adj)
            else:
                raise TypeError(f"Unsupported client model: {type(model).__name__}")
    finally:
        model.train(was_training)
    return z.detach(), q.detach()


def normalize_global_target(p_global, eps):
    p_global = np.maximum(p_global, 0.0)
    return p_global / (np.sum(p_global, axis=1, keepdims=True) + eps)


_WARMUP_CHECKPOINT_VERSION = 5


def _warmup_signature(args):
    """Return the mode-independent configuration guarded by a warm-up checkpoint."""
    fields = (
        "name",
        "missrates",
        "n_clusters",
        "view_num",
        "training_seed",
        "feat_miss_seed",
        "uncertainty_seed",
        "kmeans_seed",
        "max_epoch_pretrain",
        "local_epoch",
        "impute_warmup",
        "impute_every",
        "impute_eta",
        "impute_k",
        "impute_ref_mode",
        "knn_neighbor",
        "knn_top_ratio",
        "knn_sigma",
        "uncertainty_samples_B",
        "uncertainty_perturbation_scale",
        "ud_calibration_quantile",
        "reliability_confidence_threshold",
        "reliability_agreement_threshold",
        "reliability_floor",
        "reliable_neighbor_threshold",
        "eta_1",
        "eta_2",
        "eta_3",
        "hidden_size_gat",
        "embedding_size_gat",
        "hidden_size_gcn",
        "embedding_size_gcn",
        "lr_pretrain_gat",
        "lr_pretrain_gcn",
        "lr_gat",
        "lr_gcn",
        "weight_decay_gat",
        "weight_decay_gcn",
        "beta_gat",
        "beta_gcn",
        "zeta_min",
        "zeta_max",
        "paper_ablation_warmup_group",
    )
    signature = {"version": _WARMUP_CHECKPOINT_VERSION}
    for field in fields:
        value = getattr(args, field)
        if isinstance(value, (tuple, list)):
            value = [float(item) for item in value]
        signature[field] = value
    return signature


def _resolve_warmup_checkpoint_path(args, log_dir):
    configured = str(getattr(args, "common_warmup_checkpoint", "auto")).strip()
    if not configured or configured.lower() in {"none", "off", "disabled"}:
        return None
    if configured.lower() != "auto":
        return os.path.abspath(configured)
    rate_token = "-".join(
        np.format_float_positional(float(rate), trim="-") for rate in args.missrates
    )
    filename = (
        f"{args.name}_{rate_token}_{args.paper_ablation_warmup_group}_"
        f"seed{args.training_seed}_"
        f"warm{args.impute_warmup}_v{_WARMUP_CHECKPOINT_VERSION}.pt"
    )
    return os.path.abspath(os.path.join(log_dir, "_warmup_checkpoints", filename))


def _capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }


def _restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _load_warmup_checkpoint(path):
    """Load full protocol state across both older and newer PyTorch releases."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="train", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--name", type=str, default="BDGP_4view")
    parser.add_argument("--max_epoch_pretrain", type=int, default=100)
    parser.add_argument("--n_clusters", default=5, type=int)
    parser.add_argument("--view_num", default=2, type=int)

    parser.add_argument("--lr_pretrain_gat", type=float, default=0.005)
    parser.add_argument('--lr_gat', type=float, default=0.001)
    parser.add_argument("--hidden_size_gat", default=1024, type=int)
    parser.add_argument("--embedding_size_gat", default=128, type=int)
    parser.add_argument("--weight_decay_gat", type=float, default=5e-3)
    parser.add_argument("--alpha_gat", type=float, default=0.2, help="Alpha for the leaky_relu.")

    parser.add_argument("--lr_pretrain_gcn", type=float, default=0.005)
    parser.add_argument('--lr_gcn', type=float, default=0.001)
    parser.add_argument("--hidden_size_gcn", default=128, type=int)
    parser.add_argument("--embedding_size_gcn", default=16, type=int)
    parser.add_argument("--weight_decay_gcn", type=float, default=0.0001)

    parser.add_argument("--communication_epoch", type=int, default=20)
    parser.add_argument("--local_epoch", type=int, default=50)
    parser.add_argument("--beta_gat", type=float, default=1)
    parser.add_argument("--beta_gcn", type=float, default=1)
    parser.add_argument("--missrates", type=float, nargs="+", default=None)
    parser.add_argument("--knn_neighbor", type=int, default=50)
    parser.add_argument('--yuzhi', type=float, default=0.1)
    parser.add_argument("--feat_miss_rate", type=float, default=0.1)
    parser.add_argument("--feat_miss_seed", type=int, default=1018)
    parser.add_argument("--training-seed", dest="training_seed", type=int, default=1018)
    parser.add_argument("--knn_top_ratio", type=float, default=0.1)
    parser.add_argument("--knn_eps", type=float, default=1e-8)
    parser.add_argument("--knn_sigma", type=float, default=0.01)
    parser.add_argument("--impute_sanity_check", action="store_true")
    parser.add_argument("--use_view_impute", action="store_true", default=True)
    parser.add_argument("--no-use_view_impute", dest="use_view_impute", action="store_false")
    parser.add_argument("--impute_k", type=int, default=None)
    parser.add_argument("--impute_ref_mode", type=str, default="masked", choices=["masked", "concat"])
    parser.add_argument("--proxy-temperature", dest="proxy_temperature", type=float, default=1.0)
    parser.add_argument(
        "--proxy-coverage-threshold",
        dest="proxy_coverage_threshold",
        type=float,
        default=0.8,
    )
    parser.add_argument("--iterative_impute", action="store_true", default=True)
    parser.add_argument("--no-iterative_impute", dest="iterative_impute", action="store_false")
    parser.add_argument(
        "--impute_warmup",
        type=int,
        default=5,
        help="Plot/reference interval only; it does not delay the paper method.",
    )
    parser.add_argument("--impute_every", type=int, default=1)
    parser.add_argument("--impute_eta", type=float, default=0.25)
    parser.add_argument("--gate-min", dest="gate_min", type=float, default=0.05)
    parser.add_argument("--gate-max", dest="gate_max", type=float, default=0.5)
    parser.add_argument(
        "--ud-calibration-quantile",
        dest="ud_calibration_quantile",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--zeta-min", "--alpha_min", dest="zeta_min", type=float, default=0.05
    )
    parser.add_argument(
        "--zeta-max", "--alpha_max", dest="zeta_max", type=float, default=0.5
    )
    parser.add_argument(
        "--experiment-mode",
        choices=[
            "original",
            "no_uncertainty",
            "ud_only",
            "um_only",
            "car_gate",
            "weighted_alignment",
            "full",
        ],
        default="full",
    )
    parser.add_argument(
        "--paper-ablation",
        choices=PAPER_ABLATION_CHOICES,
        default="none",
        help=(
            "Paper-table ablation. QAA component modes keep the Full client "
            "algorithm and remove only the named reliability-score term."
        ),
    )
    parser.add_argument("--uncertainty-samples-B", dest="uncertainty_samples_B", type=int, default=5)
    parser.add_argument("--eta-1", dest="eta_1", type=float, default=0.4)
    parser.add_argument("--eta-2", dest="eta_2", type=float, default=0.4)
    parser.add_argument("--eta-3", dest="eta_3", type=float, default=0.2)
    parser.add_argument("--lambda-r", dest="lambda_r", type=float, default=1.0)
    parser.add_argument("--lambda-d", dest="lambda_d", type=float, default=1.0)
    parser.add_argument("--lambda-m", dest="lambda_m", type=float, default=1.0)
    parser.add_argument(
        "--lambda-um-loss",
        dest="lambda_um_loss",
        type=float,
        default=0.0,
        help="Deprecated compatibility option; the paper has no standalone U_m loss.",
    )
    parser.add_argument(
        "--lambda-um-balance",
        dest="lambda_um_balance",
        type=float,
        default=0.2,
        help=(
            "Deprecated compatibility option; unused by the paper method."
        ),
    )
    parser.add_argument(
        "--lambda-um-hard-target",
        dest="lambda_um_hard_target",
        type=float,
        default=2.0,
        help=(
            "Deprecated compatibility option; unused by the paper method."
        ),
    )
    parser.add_argument(
        "--um-target-power",
        dest="um_target_power",
        type=float,
        default=1.0,
        help=(
            "Deprecated compatibility option; P is constructed directly from q^2."
        ),
    )
    parser.add_argument(
        "--um-assignment-power",
        dest="um_assignment_power",
        type=float,
        default=1.0,
        help=(
            "Deprecated compatibility option; local Student-t assignments are unsharpened."
        ),
    )
    parser.add_argument(
        "--um-assignment-ramp",
        dest="um_assignment_ramp",
        type=int,
        default=10,
        help=(
            "Intervention rounds used to increase assignment power smoothly "
            "from 1 to --um-assignment-power."
        ),
    )
    parser.add_argument(
        "--um-weight-min", dest="um_weight_min", type=float, default=0.1
    )
    parser.add_argument(
        "--reliability-confidence-threshold",
        dest="reliability_confidence_threshold",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--reliability-agreement-threshold",
        dest="reliability_agreement_threshold",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--reliability-floor",
        dest="reliability_floor",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--reliable-neighbor-threshold",
        dest="reliable_neighbor_threshold",
        type=float,
        default=0.2,
    )
    parser.add_argument("--qaa-temperature", dest="qaa_temperature", type=float, default=1.0)
    parser.add_argument(
        "--qaa-eps",
        dest="qaa_eps",
        type=float,
        default=1e-8,
        help="Numerical stability constant for QAA min-max normalization.",
    )
    parser.add_argument(
        "--qaa-weight-ema",
        dest="qaa_weight_ema",
        type=float,
        default=0.0,
        help=(
            "Temporal smoothing applied to client aggregation weights. "
            "Larger values suppress one-round uncertainty noise."
        ),
    )
    parser.add_argument("--uncertainty-eps", dest="uncertainty_eps", type=float, default=1e-8)
    parser.add_argument("--uncertainty-seed", dest="uncertainty_seed", type=int, default=1018)
    parser.add_argument("--kmeans-seed", dest="kmeans_seed", type=int, default=1018)
    parser.add_argument(
        "--common-warmup-checkpoint",
        dest="common_warmup_checkpoint",
        type=str,
        default="none",
        help="Legacy experiment-only warm-up checkpoint; disabled by the paper method.",
    )
    parser.add_argument(
        "--reset-common-warmup-checkpoint",
        dest="reset_common_warmup_checkpoint",
        action="store_true",
        help="Ignore and replace an existing shared warm-up checkpoint.",
    )
    parser.add_argument(
        "--uncertainty-perturbation-scale",
        dest="uncertainty_perturbation_scale",
        type=float,
        default=0.2,
    )
    for uncertainty_switch in (
        "use_data_uncertainty",
        "use_model_uncertainty",
        "use_car_uncertainty_gate",
        "use_uncertainty_weighted_alignment",
        "use_uncertainty_qaa",
    ):
        option_name = uncertainty_switch.replace("_", "-")
        parser.add_argument(
            f"--{option_name}",
            dest=uncertainty_switch,
            action="store_true",
        )
        parser.add_argument(
            f"--no-{option_name}",
            dest=uncertainty_switch,
            action="store_false",
        )
        parser.set_defaults(**{uncertainty_switch: None})

    args = parser.parse_args()
    setup_seed(args.training_seed)
    configure_uncertainty_mode(args)
    configure_paper_ablation(args)
    validate_uncertainty_config(args)
    if args.impute_warmup < 0:
        raise ValueError("impute_warmup must be non-negative.")
    if args.impute_every <= 0:
        raise ValueError("impute_every must be a positive integer.")
    if args.proxy_temperature <= 0:
        raise ValueError("proxy_temperature must be positive.")
    if not 0.0 < args.proxy_coverage_threshold <= 1.0:
        raise ValueError("proxy_coverage_threshold must be in (0, 1].")
    if not 0.0 < args.zeta_min < args.zeta_max < 1.0:
        raise ValueError("inertia bounds must satisfy 0 < zeta_min < zeta_max < 1.")
    if not 0.0 < args.impute_eta <= 1.0:
        raise ValueError("impute_eta must be in (0, 1].")
    if not 0.0 <= args.gate_min <= args.gate_max <= 1.0:
        raise ValueError("gate bounds must satisfy 0 <= min <= max <= 1.")
    if not args.gate_min <= args.impute_eta <= args.gate_max:
        raise ValueError(
            "impute_eta must lie inside [gate_min, gate_max] for a fair "
            "equal-mean recovery gate."
        )
    if not 0.0 < args.ud_calibration_quantile <= 1.0:
        raise ValueError("ud_calibration_quantile must be in (0, 1].")
    if args.lambda_um_loss < 0:
        raise ValueError("lambda_um_loss must be non-negative.")
    if args.lambda_um_balance < 0:
        raise ValueError("lambda_um_balance must be non-negative.")
    if args.lambda_um_hard_target < 0:
        raise ValueError("lambda_um_hard_target must be non-negative.")
    if args.um_target_power < 1.0:
        raise ValueError("um_target_power must be at least 1.")
    if args.um_assignment_power < 1.0:
        raise ValueError("um_assignment_power must be at least 1.")
    if args.um_assignment_ramp <= 0:
        raise ValueError("um_assignment_ramp must be a positive integer.")
    if not 0.0 <= args.um_weight_min <= 1.0:
        raise ValueError("um_weight_min must be in [0, 1].")
    if not 0.0 <= args.reliability_confidence_threshold < 1.0:
        raise ValueError("reliability_confidence_threshold must be in [0, 1).")
    if not 0.0 <= args.reliability_agreement_threshold < 1.0:
        raise ValueError("reliability_agreement_threshold must be in [0, 1).")
    if not 0.0 <= args.reliability_floor <= 1.0:
        raise ValueError("reliability_floor must be in [0, 1].")
    if not 0.0 <= args.reliable_neighbor_threshold <= 1.0:
        raise ValueError("reliable_neighbor_threshold must be in [0, 1].")
    if args.use_data_uncertainty and not args.use_view_impute:
        raise ValueError("Data uncertainty requires the original view-imputation path.")
    if args.use_model_uncertainty and not args.use_view_impute:
        raise ValueError(
            "Model uncertainty requires proxy neighborhoods for structural consistency."
        )
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")
    # device = torch.device("cpu")
    impute_sanity_check = getattr(args, "impute_sanity_check", False)

    if args.name == 'Caltech101-7':
        args.n_clusters = 7
        args.view_num = 6
       # args.hidden_size_gat = 512
       # args.embedding_size_gat = 32
       # args.hidden_size_gcn = 512
       # args.embedding_size_gcn = 64
       # args.beta_gat = 10
       # args.beta_gcn = 10
       # args.communication_epoch = 20
        if args.missrates is None:
            args.missrates = [0.1, 0.1, 0.2, 0.2, 0.1, 0.1]
    elif args.name == 'BDGP_4view':
        args.n_clusters = 5
        args.view_num = 4
        args.hidden_size_gat = 128
        args.embedding_size_gat = 32
        args.hidden_size_gcn = 128
        args.embedding_size_gcn = 16
        args.beta_gat = 1
        args.beta_gcn = 1
         #args.communication_epoch = 30
        if args.missrates is None:
            args.missrates = [0.1, 0.3, 0.5, 0.7]
    elif args.name == 'Scene-15':
        args.n_clusters = 15
        args.view_num = 3
      #  args.hidden_size_gat = 512
       # args.embedding_size_gat = 64
       # args.hidden_size_gcn = 512
       # args.embedding_size_gcn = 64
        #args.beta_gat = 0.01
        #args.beta_gcn = 0.01
        #args.lr_gat = 0.01
        #args.lr_gcn = 0.01
        #args.communication_epoch = 20
        if args.missrates is None:
            args.missrates = [0.1, 0.5, 0.7]
    elif args.name == 'Reuters':
        args.n_clusters = 6
        args.view_num = 5
        #args.hidden_size_gat = 512
        #args.embedding_size_gat = 32
        #args.hidden_size_gcn = 512
        #args.embedding_size_gcn = 32
        #args.beta_gat = 1
        #args.beta_gcn = 1
        #args.communication_epoch = 10
        if args.missrates is None:
            args.missrates = [0.1, 0.2, 0.2, 0.1, 0.1]
    else:
        print('datasets miss')


    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = 'log_new'
    folder_name = f'{args.name}_{args.run_label}_{current_time}'
    log_folder_path = os.path.join(log_dir, folder_name)
    log_file_name = f'{args.name}_{args.run_label}_{current_time}.log'
    if not os.path.exists(log_folder_path):
        os.makedirs(log_folder_path)
    log_file_path = os.path.join(log_folder_path, log_file_name)
    logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger()
    logger.info("Command line arguments: " + str(args))
    logger.info(
        f"[paper-ablation] setting={args.paper_ablation} "
        f"proxy_initialization={args.proxy_initialization} "
        f"iterative_impute={args.iterative_impute} "
        f"uniform_aggregation={args.force_uniform_aggregation}"
    )
    logger.info(
        "[QAA] R_metric=mean(top1_probability-top2_probability)"
    )
    shutil.copyfile('main.py', log_folder_path + '/main.py')
    shutil.copyfile('cluster_quality.py', log_folder_path + '/cluster_quality.py')
    shutil.copyfile('qaa_fusion.py', log_folder_path + '/qaa_fusion.py')
    shutil.copyfile('train.py', log_folder_path + '/train.py')
    shutil.copyfile('uncertainty.py', log_folder_path + '/uncertainty.py')
    shutil.copyfile('utils_impute.py', log_folder_path + '/utils_impute.py')
    shutil.copyfile('generate_incomplete_data.py', log_folder_path + '/generate_incomplete_data.py')
    shutil.copyfile('paper_ablations.py', log_folder_path + '/paper_ablations.py')
    shutil.copyfile('experiment_recorder.py', log_folder_path + '/experiment_recorder.py')
    args_file_path = os.path.join(log_folder_path, 'args.json')
    with open(args_file_path, 'w') as args_file:
        json.dump(vars(args), args_file, indent=4)


    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(console_handler)

# load dataset
    direct_path = './datasets/'
    missrates_input = args.missrates if args.missrates is not None else args.feat_miss_rate
    needs_proxy_context = bool(
        args.use_data_uncertainty or args.use_model_uncertainty
    )
    data_out = generate_data(
        direct_path,
        args.name,
        missrates_input,
        args.knn_neighbor,
        use_view_impute=args.use_view_impute,
        impute_k=args.impute_k,
        ref_mode=args.impute_ref_mode,
        return_impute_context=needs_proxy_context,
        proxy_initialization=args.proxy_initialization,
        proxy_temperature=args.proxy_temperature,
        proxy_coverage_threshold=args.proxy_coverage_threshold,
        missing_seed=args.feat_miss_seed,
    )
    if args.use_view_impute:
        imputation_context = data_out[-1] if needs_proxy_context else None
        imputed_data_out = data_out[:-1] if needs_proxy_context else data_out
        (
            feature_list,
            labels,
            X_incomplete,
            Y_incomplete,
            X_complete,
            Y_complete,
            incomplete_indices,
            complete_indices,
            adj,
            adj_label,
            adj_c,
            adj_c_label,
            M,
            M_c,
            X_full_imputed_list,
            mask_view_list,
            X_incomplete_imputed_list,
            X_full_zero_list,
            adj_full_list,
            adj_full_label_list,
            M_full_list,
        ) = imputed_data_out
    else:
        (
            feature_list,
            labels,
            X_incomplete,
            Y_incomplete,
            X_complete,
            Y_complete,
            incomplete_indices,
            complete_indices,
            adj,
            adj_label,
            adj_c,
            adj_c_label,
            M,
            M_c,
        ) = data_out
        X_full_imputed_list = None
        mask_view_list = None
        X_incomplete_imputed_list = None
        X_full_zero_list = None
        adj_full_list = None
        adj_full_label_list = None
        M_full_list = None
        imputation_context = None
    V = len(feature_list)
    if args.use_view_impute:
        normal_recovery_list = [np.asarray(x).copy() for x in X_full_imputed_list]
        X_incomplete = X_full_imputed_list
        Y_incomplete = [labels for _ in range(V)]
        adj = adj_full_list
        adj_label = adj_full_label_list
        M = M_full_list

        sanity_view = None
        for v in range(V):
            missing_rows = np.where(np.asarray(mask_view_list[v]) == 0)[0]
            if missing_rows.size > 0:
                sanity_view = v
                break
        if sanity_view is not None:
            zero_ok = np.allclose(X_full_zero_list[sanity_view][missing_rows], 0.0)
            imputed_all_zero = np.allclose(
                X_full_imputed_list[sanity_view][missing_rows], 0.0
            )
            assert zero_ok, "[impute] missing rows in X_full_zero_list are not all zero."
            assert not imputed_all_zero, "[impute] missing rows in X_full_imputed_list are still all zero."
            logger.info(
                f"[impute] view{sanity_view} missing rows={missing_rows.size} "
                f"train_X_shape={X_full_imputed_list[sanity_view].shape}"
            )
        else:
            logger.info("[impute] no missing rows found for sanity check.")
    else:
        normal_recovery_list = None
    # Keep missrates aligned with the actual number of views.
    missrates = missrates_input
    if isinstance(missrates, (int, float, np.integer, np.floating)):
        missrates = [float(missrates)] * V
    else:
        missrates = list(missrates)
        if len(missrates) == 1:
            missrates = missrates * V
        elif len(missrates) < V:
            missrates = missrates + [missrates[-1]] * (V - len(missrates))
        elif len(missrates) > V:
            missrates = missrates[:V]
        missrates = [float(r) for r in missrates]
    if any(r > 1.0 for r in missrates):
        missrates = [r / 100.0 for r in missrates]
    args.missrates = missrates
    fair_ablation_modes = {"no_uncertainty", "ud_only", "um_only", "full"}
    # The paper applies the proposed mechanism from the first communication
    # round.  Shared baseline warm-up checkpoints belonged to an older
    # experimental protocol and are intentionally inactive here.
    common_warmup_checkpoint_path = None
    args.resolved_common_warmup_checkpoint = common_warmup_checkpoint_path
    with open(args_file_path, "w") as args_file:
        json.dump(vars(args), args_file, indent=4)
    uncertainty_recorder = UncertaintyExperimentRecorder(
        dataset_name=args.name,
        miss_rates=args.missrates,
        output_dir=log_folder_path,
        experiment_mode=args.run_label,
        warmup_rounds=args.impute_warmup,
    )
    logger.info(
        f"[uncertainty-export] Excel={uncertainty_recorder.excel_path} "
        f"plot={uncertainty_recorder.plot_path}"
    )
    num_samples = feature_list[0].shape[0]
    print(f'number of samples: {num_samples}')

    view_missing_masks = []
    view_missing_ratios = []
    for i in range(V):
        if args.use_view_impute and mask_view_list is not None:
            mask_view = np.asarray(mask_view_list[i], dtype=float)
            missing_ratio = 1.0 - float(np.mean(mask_view)) if mask_view.size else 0.0
            view_missing_ratios.append(missing_ratio)
            view_missing_masks.append((1.0 - mask_view).reshape(-1, 1))
        else:
            missing_ratio = 1.0 - (len(incomplete_indices[i]) / float(num_samples))
            view_missing_ratios.append(missing_ratio)
            if len(complete_indices) == 0:
                view_missing_masks.append(None)
            else:
                view_missing_masks.append(
                    np.full((len(complete_indices), 1), missing_ratio, dtype=float)
                )
        logger.info(f"view{i} missing ratio: {missing_ratio:.4f}")

    for i in range(len(X_incomplete)):
        adj[i] = adj[i].to(device)
        adj_c[i] = adj_c[i].to(device)
        adj_label[i] = adj_label[i].to(device)
        adj_c_label[i] = adj_c_label[i].to(device)
        X_incomplete[i] = torch.Tensor(X_incomplete[i]).to(device)
        X_complete[i] = torch.Tensor(X_complete[i]).to(device)
        M[i] = M[i].to(device)
        M_c[i] = M_c[i].to(device)

    iter_impute_enabled = bool(args.iterative_impute)
    if iter_impute_enabled and (not args.use_view_impute or mask_view_list is None):
        logger.info("[iter-impute] disabled: requires use_view_impute=True and mask_view_list.")
        iter_impute_enabled = False

# pretrain
    premodel_list = []
    for i in range(V):
        pre = GCN(
            num_features=X_incomplete[i].shape[1],
            hidden_size=args.hidden_size_gcn,
            embedding_size=args.embedding_size_gcn,
        ).to(device)
        premodel_list.append(pre)
        print(f'view{i} use model GCN')


    print('---------pretrain_start------------')
    for i in range(len(premodel_list)):
        if (isinstance(premodel_list[i], GAT)):
            pretrain_gat(premodel_list[i], adj[i], adj_label[i].to(torch.float32), M[i], X_incomplete[i], args, logger)
        elif (isinstance(premodel_list[i], GCN)):
            pretrain_gcn(premodel_list[i], adj[i], adj_label[i].to(torch.float32), X_incomplete[i], args, logger)
        else:
            print('error')


    for i in range(len(premodel_list)):
        save_path = os.path.join(log_folder_path, f"premodel{i}_{args.name}.pkl")
        torch.save(premodel_list[i].state_dict(), save_path)


    model_list = []
    for i in range(len(premodel_list)):
        if (isinstance(premodel_list[i], GAT)):
            model = DAEGC(pretrain_model=premodel_list[i], args=args).to(device)
            model_list.append(model)
        elif (isinstance(premodel_list[i], GCN)):
            model = GCNED(pretrain_model=premodel_list[i], args=args).to(device)
            model_list.append(model)
        else:
            print('error')

    num_views = len(model_list)
    p_local_list = [None] * num_views
    # All uncertainty-reduction ablations train on the same full imputed sample
    # set.  This prevents client sample count from confounding Ud-only, Um-only,
    # and Full comparisons.
    federated_full_data = bool(
        args.use_view_impute
        and (
            args.experiment_mode == "no_uncertainty"
            or args.use_data_uncertainty_reduction
            or args.use_model_uncertainty_reduction
        )
    )
    train_X = X_incomplete if federated_full_data else X_complete
    train_adj = adj if federated_full_data else adj_c
    train_adj_label = adj_label if federated_full_data else adj_c_label
    train_M = M if federated_full_data else M_c
    train_labels = labels if federated_full_data else Y_complete

    # Build P^(0) from the pretrained client representations through the original
    # QAA representation weighting, K-means, target distribution, and alignment.
    features = []
    for view in range(V):
        with torch.no_grad():
            if isinstance(premodel_list[view], GAT):
                _, feature = premodel_list[view](train_X[view], train_adj[view], train_M[view])
            elif isinstance(premodel_list[view], GCN):
                _, feature = premodel_list[view](train_X[view], train_adj[view])
            else:
                raise TypeError("Unsupported pretrain model.")
        feature_np = feature.detach().cpu().numpy()
        features.append(feature_np)

        # Algorithm 1 requires an initial local Q^(v,0). Initialize each
        # Student-t center layer from the pretrained representation instead of
        # leaving it at an unrelated Xavier-random state.
        local_kmeans = GPUKMeans(
            n_clusters=args.n_clusters,
            n_init=20,
            random_state=args.kmeans_seed,
        ).fit(feature_np)
        local_centers = torch.as_tensor(
            local_kmeans.cluster_centers_,
            dtype=model_list[view].cluster_layer.dtype,
            device=device,
        )
        with torch.no_grad():
            model_list[view].cluster_layer.copy_(local_centers)

    initial_weights = np.full(V, 1.0 / float(V), dtype=float)
    initial_blocks = [features[v] * initial_weights[v] for v in range(V)]
    z = np.hstack(initial_blocks)
    np.save(os.path.join(log_folder_path, "z_array_initial.npy"), z)

    kmean = GPUKMeans(
        n_clusters=args.n_clusters, n_init=20, random_state=args.kmeans_seed
    )
    y_pred = kmean.fit_predict(z)
    Center_init = kmean.cluster_centers_
    y_pred_global = np.copy(y_pred)
    p_global = normalize_global_target(update_P(z, Center_init), args.uncertainty_eps)
    p_global_torch = torch.from_numpy(p_global).to(device=device, dtype=torch.float32).detach()
    for view in range(num_views):
        p_local_list[view] = p_global_torch.detach().clone()
    logger.info(
        f"[IU] initialized P^(0) from pretrained representations; samples={p_global.shape[0]}"
    )

    ud_reference_scales = [None] * V
    start_com_epoch = 0
    warmup_checkpoint_loaded = False
    if (
        common_warmup_checkpoint_path is not None
        and os.path.exists(common_warmup_checkpoint_path)
        and not args.reset_common_warmup_checkpoint
    ):
        checkpoint = _load_warmup_checkpoint(common_warmup_checkpoint_path)
        expected_signature = _warmup_signature(args)
        if checkpoint.get("signature") != expected_signature:
            raise ValueError(
                "The shared warm-up checkpoint does not match the current "
                "configuration. Use --reset-common-warmup-checkpoint once to "
                "create a new checkpoint."
            )
        for view, state_dict in enumerate(checkpoint["model_states"]):
            model_list[view].load_state_dict(state_dict)
        for view, saved_x in enumerate(checkpoint["x_incomplete"]):
            X_incomplete[view] = saved_x.to(device=device, dtype=torch.float32)
        if federated_full_data:
            train_X = X_incomplete
        X_full_imputed_list = [
            saved_x.detach().cpu().numpy().copy()
            for saved_x in checkpoint["x_incomplete"]
        ]
        p_local_list = [
            saved_p.to(device=device, dtype=torch.float32).detach()
            for saved_p in checkpoint["p_local_list"]
        ]
        y_pred_global = np.asarray(checkpoint["y_pred_global"]).copy()
        ud_reference_scales = [
            float(value) for value in checkpoint["ud_reference_scales"]
        ]
        uncertainty_recorder.restore_records(checkpoint["records"])
        start_com_epoch = int(checkpoint["completed_rounds"])
        if start_com_epoch != args.impute_warmup:
            raise ValueError(
                "Shared warm-up checkpoint round count does not match impute_warmup."
            )
        warmup_checkpoint_loaded = True
        uncertainty_recorder.export()
        _restore_rng_state(checkpoint["rng_state"])
        logger.info(
            f"[warmup-checkpoint] loaded={common_warmup_checkpoint_path} "
            f"completed_rounds={start_com_epoch}"
        )

# train
    logger.info('---------train_start------------')
    logger.info(
        f"[uncertainty] mode={args.experiment_mode}, "
        f"data_reduction={args.use_data_uncertainty_reduction}, "
        f"model_reduction={args.use_model_uncertainty_reduction}, "
        f"full_client_training={federated_full_data}"
    )
    for com_epoch in range(start_com_epoch, args.communication_epoch):
        # The proposed method is active from round 1.  ``impute_warmup`` is kept
        # only as a plotting/reference setting for old experiment files.
        uncertainty_intervention_active = args.experiment_mode != "original"
        args.uncertainty_intervention_active = uncertainty_intervention_active
        # Q is the unmodified Student-t assignment in the paper; no additional
        # assignment sharpening is applied.
        assignment_power = 1.0
        for client_model in model_list:
            client_model.assignment_power = assignment_power
        logger.info(
            f"[Um-assignment] epoch={com_epoch} power={assignment_power:.6f}"
        )
        recovery_update_due = (
            iter_impute_enabled and com_epoch % args.impute_every == 0
        )
        phase = "paper-method" if uncertainty_intervention_active else "original"
        logger.info(
            f"[protocol] epoch={com_epoch} phase={phase} "
            f"recovery_update_due={recovery_update_due}"
        )

        # PRE measurements are computed from one shared model/target state.  The
        # same perturbed recoveries are reused for the corresponding POST
        # measurement, avoiding a random-sampling confound within the round.
        perturbed_by_view = None
        sample_ud_pre_list = [
            torch.zeros(num_samples, dtype=torch.float32, device=device) for _ in range(V)
        ]
        if args.use_data_uncertainty:
            perturbed_by_view = generate_perturbed_recoveries(
                X_full_zero_list,
                normal_recovery_list,
                mask_view_list,
                imputation_context,
                samples_B=args.uncertainty_samples_B,
                seed=args.uncertainty_seed + com_epoch,
                tau=args.proxy_temperature,
                eps=args.uncertainty_eps,
                perturbation_scale=args.uncertainty_perturbation_scale,
            )
            for view in range(V):
                forward_context = {"adj": adj[view]}
                if isinstance(model_list[view], DAEGC):
                    forward_context["M"] = M[view]
                sample_ud_pre_list[view] = compute_data_uncertainty(
                    perturbed_by_view[view],
                    model_list[view],
                    forward_context,
                    np.asarray(mask_view_list[view]) == 0,
                    eps=args.uncertainty_eps,
                )

        sample_um_pre_list = []
        sample_us_round_list = []
        alignment_permutation_list = []
        client_ud_pre = np.zeros(V, dtype=float)
        client_um_pre = np.zeros(V, dtype=float)
        for view in range(V):
            z_pre, q_pre = forward_client(
                model_list[view], X_incomplete[view], adj[view], M[view]
            )
            alignment_permutation = compute_alignment_permutation(
                q_pre, p_global_torch, args.uncertainty_eps
            )
            aligned_q_pre = apply_alignment_permutation(
                q_pre, alignment_permutation, args.uncertainty_eps
            )
            alignment_permutation_list.append(alignment_permutation.detach())
            if args.use_model_uncertainty:
                sample_us_round = compute_structural_inconsistency(
                    p_global_torch,
                    imputation_context[view],
                    eps=args.uncertainty_eps,
                )
                sample_um_pre = compute_model_uncertainty(
                    q_pre,
                    p_global_torch,
                    aligned_q_pre,
                    args.eta_1,
                    args.eta_2,
                    args.uncertainty_eps,
                    structural_inconsistency=sample_us_round,
                    eta_3=args.eta_3,
                )
                client_um_pre[view] = float(sample_um_pre.mean().item())
            else:
                sample_us_round = torch.zeros(
                    num_samples, dtype=torch.float32, device=device
                )
                sample_um_pre = torch.zeros(
                    num_samples, dtype=torch.float32, device=device
                )

            if args.use_data_uncertainty:
                missing = torch.as_tensor(
                    np.asarray(mask_view_list[view]) == 0, device=device
                )
                if torch.any(missing):
                    client_ud_pre[view] = float(
                        sample_ud_pre_list[view][missing].mean().item()
                    )

            # CAR must compare common cluster columns across views. The old
            # implementation averaged unaligned client columns here.
            sample_um_pre_list.append(sample_um_pre)
            sample_us_round_list.append(sample_us_round)
            logger.info(
                f"[uncertainty-pre] epoch={com_epoch} client={view} "
                f"client_U_d={client_ud_pre[view]:.6e} "
                f"client_U_m={client_um_pre[view]:.6e}"
            )

        # The paper uses only a_i=stopgrad(1-U_m,i) as sample reliability in
        # the local model loss.  The former extra consensus-reliability filter
        # is deliberately removed.
        client_reliability = np.ones(V, dtype=float)

        # Recovery is performed after local training and global-target update,
        # so it becomes the recursively refined input of the next round.
        client_gate_mean = np.zeros(V, dtype=float)
        client_gate_max = np.zeros(V, dtype=float)
        recovery_kind = "paper-recursive-car"

        features = []
        q = []
        q_after_local_list = []
        client_model_loss = np.zeros(V, dtype=float)
        for view in range(V):
            if isinstance(model_list[view], DAEGC):
                z_v, q_v, train_diagnostics = train_gat(
                    model_list[view],
                    train_adj[view],
                    train_adj_label[view].to(torch.float32),
                    train_M[view],
                    train_X[view],
                    p_local_list[view],
                    args,
                    alignment_permutation=alignment_permutation_list[view],
                    structural_inconsistency=sample_us_round_list[view],
                    uncertainty_target=p_global_torch,
                    return_diagnostics=True,
                )
            elif isinstance(model_list[view], GCNED):
                z_v, q_v, train_diagnostics = train_gcn(
                    model_list[view],
                    train_adj[view],
                    train_adj_label[view].to(torch.float32),
                    train_X[view],
                    p_local_list[view],
                    args,
                    alignment_permutation=alignment_permutation_list[view],
                    structural_inconsistency=sample_us_round_list[view],
                    uncertainty_target=p_global_torch,
                    return_diagnostics=True,
                )
            else:
                raise TypeError("Unsupported client model.")
            client_model_loss[view] = float(
                train_diagnostics["model_alignment_loss"]
            )
            logger.info(
                f"[model-alignment-loss] epoch={com_epoch} client={view} "
                f"value={client_model_loss[view]:.6e}"
            )
            features.append(z_v.detach().cpu().numpy())
            q.append(q_v.detach().cpu().numpy())
            q_after_local_list.append(q_v.detach())

        # QAA aggregates the models produced by this local-training round, so
        # its uncertainty inputs must describe those updated models.  The old
        # path used PRE values measured before recovery and local training,
        # which could assign a large weight from stale information.
        sample_ud_after_local_list = [
            torch.zeros(num_samples, dtype=torch.float32, device=device)
            for _ in range(V)
        ]
        sample_um_after_local_list = [
            torch.zeros(num_samples, dtype=torch.float32, device=device)
            for _ in range(V)
        ]
        aligned_q_after_local_list = []
        client_ud_after_local = np.zeros(V, dtype=float)
        client_um_after_local = np.zeros(V, dtype=float)
        for view in range(V):
            if args.use_data_uncertainty:
                forward_context = {"adj": adj[view]}
                if isinstance(model_list[view], DAEGC):
                    forward_context["M"] = M[view]
                sample_ud_after_local_list[view] = compute_data_uncertainty(
                    perturbed_by_view[view],
                    model_list[view],
                    forward_context,
                    np.asarray(mask_view_list[view]) == 0,
                    eps=args.uncertainty_eps,
                )
                missing = torch.as_tensor(
                    np.asarray(mask_view_list[view]) == 0,
                    dtype=torch.bool,
                    device=device,
                )
                if torch.any(missing):
                    client_ud_after_local[view] = float(
                        sample_ud_after_local_list[view][missing].mean().item()
                    )
            if args.use_model_uncertainty:
                aligned_q_after_local = apply_alignment_permutation(
                    q_after_local_list[view],
                    alignment_permutation_list[view],
                    args.uncertainty_eps,
                )
                sample_um_after_local = compute_model_uncertainty(
                    q_after_local_list[view],
                    p_global_torch,
                    aligned_q_after_local,
                    args.eta_1,
                    args.eta_2,
                    args.uncertainty_eps,
                    structural_inconsistency=sample_us_round_list[view],
                    eta_3=args.eta_3,
                )
                client_um_after_local[view] = float(
                    sample_um_after_local.mean().item()
                )
                sample_um_after_local_list[view] = sample_um_after_local
            else:
                aligned_q_after_local = apply_alignment_permutation(
                    q_after_local_list[view],
                    alignment_permutation_list[view],
                    args.uncertainty_eps,
                )
            aligned_q_after_local_list.append(
                aligned_q_after_local.detach().cpu().numpy()
            )
            logger.info(
                f"[QAA-input] epoch={com_epoch} client={view} stage=post-local "
                f"client_U_d={client_ud_after_local[view]:.6e} "
                f"client_U_m={client_um_after_local[view]:.6e}"
            )

        y_q = np.sum(np.stack(q, axis=0), axis=0)
        y_mean_pred = y_q.argmax(1)
        client_mean_acc, client_mean_nmi, client_mean_ari, client_mean_f1 = eva(
            train_labels, y_mean_pred.copy(), com_epoch
        )
        data_scope = "all" if federated_full_data else "common"
        logger.info(
            f"{data_scope} client-mean :com_epoch {com_epoch}:"
            f"acc {client_mean_acc:.4f}, nmi {client_mean_nmi:.4f}, "
            f"ari {client_mean_ari:.4f}, f1 {client_mean_f1:.4f}"
        )

        # Client discriminability is exactly the mean top-1/top-2 assignment
        # margin specified by the paper.
        r_raw = np.asarray(
            [compute_assignment_margin(q_after_local_list[v]) for v in range(V)],
            dtype=float,
        )

        # All four targeted ablations use the same softmax-QAA family from
        # round 1. Disabled terms receive lambda=0.
        common_ablation_qaa = (
            args.experiment_mode in fair_ablation_modes
            and uncertainty_intervention_active
        )
        uncertainty_qaa_active = (
            uncertainty_intervention_active and args.use_uncertainty_qaa
        )
        if args.force_uniform_aggregation and uncertainty_intervention_active:
            weights = np.full(V, 1.0 / float(V), dtype=float)
            logger.info(
                f"[QAA-ablation] epoch={com_epoch} setting=average_aggregation "
                f"omega={weights.tolist()}"
            )
        elif common_ablation_qaa or uncertainty_qaa_active:
            (
                effective_lambda_r,
                effective_lambda_d,
                effective_lambda_m,
            ) = effective_qaa_coefficients(args)
            qaa = compute_uncertainty_qaa_weights(
                r_raw,
                client_ud_after_local,
                client_um_after_local,
                effective_lambda_r,
                effective_lambda_d,
                effective_lambda_m,
                args.qaa_temperature,
                args.qaa_eps,
                logger,
            )
            candidate_weights = np.asarray(qaa["omega"], dtype=float)
            weights = candidate_weights
            score_lower = -(effective_lambda_d + effective_lambda_m) - 1e-10
            score_upper = effective_lambda_r + 1e-10
            if np.any(qaa["score"] < score_lower) or np.any(qaa["score"] > score_upper):
                raise RuntimeError("QAA score escaped its theoretical range.")
            logger.info(
                f"[QAA-ablation] epoch={com_epoch} setting={args.paper_ablation} "
                f"lambda_r={effective_lambda_r:.6f} "
                f"lambda_d={effective_lambda_d:.6f} "
                f"lambda_m={effective_lambda_m:.6f}"
            )
            for view in range(V):
                logger.info(
                    f"[QAA] epoch={com_epoch} client={view} R_raw={r_raw[view]:.6f} "
                    f"R_norm={qaa['R_norm'][view]:.6f} U_d_norm={qaa['U_d_norm'][view]:.6f} "
                    f"U_m_norm={qaa['U_m_norm'][view]:.6f} score={qaa['score'][view]:.6f} "
                    f"candidate_omega={candidate_weights[view]:.6f} "
                    f"omega={weights[view]:.6f}"
                )
            logger.info(
                f"[QAA] epoch={com_epoch} omega_sum={float(np.sum(weights)):.12f} "
                f"has_nan_or_inf={qaa['has_invalid']}"
            )
        else:
            # Original positive discriminativity QAA remains untouched.
            kmeans = GPUKMeans(
                n_clusters=args.n_clusters,
                n_init=100,
                random_state=args.kmeans_seed,
            )
            center = []
            for view in range(V):
                kmeans.fit_predict(features[view])
                center.append([np.asarray(kmeans.cluster_centers_).copy()])
            weights = get_weights_safe(center, V)

        diagnostic_weights = np.asarray(weights, dtype=float)
        diagnostic_weights = np.maximum(diagnostic_weights, 0.0)
        diagnostic_total = float(np.sum(diagnostic_weights))
        if diagnostic_total <= args.uncertainty_eps:
            diagnostic_weights = np.full(V, 1.0 / V, dtype=float)
        else:
            diagnostic_weights = diagnostic_weights / diagnostic_total
        qaa_max_weight = float(np.max(diagnostic_weights))
        qaa_weight_entropy = float(
            -np.sum(
                diagnostic_weights
                * np.log(np.maximum(diagnostic_weights, args.qaa_eps))
            )
            / math.log(float(V))
        )
        logger.info(
            f"[QAA-diagnostic] epoch={com_epoch} "
            f"max_weight={qaa_max_weight:.6f} "
            f"normalized_entropy={qaa_weight_entropy:.6f}"
        )

        reliability_fusion_active = (
            uncertainty_intervention_active
            and args.experiment_mode in fair_ablation_modes
        )
        if reliability_fusion_active:
            # Direct weighted concatenation: Z=[omega_1 Z_1 || ... || omega_V Z_V].
            feature_scales = diagnostic_weights
            logger.info(
                f"[QAA-fusion] epoch={com_epoch} "
                f"feature_scales={feature_scales.tolist()}"
            )
        else:
            # Preserve the untouched DU-FMVC fusion before intervention and in
            # experiment modes outside the controlled QAA ablation family.
            feature_scales = np.asarray(weights, dtype=float)
        n_features = [
            features[view] * feature_scales[view] for view in range(V)
        ]
        z = np.hstack(n_features)
        np.save(os.path.join(log_folder_path, f"z_array_epoch_{com_epoch}.npy"), z)

        # Original server K-means, target distribution, label alignment, and
        # entropy-driven dynamic Inertia Update.
        kmean = GPUKMeans(
            n_clusters=args.n_clusters, n_init=20, random_state=args.kmeans_seed
        )
        y_pred = kmean.fit_predict(z)
        Center_init = kmean.cluster_centers_
        server_acc, server_nmi, server_ari, server_f1 = eva(
            labels, np.asarray(y_pred).copy(), com_epoch
        )
        logger.info(
            f"server fused data :com_epoch {com_epoch}:acc {server_acc:.4f}, "
            f"nmi {server_nmi:.4f}, ari {server_ari:.4f}, f1 {server_f1:.4f}"
        )
        # Hungarian matrix rows are current clusters and columns are previous
        # global clusters, so current probabilities are right-multiplied by it.
        _, row_ind, col_ind, matrix = update_A(y_pred_global, y_pred)
        current_labels = np.asarray(y_pred, dtype=np.int64)
        y_pred_global = np.argmax(matrix[current_labels], axis=1)
        p_global = normalize_global_target(
            np.dot(update_P(z, Center_init), matrix), args.uncertainty_eps
        )
        p_global_torch = torch.from_numpy(p_global).to(
            device=device, dtype=torch.float32
        ).detach()
        zeta = compute_zeta_from_entropy(
            p_global_torch, args.n_clusters, args.zeta_min, args.zeta_max
        )
        for view in range(num_views):
            if com_epoch == 0:
                p_local_list[view] = p_global_torch.detach().clone()
            else:
                p_local_list[view] = (
                    (1.0 - zeta) * p_local_list[view] + zeta * p_global_torch
                ).detach()
        logger.info(f"[IU] epoch={com_epoch} zeta={zeta:.6f}")

        # CAR is the final step of a communication round.  Full uses the exact
        # gate g=U_d(1-U_m); single-factor modes keep only their corresponding
        # factor for controlled ablation.  No calibration, clipping range,
        # neighbor reliability filter, or fixed initial anchor is used.
        if recovery_update_due:
            gate_list = []
            for view in range(V):
                missing_mask = np.asarray(mask_view_list[view]) == 0
                if (
                    args.use_data_uncertainty_reduction
                    and args.use_model_uncertainty_reduction
                ):
                    gate = sample_ud_after_local_list[view] * (
                        1.0 - sample_um_after_local_list[view]
                    )
                    recovery_kind = "paper-dual-gate"
                elif args.use_data_uncertainty_reduction:
                    gate = sample_ud_after_local_list[view]
                    recovery_kind = "paper-ud-gate"
                elif args.use_model_uncertainty_reduction:
                    gate = 1.0 - sample_um_after_local_list[view]
                    recovery_kind = "paper-um-gate"
                else:
                    gate = torch.ones(
                        num_samples, dtype=torch.float32, device=device
                    )
                    recovery_kind = "ungated-semantic-refinement"
                gate = torch.clamp(gate, 0.0, 1.0).detach()
                gate_list.append(gate.cpu().numpy())
                missing_tensor = torch.as_tensor(
                    missing_mask, dtype=torch.bool, device=device
                )
                missing_gate = gate[missing_tensor]
                if missing_gate.numel() > 0:
                    client_gate_mean[view] = float(missing_gate.mean().item())
                    client_gate_max[view] = float(missing_gate.max().item())
                logger.info(
                    f"[recovery-gate] epoch={com_epoch} client={view} "
                    f"kind={recovery_kind} missing_mean={client_gate_mean[view]:.6e} "
                    f"missing_max={client_gate_max[view]:.6e}"
                )

            X_previous_list = [x.detach().cpu().numpy() for x in X_incomplete]
            X_refined_list = semantic_refine_missing_views(
                X_previous_list,
                mask_view_list,
                aligned_q_after_local_list,
                features,
                gate_list=gate_list,
                eps=args.uncertainty_eps,
            )
            for view in range(V):
                X_incomplete[view] = torch.as_tensor(
                    X_refined_list[view], dtype=torch.float32, device=device
                )
            X_full_imputed_list = X_refined_list
            logger.info(
                f"[iter-impute] com_epoch {com_epoch}: updated_views={V} "
                f"recovery={recovery_kind}"
            )

        # Original full/incomplete-view evaluation.
        y_pred_all = []
        q_pred_all = []
        z_pred_all = []
        q_post_torch_list = []
        for view in range(V):
            m_view = get_M(adj[view]).to(device) if isinstance(model_list[view], DAEGC) else None
            z_v, q_v = forward_client(
                model_list[view], X_incomplete[view], adj[view], m_view
            )
            q_numpy = q_v.cpu().numpy()
            z_numpy = z_v.cpu().numpy()
            q_pred_all.append(q_numpy)
            z_pred_all.append(z_numpy)
            q_post_torch_list.append(q_v)
            y_v = q_numpy.argmax(1)
            y_pred_all.append(y_v)
            view_acc, view_nmi, view_ari, view_f1 = eva(
                Y_incomplete[view], y_v.copy(), com_epoch
            )
            logger.info(
                f"view{view} data :com_epoch {com_epoch}:acc {view_acc:.4f}, "
                f"nmi {view_nmi:.4f}, ari {view_ari:.4f}, f1 {view_f1:.4f}"
            )

        if args.use_view_impute:
            y_server = np.argmax(np.mean(np.stack(q_pred_all, axis=0), axis=0), axis=1)
        else:
            y_server = average_predictions(
                q_pred_all, incomplete_indices, V, args.n_clusters, num_samples
            )
        ensemble_acc, ensemble_nmi, ensemble_ari, ensemble_f1 = eva(
            labels, y_server.copy(), com_epoch
        )
        logger.info(
            f"client ensemble data :com_epoch {com_epoch}:acc {ensemble_acc:.4f}, "
            f"nmi {ensemble_nmi:.4f}, ari {ensemble_ari:.4f}, "
            f"f1 {ensemble_f1:.4f}"
        )

        # The paper defines U_m against P^(t-1), hence the synchronized
        # post-local measurements used by QAA are also the values exported for
        # this round.  Recomputing them against the newly updated P^t would be a
        # different quantity.
        sample_ud_post_list = sample_ud_after_local_list

        client_ud_post = client_ud_after_local.copy()
        client_um_post = client_um_after_local.copy()
        for view in range(V):
            logger.info(
                f"[uncertainty-post] epoch={com_epoch} client={view} "
                f"client_U_d={client_ud_post[view]:.6e} "
                f"client_U_m={client_um_post[view]:.6e}"
            )

        uncertainty_recorder.record_round(
            communication_round=com_epoch + 1,
            client_ud=client_ud_post,
            client_um=client_um_post,
            acc=server_acc,
            nmi=server_nmi,
            ari=server_ari,
            f1=server_f1,
            client_ud_pre=client_ud_pre,
            client_um_pre=client_um_pre,
            client_gate_mean=client_gate_mean,
            client_gate_max=client_gate_max,
            client_um_loss=client_model_loss,
            client_reliability=client_reliability,
            qaa_max_weight=qaa_max_weight,
            qaa_weight_entropy=qaa_weight_entropy,
            recovery_kind=recovery_kind,
        )
        uncertainty_excel_path, uncertainty_plot_path = uncertainty_recorder.export()
        logger.info(
            f"[uncertainty-export] completed_round={com_epoch + 1} "
            f"Excel={uncertainty_excel_path} plot={uncertainty_plot_path}"
        )

        if (
            common_warmup_checkpoint_path is not None
            and not warmup_checkpoint_loaded
            and com_epoch + 1 == args.impute_warmup
        ):
            os.makedirs(
                os.path.dirname(common_warmup_checkpoint_path), exist_ok=True
            )
            checkpoint = {
                "signature": _warmup_signature(args),
                "completed_rounds": uncertainty_recorder.completed_rounds,
                "model_states": [
                    {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
                    for model in model_list
                ],
                "x_incomplete": [
                    value.detach().cpu().clone() for value in X_incomplete
                ],
                "p_local_list": [
                    value.detach().cpu().clone() for value in p_local_list
                ],
                "y_pred_global": np.asarray(y_pred_global).copy(),
                "ud_reference_scales": [
                    float(value) for value in ud_reference_scales
                ],
                "records": uncertainty_recorder.snapshot_records(),
                "rng_state": _capture_rng_state(),
            }
            temp_checkpoint_path = common_warmup_checkpoint_path + ".tmp"
            torch.save(checkpoint, temp_checkpoint_path)
            os.replace(temp_checkpoint_path, common_warmup_checkpoint_path)
            logger.info(
                f"[warmup-checkpoint] saved={common_warmup_checkpoint_path} "
                f"completed_rounds={uncertainty_recorder.completed_rounds}"
            )

    # model save
    for i in range(len(model_list)):
        save_path = os.path.join(log_folder_path, f"model{i}_{args.name}.pkl")
        torch.save(model_list[i].state_dict(), save_path)
