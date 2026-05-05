import argparse
from sklearn.cluster import KMeans
import torch
import numpy as np
from evaluation import eva
from model import GAT, DAEGC, GCN, GCNED
from pretrain import pretrain_gcn, pretrain_gat
from train import train_gat, train_gcn
from utils import get_weights, get_weights_from_U, update_A, update_P
from generate_incomplete_data import generate_data
from sklearn.preprocessing import normalize
from scipy.fftpack import fft
import random
import logging
import os
import time
from datetime import datetime
import shutil
import json
import math

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

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

def compute_alpha_from_entropy(p_global, num_clusters, alpha_min, alpha_max, eps=1e-12):
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
    alpha = float(alpha_min + (alpha_max - alpha_min) * mean_conf)
    if alpha < alpha_min:
        alpha = float(alpha_min)
    elif alpha > alpha_max:
        alpha = float(alpha_max)
    return alpha

def get_weights_safe(center, view):
    weights = []
    total = 0.0
    for i in range(view):
        var = np.var(center[i])
        total += var
        weights.append(var)
    weights = np.asarray(weights, dtype=float)
    weights = 1 + np.log2(1 + weights / total)
    return weights

if __name__ == "__main__":
    setup_seed(1018)

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
    parser.add_argument("--hidden_size_gcn", default=1024, type=int)
    parser.add_argument("--embedding_size_gcn", default=128, type=int)
    parser.add_argument("--weight_decay_gcn", type=float, default=0.0001)

    parser.add_argument("--communication_epoch", type=int, default=10)
    parser.add_argument("--local_epoch", type=int, default=50)
    parser.add_argument("--beta_gat", type=float, default=1)
    parser.add_argument("--beta_gcn", type=float, default=1)
    parser.add_argument("--missrates", type=float, nargs="+", default=None)
    parser.add_argument("--knn_neighbor", type=int, default=50)
    parser.add_argument('--yuzhi', type=float, default=0.1)
    parser.add_argument("--feat_miss_rate", type=float, default=0.1)
    parser.add_argument("--feat_miss_seed", type=int, default=1018)
    parser.add_argument("--knn_top_ratio", type=float, default=0.1)
    parser.add_argument("--knn_eps", type=float, default=1e-8)
    parser.add_argument("--knn_sigma", type=float, default=0.01)
    parser.add_argument("--impute_sanity_check", action="store_true")
    parser.add_argument("--use_view_impute", action="store_true", default=True)
    parser.add_argument("--no-use_view_impute", dest="use_view_impute", action="store_false")
    parser.add_argument("--impute_k", type=int, default=None)
    parser.add_argument("--impute_ref_mode", type=str, default="masked", choices=["masked", "concat"])
    parser.add_argument("--iterative_impute", action="store_true", default=True)
    parser.add_argument("--no-iterative_impute", dest="iterative_impute", action="store_false")
    parser.add_argument("--impute_warmup", type=int, default=0)
    parser.add_argument("--impute_every", type=int, default=1)
    parser.add_argument("--impute_eta", type=float, default=1.0)
    parser.add_argument("--alpha_min", type=float, default=0.05)
    parser.add_argument("--alpha_max", type=float, default=0.5)

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")
    # device = torch.device("cpu")
    impute_sanity_check = getattr(args, "impute_sanity_check", False)

    if args.name == 'Caltech101-7':
        #args.n_clusters = 7
       # args.view_num = 6
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
    folder_name = f'{args.name}_{current_time}'
    log_folder_path = os.path.join(log_dir, folder_name)
    log_file_name = f'{args.name}_{current_time}.log'
    if not os.path.exists(log_folder_path):
        os.makedirs(log_folder_path)
    log_file_path = os.path.join(log_folder_path, log_file_name)
    logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger()
    logger.info("Command line arguments: " + str(args))
    shutil.copyfile('main.py', log_folder_path + '/main.py')
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
    data_out = generate_data(
        direct_path,
        args.name,
        missrates_input,
        args.knn_neighbor,
        use_view_impute=args.use_view_impute,
        impute_k=args.impute_k,
        ref_mode=args.impute_ref_mode,
    )
    if args.use_view_impute:
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
        ) = data_out
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
    V = len(feature_list)
    if args.use_view_impute:
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
        if args.missrates[i]<=args.yuzhi:
            pre = GCN(
                num_features=X_incomplete[i].shape[1],
                hidden_size=args.hidden_size_gcn,
                embedding_size=args.embedding_size_gcn
            ).to(device)
            premodel_list.append(pre)
            print(f'view{i} use model GCN')
        else:
            pre = GAT(
                num_features=X_incomplete[i].shape[1],
                hidden_size=args.hidden_size_gat,
                embedding_size=args.embedding_size_gat,
                alpha=args.alpha_gat,
            ).to(device)
            premodel_list.append(pre)
            print(f'view{i} use model GAT')


    print('---------pretrain_start------------')
    for i in range(len(premodel_list)):
        if (isinstance(premodel_list[i], GAT)):
            gat_acc_dict, gat_nmi_dict, gat_ari_dict, gat_f1_dict = pretrain_gat(premodel_list[i], adj[i], adj_label[i].to(torch.float32), M[i], X_incomplete[i], Y_incomplete[i], args, logger)
        elif (isinstance(premodel_list[i], GCN)):
            gcn_acc_dict, gcn_nmi_dict, gcn_ari_dict, gcn_f1_dict = pretrain_gcn(premodel_list[i], adj[i], adj_label[i].to(torch.float32),X_incomplete[i], Y_incomplete[i], args, logger)
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

# train
    logger.info('---------train_start------------')
    U_list_prev = None
    for com_epoch in range(args.communication_epoch):
        if com_epoch==0:
            # update z
            features = []
            for i in range(len(model_list)):
                if (isinstance(model_list[i], DAEGC)):
                    _, feature = premodel_list[i](X_complete[i], adj_c[i], M_c[i])
                    features.append(feature.data.cpu().numpy())
                elif (isinstance(model_list[i], GCNED)):
                    _, feature = premodel_list[i](X_complete[i], adj_c[i])
                    features.append(feature.data.cpu().numpy())
                else:
                    print('pretrain model error')

            # update u
            kmeans = KMeans(n_clusters=args.n_clusters, n_init=100)
            y_pretrain = []
            center = []
            for view in range(V):
                y_pretrain.append(kmeans.fit_predict(features[view]))
                center.append([kmeans.cluster_centers_])

    # Server
        if U_list_prev is not None:
            print("DEBUG lens:", len(U_list_prev), len(view_missing_masks), "V=", V)
            assert len(U_list_prev) == len(view_missing_masks) == V
            weights = get_weights_from_U(U_list_prev, view_missing_masks)
        else:
            weights = get_weights_safe(center, V)
        n_features = []
        for view in range(V):
            n_features.append(features[view] * (weights[view]))
        z = np.hstack(n_features)

        save_path = os.path.join(log_folder_path, f"z_array_epoch_{com_epoch}.npy")
        np.save(save_path, z)

        # Update C
        kmean = KMeans(n_clusters=args.n_clusters, n_init=20)
        y_pred = kmean.fit_predict(z)
        Center_init = kmean.cluster_centers_

        # Update A
        if com_epoch == 0:
            y_pred_global = np.copy(y_pred)
        new_y, row_ind, col_ind, matrix = update_A(y_pred, y_pred_global)
        y_pred_global = np.copy(new_y)

        # Update P (global) and update local targets with inertia
        p_global = update_P(z, Center_init)
        p_global = np.dot(p_global, matrix)
        p_global = np.maximum(p_global, 0.0)
        p_global = p_global / (np.sum(p_global, axis=1, keepdims=True) + 1e-12)
        p_global_torch = torch.from_numpy(p_global).to(device=device, dtype=torch.float32)
        p_global_torch = p_global_torch.detach()
        alpha = compute_alpha_from_entropy(
            p_global_torch, args.n_clusters, args.alpha_min, args.alpha_max
        )
        for view in range(num_views):
            if p_local_list[view] is None:
                p_local_list[view] = p_global_torch.detach().clone()
            else:
                p_local_list[view] = (
                    (1.0 - alpha) * p_local_list[view] + alpha * p_global_torch
                ).detach()

        features = []
        q = []
        for i in range(len(model_list)):
            if (isinstance(model_list[i], DAEGC)):
                z_v, q_v = train_gat(model_list[i], adj_c[i], adj_c_label[i].to(torch.float32), M_c[i], X_complete[i], Y_complete, p_local_list[i], args)
                features.append(z_v.data.cpu().numpy())
                q.append(q_v.data.cpu().numpy())
            elif (isinstance(model_list[i], GCNED)):
                z_v, q_v = train_gcn(model_list[i], adj_c[i], adj_c_label[i].to(torch.float32), X_complete[i], Y_complete, p_local_list[i], args)
                features.append(z_v.data.cpu().numpy())
                q.append(q_v.data.cpu().numpy())
            else:
                print('model error')

        U_list_prev = q

        y_q = 0
        for view in range(V):
            y_q += q[view]

        y_mean_pred = y_q.argmax(1)
        acc, nmi, ari, f1 = eva(Y_complete, y_mean_pred, com_epoch)
        logger.info(f"common data :com_epoch {com_epoch}:acc {acc:.4f}, nmi {nmi:.4f}, ari {ari:.4f}, f1 {f1:.4f}")


        # update u
        kmeans = KMeans(n_clusters=args.n_clusters, n_init=100)
        center = []
        for view in range(V):
            kmeans.fit_predict(features[view])
            center.append([kmeans.cluster_centers_])

        # test
        y_pred_all = []
        q_pred_all = []
        z_pred_all = []
        with torch.no_grad():
            for i in range(len(model_list)):
                if (isinstance(model_list[i], DAEGC)):
                    m = get_M(adj[i])
                    m = m.to(device)
                    A_pred_v, z_v, q_v = model_list[i](X_incomplete[i], adj[i], m)
                    q_pred_all.append(q_v.data.cpu().numpy())
                    z_pred_all.append(z_v.data.cpu().numpy())
                    y_v = q_v.detach().data.cpu().numpy().argmax(1)
                    y_pred_all.append(y_v)
                    acc, nmi, ari, f1 = eva(Y_incomplete[i], y_pred_all[i], com_epoch)
                    logger.info(f"view{i} data :com_epoch {com_epoch}:acc {acc:.4f}, nmi {nmi:.4f}, ari {ari:.4f}, f1 {f1:.4f}")
                elif (isinstance(model_list[i], GCNED)):
                    A_pred_v, z_v, q_v = model_list[i](X_incomplete[i], adj[i])
                    q_pred_all.append(q_v.data.cpu().numpy())
                    z_pred_all.append(z_v.data.cpu().numpy())
                    y_v = q_v.detach().data.cpu().numpy().argmax(1)
                    y_pred_all.append(y_v)
                    acc, nmi, ari, f1 = eva(Y_incomplete[i], y_pred_all[i], com_epoch)
                    logger.info(f"view{i} data :com_epoch {com_epoch}:acc {acc:.4f}, nmi {nmi:.4f}, ari {ari:.4f}, f1 {f1:.4f}")
                else:
                    print('error')


        if args.use_view_impute:
            q_stack = np.stack(q_pred_all, axis=0)
            y_server = np.argmax(np.mean(q_stack, axis=0), axis=1)
        else:
            y_server = average_predictions(q_pred_all, incomplete_indices, V, args.n_clusters, num_samples)
        acc, nmi, ari, f1 = eva(labels, y_server, com_epoch)
        logger.info(f"all data :com_epoch {com_epoch}:acc {acc:.4f}, nmi {nmi:.4f}, ari {ari:.4f}, f1 {f1:.4f}")

        if iter_impute_enabled and com_epoch >= args.impute_warmup:
            if args.impute_every <= 0:
                args.impute_every = 1
            if (com_epoch - args.impute_warmup) % args.impute_every == 0:
                q_stack = np.stack(q_pred_all, axis=0)
                p_impute = np.mean(q_stack, axis=0)
                p_impute = np.maximum(p_impute, 0.0)
                p_impute = p_impute / (np.sum(p_impute, axis=1, keepdims=True) + 1e-12)
                if len(z_pred_all) == V:
                    X_np_list = [x.detach().cpu().numpy() for x in X_incomplete]
                    updated_views = 0
                    cluster_assign = np.argmax(p_impute, axis=1)
                    for v in range(V):
                        mask_view = np.asarray(mask_view_list[v], dtype=bool)
                        if mask_view.ndim != 1:
                            mask_view = mask_view.reshape(-1)
                        missing_rows = np.where(~mask_view)[0]
                        if missing_rows.size == 0:
                            continue
                        X_v = X_np_list[v]
                        Z_v = z_pred_all[v]
                        z_norm = np.linalg.norm(Z_v, axis=1, keepdims=True) + 1e-12
                        Z_v = Z_v / z_norm
                        present_idx = np.where(mask_view)[0]
                        if present_idx.size == 0:
                            continue
                        present_mean = np.mean(X_v[present_idx], axis=0)
                        for i in missing_rows:
                            same_cluster = cluster_assign == cluster_assign[i]
                            candidates = present_idx[same_cluster[present_idx]]
                            if candidates.size == 0:
                                X_v[i] = present_mean
                                continue
                            sim = np.dot(Z_v[candidates], Z_v[i])
                            k_eff = candidates.size
                            if args.impute_k is not None and args.impute_k > 0:
                                k_eff = min(k_eff, int(args.impute_k))
                            if k_eff < candidates.size:
                                top_idx = np.argpartition(-sim, k_eff - 1)[:k_eff]
                                candidates = candidates[top_idx]
                                sim = sim[top_idx]
                            sim = sim - np.max(sim)
                            w = np.exp(sim)
                            w_sum = np.sum(w)
                            if w_sum <= 0:
                                w = np.ones_like(w) / w.size
                            else:
                                w = w / w_sum
                            X_v[i] = np.sum(X_v[candidates] * w[:, None], axis=0)
                        X_np_list[v] = X_v
                        updated_views += 1

                    if updated_views > 0:
                        for v in range(V):
                            X_incomplete[v] = torch.Tensor(X_np_list[v]).to(device)
                        X_full_imputed_list = X_np_list
                        logger.info(f"[iter-impute] com_epoch {com_epoch}: updated_views={updated_views}")
                else:
                    logger.info("[iter-impute] skipped: missing embeddings for similarity weighting.")


    # model save
    for i in range(len(model_list)):
        save_path = os.path.join(log_folder_path, f"model{i}_{args.name}.pkl")
        torch.save(model_list[i].state_dict(), save_path)
