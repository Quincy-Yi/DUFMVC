import numpy as np
import torch
from sklearn.preprocessing import normalize

# gat gcn
def data_preprocessing(dataset):
    dataset.adj = torch.sparse_coo_tensor(
        dataset.edge_index, torch.ones(dataset.edge_index.shape[1]), torch.Size([dataset.x.shape[0], dataset.x.shape[0]])
    ).to_dense()

    dataset.adj_label = dataset.adj

    dataset.adj += torch.eye(dataset.x.shape[0])
    dataset.adj = normalize(dataset.adj, norm="l1")
    dataset.adj = torch.from_numpy(dataset.adj).to(dtype=torch.float)

    return dataset


def get_M(adj):
    adj_numpy = adj.cpu().numpy()
    t=2
    tran_prob = normalize(adj_numpy, norm="l1", axis=0)
    M_numpy = sum([np.linalg.matrix_power(tran_prob, i) for i in range(1, t + 1)]) / t
    return torch.Tensor(M_numpy)


def get_weights(center,view):
    weights = []
    sum = 0
    for i in range(view):
        var = np.var(center[i])
        sum += var
        weights.append(var)

    weights = 1 + np.log2(1 + weights / sum)

    return weights

def get_weights_from_U(U_list, feature_missing_masks, alpha=1.0, lam=1.0, eps=1e-8):
    """
    U_list: List[np.ndarray], each of shape [N_complete, K]
    feature_missing_masks: List[np.ndarray|None], missing indicator or missing ratio tensor
    return: np.ndarray of shape [view_num], normalized weights
    """
    if len(U_list) != len(feature_missing_masks):
        raise ValueError("U_list and feature_missing_masks must have the same length.")

    weights = []
    for U, mask in zip(U_list, feature_missing_masks):
        U = np.asarray(U, dtype=float)
        if U.ndim != 2:
            raise ValueError("Each U must be a 2D array with shape [N, K].")

        N, K = U.shape
        if N == 0 or K == 0:
            weights.append(eps)
            continue

        U = np.clip(U, eps, 1.0)
        row_sum = np.sum(U, axis=1, keepdims=True)
        U = U / (row_sum + eps)

        if mask is None:
            rho_bar = 0.0
        else:
            mask_arr = np.asarray(mask, dtype=float)
            if mask_arr.size == 0:
                rho_bar = 0.0
            else:
                rho_bar = float(np.mean(mask_arr))
        rho_bar = float(np.clip(rho_bar, 0.0, 1.0))

        log_k = np.log(float(K) + eps)
        ent = -np.sum(U * np.log(U + eps), axis=1) / log_k
        H_bar = float(np.mean(ent)) if ent.size else 0.0

        conf = np.max(U, axis=1)
        sigma = float(np.var(conf))

        denom = (rho_bar ** alpha) + (lam * H_bar) + eps
        s = np.log1p(sigma) / denom

        w_prime = 1.0 - np.exp(-s)
        weights.append(w_prime)

    weights = np.asarray(weights, dtype=float)
    weights = np.clip(weights, eps, None)
    weights = weights / np.sum(weights)
    return weights

def update_A(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    new_y = np.zeros(y_true.shape[0])

    matrix = np.zeros((D, D), dtype=np.int64)
    matrix[row_ind, col_ind] = 1
    for i in range(y_pred.size):
        for j in row_ind:
            if y_true[i] == col_ind[j]:
                new_y[i] = row_ind[j]
    return new_y, row_ind, col_ind, matrix

def update_P(inputs, centers):
    alpha = 1
    q = 1.0 / (1.0 + (np.sum(np.square(np.expand_dims(inputs, axis=1) - centers), axis=2) / alpha))
    q **= (alpha + 1.0) / 2.0
    q = np.transpose(np.transpose(q) / np.sum(q, axis=1))
    q = target_distribution(q)
    return q

def target_distribution(q):
    t = 2
    weight = q ** t
    return (weight.T / weight.sum(1)).T



