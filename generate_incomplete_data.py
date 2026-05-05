import os
import pdb
import time
import torch
import openpyxl
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import minmax_scale, maxabs_scale, normalize, robust_scale, scale
from functools import reduce
import h5py
from utils_impute import impute_sample_view_missing_knn

def load_Caltech_7(direct_path, dataset_name):
    target_path = direct_path + '/' + dataset_name + '.mat'
    data = sio.loadmat(target_path)
    X = data['X']

    def _extract_views(x):
        if isinstance(x, np.ndarray) and x.dtype == object:
            while isinstance(x, np.ndarray) and x.dtype == object and x.size == 1:
                x = x.item()
        if isinstance(x, np.ndarray) and x.dtype == object:
            if x.ndim == 2 and x.shape[0] == 1:
                return [x[0, i] for i in range(x.shape[1])]
            if x.ndim == 2 and x.shape[1] == 1:
                return [x[i, 0] for i in range(x.shape[0])]
            if x.ndim == 1:
                return list(x)
            return list(x.ravel())
        if isinstance(x, (list, tuple)):
            return list(x)
        if isinstance(x, np.ndarray):
            return [x]
        return [np.asarray(x)]

    views = _extract_views(X)
    feature_list = [normalize(v) for v in views]
    label_key = None
    for candidate in ['Y', 'y', 'labels', 'label', 'gt', 'gnd', 'truth']:
        if candidate in data:
            label_key = candidate
            break
    if label_key is None:
        raise KeyError(f"No label key found. Available keys: {list(data.keys())}")
    labels = data[label_key]
    labels = np.asarray(labels).squeeze()
    if labels.ndim != 1:
        labels = labels.flatten()
    labels = label_from_zero(labels).astype(np.int64)
    return feature_list, labels

def load_BDGP(direct_path, dataset_name):
    target_path = direct_path + '/' + dataset_name + '.mat'
    data = sio.loadmat(target_path)
    X = data['X']

    def _extract_views(x):
        if isinstance(x, np.ndarray) and x.dtype == object:
            while isinstance(x, np.ndarray) and x.dtype == object and x.size == 1:
                x = x.item()
        if isinstance(x, np.ndarray) and x.dtype == object:
            if x.ndim == 2 and x.shape[0] == 1:
                return [x[0, i] for i in range(x.shape[1])]
            if x.ndim == 2 and x.shape[1] == 1:
                return [x[i, 0] for i in range(x.shape[0])]
            if x.ndim == 1:
                return list(x)
            return list(x.ravel())
        if isinstance(x, (list, tuple)):
            return list(x)
        if isinstance(x, np.ndarray):
            return [x]
        return [np.asarray(x)]

    views = _extract_views(X)
    valid = list(range(len(views)))
    labels = data['y']
    labels = np.asarray(labels).squeeze()
    if labels.ndim != 1:
        labels = labels.flatten()
    labels = label_from_zero(labels).astype(np.int64)
    n = len(labels)

    feature_list = []
    for i in valid:
        v = views[i]
        while isinstance(v, np.ndarray) and v.dtype == object and v.size == 1:
            v = v.item()
        # Convert SciPy sparse matrices/arrays to dense 2D arrays first.
        if sp.issparse(v):
            v = v.toarray()
        elif hasattr(v, "toarray") and not isinstance(v, np.ndarray):
            try:
                v = v.toarray()
            except Exception:
                pass
        view = np.asarray(v)
        if view.ndim == 0:
            raise ValueError(f"[load_BDGP] view {i} is scalar after conversion: type={type(v)}")
        view = np.squeeze(view)
        if view.ndim == 1:
            view = view.reshape(-1, 1)
        elif view.ndim > 2:
            view = view.reshape(view.shape[0], -1)
        if view.ndim == 2 and view.shape[0] != n and view.shape[1] == n:
            view = view.T
        feature_list.append(normalize(view))
    if not feature_list:
        raise ValueError(f"No valid views found in X. views={views}")
    return feature_list, labels




def load_Scene15(direct_path, dataset_name):
    target_path = direct_path + '/' + dataset_name + '.mat'
    data = sio.loadmat(target_path)
    features = data['X']
    feature_list = []
    for i in range(3):
        fea = normalize(features[0][i])  # <class 'numpy.ndarray'>
        feature_list.append(fea)
        print(fea.shape)
    labels = data['Y'].flatten()
    labels = label_from_zero(labels)
    labels = labels.astype(np.int64)
    return feature_list, labels

def load_Reuters(direct_path, dataset_name):
    target_path = direct_path + '/' + dataset_name + '.mat'
    data = sio.loadmat(target_path)
    if 'X' in data:
        X = data['X']
    elif 'features' in data:
        X = data['features']
    else:
        raise KeyError(f"No feature key found. Available keys: {list(data.keys())}")

    def _extract_views(x):
        if isinstance(x, np.ndarray) and x.dtype == object:
            while isinstance(x, np.ndarray) and x.dtype == object and x.size == 1:
                x = x.item()
        if isinstance(x, np.ndarray) and x.dtype == object:
            if x.ndim == 2 and x.shape[0] == 1:
                return [x[0, i] for i in range(x.shape[1])]
            if x.ndim == 2 and x.shape[1] == 1:
                return [x[i, 0] for i in range(x.shape[0])]
            if x.ndim == 1:
                return list(x)
            return list(x.ravel())
        if isinstance(x, (list, tuple)):
            return list(x)
        if isinstance(x, np.ndarray):
            return [x]
        return [np.asarray(x)]

    def _to_dense_2d(v):
        while isinstance(v, np.ndarray) and v.dtype == object and v.size == 1:
            v = v.item()
        if sp.issparse(v):
            v = v.toarray()
        elif hasattr(v, 'toarray') and not isinstance(v, np.ndarray):
            try:
                v = v.toarray()
            except Exception:
                pass
        view = np.asarray(v)
        view = np.squeeze(view)
        if view.ndim == 1:
            view = view.reshape(-1, 1)
        elif view.ndim > 2:
            view = view.reshape(view.shape[0], -1)
        return view.astype(np.float32)

    
    views = _extract_views(X)
    feature_list = []
    if len(views) == 0:
        raise ValueError("No views found in X")
    if len(views) == 1:
        indices = [0]
    else:
        indices = list(range(len(views)))

    for i in indices:
        view = _to_dense_2d(views[i])
        if view.ndim == 0 or view.size == 0:
            print(f"[load_Reuters] warning: view {i} is empty, skip.")
            continue
        feature_list.append(normalize(view))
    if not feature_list:
        raise ValueError("No valid views after conversion")

    label_key = None
    for candidate in ['Y', 'y', 'labels', 'label', 'gt', 'gnd', 'truth']:
        if candidate in data:
            label_key = candidate
            break
    if label_key is None:
        raise KeyError(f"No label key found. Available keys: {list(data.keys())}")
    labels = data[label_key]
    labels = np.asarray(labels).squeeze()
    if labels.ndim != 1:
        labels = labels.flatten()
    labels = label_from_zero(labels).astype(np.int64)
    return feature_list, labels


def generate_data(
    direct_path,
    dataset_name,
    miss_rates,
    knn_neighbor,
    use_view_impute=False,
    impute_k=None,
    ref_mode="concat",
):
    print('load_data')
    if dataset_name == 'Caltech101-7':
        feature_list, labels = load_Caltech_7(direct_path, dataset_name)
    elif dataset_name == 'BDGP_4view':
        feature_list, labels = load_BDGP(direct_path, dataset_name)
    elif dataset_name == 'Scene-15':
        feature_list, labels = load_Scene15(direct_path, dataset_name)
    elif dataset_name == 'Reuters':
        feature_list , labels = load_Reuters(direct_path , dataset_name)
    else:
        print('datasets miss')
    # Normalize miss_rates to a per-view float list in [0, 1].
    V = len(feature_list)
    if isinstance(miss_rates, (int, float, np.integer, np.floating)):
        miss_rates = [float(miss_rates)] * V
    else:
        miss_rates = list(miss_rates)
        if len(miss_rates) == 0:
            raise ValueError("miss_rates must not be empty.")
        if len(miss_rates) == 1:
            miss_rates = miss_rates * V
        elif len(miss_rates) < V:
            miss_rates = miss_rates + [miss_rates[-1]] * (V - len(miss_rates))
        elif len(miss_rates) > V:
            miss_rates = miss_rates[:V]
        miss_rates = [float(r) for r in miss_rates]
    if any(r > 1.0 for r in miss_rates):
        miss_rates = [r / 100.0 for r in miss_rates]

    if use_view_impute:
        (
            X_incomplete,
            Y_incomplete,
            X_complete,
            Y_complete,
            incomplete_indices,
            complete_indices,
            mask_view_list,
            X_full_zero_list,
        ) = form_incomplete_data(miss_rates, feature_list, labels, return_full=True)
    else:
        (
            X_incomplete,
            Y_incomplete,
            X_complete,
            Y_complete,
            incomplete_indices,
            complete_indices,
        ) = form_incomplete_data(miss_rates, feature_list, labels, return_full=False)
        mask_view_list = None
        X_full_zero_list = None

    X_full_imputed_list = None
    X_incomplete_imputed_list = None
    adj_full_list = None
    adj_full_label_list = None
    M_full_list = None
    if use_view_impute:
        if impute_k is None:
            impute_k = knn_neighbor
        X_full_imputed_list = impute_sample_view_missing_knn(
            X_full_zero_list,
            mask_view_list,
            k=int(impute_k),
            ref_mode=ref_mode,
        )
        X_incomplete_imputed_list = []
        for v in range(len(feature_list)):
            present_idx = incomplete_indices[v]
            X_incomplete_imputed_list.append(X_full_imputed_list[v][present_idx])
        adj_full_list = []
        adj_full_label_list = []
        M_full_list = []
        for v in range(len(feature_list)):
            adj_full, adj_full_label = construct_adjacency_matrix_singleview(
                X_full_imputed_list[v],
                k_nearest_neighobrs=knn_neighbor,
                prunning_one=True,
                prunning_two=True,
                common_neighbors=2,
            )
            adj_full_list.append(adj_full)
            adj_full_label_list.append(adj_full_label)
            M_full_list.append(get_M(adj_full))

    print("----------------generate incomplete multi-view data adj-----------------------")
    view_num = len(X_incomplete)
    adj_list = []
    adj_label_list = []
    adj_c_list = []
    adj_c_label_list = []
    M_list = []
    M_c_list = []
    for i in range(view_num):
        new_index = np.where(incomplete_indices[i] == complete_indices[:, None])[1]

        adj, adj_label, adj_c, adj_c_label = construct_adjacency_matrix(X_incomplete[i], k_nearest_neighobrs=knn_neighbor, prunning_one=True, prunning_two=True, common_neighbors=2, new_index=new_index)
        adj_list.append(adj)
        adj_label_list.append(adj_label)
        adj_c_list.append(adj_c)
        adj_c_label_list.append(adj_c_label)

        m = get_M(adj)
        m_c = get_M(adj_c)
        M_list.append(m)
        M_c_list.append(m_c)

    base = (
        feature_list,
        labels,
        X_incomplete,
        Y_incomplete,
        X_complete,
        Y_complete,
        incomplete_indices,
        complete_indices,
        adj_list,
        adj_label_list,
        adj_c_list,
        adj_c_label_list,
        M_list,
        M_c_list,
    )
    if use_view_impute:
        return base + (
            X_full_imputed_list,
            mask_view_list,
            X_incomplete_imputed_list,
            X_full_zero_list,
            adj_full_list,
            adj_full_label_list,
            M_full_list,
        )
    return base

def construct_adjacency_matrix(features, k_nearest_neighobrs, prunning_one, prunning_two, common_neighbors, new_index):
    nbrs = NearestNeighbors(n_neighbors=k_nearest_neighobrs + 1, algorithm='ball_tree').fit(features)
    adj_wave = nbrs.kneighbors_graph(features)

    if prunning_one:
        # Pruning strategy 1
        original_adj_wave = adj_wave.toarray()
        judges_matrix = original_adj_wave == original_adj_wave.T
        np_adj_wave = original_adj_wave * judges_matrix
        adj_wave = sp.csc_matrix(np_adj_wave)
    else:
        # transform the matrix to be symmetric (Instead of Pruning strategy 1)
        np_adj_wave = construct_symmetric_matrix(adj_wave.toarray())
        adj_wave = sp.csc_matrix(np_adj_wave)

    # obtain the adjacency matrix without self-connection
    adj = sp.csc_matrix(np_adj_wave)
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()

    if prunning_two:
        # Pruning strategy 2
        adj = adj.toarray()
        b = np.nonzero(adj)
        rows = b[0]
        cols = b[1]
        dic = {}
        for row, col in zip(rows, cols):
            if row in dic.keys():
                dic[row].append(col)
            else:
                dic[row] = []
                dic[row].append(col)
        for row, col in zip(rows, cols):
            if len(set(dic[row]) & set(dic[col])) < common_neighbors:
                adj[row][col] = 0
        adj = sp.csc_matrix(adj)
        adj.eliminate_zeros()

    adj_coo = adj.tocoo()
    adj_tensor = torch.sparse_coo_tensor(
        torch.tensor([adj_coo.row, adj_coo.col]),
        torch.from_numpy(adj_coo.data).float(),
        size=adj_coo.shape
    )
    adj = adj_tensor.to_dense()

    adj_label = adj
    select_adj = adj[:, new_index][new_index, :]
    adj_c_label = select_adj

    adj += torch.eye(adj.shape[0])
    adj = normalize(adj, norm="l1")
    adj = torch.from_numpy(adj).to(dtype=torch.float)
    select_adj += torch.eye(select_adj.shape[0])
    select_adj = normalize(select_adj, norm="l1")
    select_adj = torch.from_numpy(select_adj).to(dtype=torch.float)

    return adj, adj_label, select_adj, adj_c_label

def construct_symmetric_matrix(original_matrix):
    """
        transform a matrix (n*n) to be symmetric
    :param np_matrix: <class 'numpy.ndarray'>
    :return: result_matrix: <class 'numpy.ndarray'>
    """
    result_matrix = np.zeros(original_matrix.shape, dtype=float)
    num = original_matrix.shape[0]
    for i in range(num):
        for j in range(num):
            if original_matrix[i][j] == 0:
                continue
            elif original_matrix[i][j] == 1:
                result_matrix[i][j] = 1
                result_matrix[j][i] = 1
            else:
                print("The value in the original matrix is illegal!")
                pdb.set_trace()
    assert (result_matrix == result_matrix.T).all() == True

    if ~(np.sum(result_matrix, axis=1) > 1).all():
        print("There existing a outlier!")
        pdb.set_trace()

    return result_matrix


def label_from_zero(labels):
    min_num = min(set(labels))
    return labels - min_num

def get_M(adj):
    adj_numpy = np.array(adj)
    # t_order
    t=2
    tran_prob = normalize(adj_numpy, norm="l1", axis=0)
    M_numpy = sum([np.linalg.matrix_power(tran_prob, i) for i in range(1, t + 1)]) / t
    return torch.Tensor(M_numpy)

def form_incomplete_data(miss_rates, X, Y, return_full=False):
    np.random.seed(0)
    size = len(Y)
    view_num = len(X)
    missdata_indices = []
    incomplete_indices = []
    complete_indices = np.arange(size)
    indices = np.arange(size)
    for i in range(view_num):
        if (i != view_num-1):
            missing_size = int(miss_rates[i] * size)
            missing_indices = np.random.choice(indices, size=missing_size, replace=False)
            missdata_indices.append(missing_indices)
            incomplete_indices.append(np.setdiff1d(indices, missing_indices))
            complete_indices = np.setdiff1d(complete_indices, missing_indices)
        else:
            missing_size = int(miss_rates[i] * size)
            missdata = missdata_indices[0]
            for j in range(1, len(missdata_indices)):
                missdata = np.intersect1d(missdata, missdata_indices[j])
            remind_indices = np.setdiff1d(indices, missdata)
            missing_indices = np.random.choice(remind_indices, size=missing_size, replace=False)
            incomplete_indices.append(np.setdiff1d(indices, missing_indices))
            complete_indices = np.setdiff1d(complete_indices, missing_indices)

    X_incomplete = [X[i][incomplete_indices[i]] for i in range(view_num)]
    Y_incomplete = [Y[incomplete_indices[i]] for i in range(view_num)]
    X_complete = [X[i][complete_indices] for i in range(view_num)]
    Y_complete = Y[complete_indices]

    if not return_full:
        return X_incomplete, Y_incomplete, X_complete, Y_complete, incomplete_indices, complete_indices

    mask_view_list = []
    X_full_zero_list = []
    for i in range(view_num):
        mask_view = np.zeros(size, dtype=np.int64)
        mask_view[incomplete_indices[i]] = 1
        mask_view_list.append(mask_view)

        X_full = np.asarray(X[i])
        if X_full.ndim == 1:
            X_full = X_full.reshape(-1, 1)
        X_full_zero = X_full.copy()
        X_full_zero[mask_view == 0] = 0.0
        X_full_zero_list.append(X_full_zero)

    return (
        X_incomplete,
        Y_incomplete,
        X_complete,
        Y_complete,
        incomplete_indices,
        complete_indices,
        mask_view_list,
        X_full_zero_list,
    )

def construct_adjacency_matrix_singleview(features, k_nearest_neighobrs, prunning_one, prunning_two, common_neighbors):
    nbrs = NearestNeighbors(n_neighbors=k_nearest_neighobrs + 1, algorithm='ball_tree').fit(features)
    adj_wave = nbrs.kneighbors_graph(features)

    if prunning_one:
        # Pruning strategy 1
        original_adj_wave = adj_wave.toarray()
        judges_matrix = original_adj_wave == original_adj_wave.T
        np_adj_wave = original_adj_wave * judges_matrix
        adj_wave = sp.csc_matrix(np_adj_wave)
    else:
        # transform the matrix to be symmetric (Instead of Pruning strategy 1)
        np_adj_wave = construct_symmetric_matrix(adj_wave.toarray())
        adj_wave = sp.csc_matrix(np_adj_wave)

    # obtain the adjacency matrix without self-connection
    adj = sp.csc_matrix(np_adj_wave)
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()

    if prunning_two:
        # Pruning strategy 2
        adj = adj.toarray()
        b = np.nonzero(adj)
        rows = b[0]
        cols = b[1]
        dic = {}
        for row, col in zip(rows, cols):
            if row in dic.keys():
                dic[row].append(col)
            else:
                dic[row] = []
                dic[row].append(col)
        for row, col in zip(rows, cols):
            if len(set(dic[row]) & set(dic[col])) < common_neighbors:
                adj[row][col] = 0
        adj = sp.csc_matrix(adj)
        adj.eliminate_zeros()
    adj_coo = adj.tocoo()
    adj_tensor = torch.sparse_coo_tensor(
        torch.tensor([adj_coo.row, adj_coo.col]),
        torch.from_numpy(adj_coo.data).float(),
        size=adj_coo.shape
    )
    adj = adj_tensor.to_dense()

    adj_label = adj

    adj += torch.eye(adj.shape[0])
    adj = normalize(adj, norm="l1")
    adj = torch.from_numpy(adj).to(dtype=torch.float)

    return adj, adj_label
