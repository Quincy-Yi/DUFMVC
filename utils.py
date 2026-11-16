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



