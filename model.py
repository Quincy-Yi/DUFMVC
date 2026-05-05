import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from layer import GATLayer
from layer import GraphConvolution
from layer import GraphConvSparse

# gat
class GAT(nn.Module):
    def __init__(self, num_features, hidden_size, embedding_size, alpha):
        super(GAT, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.alpha = alpha
        self.conv1 = GATLayer(num_features, hidden_size, alpha)
        self.conv2 = GATLayer(hidden_size, embedding_size, alpha)

    def forward(self, x, adj, M):
        h = self.conv1(x, adj, M)
        h = self.conv2(h, adj, M)
        z = F.normalize(h, p=2, dim=1)
        A_pred = self.dot_product_decode(z)
        return A_pred, z

    def dot_product_decode(self, Z):
        A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
        return A_pred

class DAEGC(nn.Module):
    def __init__(self, pretrain_model, args, v=1):
        super(DAEGC, self).__init__()
        self.num_clusters = args.n_clusters
        self.v = v

        # get pretrain model
        self.gat = pretrain_model

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(args.n_clusters, args.embedding_size_gat))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def forward(self, x, adj, M):
        A_pred, z = self.gat(x, adj, M)
        q = self.get_Q(z)

        return A_pred, z, q

    def get_Q(self, z):
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q


# gcn
class GCN(nn.Module):
    def __init__(self, num_features, hidden_size, embedding_size):
        super(GCN, self).__init__()
        self.num_feature = num_features
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.hidden1 = GraphConvolution(input_dim=num_features, output_dim=hidden_size, act=F.relu)
        self.embeddings = GraphConvolution(input_dim=hidden_size, output_dim=embedding_size, act=lambda x: x)

    def forward(self, inputs, adj):
        h = self.hidden1(inputs, adj)
        h = self.embeddings(h, adj)
        z = F.normalize(h, p=2, dim=1)
        A_pred = self.dot_product_decode(z)
        return A_pred, z

    def dot_product_decode(self, Z):
        A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
        return A_pred


class GCNED(nn.Module):
    def __init__(self, pretrain_model, args, v=1):
        super(GCNED, self).__init__()
        self.num_clusters = args.n_clusters
        self.v = v

        # get pretrain model
        self.gcn = pretrain_model

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(args.n_clusters, args.embedding_size_gcn))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def forward(self, x, adj):
        A_pred, z = self.gcn(x, adj)
        q = self.get_Q(z)

        return A_pred, z, q

    def get_Q(self, z):
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q



