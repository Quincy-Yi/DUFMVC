import argparse

import torch
import torch.nn.functional as F
from torch.optim import Adam

#from multiview import utils
from model import GAT, DAEGC, GCN, GCNED
from evaluation import eva
from pretrain import pretrain_gcn, pretrain_gat

import warnings
warnings.filterwarnings("ignore")

def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

def train_gat(model, adj, adj_label, M, data, y, p, args):
    # print(model)
    optimizer = Adam(model.parameters(), lr=args.lr_gat, weight_decay=args.weight_decay_gat)

    acc_dict = []
    nmi_dict = []
    ari_dict = []
    f1_dict = []

    for epoch in range(args.local_epoch):
        model.train()
        A_pred, z, q = model(data, adj, M)
        y_pred = q.detach().data.cpu().numpy().argmax(1)

        acc, nmi, ari, f1 = eva(y, y_pred, epoch)
        acc_dict.append(acc)
        nmi_dict.append(nmi)
        ari_dict.append(ari)
        f1_dict.append(f1)

        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        re_loss = F.binary_cross_entropy(A_pred.view(-1), adj_label.view(-1))
        loss = args.beta_gat * kl_loss + re_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return z, q

def train_gcn(model, adj, adj_label, data, y, p, args):
    # print(model)
    optimizer = Adam(model.parameters(), lr=args.lr_gcn, weight_decay=args.weight_decay_gcn)

    acc_dict = []
    nmi_dict = []
    ari_dict = []
    f1_dict = []

    for epoch in range(args.local_epoch):
        model.train()
        A_pred, z, q = model(data, adj)
        y_pred = q.detach().data.cpu().numpy().argmax(1)

        acc, nmi, ari, f1 = eva(y, y_pred, epoch)
        acc_dict.append(acc)
        nmi_dict.append(nmi)
        ari_dict.append(ari)
        f1_dict.append(f1)

        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        re_loss = F.binary_cross_entropy(A_pred.view(-1), adj_label.view(-1))


        loss = args.beta_gcn * kl_loss + re_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return z, q



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='Cora')
    #    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument("--max_epoch_pretrain", type=int, default=100)
    parser.add_argument('--epoch', type=int, default=95)  # 加载预训练的epoch
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--n_clusters', default=6, type=int)
    parser.add_argument('--update_interval', default=1, type=int)  # [1,3,5]，PQ更新频率

    parser.add_argument("--lr_pretrain_gat", type=float, default=0.001)
    parser.add_argument('--lr_gat', type=float, default=0.0001)
    parser.add_argument('--hidden_size_gat', default=256, type=int)
    parser.add_argument('--embedding_size_gat', default=16, type=int)
    parser.add_argument('--weight_decay_gat', type=int, default=5e-3)
    parser.add_argument('--alpha_gat', type=float, default=0.2, help='Alpha for the leaky_relu.')


    parser.add_argument("--lr_pretrain_gcn", type=float, default=0.001)
    parser.add_argument('--lr_gcn', type=float, default=0.0001)
    parser.add_argument('--hidden_size_gcn', default=1024, type=int)
    parser.add_argument('--embedding_size_gcn', default=16, type=int)
    parser.add_argument('--weight_decay_gcn', type=int, default=0.0001)


    parser.add_argument("--lr_pretrain_vgae", type=float, default=0.005)
    parser.add_argument('--lr_vgae', type=float, default=0.005)
    parser.add_argument('--hidden_size_vgae', default=32, type=int)
    parser.add_argument('--embedding_size_vgae', default=16, type=int)

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")

# load dataset
    datasets = utils.get_dataset(args.name)
    dataset = datasets[0]
    # data process
    dataset = utils.data_preprocessing(dataset)
    adj = dataset.adj.to(device)
    adj_label = dataset.adj_label.to(device)
    M = utils.get_M(adj).to(device)
    # data and label
    x= torch.Tensor(dataset.x).to(device)
    y = dataset.y.cpu().numpy()



    args.input_dim = dataset.num_features
    print(args)

# gat
    gat_pre = GAT(
        num_features=args.input_dim_gat,
        hidden_size=args.hidden_size_gat,
        embedding_size=args.embedding_size_gat,
        alpha=args.alpha_gat,
    ).to(device)
    pretrain_gat(gat_pre, adj, adj_label, M, x, y, args)

    gat = DAEGC(pretrain_model=gat_pre, args=args).to(device)
    z = train_gat(gat, adj, adj_label, M, x, y, args)
    print(z)

# gcn
    gcn_pre = GCN(
        num_features=args.input_dim_gcn,
        hidden_size=args.hidden_size_gcn,
        embedding_size=args.embedding_size_gcn,
    ).to(device)
    pretrain_gcn(gcn_pre, adj, adj_label, x, y, args)

    # train(include PQ)
    gcn = GCNED(pretrain_model=gcn_pre, args=args).to(device)
    z = train_gcn(gcn, adj, adj_label, x, y, args)
    print(z)





