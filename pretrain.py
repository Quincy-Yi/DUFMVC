import argparse
from sklearn.cluster import KMeans
import torch
import torch.nn.functional as F
from torch.optim import Adam
from evaluation import eva
from model import GAT, GCN

import warnings
warnings.filterwarnings("ignore")

def pretrain_gcn(model, adj, adj_label, x, y, args, logger):
    # 还原：去掉 pos_weight
    optimizer = Adam(model.parameters(), lr=args.lr_pretrain_gcn, weight_decay=args.weight_decay_gcn)

    acc_dict = []
    nmi_dict = []
    ari_dict = []
    f1_dict = []

    for epoch in range(args.max_epoch_pretrain):
        model.train()
        optimizer.zero_grad()
        A_pred, z = model(x, adj)
        
        # 还原：标准 BCE Loss
        loss = F.binary_cross_entropy(A_pred.view(-1), adj_label.view(-1))
        
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            kmeans = KMeans(n_clusters=args.n_clusters, n_init=20).fit(
                z.data.cpu().numpy()
            )
            acc, nmi, ari, f1 = eva(y, kmeans.labels_, epoch)
            if epoch % 10 == 0:
                logger.info(f"pre_epoch {epoch}:rebuild_loss {loss:.4f}, acc {acc:.4f}, nmi {nmi:.4f}, ari {ari:.4f}, f1 {f1:.4f}")

            acc_dict.append(acc)
            nmi_dict.append(nmi)
            ari_dict.append(ari)
            f1_dict.append(f1)

    return acc_dict, nmi_dict, ari_dict, f1_dict

def pretrain_gat(model, adj, adj_label, M, x, y, args, logger):
    # 还原：去掉 pos_weight
    optimizer = Adam(model.parameters(), lr=args.lr_pretrain_gat, weight_decay=args.weight_decay_gat)

    acc_dict = []
    nmi_dict = []
    ari_dict = []
    f1_dict = []

    for epoch in range(args.max_epoch_pretrain):
        model.train()
        A_pred, z = model(x, adj, M)
        
        # 还原：标准 BCE Loss
        loss = F.binary_cross_entropy(A_pred.view(-1), adj_label.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            kmeans = KMeans(n_clusters=args.n_clusters, n_init=20).fit(
                z.data.cpu().numpy()
            )
            acc, nmi, ari, f1 = eva(y, kmeans.labels_, epoch)
            if epoch % 10 == 0:
                logger.info(f"pre_epoch {epoch}:loss {loss:.4f}, acc {acc:.4f}, nmi {nmi:.4f}, ari {ari:.4f}, f1 {f1:.4f}")

            acc_dict.append(acc)
            nmi_dict.append(nmi)
            ari_dict.append(ari)
            f1_dict.append(f1)

    return acc_dict, nmi_dict, ari_dict, f1_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="train", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--name", type=str, default="3sourceIncomplete")
    parser.add_argument("--max_epoch_pretrain", type=int, default=500)
    parser.add_argument("--n_clusters", default=5, type=int)
    parser.add_argument("--view_num", default=2, type=int)

    parser.add_argument("--lr_pretrain_gat", type=float, default=0.005)
    parser.add_argument('--lr_gat', type=float, default=0.001)
    parser.add_argument("--hidden_size_gat", default=512, type=int)
    parser.add_argument("--embedding_size_gat", default=64, type=int)
    parser.add_argument("--weight_decay_gat", type=int, default=5e-3)
    parser.add_argument("--alpha_gat", type=float, default=0.2, help="Alpha for the leaky_relu.")

    parser.add_argument("--lr_pretrain_gcn", type=float, default=0.005)
    parser.add_argument('--lr_gcn', type=float, default=0.001)
    parser.add_argument("--hidden_size_gcn", default=512, type=int)
    parser.add_argument("--embedding_size_gcn", default=16, type=int)
    parser.add_argument("--weight_decay_gcn", type=int, default=0.0001)

    parser.add_argument("--communication_epoch", type=int, default=10)
    parser.add_argument("--local_epoch", type=int, default=50)


    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")
    # device = torch.device("cpu")

    x, y, X_incomplete, Y_incomplete, X_complete, Y_complete, incomplete_indices, complete_indices, adj, adj_label, adj_c, adj_c_label, M, M_c = generate_data(args.name)
    for i in range(len(X_incomplete)):
        adj[i] = adj[i].to(device)
        adj_c[i] = adj_c[i].to(device)
        adj_label[i] = adj_label[i].to(device)
        adj_c_label[i] = adj_c_label[i].to(device)
        X_incomplete[i] = torch.Tensor(X_incomplete[i]).to(device)
        M[i] = M[i].to(device)
        M_c[i] = M_c[i].to(device)
        X_complete[i] = torch.Tensor(X_complete[i]).to(device)


# 加载预训练模型
    pre1 = GAT(
        num_features=X_incomplete[0].shape[1],
        hidden_size=args.hidden_size_gat,
        embedding_size=args.embedding_size_gat,
        alpha=args.alpha_gat,
    ).to(device)

    pre2 = GAT(
        num_features=X_incomplete[1].shape[1],
        hidden_size=args.hidden_size_gat,
        embedding_size=args.embedding_size_gat,
        alpha=args.alpha_gat,
    ).to(device)

    pre3 = GCN(
        num_features=X_incomplete[2].shape[1],
        hidden_size=args.hidden_size_gcn,
        embedding_size=args.embedding_size_gcn
    ).to(device)

    premodel_list = [pre1,pre2,pre3]

# 预训练
    print('---------pretrain_start------------')
    for i in range(len(premodel_list)):
        if (isinstance(premodel_list[i], GAT)):
            gat_acc_dict, gat_nmi_dict, gat_ari_dict, gat_f1_dict = pretrain_gat(premodel_list[i], adj[i], adj_label[i].to(torch.float32), M[i], X_incomplete[i], Y_incomplete[i], args)
        elif (isinstance(premodel_list[i], GCN)):
            gcn_acc_dict, gcn_nmi_dict, gcn_ari_dict, gcn_f1_dict = pretrain_gcn(premodel_list[i], adj[i], adj_label[i].to(torch.float32),X_incomplete[i], Y_incomplete[i], args)
        else:
            print('error')

    for i in range(len(premodel_list)):
        torch.save(premodel_list[i].state_dict(), f"./pretrain/premodel{i}_{args.name}.pkl")









