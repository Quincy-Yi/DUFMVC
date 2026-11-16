import torch.nn.functional as F
from torch.optim import Adam
import warnings
warnings.filterwarnings("ignore")

def pretrain_gcn(model, adj, adj_label, x, args, logger):
    # 还原：去掉 pos_weight
    optimizer = Adam(model.parameters(), lr=args.lr_pretrain_gcn, weight_decay=args.weight_decay_gcn)

    for epoch in range(args.max_epoch_pretrain):
        model.train()
        optimizer.zero_grad()
        A_pred, z = model(x, adj)
        
        # 还原：标准 BCE Loss
        loss = F.binary_cross_entropy(A_pred.view(-1), adj_label.view(-1))
        
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            logger.info(f"pre_epoch {epoch}:rebuild_loss {loss:.4f}")

def pretrain_gat(model, adj, adj_label, M, x, args, logger):
    # 还原：去掉 pos_weight
    optimizer = Adam(model.parameters(), lr=args.lr_pretrain_gat, weight_decay=args.weight_decay_gat)

    for epoch in range(args.max_epoch_pretrain):
        model.train()
        A_pred, z = model(x, adj, M)
        
        # 还原：标准 BCE Loss
        loss = F.binary_cross_entropy(A_pred.view(-1), adj_label.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            logger.info(f"pre_epoch {epoch}:loss {loss:.4f}")

