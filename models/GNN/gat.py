from torch_geometric.nn import GAT, GCNConv, GATConv
from torch.nn import Linear, Module
import torch.nn as nn
from torch import tanh, cat
from torch import argmax, tensor, Tensor
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from torch_geometric.nn.pool import SAGPooling, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch
from torch_scatter import scatter

logger = TensorBoardLogger(name="EXPERIMENT_DRUGO_SYMP", save_dir="../tb_logs", version=1)


# Custom filter_adj function
def filter_adj(edge_index, edge_attr, perm, num_nodes=None):
    device = edge_index.device

    if num_nodes is None:
        num_nodes = edge_index.max().item() + 1

    mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    mask[perm] = 1

    edge_mask = mask[edge_index[0]] & mask[edge_index[1]]

    edge_index = edge_index[:, edge_mask]
    if edge_attr is not None:
        edge_attr = edge_attr[edge_mask]

    idx = torch.zeros(num_nodes, dtype=torch.long, device=device)
    idx[perm] = torch.arange(perm.size(0), device=device)

    edge_index = idx[edge_index]

    return edge_index, edge_attr


# Custom topk function
def topk(x, ratio, batch):
    num_nodes = scatter(batch, batch, reduce="max") + 1
    k = (ratio * num_nodes).ceil().long()

    batch_size = batch.max() + 1
    perm = torch.cat([x[batch == i].argsort(descending=True)[:k[i]] + (batch == i).nonzero(as_tuple=False)[0].item()
                      for i in range(batch_size)], dim=0)
    return perm


class SAGPool(nn.Module):
    def __init__(self, in_channels, ratio=0.8, Conv=GCNConv, non_linearity=tanh):
        super(SAGPool, self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.score_layer = Conv(in_channels, 1)
        self.non_linearity = non_linearity

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        score = self.score_layer(x, edge_index).squeeze()

        perm = topk(score, self.ratio, batch)
        x = x[perm] * self.non_linearity(score[perm]).view(-1, 1)
        batch = batch[perm]
        edge_index, edge_attr = filter_adj(
            edge_index, edge_attr, perm, num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm


class GraphAttentionModel(pl.LightningModule):
    def __init__(self, config=None, out_size=14, num_node_features=700, gnn_hidden_size=700, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not config:
            config = {
                'graph_hidden_layers': 3,
                'num_node_features': num_node_features,
                "num_encoded_features": 200,
                "num_gnn_hidden": 12,
                "gnn_hidden_size": gnn_hidden_size,
                "num_labels": out_size,
                "lr": 1e-4,
                "weight_decay": 5e-4,
                "train_index": None,  # tensor(train_index) == True,
                "val_index": None  # tensor(val_index) == True
            }
        self.graph_hidden_layers = config['graph_hidden_layers']
        self.num_node_features = config["num_node_features"]
        self.num_encoded_features = config['num_encoded_features']
        self.num_gnn_hidden = config['num_gnn_hidden']
        self.gnn_hidden_size = config['gnn_hidden_size']
        self.num_labels = config['num_labels']
        self.lr = config['lr']
        self.weight_decay = config['weight_decay']
        self.val_index = config['val_index']
        self.train_index = config['train_index']
        self.graph_encoder = GAT(self.num_node_features, self.gnn_hidden_size, self.num_gnn_hidden,
                                 self.num_encoded_features)
        self.classifier = Linear(self.num_encoded_features, self.num_labels)
        # self.logger = TensorBoardLogger(name="EXPERIMENT_DRUGO_SYMP", save_dir="../tb_logs", version=1)

    def log(self, name, value, idx):
        logger.log_metrics({name: value}, idx)

    def forward(self, x, edge_index):
        graph_out = self.graph_encoder(x, edge_index)
        return self.classifier(graph_out)

    def training_step(self, graph, epoch_idx, *args, **kwargs):
        x, edge_index, y = graph.x, graph.edge_index, graph.y
        logits = self(x, edge_index)
        y, logits = y[self.train_index], logits[self.train_index]
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss, epoch_idx)
        return loss

    def validation_step(self, graph, epoch_idx, *args, **kwargs):
        x, edge_index, y = graph.x, graph.edge_index, graph.y
        logits = self(x, edge_index)
        y, logits = y[self.val_index], logits[self.val_index]
        loss = F.cross_entropy(logits, y)
        y_hat = argmax(logits, dim=1)
        accuracy = accuracy_score(y, y_hat)
        self.log("val_loss", loss, epoch_idx)
        self.log("val_acc", accuracy, epoch_idx)
        print(f"Current validation_loss at epoch: {epoch_idx}: {loss.item()}")
        print(f"Current validation_accuracy at epoch: {epoch_idx}: {accuracy}")
        return loss.item()

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def get_embeddings(self, graph):
        return self.graph_encoder(graph.x, graph.edge_index)


class GraphAttentionModelPooling(nn.Module):
    def __init__(self, config=None, num_node_features=700, gnn_hidden_size=700, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not config:
            config = {
                'graph_hidden_layers': 3,
                'num_node_features': num_node_features,
                "num_encoded_features": 200,
                "num_gnn_hidden": 12,
                "gnn_hidden_size": gnn_hidden_size,
                "num_labels": -1,
                "lr": 1e-4,
                "weight_decay": 5e-4,
                "train_index": None,  # tensor(train_index) == True,
                "val_index": None  # tensor(val_index) == True
            }
        self.graph_hidden_layers = config['graph_hidden_layers']
        self.num_node_features = config["num_node_features"]
        self.num_encoded_features = config['num_encoded_features']
        self.num_gnn_hidden = config['num_gnn_hidden']
        self.gnn_hidden_size = config['gnn_hidden_size']
        # self.num_labels = config['num_labels']
        self.lr = config['lr']
        self.weight_decay = config['weight_decay']
        self.val_index = config['val_index']
        self.train_index = config['train_index']
        self.graph