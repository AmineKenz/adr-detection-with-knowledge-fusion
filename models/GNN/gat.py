# %%
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
from torch_geometric.nn.pool.topk_pool import topk,filter_adj
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
logger = TensorBoardLogger(name="EXPERIMENT_DRUGO_SYMP", save_dir="../tb_logs", version=1)




class SAGPool(nn.Module):
    def __init__(self,in_channels,ratio=0.8,Conv=GCNConv,non_linearity=tanh):
        super(SAGPool,self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.score_layer = Conv(in_channels,1)
        self.non_linearity = non_linearity
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        #x = x.unsqueeze(-1) if x.dim() == 1 else x
        score = self.score_layer(x,edge_index).squeeze()

        perm = topk(score, self.ratio, batch)
        x = x[perm] * self.non_linearity(score[perm]).view(-1, 1)
        batch = batch[perm]
        edge_index, edge_attr = filter_adj(
            edge_index, edge_attr, perm, num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm

class GraphAttentionModel(pl.LightningModule):
    def __init__(self, config=None, out_size=14, num_node_features=700, gnn_hidden_size = 700, *args, **kwargs):
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
                "train_index": None, #tensor(train_index) == True,
                "val_index": None #tensor(val_index) == True
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
        self.graph_encoder = GAT(self.num_node_features, self.gnn_hidden_size, self.num_gnn_hidden, self.num_encoded_features)
        self.classifier = Linear(self.num_encoded_features, self.num_labels)
        #self.logger = TensorBoardLogger(name="EXPERIMENT_DRUGO_SYMP", save_dir="../tb_logs", version=1)

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
        "train_index": None, #tensor(train_index) == True,
        "val_index": None #tensor(val_index) == True
    }
        self.graph_hidden_layers = config['graph_hidden_layers']
        self.num_node_features = config["num_node_features"] 
        self.num_encoded_features = config['num_encoded_features']
        self.num_gnn_hidden = config['num_gnn_hidden']
        self.gnn_hidden_size = config['gnn_hidden_size']
        #self.num_labels = config['num_labels']
        self.lr = config['lr']
        self.weight_decay = config['weight_decay']
        self.val_index = config['val_index']
        self.train_index = config['train_index']
        self.graph_encoder = GAT(self.num_node_features, self.gnn_hidden_size, self.num_gnn_hidden, self.num_encoded_features)
        
        self.pooler = SAGPooling(self.num_encoded_features, ratio=0.5, Conv=GATConv)
        
    def forward(self, x, edge_index):
        graph_out = self.graph_encoder(x, edge_index)

        ## pooling

        out = self.pooler(graph_out, edge_index)
        x2 = cat([gmp(out[0], out[3]), gap(out[0], out[3])], dim=1).squeeze()

        return x2
    

class SAGNet(nn.Module):
    def __init__(self,**kwargs):
        super(SAGNet, self).__init__()
        
        self.num_features = kwargs['num_features']
        self.nhid = kwargs['nhid']
        #self.num_classes = args.num_classes
        self.pooling_ratio = kwargs['pooling_ratio']
        #self.dropout_ratio =  kwargs['dropout_ratio']
        
        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.pool1 = SAGPool(self.nhid, ratio=self.pooling_ratio)
        self.conv2 = GCNConv(self.nhid, self.nhid)
        self.pool2 = SAGPool(self.nhid, ratio=self.pooling_ratio)
        self.conv3 = GCNConv(self.nhid, self.nhid)
        self.pool3 = SAGPool(self.nhid, ratio=self.pooling_ratio)

        self.lin1 = nn.Linear(self.nhid*2, self.nhid)
        # self.lin2 = nn.Linear(self.nhid, self.nhid//2)
        # self.lin3 = nn.Linear(self.nhid//2, self. num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _ = self.pool1(x, edge_index, None, batch)
        x1 = cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _ = self.pool2(x, edge_index, None, batch)
        x2 = cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _ = self.pool3(x, edge_index, None, batch)
        x3 = cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3

        x = F.relu(self.lin1(x))
        # x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        # x = F.relu(self.lin2(x))
        # x = F.log_softmax(self.lin3(x), dim=-1)

        return x

# %%
graphs = {
    'GAT_BERT_SYMP': "../../knowledge_graphs/BERT_embs_graph_data_SYMP.pkl",    
    'GAT_BERT_DRUGO': "../../knowledge_graphs/BERT_embs_graph_data_DRUGO.pkl",
    'GAT_BERT_DRUGO_SYMP': "../../knowledge_graphs/BERT_embs_graph_data_DRUGO_SYMP.pkl",

    'GAT_BioBERT_SYMP': "../../knowledge_graphs/BioBERT_embs_graph_data_SYMP.pkl",
    'GAT_BioBERT_DRUGO': "../../knowledge_graphs/BioBERT_embs_graph_data_DRUGO.pkl",
    'GAT_BioBERT_DRUGO_SYMP': "../../knowledge_graphs/BioBERT_embs_graph_data_DRUGO_SYMP.pkl",
}

if __name__ == "__main__":
    import pickle, torch

    for key, value in graphs.items():
        print(f"Training {key}")
        graph_data = pickle.load(open(value, "rb"))
        train_index, val_index = train_test_split(list(range(len(graph_data.x))), test_size=0.2, random_state=42)
        val_index, test_index = train_test_split(val_index, test_size=0.5, random_state=34)
        train_index, val_index, test_index  = [True if x in train_index else False for x in range(len(graph_data.x))],  [True if x in val_index else False for x in range(len(graph_data.x))], [True if x in test_index else False for x in range(len(graph_data.x))]

        config = {
            'graph_hidden_layers': 3, 
            'num_node_features': 768,
            "num_encoded_features": 200, 
            "num_gnn_hidden": 12, 
            "gnn_hidden_size": 768,
            "num_labels": 14,
            "lr": 1e-4, 
            "weight_decay": 5e-4,
            "train_index": tensor(train_index) == True, 
            "val_index": tensor(val_index) == True
        }

        if 'DRUGO_SYMP' in key:
            config['num_labels'] = 28

        ## training
        model = GraphAttentionModel(config)
        optimizer = model.configure_optimizers()
        NUM_EPOCHS = 200

        current_val_loss = 1e10
        val_losses = []
        for i in range(NUM_EPOCHS):
            
            optimizer.zero_grad()
            loss = model.training_step(graph_data, i)
            print(f"Train loss at epoch: {i}: {loss.item()}")
            loss.backward()
            optimizer.step()
            if i % 5 == 0:
                current_val_loss = model.validation_step(graph_data, i)
                val_losses.append(current_val_loss)

                ## early stopping
                if len(val_losses) > 1:
                    if current_val_loss > val_losses[-2] and current_val_loss > val_losses[-1]:
                        print("Early stopping")
                        print("Best validation loss was: ", min(val_losses))
                        print(f"Epoch: {i}")
                        break
        torch.save(model.state_dict(), f"{key}_state_dict.pt")
                        
                



# %%


# %%

def get_gat_model():
    import torch 
    #model = torch.load("../GNN/GAT_SYMP.pt")
    model = torch.load("../GNN/GAT_DRUGO_SYMP.pt")
    return model

# %%

#m = get_gat_model()

#import torch

#torch.save(m.state_dict(),"GAT_DRUGO_SYMP_state_dict.pt" )



# %%
