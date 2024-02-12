# %%
from collections import OrderedDict
import pytorch_lightning as pl
from transformers import AutoModel
from torch.nn import Linear, Dropout, Sequential, ReLU
from torch import cat, Tensor, argmax
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import AdamW
from sklearn.metrics import f1_score, recall_score, precision_score, classification_report, roc_auc_score, precision_recall_curve, auc

from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import pandas as pd
import sys, pickle
import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from torch_geometric.data.data import Data
from torch_geometric.utils import k_hop_subgraph
from helpers import settings
from  pytorch_lightning.utilities.seed import seed_everything
from torch_geometric.utils import to_undirected, is_undirected

from models.NER.utils import calc_scores_relaxed, calc_scores_strict
import os.path as path

BERTTWEET = False

# sys.path.append("../..")

# sys.path.append("..")


from models.GNN.gat import GraphAttentionModel, GraphAttentionModelPooling, SAGNet
from helpers.my_dataloader import MyDataLoader
from pytorch_lightning.loggers import TensorBoardLogger

# %%

def get_k_hop_neighborhood(graph, node_idx, k):
    subset, edge_index, mapping, edge_mask  = k_hop_subgraph(node_idx, k, graph.edge_index, num_nodes=graph.num_nodes)
    subgraph = Data(edge_index=edge_index, x=graph.x)
    return subgraph


class GNNConcatModel(pl.LightningModule):
    def __init__(self, config=None) -> None:
        super().__init__()
        self.best_epoch = None
        self.best_f1 = 0 
        if not config:
            config = {
                'version': "bert-base-uncased", 
                # 'version': "dmis-lab/biobert-v1.1",
                "kg_name": "SYMP",
                # "kg_name": "DRUG",
                'transformer_last_layer_size': 768,
                'embedding_dim': 200, 
                'num_cl_layers': 3,
                'num_labels': 3,
                'dropout': 0.5,
                'lr': 1e-5,
                'classifier_lr': 1e-3,
                'weight_decay': 1e-2,
                'gnn_type': "GATPool",
                'k': 50,
                'node_embedding_type': "bert",
            }
        self.transformer = AutoModel.from_pretrained(config['version'])
        self.transformer_last_layer_size = config['transformer_last_layer_size']
        self.embedding_dim = config['embedding_dim']
        self.classifier_depth = config['num_cl_layers']
        self.num_labels = config['num_labels']
        self.dropout = Dropout(config['dropout'])
        self.lr = config['lr']
        self.lr_classifier = config['classifier_lr']
        self.weight_decay = config['weight_decay']
        self.gnn_lr = config['gnn_lr']
        self.kg_name = config["kg_name"]
        self.k = config["k"]
        self.node_embedding_type = config["node_embedding_type"]
        if self.classifier_depth == 1:
            self.classifier = Linear(self.transformer_last_layer_size + self.embedding_dim*2, self.num_labels)
        else:
            layers = []
            concat_layer_size = self.transformer_last_layer_size + self.embedding_dim
            for k in range(1, self.classifier_depth):
                layers.append((f'classifier_layer_{k}', Linear(int(concat_layer_size/k), int(concat_layer_size/(k+1)))))
                layers.append((f"ReLu_{k}", ReLU()))

            layers.append(("classifier_last", Linear(int(concat_layer_size/(self.classifier_depth)), self.num_labels)))
            self.classifier = Sequential(OrderedDict(layers))
            #self.classifier = Sequential(OrderedDict([(f'classifier_layer_{k}', Linear(int(concat_layer_size/k), int(concat_layer_size/(k+1)))) for k in range(
                #1, self.classifier_depth)] + [("classifier_last", Linear(int(concat_layer_size/(self.classifier_depth)), self.num_labels))]))
        
        
        self.gnn = GraphAttentionModelPooling(num_node_features=768, gnn_hidden_size=768)

        if self.kg_name == "DRUG":
            
            
            if self.node_embedding_type == "bert":
                    #self.gnn.load_state_dict(torch.load(settings.BASE_DIR + "/models/GNN/GAT_BERT_DRUGO_state_dict.pt"))
                    
                    with open(settings.BASE_DIR + "/knowledge_graphs/BERT_embs_graph_data_DRUGO.pkl", "rb") as f:
                        self.graph_data: Data = pickle.load(f)
            elif self.node_embedding_type == "biobert":
                #self.gnn.load_state_dict(torch.load(settings.BASE_DIR + "/models/GNN/GAT_BioBERT_DRUGO_state_dict.pt"))
                
                with open(settings.BASE_DIR + "/knowledge_graphs/BioBERT_embs_graph_data_DRUGO.pkl", "rb") as f:
                    self.graph_data: Data = pickle.load(f)
            else:
                raise Exception("Node embedding type unknown", self.node_embedding_type)
            self.graph_data.to(self.device)

        elif self.kg_name == "SYMP":
            

            if self.node_embedding_type == "bert":
                #self.gnn.load_state_dict(torch.load(settings.BASE_DIR + "/models/GNN/GAT_BERT_SYMP_state_dict.pt"))
                with open(settings.BASE_DIR + "/knowledge_graphs/BERT_embs_graph_data_SYMP.pkl", "rb") as f:
                    self.graph_data: Data = pickle.load(f)
            elif self.node_embedding_type == "biobert":
                #self.gnn.load_state_dict(torch.load(settings.BASE_DIR + "/models/GNN/GAT_BioBERT_SYMP_state_dict.pt"))
                with open(settings.BASE_DIR + "/knowledge_graphs/BioBERT_embs_graph_data_SYMP.pkl", "rb") as f:
                    self.graph_data: Data = pickle.load(f)
            else:
                raise Exception("Unknown node embedding type", self.node_embedding_type)
            
            self.graph_data.to(self.device)
        elif self.kg_name == "DRUGO_SYMP":
            
            
            if self.node_embedding_type == "bert":
                #self.gnn.load_state_dict(torch.load(settings.BASE_DIR + "/models/GNN/GAT_BERT_DRUGO_SYMP_state_dict.pt"))
                with open(settings.BASE_DIR + "/knowledge_graphs/BERT_embs_graph_data_DRUGO_SYMP.pkl", "rb") as f:
                    self.graph_data: Data = pickle.load(f)
            elif self.node_embedding_type == "biobert":
                #self.gnn.load_state_dict(torch.load(settings.BASE_DIR + "/models/GNN/GAT_BioBERT_DRUGO_SYMP_state_dict.pt"))
                with open(settings.BASE_DIR + "/knowledge_graphs/BioBERT_embs_graph_data_DRUGO_SYMP.pkl", "rb") as f:
                    self.graph_data: Data = pickle.load(f)
            else:
                raise Exception("Unknown node embedding type", self.node_embedding_type)
            self.graph_data.to(self.device)
        else:
            raise Exception("KG name unknown", self.kg_name)


        # if self.kg_name == "DRUG":
        #     #self.gnn = GraphAttentionModel()
        #     #self.gnn.load_state_dict(torch.load(settings.BASE_DIR + "/models/GNN/GAT_UATC_state_dict.pt"))
        #     with open(settings.BASE_DIR + "/knowledge_graphs/graph_data_ATC.pkl", "rb") as f:
        #         self.graph_data: Data = pickle.load(f)
        #         self.graph_data.to(self.device)
        # elif self.kg_name == "SYMP":
        #     #self.gnn = GraphAttentionModel()
        #     #self.gnn.load_state_dict(torch.load(settings.BASE_DIR + "/models/GNN/GAT_SYMP_state_dict.pt"))
        #     with open(settings.BASE_DIR + "/knowledge_graphs/graph_data_SYMP.pkl", "rb") as f:
        #         self.graph_data: Data = pickle.load(f)
        #         self.graph_data.to(self.device)
        # elif self.kg_name == "DRUGO_SYMP":
        #     #self.gnn = GraphAttentionModel(out_size=28)
        #     #self.gnn.load_state_dict(torch.load(settings.BASE_DIR + "/models/GNN/GAT_DRUGO_SYMP_state_dict.pt"))
        #     with open(settings.BASE_DIR + "/knowledge_graphs/graph_data_DRUGO_SYMP.pkl", "rb") as f:
        #         self.graph_data: Data = pickle.load(f)
        #         self.graph_data.to(self.device)
        # else:
        #     raise Exception("KG name unknown", self.kg_name)
        
        ## make graph undirected if not 
        if not is_undirected(self.graph_data.edge_index):
            print("Graph is not undirected... Changing that")
            edge_index = to_undirected(self.graph_data.edge_index)
            self.graph_data.edge_index = edge_index

        self.save_hyperparameters() 

    def forward(self, x, ents_idx,**kwargs):
        k=self.k
        if BERTTWEET:
            out_transformer = self.transformer(x['input_ids'])
        else:
            out_transformer = self.transformer(**x)

        raw_transformer_out = out_transformer.last_hidden_state

        ## extract subgraph for those entities where the ent idx is not -1 or -2

        ents = []
        self.graph_data.to(self.device)


        for b in ents_idx:
            ent_single = []
            for e in b:
                if e.item() in [-2,-1]:
                    vec = [0]*(self.embedding_dim*2)
                else:

                    ## extract subraph and pass through network 
                    subgraph = get_k_hop_neighborhood(self.graph_data, e.item(), k)

                    vec = self.gnn(subgraph.x, subgraph.edge_index)


                    #vec = self.graph_data.x[e]
                ent_single.append(vec)
            ents.append(ent_single)


        
        self.graph_data.to(self.device)
        # gnn_out = self.gnn.get_embeddings(self.graph_data)
        # ents = []
        # for b in ents_idx:
        #     ent_single = []
        #     for e in b:
        #         if e.item() in [-2,-1]:
        #             vec = [0]*self.embedding_dim
        #         else:
        #             vec = gnn_out[e]
        #         ent_single.append(vec)
        #     ents.append(ent_single)
                

        ents = Tensor(ents).to(self.device)
        concat = cat((raw_transformer_out, ents), dim=2)
        
        output = self.dropout(concat)
        output = self.classifier(output)
        return output

    def test_step(self,batch, batch_idx):
        (x,y), ents = batch
        #sequence_length = int(torch.count_nonzero(x['attention_mask']).item())
        logits = self(x, ents)
        y_hat = logits.view(-1, 3)
        y = y.view(-1)
        class_prob = torch.softmax(y_hat, dim=1)
        class_prob_ = []
        for c in class_prob:
            class_prob_.append(c[1].item()+ c[2].item())
        loss = F.cross_entropy(y_hat, y)
        pred = argmax(y_hat,dim=1)
        self.log("test_loss", loss)
        return {'loss': loss, 'pred': pred, "true": y, "class_prob": torch.tensor(class_prob_)}
        

    def test_epoch_end(self, test_step_outputs):
        all_true = []
        all_preds = []
        all_class_prob = []
        for scores in test_step_outputs:
            all_true.extend([int(val.item()) for val in scores['true'].to("cpu")])
            all_preds.extend([int(val.item()) for val in scores['pred'].to("cpu")])
            all_class_prob.extend([float(val.item()) for val in scores['class_prob'].to("cpu")])
        if len(all_true) == 0 or len(all_preds) == 0:
            print("Empty prediction...")
            return

        all_true_entity_based = [x if x in [0,1,-100] else 1 for x in all_true]
        all_pred_entity_based = [x if x in [0,1] else 1 for x in all_preds]

        ## get rid of -100 


        indeces = [i for i, x in enumerate(all_true_entity_based) if x == -100]
        for index in sorted(indeces, reverse=True):
            del all_pred_entity_based[index]
            del all_true_entity_based[index]


        report = classification_report(all_true_entity_based, all_pred_entity_based, output_dict=True)
        print(report)
        self.log_dict({"f1_entity_based": f1_score(all_true_entity_based, all_pred_entity_based), "precision_entity_based":precision_score(all_true_entity_based, all_pred_entity_based), "recall_entity_based": recall_score(all_true_entity_based, all_pred_entity_based)})


        self.log("support_entity_based", report["1"]['support'])

        #self.log("auroc",roc_auc_score(all_true_entity_based, all_pred_entity_based))
        #self.log("f1_entity-based_1", report["1"]["f1-score"])
        #self.log_dict({"f1_macro":  f1_score(y_pred = all_preds,y_true= all_true, average='macro'),
                                 #"precision_macro": precision_score(y_pred = all_preds,y_true= all_true, average='macro'),
                                 #"recall_macro": recall_score(y_pred = all_preds,y_true= all_true, average='macro')})
        #self.log("precision_entity-based_1", report['1']['precision'])
        #self.log("recall_entity-based_1", report['1']['recall'])
        precision, recall, thresholds = precision_recall_curve(all_true_entity_based, all_pred_entity_based)
       # self.log("aupr", auc(recall, precision))

        ### Calculate strict and relaxed scores

        precision_strict, recall_strict, f1_strict, support_strict = calc_scores_strict(all_true_entity_based, all_pred_entity_based)
        
        
        precision_relaxed, recall_relaxed, f1_relaxed, support_relaxed = calc_scores_relaxed(all_true_entity_based, all_pred_entity_based)


        self.log("precision_strict", precision_strict)
        self.log("recall_strict", recall_strict)
        self.log("f1_strict", f1_strict)


        self.log("f1_relaxed", f1_relaxed)
        self.log("precision_relaxed", precision_relaxed)
        self.log("recall_relaxed", recall_relaxed)

        self.log("support_relaxed", support_relaxed)

        #data = list(zip(all_true_entity_based, all_pred_entity_based, all_class_prob))
        #pd.DataFrame(data=data, columns=['true', 'pred', 'prob']).to_csv(f"{EXP}_predictions.csv", index=False)
        #sns.lineplot(recall, precision)
    
    def training_step(self,batch, batch_idx, *args, **kwargs):
        (x,y), ents = batch
        logits = self(x, ents)
        y_hat = logits.view(-1, 3)
        y = y.view(-1)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self,batch, batch_idx, *args, **kwargs):
        (x,y),ents = batch
        #sequence_length = int(torch.count_nonzero(x['attention_mask']).item())
        logits = self(x,ents)
        y_hat = logits.view(-1, 3)
        y = y.view(-1)
        loss = F.cross_entropy(y_hat, y)
        pred = argmax(y_hat,dim=1)
        self.log("val_loss", loss, prog_bar=True)
        return {'loss': loss, 'pred': pred , "true": y}

    def validation_epoch_end(self, validation_step_outputs):
        all_true = []
        all_preds = []
        for scores in validation_step_outputs:
            all_true.extend([int(val.item()) for val in scores['true'].to("cpu")])
            all_preds.extend([int(val.item()) for val in scores['pred'].to("cpu")])
        if len(all_true) == 0 or len(all_preds) == 0:
            print("Empty prediction...")
            return
        report = classification_report(all_true, all_preds, output_dict=True)
        self.log_dict({"f1_weighted": f1_score(y_pred = all_preds, y_true=all_true, average='weighted'),
                                 "precision_macro": precision_score(y_pred = all_preds, y_true=all_true, average='weighted'),
                                 "recall_macro": recall_score(y_pred = all_preds,y_true= all_true, average='weighted')})	
        self.log_dict({"f1_macro":  f1_score(y_pred = all_preds,y_true= all_true, average='macro'),
                                 "precision_macro": precision_score(y_pred = all_preds,y_true= all_true, average='macro'), 
                                 "recall_macro": recall_score(y_pred = all_preds,y_true= all_true, average='macro')})
        self.log_dict(                     
                                 {"f1_micro":  f1_score(y_pred = all_preds,y_true= all_true, average='micro'), 
                                 "precision_micro": precision_score(y_pred = all_preds, y_true=all_true, average='micro'), 
                                 "recall_micro": recall_score(y_pred = all_preds, y_true=all_true, average='micro')})
        self.log_dict({"f1_B-ADR": report['1']['f1-score'], "f1_I-ADR": report['2']["f1-score"]})
        all_true_entity_based = [x if x in [0,1,-100] else 1 for x in all_true] ## allow binary classification evaluation
        all_pred_entity_based = [x if x in [0,1] else 1 for x in all_preds]

        ## get rid of -100 


        indeces = [i for i, x in enumerate(all_true_entity_based) if x == -100]
        for index in sorted(indeces, reverse=True):
            del all_pred_entity_based[index]
            del all_true_entity_based[index]



        report = classification_report(all_true_entity_based, all_pred_entity_based, output_dict=True)
        print(report)
        print("\n")
        self.log_dict({"f1_entity_based": f1_score(all_true_entity_based, all_pred_entity_based), "precision_entity_based":precision_score(all_true_entity_based, all_pred_entity_based), "recall_entity_based": recall_score(all_true_entity_based, all_pred_entity_based)})
        self.log("auroc",roc_auc_score(all_true_entity_based, all_pred_entity_based))
        self.log("f1_entity-based_1", report["1"]["f1-score"])
        self.log("f1_entity-based_0", report["0"]["f1-score"])
        precision, recall, thresholds = precision_recall_curve(all_true_entity_based, all_pred_entity_based)
        self.log("aupr", auc(recall, precision))
        if report['1']['f1-score'] > self.best_f1:
            self.best_epoch = self.current_epoch
            self.best_f1 = report['1']['f1-score']
        if self.best_epoch:
            self.log("best_epoch", self.best_epoch + 1)
    def configure_optimizers(self):
        transformer_params = []
        classifier_params = []
        gnn_params = []
        for name, param in self.named_parameters():
            if "transformer" in name:
                transformer_params.append(param)
            elif 'classifier' in name:
                classifier_params.append(param)
            elif 'gnn' in name:
                gnn_params.append(param)
            else:
                raise Exception(f"Undefined parameter group: {name}")
        optimizer = AdamW([{'params': transformer_params},{'params': classifier_params}, {'params': gnn_params}], lr=self.lr, weight_decay=self.weight_decay)
        optimizer.param_groups[0]['lr'] = self.lr
        optimizer.param_groups[1]['lr'] = self.lr_classifier
        optimizer.param_groups[2]['lr'] = self.gnn_lr

        return optimizer

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_exp_id", help="id of test_exp (range from 0 to 59)", type=int)
    return parser.parse_args()

# %%
if __name__ == "__main__":
    args = parse_args()
    import optuna
    test_exps = [
        {
            "study_name": "graph-adaptive-weights-subg-optimization-new_SMM4H_SYMP_biobert_bert",
            "dataset_name": "SMM4H",
            "kg_name": "SYMP",
            "model_name": "dmis-lab/biobert-v1.1",
            "model_short_name": "biobert",
            "node_embedding_type": "bert",
        },
        {
            "study_name": "graph-adaptive-weights-subg-optimization-new_SMM4H_SYMP_biobert_biobert",
            "dataset_name": "SMM4H",
            "kg_name": "SYMP",
            "model_name": "dmis-lab/biobert-v1.1",
            "model_short_name": "biobert",
            "node_embedding_type": "biobert",
        },
        {
            "study_name": "graph-adaptive-weights-subg-optimization-new_SMM4H_SYMP_bert_bert",
            "dataset_name": "SMM4H",
            "kg_name": "SYMP",
            "model_name": "bert-base-uncased",
            "model_short_name": "bert",
            "node_embedding_type": "bert",
        },
        {
            "study_name": "graph-adaptive-weights-subg-optimization-new_SMM4H_SYMP_bert_biobert",
            "dataset_name": "SMM4H",
            "kg_name": "SYMP",
            "model_name": "bert-base-uncased",
            "model_short_name": "bert",
            "node_embedding_type": "biobert",
        },
        {
            "study_name": "graph-adaptive-weights-subg-optimization-new_SMM4H_DRUG_biobert_bert",
            "dataset_name": "SMM4H",
            "kg_name": "DRUG",
            "model_name": "dmis-lab/biobert-v1.1",
            "model_short_name": "biobert",
            "node_embedding_type": "bert",
        },
        {
            "study_name": "graph-adaptive-weights-subg-optimization-new_SMM4H_DRUG_biobert_biobert",
            "dataset_name": "SMM4H",
            "kg_name": "DRUG",
            "model_name": "dmis-lab/biobert-v1.1",
            "model_short_name": "biobert",
            "node_embedding_type": "biobert",
        },
        {
            "study_name": "graph-adaptive-weights-subg-optimization-new_SMM4H_DRUG_bert_bert",
            "dataset_name": "SMM4H",
            "kg_name": "DRUG",
            "model_name": "bert-base-uncased",
            "model_short_name": "bert",
            "node_embedding_type": "bert",
        },
        {
            "study_name": "graph-adaptive-weights-subg-optimization-new_SMM4H_DRUG_bert_biobert",
            "dataset_name": "SMM4H",
            "kg_name": "DRUG",
            "model_name": "bert-base-uncased",
            "model_short_name": "bert",
            "node_embedding_type": "biobert",
        },
        {
            "study_name": "graph-adaptive-weights-subg-optimization-new_SMM4H_DRUGO_SYMP_biobert_bert",
            "dataset_name": "SMM4H",
            "kg_name": "DRUGO_SYMP",
            "model_name": "dmis-lab/biobert-v1.1",
            "model_short_name": "biobert",
            "node_embedding_type": "bert",
        },
        {
            "study_name": "graph-adaptive-weights-subg-optimization-new_SMM4H_DRUGO_SYMP_biobert_biobert",
            "dataset_name": "SMM4H",
            "kg_name": "DRUGO_SYMP",
            "model_name": "dmis-lab/biobert-v1.1",
            "model_short_name": "biobert",
            "node_embedding_type": "biobert",
        },
        {
            "study_name": "graph-adaptive-weights-subg-optimization-new_SMM4H_DRUGO_SYMP_bert_bert",
            "dataset_name": "SMM4H",
            "kg_name": "DRUGO_SYMP",
            "model_name": "bert-base-uncased",
            "model_short_name": "bert",
            "node_embedding_type": "bert",
        },
        {
            "study_name": "graph-adaptive-weights-subg-optimization-new_SMM4H_DRUGO_SYMP_bert_biobert",
            "dataset_name": "SMM4H",
            "kg_name": "DRUGO_SYMP",
            "model_name": "bert-base-uncased",
            "model_short_name": "bert",
            "node_embedding_type": "biobert",
        },
        {
            "study_name": "graph-adaptive-weights-subg-optimization-new_CADEC_SYMP_biobert_bert",
            "dataset_name": "CADEC",
            "kg_name": "SYMP",
            "model_name": "dmis-lab/biobert-v1.1",
            "model_short_name": "biobert",
            "node_embedding_type": "bert",
        },
        {
            "study_name": "graph-adaptive-weights-subg-optimization-new_CADEC_SYMP_biobert_biobert",
            "dataset_name": "CADEC",
            "kg_name": "SYMP",
            "model_name": "dmis-lab/biobert-v1.1",
            "model_short_name": "biobert",
            "node_embedding_type": "biobert",
        },
        {
            "study_name": "graph-adaptive-weights-subg-optimization-new_CADEC_SYMP_bert_bert",
            "dataset_name": "CADEC",
            "kg_name": "SYMP",
            "model_name": "bert-base-uncased",
            "model_short_name": "bert",
            "node_embedding_type": "bert",
        },
        {
            "study_name": "graph-adaptive-weights-subg-optimization-new_CADEC_SYMP_bert_biobert",
            "dataset_name": "CADEC",
            "kg_name": "SYMP",
            "model_name": "bert-base-uncased",
            "model_short_name": "bert",
            "node_embedding_type": "biobert",
        },
        {
            "study_name": "graph-adaptive-weights-subg-optimization-new_CADEC_DRUG_biobert_bert",
            "dataset_name": "CADEC",
            "kg_name": "DRUG",
            "model_name": "dmis-lab/biobert-v1.1",
            "model_short_name": "biobert",
            "node_embedding_type": "bert",
        },
        {
            "study_name": "graph-adaptive-weights-subg-optimization-new_CADEC_DRUG_biobert_biobert",
            "dataset_name": "CADEC",
            "kg_name": "DRUG",
            "model_name": "dmis-lab/biobert-v1.1",
            "model_short_name": "biobert",
            "node_embedding_type": "biobert",
        },
        {
            "study_name": "graph-adaptive-weights-subg-optimization-new_CADEC_DRUG_bert_bert",
            "dataset_name": "CADEC",
            "kg_name": "DRUG",
            "model_name": "bert-base-uncased",
            "model_short_name": "bert",
            "node_embedding_type": "bert",
        },
        {
            "study_name": "graph-adaptive-weights-subg-optimization-new_CADEC_DRUG_bert_biobert",
            "dataset_name": "CADEC",
            "kg_name": "DRUG",
            "model_name": "bert-base-uncased",
            "model_short_name": "bert",
            "node_embedding_type": "biobert",
        },
        {
            "study_name": "graph-adaptive-weights-subg-optimization-new_CADEC_DRUGO_SYMP_biobert_bert",
            "dataset_name": "CADEC",
            "kg_name": "DRUGO_SYMP",
            "model_name": "dmis-lab/biobert-v1.1",
            "model_short_name": "biobert",
            "node_embedding_type": "bert",
        },
        {
            "study_name": "graph-adaptive-weights-subg-optimization-new_CADEC_DRUGO_SYMP_biobert_biobert",
            "dataset_name": "CADEC",
            "kg_name": "DRUGO_SYMP",
            "model_name": "dmis-lab/biobert-v1.1",
            "model_short_name": "biobert",
            "node_embedding_type": "biobert",
        },
        {
            "study_name": "graph-adaptive-weights-subg-optimization-new_CADEC_DRUGO_SYMP_bert_bert",
            "dataset_name": "CADEC",
            "kg_name": "DRUGO_SYMP",
            "model_name": "bert-base-uncased",
            "model_short_name": "bert",
            "node_embedding_type": "bert",
        },
        {
            "study_name": "graph-adaptive-weights-subg-optimization-new_CADEC_DRUGO_SYMP_bert_biobert",
            "dataset_name": "CADEC",
            "kg_name": "DRUGO_SYMP",
            "model_name": "bert-base-uncased",
            "model_short_name": "bert",
            "node_embedding_type": "biobert",
        },
        {
            "study_name": "graph-adaptive-weights-subg-optimization-new_PSYTAR_SYMP_biobert_bert",
            "dataset_name": "PSYTAR",
            "kg_name": "SYMP",
            "model_name": "dmis-lab/biobert-v1.1",
            "model_short_name": "biobert",
            "node_embedding_type": "bert",
        },
        {
            "study_name": "graph-adaptive-weights-subg-optimization-new_PSYTAR_SYMP_biobert_biobert",
            "dataset_name": "PSYTAR",
            "kg_name": "SYMP",
            "model_name": "dmis-lab/biobert-v1.1",
            "model_short_name": "biobert",
            "node_embedding_type": "biobert",
        },
        {
            "study_name": "graph-adaptive-weights-subg-optimization-new_PSYTAR_SYMP_bert_bert",
            "dataset_name": "PSYTAR",
            "kg_name": "SYMP",
            "model_name": "bert-base-uncased",
            "model_short_name": "bert",
            "node_embedding_type": "bert",
        },
        {
            "study_name": "graph-adaptive-weights-subg-optimization-new_PSYTAR_SYMP_bert_biobert",
            "dataset_name": "PSYTAR",
            "kg_name": "SYMP",
            "model_name": "bert-base-uncased",
            "model_short_name": "bert",
            "node_embedding_type": "biobert",
        },
        {
            "study_name": "graph-adaptive-weights-subg-optimization-new_PSYTAR_DRUG_biobert_bert",
            "dataset_name": "PSYTAR",
            "kg_name": "DRUG",
            "model_name": "dmis-lab/biobert-v1.1",
            "model_short_name": "biobert",
            "node_embedding_type": "bert",
        },
        {
            "study_name": "graph-adaptive-weights-subg-optimization-new_PSYTAR_DRUG_biobert_biobert",
            "dataset_name": "PSYTAR",
            "kg_name": "DRUG",
            "model_name": "dmis-lab/biobert-v1.1",
            "model_short_name": "biobert",
            "node_embedding_type": "biobert",
        },
        {
            "study_name": "graph-adaptive-weights-subg-optimization-new_PSYTAR_DRUG_bert_bert",
            "dataset_name": "PSYTAR",
            "kg_name": "DRUG",
            "model_name": "bert-base-uncased",
            "model_short_name": "bert",
            "node_embedding_type": "bert",
        },
        {
            "study_name": "graph-adaptive-weights-subg-optimization-new_PSYTAR_DRUG_bert_biobert",
            "dataset_name": "PSYTAR",
            "kg_name": "DRUG",
            "model_name": "bert-base-uncased",
            "model_short_name": "bert",
            "node_embedding_type": "biobert",
        },
        {
            "study_name": "graph-adaptive-weights-subg-optimization-new_PSYTAR_DRUGO_SYMP_biobert_bert",
            "dataset_name": "PSYTAR",
            "kg_name": "DRUGO_SYMP",
            "model_name": "dmis-lab/biobert-v1.1",
            "model_short_name": "biobert",
            "node_embedding_type": "bert",
        },
        {
            "study_name": "graph-adaptive-weights-subg-optimization-new_PSYTAR_DRUGO_SYMP_biobert_biobert",
            "dataset_name": "PSYTAR",
            "kg_name": "DRUGO_SYMP",
            "model_name": "dmis-lab/biobert-v1.1",
            "model_short_name": "biobert",
            "node_embedding_type": "biobert",
        },
        {
            "study_name": "graph-adaptive-weights-subg-optimization-new_PSYTAR_DRUGO_SYMP_bert_bert",
            "dataset_name": "PSYTAR",
            "kg_name": "DRUGO_SYMP",
            "model_name": "bert-base-uncased",
            "model_short_name": "bert",
            "node_embedding_type": "bert",
        },
        {
            "study_name": "graph-adaptive-weights-subg-optimization-new_PSYTAR_DRUGO_SYMP_bert_biobert",
            "dataset_name": "PSYTAR",
            "kg_name": "DRUGO_SYMP",
            "model_name": "bert-base-uncased",
            "model_short_name": "bert",
            "node_embedding_type": "biobert",
        },
        {
            "study_name": "graph-adaptive-weights-subg-optimization-new_ADE_SYMP_biobert_bert",
            "dataset_name": "ADE",
            "kg_name": "SYMP",
            "model_name": "dmis-lab/biobert-v1.1",
            "model_short_name": "biobert",
            "node_embedding_type": "bert",
        },
        {
            "study_name": "graph-adaptive-weights-subg-optimization-new_ADE_SYMP_biobert_biobert",
            "dataset_name": "ADE",
            "kg_name": "SYMP",
            "model_name": "dmis-lab/biobert-v1.1",
            "model_short_name": "biobert",
            "node_embedding_type": "biobert",
        },
        {
            "study_name": "graph-adaptive-weights-subg-optimization-new_ADE_SYMP_bert_bert",
            "dataset_name": "ADE",
            "kg_name": "SYMP",
            "model_name": "bert-base-uncased",
            "model_short_name": "bert",
            "node_embedding_type": "bert",
        },
        {
            "study_name": "graph-adaptive-weights-subg-optimization-new_ADE_SYMP_bert_biobert",
            "dataset_name": "ADE",
            "kg_name": "SYMP",
            "model_name": "bert-base-uncased",
            "model_short_name": "bert",
            "node_embedding_type": "biobert",
        },
        {
            "study_name": "graph-adaptive-weights-subg-optimization-new_ADE_DRUG_biobert_bert",
            "dataset_name": "ADE",
            "kg_name": "DRUG",
            "model_name": "dmis-lab/biobert-v1.1",
            "model_short_name": "biobert",
            "node_embedding_type": "bert",
        },
        {
            "study_name": "graph-adaptive-weights-subg-optimization-new_ADE_DRUG_biobert_biobert",
            "dataset_name": "ADE",
            "kg_name": "DRUG",
            "model_name": "dmis-lab/biobert-v1.1",
            "model_short_name": "biobert",
            "node_embedding_type": "biobert",
        },
        {
            "study_name": "graph-adaptive-weights-subg-optimization-new_ADE_DRUG_bert_bert",
            "dataset_name": "ADE",
            "kg_name": "DRUG",
            "model_name": "bert-base-uncased",
            "model_short_name": "bert",
            "node_embedding_type": "bert",
        },
        {
            "study_name": "graph-adaptive-weights-subg-optimization-new_ADE_DRUG_bert_biobert",
            "dataset_name": "ADE",
            "kg_name": "DRUG",
            "model_name": "bert-base-uncased",
            "model_short_name": "bert",
            "node_embedding_type": "biobert",
        },
        {
            "study_name": "graph-adaptive-weights-subg-optimization-new_ADE_DRUGO_SYMP_biobert_bert",
            "dataset_name": "ADE",
            "kg_name": "DRUGO_SYMP",
            "model_name": "dmis-lab/biobert-v1.1",
            "model_short_name": "biobert",
            "node_embedding_type": "bert",
        },
        {
            "study_name": "graph-adaptive-weights-subg-optimization-new_ADE_DRUGO_SYMP_biobert_biobert",
            "dataset_name": "ADE",
            "kg_name": "DRUGO_SYMP",
            "model_name": "dmis-lab/biobert-v1.1",
            "model_short_name": "biobert",
            "node_embedding_type": "biobert",
        },
        {
            "study_name": "graph-adaptive-weights-subg-optimization-new_ADE_DRUGO_SYMP_bert_bert",
            "dataset_name": "ADE",
            "kg_name": "DRUGO_SYMP",
            "model_name": "bert-base-uncased",
            "model_short_name": "bert",
            "node_embedding_type": "bert",
        },
        {
            "study_name": "graph-adaptive-weights-subg-optimization-new_ADE_DRUGO_SYMP_bert_biobert",
            "dataset_name": "ADE",
            "kg_name": "DRUGO_SYMP",
            "model_name": "bert-base-uncased",
            "model_short_name": "bert",
            "node_embedding_type": "biobert",
        },
        {
            "study_name": "graph-adaptive-weights-subg-optimization-new_TAC_SYMP_biobert_bert",
            "dataset_name": "TAC",
            "kg_name": "SYMP",
            "model_name": "dmis-lab/biobert-v1.1",
            "model_short_name": "biobert",
            "node_embedding_type": "bert",
        },
        {
            "study_name": "graph-adaptive-weights-subg-optimization-new_TAC_SYMP_biobert_biobert",
            "dataset_name": "TAC",
            "kg_name": "SYMP",
            "model_name": "dmis-lab/biobert-v1.1",
            "model_short_name": "biobert",
            "node_embedding_type": "biobert",
        },
        {
            "study_name": "graph-adaptive-weights-subg-optimization-new_TAC_SYMP_bert_bert",
            "dataset_name": "TAC",
            "kg_name": "SYMP",
            "model_name": "bert-base-uncased",
            "model_short_name": "bert",
            "node_embedding_type": "bert",
        },
        {
            "study_name": "graph-adaptive-weights-subg-optimization-new_TAC_SYMP_bert_biobert",
            "dataset_name": "TAC",
            "kg_name": "SYMP",
            "model_name": "bert-base-uncased",
            "model_short_name": "bert",
            "node_embedding_type": "biobert",
        },
        {
            "study_name": "graph-adaptive-weights-subg-optimization-new_TAC_DRUG_biobert_bert",
            "dataset_name": "TAC",
            "kg_name": "DRUG",
            "model_name": "dmis-lab/biobert-v1.1",
            "model_short_name": "biobert",
            "node_embedding_type": "bert",
        },
        {
            "study_name": "graph-adaptive-weights-subg-optimization-new_TAC_DRUG_biobert_biobert",
            "dataset_name": "TAC",
            "kg_name": "DRUG",
            "model_name": "dmis-lab/biobert-v1.1",
            "model_short_name": "biobert",
            "node_embedding_type": "biobert",
        },
        {
            "study_name": "graph-adaptive-weights-subg-optimization-new_TAC_DRUG_bert_bert",
            "dataset_name": "TAC",
            "kg_name": "DRUG",
            "model_name": "bert-base-uncased",
            "model_short_name": "bert",
            "node_embedding_type": "bert",
        },
        {
            "study_name": "graph-adaptive-weights-subg-optimization-new_TAC_DRUG_bert_biobert",
            "dataset_name": "TAC",
            "kg_name": "DRUG",
            "model_name": "bert-base-uncased",
            "model_short_name": "bert",
            "node_embedding_type": "biobert",
        },
        {
            "study_name": "graph-adaptive-weights-subg-optimization-new_TAC_DRUGO_SYMP_biobert_bert",
            "dataset_name": "TAC",
            "kg_name": "DRUGO_SYMP",
            "model_name": "dmis-lab/biobert-v1.1",
            "model_short_name": "biobert",
            "node_embedding_type": "bert",
        },
        {
            "study_name": "graph-adaptive-weights-subg-optimization-new_TAC_DRUGO_SYMP_biobert_biobert",
            "dataset_name": "TAC",
            "kg_name": "DRUGO_SYMP",
            "model_name": "dmis-lab/biobert-v1.1",
            "model_short_name": "biobert",
            "node_embedding_type": "biobert",
        },
        {
            "study_name": "graph-adaptive-weights-subg-optimization-new_TAC_DRUGO_SYMP_bert_bert",
            "dataset_name": "TAC",
            "kg_name": "DRUGO_SYMP",
            "model_name": "bert-base-uncased",
            "model_short_name": "bert",
            "node_embedding_type": "bert",
        },
        {
            "study_name": "graph-adaptive-weights-subg-optimization-new_TAC_DRUGO_SYMP_bert_biobert",
            "dataset_name": "TAC",
            "kg_name": "DRUGO_SYMP",
            "model_name": "bert-base-uncased",
            "model_short_name": "bert",
            "node_embedding_type": "biobert",
        },
    ]

    # for test_exp in test_exps:
    if args.test_exp_id >= 0 and args.test_exp_id <= 59:
        test_exp = test_exps[args.test_exp_id]
        study_name = test_exp["study_name"]
        results_file_path = settings.BASE_DIR + '/graph_adaptive_weights_subgraph_test_results_with_node_emb_type_30_' + study_name + '.csv'
        
        for run_number in range(0,30):
            seed = settings.SEEDS[run_number]
            seed_everything(seed)
            
            if path.exists(results_file_path):
                df = pd.read_csv(results_file_path, header=0, sep="\t")
                found_df = df[(df["study_name"] == study_name) & (df["run_number"] == run_number)]
                if len(found_df) > 0:
                    continue

            kg_name = test_exp["kg_name"]
            dataset_name = test_exp["dataset_name"]
            model_name = test_exp["model_name"]
            model_short_name = test_exp["model_short_name"]
            node_embedding_type = test_exp["node_embedding_type"]
            print("RUNNING " + study_name + " RUN " + str(run_number))
                        
            # study_name = "graph-adaptive-weights-optimization-new_" + dataset_name + "_" + kg_name + "_" + model_short_name 
            storage_name = "sqlite:///" + settings.BASE_DIR + "/models/HP-Tuning/{}.db".format(study_name)

            study = optuna.load_study(
                study_name = study_name, 
                storage = storage_name
            )
            # config = study.best_params

            loader = MyDataLoader(model_name = model_name, kg_name = kg_name, dataset_name = dataset_name, model_type="CONCAT_MODEL_AW", node_embedding_type=node_embedding_type)
            # config['layer_num'] = 1
            BATCH_SIZE = study.best_params["batch_size"]
#             BATCH_SIZE = int(study.best_params["batch_size"] / 2.0)
            EXP = study_name # "410"
            DEVICE = "gpu" if torch.cuda.is_available() else "cpu"
            #EXP = "ADAPTIVE_WEIGHTS_" + dataset_name
            
            # prepare data
            if dataset_name == "CADEC":
                corpus_train = loader.get_cadec_train()
                corpus_test = loader.get_cadec_test()
            elif dataset_name == "SMM4H":
                corpus_train = loader.get_smm4h_train()
                corpus_test = loader.get_smm4h_test()
            elif dataset_name == "PSYTAR":
                corpus_train = loader.get_psytar_train()
                corpus_test = loader.get_psytar_test()
            elif dataset_name == "ADE":
                corpus_train = loader.get_adev2_train()
                corpus_test = loader.get_adev2_test()
            elif dataset_name == "TAC":
                corpus_train = loader.get_tac_train()
                corpus_test = loader.get_tac_test()
            else:
                raise Exception("dataset name unknown: " + dataset_name)
            
            # print(corpus_train)
            
            train_data_optim, val_data_optim = train_test_split(corpus_train, test_size=0.2, random_state=seed)
            
            train_data_optim = DataLoader(train_data_optim, batch_size=BATCH_SIZE)
            val_data_optim = DataLoader(val_data_optim, batch_size=BATCH_SIZE)
            
            train_data = DataLoader(corpus_train, batch_size=BATCH_SIZE)
            test_data = DataLoader(corpus_test, batch_size=BATCH_SIZE)

            config = {
                'version': model_name,
                "kg_name": kg_name,
                'transformer_last_layer_size': 768,
                'embedding_dim': 200, 
                'num_cl_layers': 1, # study.best_params["num_layers"],
                'num_labels': 3,
                'dropout': study.best_params["dropout"],
                'lr': study.best_params["lr"],
                'classifier_lr': study.best_params["lr_classifier"],
                'weight_decay': study.best_params["weight_decay"],
                'gnn_lr': study.best_params["lr_gnn"], 
                'k': 15 if 'k' not in study.best_params.keys() else study.best_params["k"],
                "node_embedding_type": node_embedding_type
            }
            
            # config = {
            #         'version': model_name,
            #         'transformer_last_layer_size': 768,
            #         'embedding_dim': 200, 
            #         'num_cl_layers': 3,
            #         'num_labels': 3,
            #         'dropout': 0.2137,
            #         'lr': 1.7971e-5,
            #         'classifier_lr': 1.1437e-3,
            #         'gnn_lr': 1e-4,
            #         'weight_decay': 0.03345
            # }

            print("Getting epochs...")
            logger = TensorBoardLogger(name=f"EXPERIMENT_TEST_{EXP}", save_dir=settings.BASE_DIR + "/models/tb_logs")
            seed_everything(seed)
            trainer = pl.Trainer(max_epochs=30, accelerator=DEVICE, devices=1, logger=logger, callbacks=[EarlyStopping(monitor='val_loss')])
            model = GNNConcatModel(config)
            trainer.fit(model, train_dataloaders=train_data_optim, val_dataloaders=val_data_optim)
            epoch = model.best_epoch + 1
            print(f"BEST EPOCH: {epoch}")

            print("Testing...")
            logger = TensorBoardLogger(name=f"EXPERIMENT_TEST_{EXP}", save_dir=settings.BASE_DIR + "/models/tb_logs")
            seed_everything(seed)
            trainer = pl.Trainer(max_epochs=epoch, accelerator=DEVICE, devices=1, logger=logger)
            model = GNNConcatModel(config)
            trainer.fit(model, train_dataloaders=train_data)
            # trainer.test(model, dataloaders=test_data)
            results = trainer.test(model, dataloaders=test_data)
            results[0]["study_name"] = study_name
            results[0]["run_number"] = run_number
            print(results)

            if path.exists(results_file_path):
                df = pd.read_csv(results_file_path, header=0, sep="\t")
            else:

                columns = ["study_name", "run_number", 'test_loss','f1_entity_based','precision_entity_based', 'recall_entity_based', "support_entity_based", "precision_strict", "recall_strict", 'f1_strict', "precision_relaxed", "recall_relaxed", 'f1_relaxed', "support_relaxed"]
                df = pd.DataFrame(columns=columns)

            # df = df.append(pd.DataFrame.from_dict(results))
            df.loc[len(df)] = results[0]
            df.to_csv(results_file_path,sep="\t", header=True, index=False)            
