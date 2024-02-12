# %%
from collections import OrderedDict
import sys
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import KFold
import spacy, torch, json
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import pandas as pd 
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import seaborn as sn
#sys.path.append("../pretrained_models")
#sys.path.append("..")
#sys.path.append("../..")
import matplotlib.pyplot as plt
from pretrained_models.knowledge_bert import BertModel, BertTokenizer, BertAdam
from helpers.my_dataloader import MyDataLoader
import torch.nn.functional as F
from torch.optim import AdamW
from sklearn.metrics import f1_score, recall_score, roc_auc_score, precision_score, classification_report, precision_recall_curve, auc
from sklearn.model_selection import train_test_split
from helpers import settings

from pytorch_lightning.callbacks import EarlyStopping
from models.NER.utils import calc_scores_relaxed, calc_scores_strict
from  pytorch_lightning.utilities.seed import seed_everything
import os.path as path

# %%
#TRANSFORMER_HIDDEN_SIZE = 768
#NUM_LABELS = 3
# BATCH_SIZE=16
#VERSION="/Users/philippwegner/Desktop/research/NLP/biobert-v1.1"
#VERSION = "/home/pwegner/execution_projects/biobert/"
# VERSION ="bert-base-uncased"

# %%
class ERNIEModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.best_epoch = None
        self.best_f1 = 0
        ernie, _ = BertModel.from_pretrained(config['version'])
        self.transformer = ernie
        self.lr = config['lr']
        self.lr_classifier = config['lr_classifier']
        dropout_prop = config['dropout']
        self.dropout = nn.Dropout(p=dropout_prop)
        self.classifier = nn.Linear(config["transformer_last_layer_size"], config["num_labels"])
        self.weight_decay = config['weight_decay']
        self.save_hyperparameters()

    def forward(self, x, ents, ent_mask):
        out, _ = self.transformer(**x, input_ent=ents, ent_mask=ent_mask, output_all_encoded_layers=False)
        out = self.dropout(out)
        out = self.classifier(out)
        return out

    def test_step(self, batch, batch_idx):
        (x,y),ents,ent_mask = batch
        logits = self(x,ents,ent_mask)
        y_hat = logits.view(-1, 3)
        sequence_length = int(torch.count_nonzero(x['attention_mask']).item())
        y = y.view(-1)
        loss = F.cross_entropy(y_hat, y)
        pred = torch.argmax(y_hat,dim=1)
        class_prob = torch.softmax(y_hat, dim=1)
        class_prob_ = []
        for c in class_prob:
            class_prob_.append(c[1].item()+ c[2].item())
        self.log("test_loss", loss)
        return {'loss': loss, 'pred': pred , "true": y, "class_prob": torch.tensor(class_prob_) }

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

        all_true_entity_based = [x if x in [0,1, -100] else 1 for x in all_true]
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
        #precision, recall, thresholds = precision_recall_curve(all_true_entity_based, all_pred_entity_based)
        #self.log("aupr", auc(recall, precision))
        
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
        
    def training_step(self, batch, batch_idx, *args, **kwargs):
        (x,y),ents,ent_mask = batch
        logits= self(x,ents,ent_mask)
        #y = torch.squeeze(y)
        y_hat = logits.view(-1, 3)
        y = y.view(-1)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        (x,y),ents,ent_mask = batch
        logits = self(x,ents,ent_mask)
        y_hat = logits.view(-1, 3)
        #sequence_length = int(torch.count_nonzero(x['attention_mask']).item())
        y = y.view(-1)
        loss = F.cross_entropy(y_hat, y)
        pred = torch.argmax(y_hat,dim=1)
        
        self.log("val_loss", loss, prog_bar=True)
        return {'loss': loss, 'pred': pred , "true": y}

    def validation_epoch_end(self, validation_step_outputs):
        all_true = []
        all_preds = []
        all_class_prob = []
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
        all_true_entity_based = [x if x in [0,1,-100] else 1 for x in all_true]
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
        for name, param in self.named_parameters():
            if "transformer" in name:
                transformer_params.append(param)
            else:
                classifier_params.append(param)

        optimizer = AdamW([{'params': transformer_params},{'params': classifier_params}], lr=self.lr, weight_decay=self.weight_decay)
        optimizer.param_groups[0]['lr'] = self.lr
        optimizer.param_groups[1]['lr'] = self.lr_classifier
        return optimizer
        

# %%
# main
if __name__ == "__main__":

    # set exp to run:
    import optuna
    # ERNIE Experimenst are all with BERT
    test_exps = [
        # {
        # "study_name":"ernie-optimization_SYMP_SMM4H_TRANS_E",
        # "model_name": "bert-base-uncased",
        # "kg_name": "SYMP",
        # "dataset_name": "SMM4H"
        # },
        # {
        # "study_name": "ernie-optimization_DRUG_SMM4H_TRANS_E",
        # "model_name": "bert-base-uncased",
        # "kg_name": "DRUG",
        # "dataset_name": "SMM4H"
        # },
        # {
        # "study_name": "ernie-optimization_SYMP_CADEC_TRANS_E",
        # "model_name": "bert-base-uncased",
        # "kg_name":"SYMP",
        # "dataset_name": "CADEC"
        # },
        # {
        # "study_name": "ernie-optimization_DRUG_CADEC_TRANS_E",
        # "model_name": "bert-base-uncased",
        # "kg_name": "DRUG",
        # "dataset_name": "CADEC"
        # },
        # {
        # "study_name": "ernie-optimization_CADEC_DRUGO_SYMP_bert",
        # "model_name": "bert-base-uncased",
        # "kg_name":"DRUGO_SYMP",
        # "dataset_name": "CADEC"
        # },
        # {
        # "study_name": "ernie-optimization_SMM4H_DRUGO_SYMP_bert",
        # "model_name": "bert-base-uncased",
        # "kg_name": "DRUGO_SYMP",
        # "dataset_name": "SMM4H"
        # },
        # {
        #     "study_name": "ernie-optimization_PSYTAR_SYMP_bert",
        #     "model_name": "bert-base-uncased",
        #     "kg_name": "SYMP",
        #     "dataset_name": "PSYTAR"
        # },
        # {
        #     "study_name": "ernie-optimization_PSYTAR_DRUG_bert",
        #     "model_name": "bert-base-uncased",
        #     "kg_name": "DRUG",
        #     "dataset_name": "PSYTAR"
        # },
        # {
        #      "study_name": "ernie-optimization_PSYTAR_DRUGO_SYMP_bert",
        #      "model_name": "bert-base-uncased",
        #     "kg_name": "DRUGO_SYMP",
        #     "dataset_name": "PSYTAR"
        # }, 
        {
            "study_name": "ernie-optimization_ADE_SYMP_bert",
            "model_name": "bert-base-uncased",
            "kg_name": "SYMP",
            "dataset_name": "ADE"
        },
        {
            "study_name": "ernie-optimization_ADE_DRUG_bert",
            "model_name": "bert-base-uncased",
            "kg_name": "DRUG",
            "dataset_name": "ADE"
        },
        {
            "study_name": "ernie-optimization_ADE_DRUGO_SYMP_bert",
            "model_name": "bert-base-uncased",
            "kg_name": "DRUGO_SYMP",
            "dataset_name": "ADE"
        },
        {
            "study_name": "ernie-optimization_TAC_SYMP_bert",
            "model_name": "bert-base-uncased",
            "kg_name": "SYMP",
            "dataset_name": "TAC"
        },
        {
            "study_name": "ernie-optimization_TAC_DRUG_bert",
            "model_name": "bert-base-uncased",
            "kg_name": "DRUG",
            "dataset_name": "TAC"
        },
        {
            "study_name": "ernie-optimization_TAC_DRUGO_SYMP_bert",
            "model_name": "bert-base-uncased",
            "kg_name": "DRUGO_SYMP",
            "dataset_name": "TAC"
        }

    ]
    
    results_file_path = settings.BASE_DIR + '/ernie_test_results_30_new.csv'
    for test_exp in test_exps:
        for run_number in range(0,30):
            seed = settings.SEEDS[run_number]
            seed_everything(seed)
            
            study_name = test_exp["study_name"]
            
            if path.exists(results_file_path):
                df = pd.read_csv(results_file_path, header=0, sep="\t")
                found_df = df[(df["study_name"] == study_name) & (df["run_number"] == run_number)]
                if len(found_df) > 0:
                    continue

            kg_name = test_exp["kg_name"]
            dataset_name = test_exp["dataset_name"]
            model_name = test_exp["model_name"]
            print("RUNNING " + study_name + " RUN " + str(run_number))

            # get best params:
            study = optuna.load_study(study_name= study_name, storage="sqlite:///" + settings.BASE_DIR + "/models/HP-Tuning/" + study_name + ".db")
            batch_size = study.best_params["batch_size"]
            EXP = study_name # "310"
            accelerator = "gpu" if torch.cuda.is_available() else "cpu"

            # prepare loader
            loader = MyDataLoader(model_name=model_name, kg_name=kg_name, dataset_name=dataset_name, model_type="ERNIE")

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

            train_data_optim, val_data_optim = train_test_split(corpus_train, test_size=0.2, random_state=seed)
            
            train_data_optim = DataLoader(train_data_optim, batch_size=batch_size)
            val_data_optim = DataLoader(val_data_optim, batch_size=batch_size)
            
            train_data = DataLoader(corpus_train, batch_size=batch_size)
            test_data = DataLoader(corpus_test, batch_size=batch_size)

            config = {
                'version': model_name,
                'transformer_last_layer_size': 768,
                'num_labels': 3,
                'dropout': study.best_params["dropout"],
                'lr': study.best_params["lr"],
                'lr_classifier': study.best_params["lr_classifier"],
                'weight_decay': study.best_params["weight_decay"],
                "batch_size": batch_size
            }

            print("Getting epochs...")

            logger = TensorBoardLogger(name=f"EXPERIMENT_TEST_{EXP}", save_dir=settings.BASE_DIR + "/models/tb_logs", version=1)
            seed_everything(seed)
            trainer = pl.Trainer(max_epochs=25,accelerator=accelerator,devices=1, logger=logger, callbacks=[EarlyStopping(monitor='val_loss', patience=5)])
            model = ERNIEModel(config)
            trainer.fit(model, train_dataloaders=train_data_optim, val_dataloaders=val_data_optim)
            epoch = model.best_epoch + 1
            print(f"BEST EPOCH: {epoch}")

            print("Testing...")
            logger = TensorBoardLogger(name=f"EXPERIMENT_TEST_{EXP}", save_dir=settings.BASE_DIR + "/models/tb_logs", version=2)
            seed_everything(seed)
            trainer = pl.Trainer(max_epochs=epoch,accelerator=accelerator,devices=1, logger=logger)
            model = ERNIEModel(config)
            trainer.fit(model, train_dataloaders=train_data)
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
