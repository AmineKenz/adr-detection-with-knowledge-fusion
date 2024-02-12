# %%
import json
#from pkgutil import get_data
import torch , pickle
import spacy
from spacy.training.iob_utils import offsets_to_biluo_tags
from transformers import BertTokenizer, RobertaTokenizer
#import seaborn as sns
from matplotlib import pyplot as plt
import gensim
from helpers import settings
from helpers.my_knowledgegraph import MyKnowledgeGraph

# import sys

# sys.path.append("../models/")
# sys.path.append("../../models/")

# from models.GNN.gat import get_gat_model
from models.GNN.gat import GraphAttentionModel

# %%
# VERSION = "bert-base-uncased"
#VERSION = "dmis-lab/biobert-v1.1"
# tokenizer_global = BertTokenizer.from_pretrained(VERSION)

ADD_KNOWLEDGE = False
KBERT = False
EMBEDDING_DIMENSION = 200
WITH_MEDDRA =False
CONCAT_MODEL = False
EMBEDDINGS_AS_DICT = True
CONCAT_MODEL_AW = False
BASELINE = False
MAX_LENGTH_PADDING=512
#MAX_LENGTH_PADDING = 80
# %%

# KGs = [
#     #'../../knowledge_graphs/UATC.spo',
#     '../../knowledge_graphs/SYMP.spo'
#     ]

label2ids = {
    'O':0,
    'B-ADR':1,
    'I-ADR':2,
    "[ENT]": -100,
    "[PAD]": -100,
    "[CLS]": -100,
    "[SEP]": -100
}

ids2label = {
    0:'O',
    1:'B-ADR',
    2:'I-ADR'
}




# %%

#tokenizer("Hello, my name is steve!", "And my name is Bob", padding="max_length", max_length=20, return_tensors='pt')


# %%
def filter_overlapping(json_data):
    filtered = []
    total_removed = 0
    for line in json_data:
        remove = False
        for i, ani in enumerate(line['annotations']):
            for k, ank in enumerate(line['annotations']):
                if i==k: continue
                else:
                    if ani['start']<= ank['start'] <= ani['end']:
                        remove = True
                    if ank['start']<= ani['start'] <= ank['end']:
                        remove = True
        if not remove:
            filtered.append(line)
        else:
            total_removed+=1
    print(f"Total removed: {total_removed}")
    return filtered



# %%
class MyDataLoader():
    def __init__(self, model_name, gnn_instance=None, kg_name="SYMP", dataset_name = "SMM4H", model_type="ERNIE", node_embedding_type="bert") -> None:
        self.cadec_test = settings.BASE_DIR + "/data/cadec_test.json"
        self.cadec_train = settings.BASE_DIR + "/data/cadec_train.json"

        #self.cadec_train = "../../data/cadec_train_with_annos.json"
        self.smm4h_test = settings.BASE_DIR + "/data/smm4h_test.json"
        self.smm4h_train = settings.BASE_DIR + "/data/smm4h_train.json"

        self.psytar_test = settings.BASE_DIR + "/data/psyTAR_test.json"
        self.psytar_train = settings.BASE_DIR + "/data/psyTAR_train.json"

        self.adev2_train = settings.BASE_DIR + "/data/ade_V2_train.json"
        self.adev2_test = settings.BASE_DIR + "/data/ade_V2_test.json"

        self.tac_train = settings.BASE_DIR + "/data/tac_train.json"
        self.tac_test = settings.BASE_DIR + "/data/tac_test.json"

        self.removed_due_to_no_label = 0
        
        self.total_ents_count = 0 
        self.total_ents_count_only_start = 0
        if model_name == "roberta-base":
            self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(model_name)

        if model_type == "CONCAT_MODEL":
            global CONCAT_MODEL
            CONCAT_MODEL = True
        if model_type == "ERNIE":
            global EMBEDDING_DIMENSION
            EMBEDDING_DIMENSION = 100
            global ADD_KNOWLEDGE
            ADD_KNOWLEDGE = True
        if model_type == "CONCAT_MODEL_AW":
            global CONCAT_MODEL_AW
            CONCAT_MODEL_AW = True
        if model_type == "KBERT":
            global KBERT
            KBERT = True
        if model_type == "BASELINE":
            global BASELINE
            BASELINE = True

        self.EMBEDDING_DIMENSION= EMBEDDING_DIMENSION

        
        global MAX_LENGTH_PADDING
        if dataset_name == "SMM4H":
            MAX_LENGTH_PADDING = 80
        elif dataset_name == "CADEC":
            MAX_LENGTH_PADDING = 512
        elif dataset_name == "PSYTAR":
            MAX_LENGTH_PADDING = 200
        elif dataset_name == "ADE":
            MAX_LENGTH_PADDING = 512
        elif dataset_name == "TAC":
            MAX_LENGTH_PADDING = 512
        else:
            raise Exception("Unknown dataset name: ", dataset_name)

        if KBERT:
            if kg_name == "DRUG":
                self.kg = MyKnowledgeGraph([settings.BASE_DIR + '/knowledge_graphs/UATC.spo'])
            elif kg_name == "SYMP":
                self.kg = MyKnowledgeGraph([settings.BASE_DIR + '/knowledge_graphs/SYMP.spo'])
            elif kg_name == "DRUGO_SYMP":
                self.kg = MyKnowledgeGraph([settings.BASE_DIR + '/knowledge_graphs/SYMP.spo',settings.BASE_DIR + '/knowledge_graphs/UATC.spo' ])
        if CONCAT_MODEL:
            if kg_name == "DRUGO_SYMP":
                if node_embedding_type in ['bert', 'biobert']:
                    self.gnn = GraphAttentionModel(out_size=28, gnn_hidden_size=768, num_node_features=768)
                else:
                    self.gnn = GraphAttentionModel(out_size=28)
            else:
                if node_embedding_type in ['bert', 'biobert']:
                    self.gnn = GraphAttentionModel(gnn_hidden_size=768, num_node_features=768)
                else:
                    self.gnn = GraphAttentionModel()
            if kg_name == "DRUG":
                if node_embedding_type == "bert":
                    self.gnn.load_state_dict(torch.load(settings.BASE_DIR + "/models/GNN/GAT_BERT_DRUGO_state_dict.pt"))
                    self.gnn.eval()
                    with open(settings.BASE_DIR + "/knowledge_graphs/BERT_embs_graph_data_DRUGO.pkl", "rb") as f:
                        self.graph_data = pickle.load(f)
                elif node_embedding_type == "biobert":
                    self.gnn.load_state_dict(torch.load(settings.BASE_DIR + "/models/GNN/GAT_BioBERT_DRUGO_state_dict.pt"))
                    self.gnn.eval()
                    with open(settings.BASE_DIR + "/knowledge_graphs/BioBERT_embs_graph_data_DRUGO.pkl", "rb") as f:
                        self.graph_data = pickle.load(f)
                else:
                    raise Exception("Unknown node embedding type", node_embedding_type)
                # self.gnn.load_state_dict(torch.load(settings.BASE_DIR + "/models/GNN/GAT_UATC_state_dict.pt"))
                # self.gnn.eval()
                # with open(settings.BASE_DIR + "/knowledge_graphs/graph_data_ATC.pkl", "rb") as f:
                #     self.graph_data = pickle.load(f)
                self.nlp = spacy.load(settings.BASE_DIR + "/tagger/atc_tagger_DB")
            elif kg_name == "SYMP":
                if node_embedding_type == "bert":
                    self.gnn.load_state_dict(torch.load(settings.BASE_DIR + "/models/GNN/GAT_BERT_SYMP_state_dict.pt"))
                    self.gnn.eval()
                    with open(settings.BASE_DIR + "/knowledge_graphs/BERT_embs_graph_data_SYMP.pkl", "rb") as f:
                        self.graph_data = pickle.load(f)
                elif node_embedding_type == "biobert":
                    self.gnn.load_state_dict(torch.load(settings.BASE_DIR + "/models/GNN/GAT_BioBERT_SYMP_state_dict.pt"))
                    self.gnn.eval()
                    with open(settings.BASE_DIR + "/knowledge_graphs/BioBERT_embs_graph_data_SYMP.pkl", "rb") as f:
                        self.graph_data = pickle.load(f)
                else:
                    raise Exception("Unknown node embedding type", node_embedding_type)

                # self.gnn.load_state_dict(torch.load(settings.BASE_DIR + "/models/GNN/GAT_SYMP_state_dict.pt"))
                # self.gnn.eval()
                # with open(settings.BASE_DIR + "/knowledge_graphs/graph_data_SYMP.pkl", "rb") as f:
                #     self.graph_data = pickle.load(f)
                self.nlp = spacy.load(settings.BASE_DIR + "/tagger/Symp_tagger")
            elif kg_name == "DRUGO_SYMP":
                if node_embedding_type == "bert":
                    self.gnn.load_state_dict(torch.load(settings.BASE_DIR + "/models/GNN/GAT_BERT_DRUGO_SYMP_state_dict.pt"))
                    self.gnn.eval()
                    with open(settings.BASE_DIR + "/knowledge_graphs/BERT_embs_graph_data_DRUGO_SYMP.pkl", "rb") as f:
                        self.graph_data = pickle.load(f)
                elif node_embedding_type == "biobert":
                    self.gnn.load_state_dict(torch.load(settings.BASE_DIR + "/models/GNN/GAT_BioBERT_DRUGO_SYMP_state_dict.pt"))
                    self.gnn.eval()
                    with open(settings.BASE_DIR + "/knowledge_graphs/BioBERT_embs_graph_data_DRUGO_SYMP.pkl", "rb") as f:
                        self.graph_data = pickle.load(f)

                else:
                    raise Exception("Unknown node embedding type", node_embedding_type)

                # self.gnn.load_state_dict(torch.load(settings.BASE_DIR + "/models/GNN/GAT_DRUGO_SYMP_state_dict.pt"))
                # self.gnn.eval()
                # with open(settings.BASE_DIR + "/knowledge_graphs/graph_data_DRUGO_SYMP.pkl", "rb") as f:
                #     self.graph_data = pickle.load(f)
                self.nlp = spacy.load(settings.BASE_DIR + "/tagger/DRUGO_SYMP_tagger")
            else:
                raise Exception("Unknown KG name", kg_name)
            
            
            print(self.nlp.analyze_pipes())
            self.graph_embeddings = self.gnn.get_embeddings(self.graph_data)
            print(self.graph_embeddings.shape)
        if CONCAT_MODEL_AW:
            if kg_name == "DRUG":
                self.nlp = spacy.load(settings.BASE_DIR + "/tagger/atc_tagger_DB")
                with open(settings.BASE_DIR + "/knowledge_graphs/graph_data_ATC.pkl", "rb") as f:
                    self.graph_data = pickle.load(f)
            elif kg_name == "SYMP":
                self.nlp = spacy.load(settings.BASE_DIR + "/tagger/Symp_tagger")
                with open(settings.BASE_DIR + "/knowledge_graphs/graph_data_SYMP.pkl", "rb") as f:
                    self.graph_data = pickle.load(f)
            elif kg_name == "DRUGO_SYMP":
                with open(settings.BASE_DIR + "/knowledge_graphs/graph_data_DRUGO_SYMP.pkl", "rb") as f:
                    self.graph_data = pickle.load(f)
                self.nlp = spacy.load(settings.BASE_DIR + "/tagger/DRUGO_SYMP_tagger")
            else:
                raise Exception("Unknown KG name", kg_name)
        
        if ADD_KNOWLEDGE:
            #self.nlp = spacy.load("../../tagger/atc_tagger")
            try:
                if kg_name == "DRUG":
                    self.nlp = spacy.load(settings.BASE_DIR + "/tagger/atc_tagger_DB")
                    self.embed = pickle.load(open(settings.BASE_DIR + "/embeddings/embedded_ontologies/atc/uatc_trans_e.pkl", "rb"))
                elif kg_name == "SYMP":
                    self.nlp = spacy.load(settings.BASE_DIR + "/tagger/Symp_tagger")
                    self.embed = pickle.load(open(settings.BASE_DIR + "/embeddings/embedded_ontologies/symp/symp_trans_e.pkl", "rb"))
                elif kg_name == "DRUGO_SYMP":
                    self.nlp = spacy.load(settings.BASE_DIR + "/tagger/DRUGO_SYMP_tagger")
                    self.embed = pickle.load(open(settings.BASE_DIR + "/embeddings/embedded_ontologies/DRUGO_SYMP/drugo_symp_trans_e.pkl", "rb"))
                else:
                    raise Exception("Unknown KG name", kg_name)
                
                #self.nlp = spacy.load("../../tagger/atc_tagger_DB")
                #self.embed = gensim.models.Word2Vec.load("../../embeddings/embedded_ontologies/symp/ontology.embeddings")
                #self.embed = gensim.models.Word2Vec.load("../../embeddings/embedded_ontologies/atc/ontology.embeddings")
                
            except:
                raise Exception()
                self.nlp = spacy.load(settings.BASE_DIR + "/tagger/Symp_tagger")
                self.embed = gensim.models.Word2Vec.load(settings.BASE_DIR + "/embeddings/embedded_ontologies/symp/ontology.embeddings")
        if (not ADD_KNOWLEDGE) and (not CONCAT_MODEL) and (not CONCAT_MODEL_AW):
            print("Falling back on default spacy model")
            self.nlp = spacy.load("en_core_web_sm")


    def tokenize_and_preserve_labels(self, sentence, text_labels, doc=None):

        """
        Word piece tokenization makes it difficult to match word labels
        back up with individual word pieces. This function tokenizes each
        word one at a time so that it is easier to preserve the correct
        label for each subword. It is, of course, a bit slower in processing
        time, but it will help our model achieve higher accuracy.
        """

        tokenized_sentence = []
        labels = []
        tokenizer = self.tokenizer
        

        if ADD_KNOWLEDGE:
            #print([tok.ent_id_ for tok in doc])
            embed_ents = []
            ent_mask = []
            tagged_entities = [(ent.start_char,ent.end_char, ent.ent_id_) for ent in doc.ents if (ent.label_.startswith("SYMP") or ent.label_.startswith("ATC-DRUG") or ent.label_.startswith("DRUGO_SYMP"))]
            bilou_tags = [str(tag).replace("L", "I").replace("U", "B") for tag in offsets_to_biluo_tags(doc, tagged_entities)] ## diese Zeile hat gefehlt
            bilou_tags = offsets_to_biluo_tags(doc, tagged_entities)
        
        if CONCAT_MODEL:
            embed_ents = []
            tagged_entities = [(ent.start_char,ent.end_char, ent.ent_id_) for ent in doc.ents if (ent.label_.startswith("SYMP") or ent.label_.startswith("ATC-DRUG") or ent.label_.startswith("DRUGO_SYMP"))]
            bilou_tags = [str(tag).replace("L", "I").replace("U", "B") for tag in offsets_to_biluo_tags(doc, tagged_entities)] ## diese Zeile hat gefehlt
            bilou_tags = offsets_to_biluo_tags(doc, tagged_entities)

        if CONCAT_MODEL_AW:
            embed_ents_indexes = []
            tagged_entities = [(ent.start_char,ent.end_char, ent.ent_id_) for ent in doc.ents if (ent.label_.startswith("SYMP") or ent.label_.startswith("ATC-DRUG") or ent.label_.startswith("DRUGO_SYMP"))]
            bilou_tags = [str(tag).replace("L", "I").replace("U", "B") for tag in offsets_to_biluo_tags(doc, tagged_entities)] ## diese Zeile hat gefehlt
            self.total_ents_count += len(tagged_entities)
            if len(tagged_entities)>0:
                #print(bilou_tags)
                self.total_ents_count_only_start +=1



        for i, (word, label) in enumerate(zip(sentence, text_labels)):
                   

            # Tokenize the word and count # of subwords the word is broken into
            tokenized_word = tokenizer.tokenize(word)
            n_subwords = len(tokenized_word)

            # Add the tokenized word to the final tokenized word list
            tokenized_sentence.extend(tokenized_word)

            # Add the same label to the new list of labels `n_subwords` times
            labels.extend([label] * n_subwords)
            if ADD_KNOWLEDGE:
                if bilou_tags[i].startswith("B-"):
                    if EMBEDDINGS_AS_DICT:
                        embed_ents.append(self.embed[bilou_tags[i].replace("B-", "")].tolist())
                    else:
                        embed_ents.append(self.embed.wv.get_vector(bilou_tags[i].replace("B-", "")).tolist())
                    if n_subwords>1:
                        embed_ents.extend([[0]*EMBEDDING_DIMENSION]*(n_subwords-1))
                else:
                    embed_ents.extend([[0]*EMBEDDING_DIMENSION]*n_subwords)
                #print(f"EMBED ENTS: {len(embed_ents)}")
            if CONCAT_MODEL:
                if bilou_tags[i].startswith("B-"):
                    iri = bilou_tags[i].replace("B-", "")
                    #print(iri)
                    index_ = self.graph_data.node_index.index(iri)
                    #print(index_)
                    emb_vec = self.graph_embeddings[index_]
                    embed_ents.append(emb_vec.tolist())
                    if n_subwords>1:
                        embed_ents.extend([[0]*EMBEDDING_DIMENSION]*(n_subwords-1))
                else:
                    embed_ents.extend([[0]*EMBEDDING_DIMENSION]*n_subwords)
            if CONCAT_MODEL_AW:
                if bilou_tags[i].startswith("B-"):
                    iri = bilou_tags[i].replace("B-", "")
                    try:
                        index_ = self.graph_data.node_index.index(iri)
                        embed_ents_indexes.append(index_)
                        if n_subwords>1:
                            embed_ents_indexes.extend([-1]*(n_subwords-1)) ## -1 for "padding"
                    except ValueError:
                        embed_ents_indexes.extend([-2]*(n_subwords))
                else:
                    embed_ents_indexes.extend([-2]*(n_subwords)) ## -2 for no entity

                #print(f"EMBED ENTS: {len(embed_ents)}")
        if ADD_KNOWLEDGE:
            for emb in embed_ents:
                if emb != [0]*EMBEDDING_DIMENSION:
                    ent_mask.append(1)
                else:
                    ent_mask.append(0)
            
           
            #embed_ents = [[0]*EMBEDDING_DIMENSION]+ embed_ents + [[0]*EMBEDDING_DIMENSION]
            if len(tokenized_sentence)>MAX_LENGTH_PADDING-2:
                return tokenized_sentence[:MAX_LENGTH_PADDING-2], labels[:MAX_LENGTH_PADDING-2], embed_ents[:MAX_LENGTH_PADDING-2], ent_mask[:MAX_LENGTH_PADDING-2]
            else:
                return tokenized_sentence, labels,embed_ents, ent_mask
        else:
            if CONCAT_MODEL:
                if len(tokenized_sentence)>MAX_LENGTH_PADDING-2:
                    return tokenized_sentence[:MAX_LENGTH_PADDING-2], labels[:MAX_LENGTH_PADDING-2], embed_ents[:MAX_LENGTH_PADDING-2]
                else:
                    return tokenized_sentence, labels,embed_ents
            if CONCAT_MODEL_AW:
                if len(tokenized_sentence)>MAX_LENGTH_PADDING-2:
                    return tokenized_sentence[:MAX_LENGTH_PADDING-2], labels[:MAX_LENGTH_PADDING-2], embed_ents_indexes[:MAX_LENGTH_PADDING-2]
                else:
                    return tokenized_sentence, labels,embed_ents_indexes
            if KBERT:
                if len(tokenized_sentence)>MAX_LENGTH_PADDING:
                    return tokenized_sentence[:MAX_LENGTH_PADDING], labels[:MAX_LENGTH_PADDING]
                else:
                    return tokenized_sentence, labels
            else:
            
                if len(tokenized_sentence)>MAX_LENGTH_PADDING-2:
                    return tokenized_sentence[:MAX_LENGTH_PADDING-2], labels[:MAX_LENGTH_PADDING-2]
                else:
                    return tokenized_sentence, labels



    def do_iob_labelling(self, data, type="ADR"):
        iob_labelled = []
        total_removed = 0
        for d in data:
            if WITH_MEDDRA:
                annos_ordered = dict.fromkeys([an['anno'] for an in d['annotations'] if an['type'] == type])
                textual_annotations = list(annos_ordered)
            annos = list(set([(an['start'], an['end'], an['type']) for an in d['annotations'] if an['type'] == type]))
            
            if len(annos) == 0:
                print("No annotations of type found")
                continue
            ## get rid of nested annotations
            remove_list = []
            for i, a in enumerate(annos):
                for k, b in enumerate(annos):
                    if i!= k:
                        if a[0] == b[0]:
                            if a[1]>b[1]:
                                remove_list.append(k)
                            elif a[1]<b[1]:
                                remove_list.append(i)
                            else:
                                raise Exception("Duplicated")
                        if a[1] == b[1]:
                            if a[0]>b[0]:
                                remove_list.append(i)
                            elif a[0]<b[0]:
                                remove_list.append(k)
                            else:
                                raise Exception("Duplicated")
            if len(remove_list)>0:
                total_removed+=len(remove_list)
                remove_list = list(set(remove_list))
                remove_list.sort()
                for i, r in enumerate(remove_list):
                    annos = annos[:(r-i)] + annos[(r-i+1):]


                #print(remove_list)
            doc = self.nlp(d['text'])
            #print([ent.label_ for ent in doc.ents])
            #print([tok for tok in doc])
            try:
                tags = [str(tag).replace("L", "I").replace("U", "B") for tag in offsets_to_biluo_tags(doc, annos)]
                #tags = offsets_to_biluo_tags(doc, annos)
            except:
                print(d)
                raise
            if WITH_MEDDRA:
                iob_labelled.append({'text': d['text'], "text_tokens": [tok.text for tok in doc], 'label': tags, 'doc': doc, "textual_annos": textual_annotations})
            else:
                iob_labelled.append({'text': d['text'], "text_tokens": [tok.text for tok in doc], 'label': tags, 'doc': doc})
        print(f"Total removed due to overlapping: {total_removed}")
        return iob_labelled


    def get_psytar_train(self, annotation_type="ADR"):
        with open(self.psytar_train, "r") as f:
            data = json.load(f)
            return self.get_data(data, annotation_type)
        

    def get_psytar_test(self, annotation_type="ADR"):
        with open(self.psytar_test, "r") as f:
            data = json.load(f)
            return self.get_data(data, annotation_type)
        
    def get_adev2_train(self, annotation_type="ADR"):
        with open(self.adev2_train, "r") as f:
            data = json.load(f)
            return self.get_data(data, annotation_type)
        
    def get_adev2_test(self, annotation_type="ADR"):
        with open(self.adev2_test, "r") as f:
            data = json.load(f)
            return self.get_data(data, annotation_type)
        
    def get_tac_train(self, annotation_type="ADR"):
        with open(self.tac_train, "r") as f:
            data = json.load(f)
            return self.get_data(data, annotation_type)
        
    def get_tac_test(self, annotation_type="ADR"):
        with open(self.tac_test, "r") as f:
            data = json.load(f)
            return self.get_data(data, annotation_type)
    

    def get_cadec_train(self,annotation_type="ADR"):
        
        with open(self.cadec_train, "r") as f:
            data = json.load(f)
            return self.get_data(data, annotation_type)

    def get_cadec_test(self, annotation_type="ADR"):
        
        with open(self.cadec_test, "r") as f:
            data = json.load(f)
            return self.get_data(data, annotation_type)

    def get_smm4h_train(self, annotation_type="ADR" ):
        
        with open(self.smm4h_train, "r") as f:
            data = json.load(f)
            data = filter_overlapping(data)
            return self.get_data(data, annotation_type)
    def get_smm4h_test(self, annotation_type="ADR" ):
        
        with open(self.smm4h_test, "r") as f:
            data = json.load(f)
            data = filter_overlapping(data)
            return self.get_data(data, annotation_type)
        
    def get_psytar_for_kbert_train(self, annotation_type="ADR"):
        with open(self.psytar_train, "r") as f:
            data = json.load(f)
            return self.get_data_kbert(data, annotation_type)
        
    def get_psytar_for_kbert_test(self, annotation_type="ADR"):
        with open(self.psytar_test, "r") as f:
            data = json.load(f)
            return self.get_data_kbert(data, annotation_type)
        
    def get_adev2_for_kbert_train(self, annotation_type="ADR"):
        with open(self.adev2_train, "r") as f:
            data = json.load(f)
            return self.get_data_kbert(data, annotation_type)
    
    def get_adev2_for_kbert_test(self, annotation_type="ADR"):
        with open(self.adev2_test, "r") as f:
            data = json.load(f)
            return self.get_data_kbert(data, annotation_type)
        
    def get_tac_for_kbert_train(self, annotation_type="ADR"):
        with open(self.tac_train, "r") as f:
            data = json.load(f)
            return self.get_data_kbert(data, annotation_type)
        
    def get_tac_for_kbert_test(self, annotation_type="ADR"):
        with open(self.tac_test, "r") as f:
            data = json.load(f)
            return self.get_data_kbert(data, annotation_type)

    def get_cadec_for_kbert_train(self, annotation_type="ADR"):
        with open(self.cadec_train, "r") as f:
            data = json.load(f)
            return self.get_data_kbert(data, annotation_type)
    def get_cadec_for_kbert_test(self, annotation_type="ADR"):
        with open(self.cadec_test, "r") as f:
            data = json.load(f)
            return self.get_data_kbert(data, annotation_type)
   
    def get_smm4h_for_kbert_train(self, annotation_type="ADR"):
        with open(self.smm4h_train, "r") as f:
            data = json.load(f)
            data = filter_overlapping(data)
            return self.get_data_kbert(data, annotation_type)

    def get_smm4h_for_kbert_test(self, annotation_type="ADR"):
        with open(self.smm4h_test, "r") as f:
            data = json.load(f)
            data = filter_overlapping(data)
            return self.get_data_kbert(data, annotation_type)

    def get_data_kbert(self, data, annotation_type="ADR"):
        miss_aligned = 0
        print("Total size of data:" + str(len(data)))
        data= self.do_iob_labelling(data, annotation_type)
        tokenized_data = []
        for line in data:
            if "-" in line['label']:
                miss_aligned+=1
                print("Missalignment")# + line['text'])
                continue
            doc = line['doc']
            label = line['label']
            for i , word in enumerate(doc):
                word.tag_ = label[i]
            # Code from https://github.com/autoliuweijie/K-BERT/blob/da9358edb3b3f59e3ad6b5aab6af6df624b881ab/run_kbert_ner.py#L225
            tokens, pos, vm, tag, labels = self.kg.add_knowledge_with_vm([doc], max_length=MAX_LENGTH_PADDING)
            tokens = tokens[0]
            pos = pos[0]
            vm = vm[0]
            tag = tag[0]
            att_mask = [1]*len(tokens)

            labels = [label2ids[label] for label in labels[0]]

            #toks , toks_labels = self.tokenize_and_preserve_labels(line['text'], line['label'])

            ## generate new labelling according to wordpiece tokenization

            #l = len(tokens) - len([tok for tok in tokens if tok.startswith("##")]) - len([t for t in tag if t==1]) - len([tok for tok in tokens if not tok != "[PAD]"])
            
            #labels = line['label']
            #print(l - len(label))
            new_labels = []
            j = 0
            for i in range(len(tokens)):
                if tag[i] == 0 and tokens[i] != label2ids["[PAD]"]:
                    new_labels.append(labels[j])
                    j += 1
                elif tag[i] == 1 and tokens[i] != label2ids["[PAD]"]:  # 是添加的实体
                    new_labels.append(label2ids['[ENT]'])
                else:
                    new_labels.append(label2ids["PAD"])

            # variables are there just to check the results in debug mode ()
            # old_label_names = [ids2label[label_id] for label_id in labels]
            # new_label_names = [ids2label[label_id] for label_id in new_labels]

            '''
            new_labels = []
            current_label = labels[0]
            k = 0
            for i, tok in enumerate(tokens):
                if tag[i] == 1:
                    new_labels.append(label2ids["[ENT]"])
                else:
                    if tok.startswith("##"):
                        new_labels.append(label2ids[current_label])
                    else:
                        if tok == "[PAD]":
                            new_labels.append(label2ids['O'])
                        else:
                            try:
                                #print(tok, i)
                                new_labels.append(label2ids[labels[k]])
                                current_label = labels[k]
                                k+=1
                            except:
                                #print(l)
                                print("test")
            '''

            tokens_tensor = torch.LongTensor(self.tokenizer.encode(tokens, add_special_tokens=False))
            pos_tensor = torch.LongTensor(pos)
            vm_tensor = torch.BoolTensor(vm)
            tag_tensor = torch.LongTensor(tag)
            labels_tensor = torch.LongTensor(new_labels)
            att_mask_tensor = torch.LongTensor(att_mask)
            tokenized_data.append([tokens_tensor, pos_tensor, vm_tensor, tag_tensor, labels_tensor, att_mask_tensor])
        
        return tokenized_data

    def get_data(self, data, annotation_type="ADR"):
        miss_aligned = 0
        if True:
            #data = json.load(f)
            print("Total size of data:" + str(len(data)))
            data = self.do_iob_labelling(data, annotation_type)
            embedded_ents = []
            embedded_ents_index = []
            ent_masks = []
             
            tokenized_data = []
            for line in data:
                if "-" in line['label']:
                    miss_aligned+=1
                    print("Missalignment")# + line['text'])
                    continue
                if ADD_KNOWLEDGE:
                    tokens, label, entities, ent_mask = self.tokenize_and_preserve_labels(line['text_tokens'], line['label'], line['doc'])
                    entities = [[0]*EMBEDDING_DIMENSION] + entities + [[0]*EMBEDDING_DIMENSION]
                    entities = entities + (MAX_LENGTH_PADDING - len(entities))*[[0]*EMBEDDING_DIMENSION]
                    embedded_ents.append(torch.Tensor(entities))
                    ent_mask[0] = 1 ## why? 
                    ent_mask = ent_mask + (MAX_LENGTH_PADDING- len(ent_mask))*[0]
                    ent_masks.append(torch.Tensor(ent_mask))

                elif CONCAT_MODEL:
                    tokens, label, entities = self.tokenize_and_preserve_labels(line['text_tokens'], line['label'], line['doc'])
                    entities = [[0]*EMBEDDING_DIMENSION] + entities + [[0]*EMBEDDING_DIMENSION]
                    entities = entities + (MAX_LENGTH_PADDING - len(entities))*[[0]*EMBEDDING_DIMENSION]
                    embedded_ents.append(torch.Tensor(entities))
                elif CONCAT_MODEL_AW:
                    tokens, label, entity_indexes= self.tokenize_and_preserve_labels(line['text_tokens'], line['label'], line['doc'])
                    entity_indexes = [-2] + entity_indexes + [-2]
                    entity_indexes = entity_indexes + (MAX_LENGTH_PADDING - len(entity_indexes))*[-2]
                    embedded_ents_index.append(torch.tensor(entity_indexes, dtype=torch.long))
                else:
                    tokens, label = self.tokenize_and_preserve_labels(line['text_tokens'], line['label'], line['doc'])
                #all_lengths.append(len(tokens) + 2)
                input_ids = self.tokenizer.convert_tokens_to_ids(["[CLS]"] + tokens + ["[SEP]"])
                label = ["[CLS]"] + label + ["[SEP]"]
                attention_mask = len(input_ids)*[1]
                #print("Attention mask length: " + str(len(attention_mask)))
                attention_mask = attention_mask + (MAX_LENGTH_PADDING - len(input_ids))*[0]
                input_ids = input_ids + (MAX_LENGTH_PADDING - len(input_ids))*[0]
                label = label + (MAX_LENGTH_PADDING - len(label))*["[PAD]"]
                toke_type_ids = MAX_LENGTH_PADDING*[0]
                
                try:
                    assert(len(label) == len(input_ids)== MAX_LENGTH_PADDING)
                except:
                    print(len(label))
                    print(len(input_ids))
                    #print(len(entities))
                    #print(len(ent_mask))
                    raise


                ## labels to ids
                label = [label2ids[l] for l in label]
                if label == toke_type_ids:
                    self.removed_due_to_no_label +1
                    continue
                    #raise Exception("Empty labels")
                #print(label)
                
                if WITH_MEDDRA:
                    tokenized_data.append([{'input_ids': torch.Tensor(input_ids).long(), 'token_type_ids':torch.Tensor(toke_type_ids).long(), 'attention_mask':torch.Tensor(attention_mask).long()}, torch.Tensor(label).long(), line['text'],line["textual_annos"]])
                else:
                    tokenized_data.append([{'input_ids': torch.Tensor(input_ids).long(), 'token_type_ids':torch.Tensor(toke_type_ids).long(), 'attention_mask':torch.Tensor(attention_mask).long()}, torch.Tensor(label).long()])
        print(f"Total missaligned: {miss_aligned}")
        if ADD_KNOWLEDGE:
            return list(zip(tokenized_data, embedded_ents, ent_masks))
            #return [[tokenized_data[k], embedded_ents[k], ent_masks[k]] for k in range(0, len(tokenized_data))]
        if CONCAT_MODEL:
            return list(zip(tokenized_data, embedded_ents))
        if CONCAT_MODEL_AW:
            return list(zip(tokenized_data, embedded_ents_index))
        return tokenized_data


# %%        
#all_lengths = []
#loader = MyDataLoader()

# %%

#out = loader.get_cadec_for_kbert_train()

# %%
