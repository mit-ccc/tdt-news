import json, random, logging, os, shutil
import math, pickle, queue
from collections import OrderedDict, defaultdict
from typing import List, Dict, Tuple, Iterable, Type, Union, Callable
from zipfile import ZipFile
import requests
import numpy as np
from numpy import ndarray
import argparse

import transformers
import torch
from torch import nn, Tensor, device
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from tqdm.autonotebook import trange

from sentence_transformers import __DOWNLOAD_SERVER__
from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers.util import import_from_string, batch_to_device, http_get
from sentence_transformers.models import Transformer, Pooling
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, util
from sentence_transformers import __version__
from esbert.transformer_entity import BertEntityEmbeddings, EntityBertModel
from esbert.transformer_entity import EntityTransformer
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
import sys
from datetime import datetime
sys.path.insert(0,"/mas/u/hjian42/tdt-twitter/baselines/news-clustering/")
import load_corpora
import clustering
from Model import Date2VecConvert
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Date2VecConvert(nn.Module):
    def __init__(self, model_path="./d2v_model/d2v_98291_17.169918439404636.pth"):
        super(Date2VecConvert, self).__init__()
        self.model = torch.load(model_path, map_location=device).to(device)
    
    def forward(self, x):
#         print("self.model", next(self.model.parameters()).is_cuda)
#         print("x", x.shape)
#         x = torch.Tensor(x).cuda()
        return self.model.encode(x)



def convert_str_to_date_tensor(string):
    date_obj = datetime.strptime(string, "%Y-%m-%d %H:%M:%S")
    date_list = [date_obj.hour, date_obj.minute, date_obj.second,
                 date_obj.year, date_obj.month, date_obj.day]
    return date_list


def custom_collate_fn(batch):
    """collate for List of InputExamples, not triplet examples"""
    texts = []
    entities = []
    dates = []

    for example in batch:
        texts.append(example.texts)
        dates.append(convert_str_to_date_tensor(example.times))

        entity_list = np.array(example.entities)
        entity_list = entity_list[entity_list < 512]
        new_entity_list = np.zeros(512, dtype=int)
        new_entity_list[entity_list] = 1
        entities.append(new_entity_list)

    tokenized = entity_transformer.tokenize(texts) # HACK: use the model's internal tokenize() function
    tokenized['entity_type_ids'] = torch.tensor(entities)
    tokenized['dates'] = torch.tensor(dates).float()
    
    return tokenized


class InputExample:
    """
    Structure for one input example with texts, the label and a unique id
    """
    def __init__(self, 
                 guid: str = '', 
                 texts: List[str] = None,  
                 label: Union[int, float] = 0, 
                 entities: List = None,
                 times: List = None
                ):
        """
        Creates one InputExample with the given texts, guid and label
        :param guid
            id for the example
        :param texts
            the texts for the example. Note, str.strip() is called on the texts
        :param label
            the label for the example
        """
        self.guid = guid
        self.texts = texts
        self.label = label
        self.entities = entities
        self.times = times

    def __str__(self):
        return "<InputExample> label: {}, texts: {}".format(str(self.label), "; ".join(self.texts[:10]))


def triplets_from_labeled_dataset(input_examples):
    # Create triplets for a [(label, sentence), (label, sentence)...] dataset
    # by using each example as an anchor and selecting randomly a
    # positive instance with the same label and a negative instance with a different label
    triplets = []
    label2sentence = defaultdict(list)
    for inp_example in input_examples:
        label2sentence[inp_example.label].append(inp_example)

    for inp_example in input_examples:
        anchor = inp_example

        if len(label2sentence[inp_example.label]) < 2: #We need at least 2 examples per label to create a triplet
            continue

        positive = None
        while positive is None or positive.guid == anchor.guid:
            positive = random.choice(label2sentence[inp_example.label])

        negative = None
        while negative is None or negative.label == anchor.label:
            negative = random.choice(input_examples)

        triplets.append(InputExample(texts=[anchor.texts, positive.texts, negative.texts],
                                     entities=[anchor.entities, positive.entities, negative.entities],
                                     times=[anchor.times, positive.times, negative.times]
                                    ))
    
    return triplets


class BertPooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        out = (x - mean) / (std + self.eps)
        out = self.gamma * out + self.beta
        return out


class TimeESBert(nn.Module):
    
    def __init__(self, esbert_model, time_model, fuse_method="selfatt_pool", device="cuda"):
        super(TimeESBert, self).__init__()
        self.esbert_model = esbert_model.to(device)
        self.time_model = time_model.to(device)
        self.fuse_method = fuse_method
        self.pooler = BertPooler(768).to(device)
        self.concat_linear = nn.Linear(832, 832).to(device)
        if "att" in fuse_method:
            self.multi_att = nn.MultiheadAttention(832, 8, 0.1).to(device)
            self.norm_layer = LayerNorm(832).to(device)
            self.pooler = BertPooler(832).to(device)
            
    def forward(self, features):
                
        batch_to_device(features, device)
        bert_features = self.esbert_model(features)
        cls_embeddings = bert_features['cls_token_embeddings']
        token_embeddings = bert_features['token_embeddings']
        
        # fuse temporal and linguistic features
        time_features = self.time_model(features['dates'])
#         print("time_features", time_features.shape)
        
        # 1. concatenation + pool
        if self.fuse_method == "concat_pool":
            pooled_features = self.pooler(token_embeddings)
            fused_features = torch.cat([pooled_features, time_features], dim=1)
        
        # concat + pool + linear transformation
        elif self.fuse_method == "concat_pool_linear":
            pooled_features = self.pooler(token_embeddings)
            fused_features = torch.cat([pooled_features, time_features], dim=1)
            fused_features = self.concat_linear(fused_features)
        
        # concat + selfatt + normalization
        elif self.fuse_method == "selfatt_pool":
            repeat_time_vec = time_features.unsqueeze(1).repeat(1, token_embeddings.shape[1], 1)
            concat_time_token_emb = torch.cat([token_embeddings, repeat_time_vec], 2)
            attn_output, attn_output_weights = self.multi_att(concat_time_token_emb, concat_time_token_emb, concat_time_token_emb)
            norm_attn_output = self.norm_layer(attn_output + concat_time_token_emb)
            fused_features = self.pooler(norm_attn_output)
        
        features.update({"sentence_embedding": fused_features})
        return features
    

def smart_batching_collate(batch):
        """
        Transforms a batch from a SmartBatchingDataset to a batch of tensors for the model
        Here, batch is a list of tuples: [(tokens, label), ...]
        :param batch:
            a batch from a SmartBatchingDataset
        :return:
            a batch of tensors for the model
        """
        num_texts = len(batch[0].texts)
        texts = [[] for _ in range(num_texts)]
        labels = []
        entities = [[] for _ in range(num_texts)]
        dates = [[] for _ in range(num_texts)]

        for example in batch:
            for idx, text in enumerate(example.texts):
                texts[idx].append(text)
            for idx, entity_list in enumerate(example.entities):
                entity_list = np.array(entity_list)
                entity_list = entity_list[entity_list < 512]
                new_entity_list = np.zeros(512, dtype=int)
                new_entity_list[entity_list] = 1
                entities[idx].append(new_entity_list)
            for idx, date_str in enumerate(example.times):
                dates[idx].append(convert_str_to_date_tensor(date_str))

            labels.append(example.label)

        labels = torch.tensor(labels).to(device)
    
#         print(dates)
        sentence_features = []
        for idx in range(num_texts):
            tokenized = entity_transformer.tokenize(texts[idx])
            tokenized['entity_type_ids'] = torch.tensor(entities[idx]).to(device)
            tokenized['dates'] = torch.tensor(dates[idx]).float().to(device)
            sentence_features.append(tokenized)

        return sentence_features, labels


def train(loss_model, dataloader, epochs=2, train_batch_size=2, warmup_steps=1000, weight_decay=0.01,  max_grad_norm=1.0, device='cuda'):
    
    # initialization
    global_step = 0
    total_loss = 0
    
    dataloader.collate_fn = smart_batching_collate
    steps_per_epoch = len(dataloader)
    num_training_steps = int(steps_per_epoch * epochs)
    data_iterator = iter(dataloader)

    # prepare optimizer
    param_optimizer = list(loss_model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=2e-5, eps=1e-06)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, 
                                                                 num_warmup_steps=warmup_steps, 
                                                                 num_training_steps=num_training_steps)
    
    for epoch in trange(epochs, desc="Epoch"):
        loss_model.zero_grad()
        loss_model.train()
        
        training_steps = 0
        
        for _ in trange(steps_per_epoch, desc="Iteration", smoothing=0.05):
            try:
                data = next(data_iterator)
            except StopIteration:
                data_iterator = iter(dataloader)
                data = next(data_iterator)
            
            features, labels = data
#             for each_f in features:
#                 print(each_f.keys(), each_f['input_ids'].shape)
            loss_value = loss_model(features, labels)
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
            optimizer.step()
            
            optimizer.zero_grad()
            scheduler.step()
            total_loss += loss_value
        training_steps += 1
        global_step += 1
        print("Avg loss is {} on training data".format(total_loss / (epoch+1)))



# global variable
entity_transformer = EntityTransformer("/mas/u/hjian42/tdt-twitter/baselines/news-clustering/entity-bert/pretrained/0_Transformer/")

def main():
    random.seed(123)

    parser = argparse.ArgumentParser(description="main training script for word2vec dynamic word embeddings...")
    parser.add_argument("--source_dir", type=str, default="./data/english-uk/", help="source dir")
    parser.add_argument("--dest_dir", type=str, default="./output/english-uk/", help="dest dir")
    parser.add_argument('--dim', type=int, default=100, help='Number of dimensions. Default is 100.')
    args = parser.parse_args()

    with open('/mas/u/hjian42/tdt-twitter/baselines/news-clustering/entity-bert/test.pickle', 'rb') as handle:
        test_corpus = pickle.load(handle)
    print("finished loading pickle files")

    # initialize a model
    # entity_transformer = EntityTransformer("/mas/u/hjian42/tdt-twitter/baselines/news-clustering/entity-bert/pretrained/0_Transformer/")
    entity_transformer.max_seq_length = 512
    date2vec_model = Date2VecConvert(model_path="/mas/u/hjian42/tdt-twitter/baselines/T-ESBERT/Date2Vec/d2v_model/d2v_98291_17.169918439404636.pth")
    print("finished loading date2vec")

    time_esbert = TimeESBert(entity_transformer, date2vec_model, fuse_method="concat_pool")

    dev_examples = [InputExample(texts=d['text'], 
                                label=d['cluster'],
                                guid=d['id'], 
                                entities=d['bert_entities'], 
                                times=d['date']
                                ) for d in test_corpus.documents]

    num_epochs = 5
    train_batch_size = 2
    margin = 2.0
    max_grad_norm = 1.0

    sampled_examples = random.sample(dev_examples, 30)

    train_trip_examples = triplets_from_labeled_dataset(sampled_examples)
    train_dataloader = DataLoader(train_trip_examples, shuffle=True, batch_size=train_batch_size)
    loss_model = losses.BatchHardTripletLoss(model=time_esbert, 
                                            distance_metric=losses.BatchHardTripletLossDistanceFunction.cosine_distance,
                                            margin=margin)
    warmup_steps = math.ceil(len(train_trip_examples)*num_epochs/train_batch_size*0.1) #10% of train data for warm-up

    train(loss_model, 
        train_dataloader, 
        epochs=num_epochs, 
        train_batch_size=train_batch_size, 
        warmup_steps=warmup_steps, 
        max_grad_norm=max_grad_norm,
        device=device)


    # sanity checking
    # dev_dataloader = DataLoader(dev_examples, shuffle=False, batch_size=2)

    # with torch.no_grad():
    #     sentence_embeddings_list = []
    #     dev_dataloader.collate_fn = custom_collate_fn
    #     for batch in iter(dev_dataloader):
    #         output_features = time_esbert.forward(batch)
    #         print("output_features", output_features.shape)
    #         break



if __name__ == "__main__":
    main()