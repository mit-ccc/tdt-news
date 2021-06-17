"""
train the entity-aware sentenceBERT (https://arxiv.org/abs/2101.11059)
"""
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
from sentence_transformers.evaluation import TripletEvaluator
from sklearn import preprocessing
from torch.utils.data.sampler import Sampler

class MyBatchSampler(Sampler):
    '''
    randomly sample two instances from the same class at once and order them in order
    e.g. [ 80, 80, 342, 342, 213, 213, 342, 342]
    '''

    def __init__(self, labels):
        self.label2idx = {}
        indices = list(range(len(labels)))
        for idx in range(len(labels)):
            label = labels[idx]
            if label in self.label2idx:
                self.label2idx[label].append(idx)
            else:
                self.label2idx[label] = [idx]
        
        # sample two instances consecutively
        sample_n = 2
        self.idx = []
        for label in labels:
            sampled_instances = random.sample(self.label2idx[label], sample_n)
            self.idx += sampled_instances
            neg_sampled_instances = random.sample(indices, sample_n)
            self.idx += neg_sampled_instances

    def __iter__(self):
        return iter(self.idx)

    def __len__(self):
        return len(self.idx)


def custom_collate_fn(batch):
    """collate for List of InputExamples, not triplet examples"""
    texts = []
    entities = []

    for example in batch:
        texts.append(example.texts)

        entity_list = np.array(example.entities)
        entity_list = entity_list[entity_list < 512]
        new_entity_list = np.zeros(512, dtype=int)
        new_entity_list[entity_list] = 1
        entities.append(new_entity_list)

    tokenized = entity_transformer.tokenize(texts) # HACK: use the model's internal tokenize() function
    tokenized['entity_type_ids'] = torch.tensor(entities)
    
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
                                     entities=[anchor.entities, positive.entities, negative.entities]
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


class EntitySBert(nn.Module):
    """entity-aware BERT"""
    
    def __init__(self, esbert_model, device="cuda"):
        super(EntitySBert, self).__init__()
        self.esbert_model = esbert_model.to(device)
        self.pooler = BertPooler(768).to(device)
            
    def forward(self, features):
                
        batch_to_device(features, device)
        bert_features = self.esbert_model(features)
        cls_embeddings = bert_features['cls_token_embeddings']
        token_embeddings = bert_features['token_embeddings']
                
        pooled_features = self.pooler(token_embeddings)
        
        features.update({"sentence_embedding": pooled_features})
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
        texts = []
        entities = []
        labels = []

        for example in batch:
            texts.append(example.texts)

            entity_list = np.array(example.entities)
            entity_list = entity_list[entity_list < 512]
            new_entity_list = np.zeros(512, dtype=int)
            new_entity_list[entity_list] = 1
            entities.append(new_entity_list)
            
            labels.append(example.label)
        labels = torch.tensor(labels).to(device)

        tokenized = entity_transformer.tokenize(texts) # HACK: use the model's internal tokenize() function
        tokenized['entity_type_ids'] = torch.tensor(entities)
        batch_to_device(tokenized, device)
        
        return [tokenized], labels

def train(loss_model, dataloader, epochs=2, train_batch_size=2, warmup_steps=1000, weight_decay=0.01,  max_grad_norm=1.0, device='cuda', folder_name=None, esbert_model=None):
    
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

        # save models at certain checkpoints
        # if epoch+1 in set([2, 5, 10, 30]):
        torch.save(esbert_model, "{}/esbert_model_ep{}.pt".format(folder_name, epochs))
        print("saving checkpoint: epoch {}".format(epochs))


# global variable
entity_transformer = EntityTransformer("/mas/u/hjian42/tdt-twitter/baselines/news-clustering/entity-bert/pretrained/0_Transformer/")

def main():
    random.seed(123)

    parser = argparse.ArgumentParser(description="main training script for word2vec dynamic word embeddings...")
    parser.add_argument("--num_epochs", type=int, default=2, help="num_epochs")
    parser.add_argument("--train_batch_size", type=int, default=64, help="train_batch_size")
    parser.add_argument("--margin", type=float, default=2.0, help="margin")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="max_grad_norm")
    parser.add_argument("--max_seq_length", type=int, default=512, help="max_seq_length")
#     parser.add_argument("--dest_dir", type=str, default="./output/exp_time_esbert_ep2_m2/", help="dest dir")
#     parser.add_argument('--dim', type=int, default=100, help='Number of dimensions. Default is 100.')
    args = parser.parse_args()
    
    with open('/mas/u/hjian42/tdt-twitter/baselines/news-clustering/entity-bert/train_dev.pickle', 'rb') as handle:
        train_corpus = pickle.load(handle)

#     with open('/mas/u/hjian42/tdt-twitter/baselines/news-clustering/entity-bert/test.pickle', 'rb') as handle:
#         test_corpus = pickle.load(handle)
    print("finished loading pickle files")

    # initialize a model
    # entity_transformer = EntityTransformer("/mas/u/hjian42/tdt-twitter/baselines/news-clustering/entity-bert/pretrained/0_Transformer/")
    entity_transformer.max_seq_length = args.max_seq_length
#     date2vec_model = Date2VecConvert(model_path="/mas/u/hjian42/tdt-twitter/baselines/T-ESBERT/Date2Vec/d2v_model/d2v_98291_17.169918439404636.pth")
#     print("finished loading date2vec")

#     time_esbert = TimeESBert(entity_transformer, date2vec_model, fuse_method="selfatt_pool")
    esbert = EntitySBert(entity_transformer)
    
    # testing on a sample sample whether we can overfit
#     train_corpus.documents = train_corpus.documents[:100]

    labels = [d['cluster'] for d in train_corpus.documents]
    le = preprocessing.LabelEncoder()
    targets = le.fit_transform(labels)
    for d, target in zip(train_corpus.documents, targets):
        d['cluster_label'] = target
    train_examples = [InputExample(texts=d['text'], 
                                label=d['cluster_label'],
                                guid=d['id'], 
                                entities=d['bert_entities'], 
                                ) for d in train_corpus.documents]

    num_epochs = args.num_epochs
    train_batch_size = args.train_batch_size
    margin = args.margin
    max_grad_norm = args.max_grad_norm

#     sampled_examples = random.sample(dev_examples, 30)

#     train_trip_examples = triplets_from_labeled_dataset(train_examples)
    # sampler = MyBatchSampler(labels)
    # train_dataloader = DataLoader(train_examples, sampler=sampler, batch_size=train_batch_size)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=train_batch_size)
    loss_model = losses.BatchHardTripletLoss(model=esbert, 
                                            distance_metric=losses.BatchHardTripletLossDistanceFunction.cosine_distance,
                                            margin=margin)
    warmup_steps = math.ceil(len(train_examples)*num_epochs/train_batch_size*0.1) #10% of train data for warm-up
    folder_name = "output/{}_ep{}_mgn{}_btch{}_norm{}_max_seq_{}".format("exp_esbert", num_epochs, margin, train_batch_size, max_grad_norm, args.max_seq_length)
    os.makedirs(folder_name, exist_ok=True)
    train(loss_model, 
        train_dataloader, 
        epochs=num_epochs, 
        train_batch_size=train_batch_size, 
        warmup_steps=warmup_steps, 
        max_grad_norm=max_grad_norm,
        device=device,
        folder_name=folder_name, esbert_model=esbert)

#     folder_name = "output/{}_ep{}_mgn{}_btch{}_norm{}".format("exp_esbert", num_epochs, margin, train_batch_size, max_grad_norm)
#     os.makedirs(folder_name, exist_ok=True)
#     torch.save(esbert, "{}/esbert_model.pt".format(folder_name))


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