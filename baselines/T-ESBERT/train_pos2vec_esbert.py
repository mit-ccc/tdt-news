"""
train time-entity sentenceBERT (our model)
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
sys.path.insert(0,"./")
import load_corpora
# import clustering
# from Model import Date2VecConvert
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from sentence_transformers.evaluation import TripletEvaluator
from sklearn import preprocessing
from torch.utils.data.sampler import Sampler

from utils import CorpusClass, InputExample, cosine_distance


##############################################
### For inference use: Feature Extraction ####
##############################################
def custom_collate_fn(batch):
    """collate for List of InputExamples, not triplet examples"""
    texts = []
    entities = []
    dates = []

    for example in batch:
        texts.append(example.texts)
        dates.append(compute_time_stamp(example.times))

        entity_list = np.array(example.entities)
        entity_list = entity_list[entity_list < 512]
        new_entity_list = np.zeros(512, dtype=int)
        if len(entity_list) > 0: # sometimes there is no entity in the document
            new_entity_list[entity_list] = 1
        entities.append(new_entity_list)

    tokenized = entity_transformer.tokenize(texts) # HACK: use the model's internal tokenize() function
    tokenized['entity_type_ids'] = torch.tensor(entities)
    tokenized['dates'] = torch.tensor(dates).float()
    
    return tokenized


#######################
### offline training ##
#######################
def triplets_from_offline_sampling(input_examples, offline_triplet_idxes, mode="EPHN_triplets"):
    """
    use pre-determined triplets from offline sampling algorithms including EPHN, EPEN, HPHN, HPEN
    """
    triplets = []

    for (anchor_idx, pos_idx, neg_idx) in offline_triplet_idxes[mode]:
        anchor = input_examples[anchor_idx]

        positive = input_examples[pos_idx]

        negative = input_examples[neg_idx]
        
        triplets.append([anchor, positive, negative])
    
    return triplets


def triplet_batching_collate(batch):
        """
        Transforms a batch from a SmartBatchingDataset to a batch of tensors for the model
        Here, batch is a list of tuples: [(tokens, label), ...]
        :param batch:
            a batch from a SmartBatchingDataset
        :return:
            a batch of tensors for the model
        """
        
        tokenized_dict = {}
        labels_dict = {}
        for idx in range(3):
            texts = []
            entities = []
            dates = []
            labels = []

            for triplet in batch:
                example = triplet[idx]
                texts.append(example.texts)
                dates.append(compute_time_stamp(example.times))

                entity_list = np.array(example.entities)
                entity_list = entity_list[entity_list < 512]
                new_entity_list = np.zeros(512, dtype=int)
                if len(entity_list) > 0: # sometimes there is no entity in the document
                    new_entity_list[entity_list] = 1
                entities.append(new_entity_list)

                labels.append(example.label)

            tokenized = entity_transformer.tokenize(texts) # HACK: use the model's internal tokenize() function
            tokenized['entity_type_ids'] = torch.tensor(entities)
            tokenized['dates'] = torch.tensor(dates).float()
        
            tokenized_dict[idx] = tokenized
            labels_dict[idx] = labels
        batch_to_device(tokenized_dict, device)
        batch_to_device(labels_dict, device)

        return tokenized_dict, labels_dict


class BatchOfflineTripletLoss(nn.Module):
    
    def __init__(self, model: SentenceTransformer, distance_metric = cosine_distance, margin: float = 5):
        super(BatchOfflineTripletLoss, self).__init__()
        self.sentence_embedder = model
        self.triplet_margin = margin
        self.distance_metric = distance_metric
    
    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        anchor_embed = self.sentence_embedder(sentence_features[0])['sentence_embedding']
        pos_embed = self.sentence_embedder(sentence_features[1])['sentence_embedding']
        neg_embed = self.sentence_embedder(sentence_features[2])['sentence_embedding']
        
        anchor_pos_distance = self.distance_metric(anchor_embed, pos_embed)
        anchor_neg_distance = self.distance_metric(anchor_embed, neg_embed)

        tl = anchor_pos_distance - anchor_neg_distance + self.triplet_margin
        tl[tl < 0] = 0
        triplet_loss = tl.mean()
        return triplet_loss


#######################
#### Time ESBERT ######
#######################
class BertMeanPooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        mean_token_tensor = torch.mean(hidden_states, 1)
        pooled_output = self.dense(mean_token_tensor)
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


class PositionTimeESBert(nn.Module):
    
    def __init__(self, esbert_model, time_model, fuse_method="selfatt_pool", device="cuda", freeze_time_module=False):
        super(PositionTimeESBert, self).__init__()
        self.esbert_model = esbert_model.to(device)
        self.time_model = time_model.to(device)
        self.fuse_method = fuse_method
        # self.pooler = BertPooler(768).to(device)
        # self.concat_linear = nn.Linear(832, 832).to(device)
        if "additive_selfatt_pool" == fuse_method:
            self.multi_att = nn.MultiheadAttention(768, 8, 0.1).to(device)
            self.norm_layer = LayerNorm(768).to(device)
            self.pooler = BertMeanPooler(768).to(device)
        elif "selfatt_pool" in fuse_method:
            self.multi_att = nn.MultiheadAttention(768*2, 8, 0.1).to(device)
            self.norm_layer = LayerNorm(768*2).to(device)
            self.pooler = BertMeanPooler(768*2).to(device)
        # if freeze_time_module:
        #     for param in time_model.parameters():
        #         param.requires_grad = False 
            
    def forward(self, features):
                
        batch_to_device(features, device)
        bert_features = self.esbert_model(features)
        cls_embeddings = bert_features['cls_token_embeddings']
        token_embeddings = bert_features['token_embeddings']
        attention_mask = bert_features['attention_mask']
        
        # fuse temporal and linguistic features
        time_features = self.time_model(features['dates'])
        # print("time_features", time_features.shape)
        
        # 1. additive
        if self.fuse_method == "additive":
            output_vectors = []
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1) # tokens not weighted
            output_vectors.append(sum_embeddings / sum_mask)
            pooled_features = torch.cat(output_vectors, 1)
            fused_features = pooled_features + time_features
        
        # additive + selfatt + normalization
        elif self.fuse_method == "additive_selfatt_pool":
            repeat_time_vec = time_features.unsqueeze(1).repeat(1, token_embeddings.shape[1], 1)
            concat_time_token_emb = token_embeddings + repeat_time_vec
            attn_output, attn_output_weights = self.multi_att(concat_time_token_emb, concat_time_token_emb, concat_time_token_emb)
            norm_attn_output = self.norm_layer(attn_output + concat_time_token_emb)
            fused_features = self.pooler(norm_attn_output)
        
        # concat + selfatt + normalization
        elif self.fuse_method == "selfatt_pool":
            repeat_time_vec = time_features.unsqueeze(1).repeat(1, token_embeddings.shape[1], 1)
            concat_time_token_emb = torch.cat([token_embeddings, repeat_time_vec], 2)
            attn_output, attn_output_weights = self.multi_att(concat_time_token_emb, concat_time_token_emb, concat_time_token_emb)
            norm_attn_output = self.norm_layer(attn_output + concat_time_token_emb)
            fused_features = self.pooler(norm_attn_output)

         # additive + concat + selfatt + normalization
        elif self.fuse_method == "additive_concat_selfatt_pool":
            repeat_time_vec = time_features.unsqueeze(1).repeat(1, token_embeddings.shape[1], 1)
            concat_time_token_emb = token_embeddings + repeat_time_vec
            concat_time_token_emb = torch.cat([concat_time_token_emb, repeat_time_vec], 2)
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
        texts = []
        entities = []
        dates = []
        labels = []

        for example in batch:
            texts.append(example.texts)
            dates.append(compute_time_stamp(example.times))

            entity_list = np.array(example.entities)
            entity_list = entity_list[entity_list < 512]
            new_entity_list = np.zeros(512, dtype=int)
            if len(entity_list) > 0: # sometimes there is no entity in the document
                new_entity_list[entity_list] = 1
            entities.append(new_entity_list)
            
            labels.append(example.label)
        labels = torch.tensor(labels).to(device)

        tokenized = entity_transformer.tokenize(texts) # HACK: use the model's internal tokenize() function
        tokenized['entity_type_ids'] = torch.tensor(entities)
        tokenized['dates'] = torch.tensor(dates).float()
        batch_to_device(tokenized, device)
        
        return [tokenized], labels

def train(loss_model, dataloader, epochs=2, train_batch_size=2, warmup_steps=1000, weight_decay=0.01,  max_grad_norm=1.0, device='cuda', folder_name=None, esbert_model=None):
    
    # initialization
    global_step = 0
    total_loss = 0
    
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

        # save models at the last checkpoint
        torch.save(esbert_model, "{}/model_ep{}.pt".format(folder_name, epochs))
        print("saving checkpoint: epoch {}".format(epochs))


class pos2vec_model(nn.Module):
    def __init__(self, model_path='./dataset/pos2vec_embed_size768_time_steps_1024.pt'):
        super(pos2vec_model, self).__init__()
        self.pos2vec_mat = torch.load(model_path).cuda()
    
    def forward(self, x):
        x = x.flatten().long()
        return self.pos2vec_mat[x]


def compute_time_stamp(string):
    """the args.time_encoding will have some potential bad behavior when we import this module to other modules
    REMEMBER to change args.time_encoding to the proper one

    HACK: use entity_transformer to pass parameters (split, time_encoding)
    """
    # the start date is fixed
    if entity_transformer.split == "train":
        anchor_date = datetime.strptime("2013-12-18 12:27:00", "%Y-%m-%d %H:%M:%S")
    elif entity_transformer.split == "test":
        anchor_date = datetime.strptime("2014-11-02 21:18:00", "%Y-%m-%d %H:%M:%S")
    else:
        print("entity_transformer.split IS WRONG")
    date_obj = datetime.strptime(string, "%Y-%m-%d %H:%M:%S")
    delta = date_obj - anchor_date

    if entity_transformer.time_encoding == "day":
        return [delta.days]
    else:
        print("entity_transformer.time_encoding IS WRONG")
    # else: # if nothing happens
    #     return [delta.days]


# global variable
entity_transformer = EntityTransformer("/mas/u/hjian42/tdt-twitter/baselines/T-ESBERT/pretrained_bert/0_Transformer/")
entity_transformer.split = "train"
entity_transformer.time_encoding = "day"

def main():

    random.seed(123)
    parser = argparse.ArgumentParser(description="main training script for word2vec dynamic word embeddings...")
    parser.add_argument("--dataset_name", type=str, default="news2013", help="dest dir")
    parser.add_argument("--num_epochs", type=int, default=2, help="num_epochs")
    parser.add_argument("--train_batch_size", type=int, default=64, help="train_batch_size")
    parser.add_argument("--margin", type=float, default=2.0, help="margin")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="max_grad_norm")
    parser.add_argument("--max_seq_length", type=int, default=512, help="max_seq_length")
    parser.add_argument("--time_encoding", type=str, default="day", help="dest dir")
    parser.add_argument("--fuse_method", type=str, default="selfatt_pool", help="dest dir")
    parser.add_argument("--sample_method", type=str, default="random", help="dest dir")
    parser.add_argument("--loss_function", type=str, default="BatchHardTripletLoss", help="dest dir")
    parser.add_argument("--freeze_time_module", type=int, default=0, help="max_seq_length")
    parser.add_argument("--offline_triplet_data_path", type=str, default="/mas/u/hjian42/tdt-twitter/baselines/T-ESBERT/output/exp_time_esbert_ep3_mgn2.0_btch64_norm1.0_max_seq_128_fuse_selfatt_pool_random_sample_BatchHardTripletLoss/train_dev_offline_triplets.pickle", help="dest dir")
    parser.add_argument("--continue_model_path", type=str, default=None, help="dest dir")
    args = parser.parse_args()

    if args.dataset_name == "vaccine":
        with open('./news_data/train_dev_entity.pickle', 'rb') as handle:
            train_corpus = pickle.load(handle)
    elif args.dataset_name == "news2013": # default News dataset
        with open('./dataset/train_dev.pickle', 'rb') as handle:
            train_corpus = pickle.load(handle)
    else:
        print("Please specify a dataset!")
#     with open('/mas/u/hjian42/tdt-twitter/baselines/news-clustering/entity-bert/test.pickle', 'rb') as handle:
#         test_corpus = pickle.load(handle)
    # print("finished loading pickle files")

    # initialize a model
    # entity_transformer = EntityTransformer("/mas/u/hjian42/tdt-twitter/baselines/news-clustering/entity-bert/pretrained/0_Transformer/")
    entity_transformer.max_seq_length = args.max_seq_length
    entity_transformer.time_encoding = args.time_encoding
    entity_transformer.split = "train"
    print("Running with split {};time_encoding {}".format(entity_transformer.split, entity_transformer.time_encoding))
    # date2vec_model = Date2VecConvert(model_path="/mas/u/hjian42/tdt-twitter/baselines/T-ESBERT/Date2Vec/d2v_model/d2v_98291_17.169918439404636.pth")
    
    time_position_model = pos2vec_model(model_path='./dataset/pos2vec_embed_size768_time_steps_1024.pt')
   
    print("finished loading time model based on time position embedding")

    if args.continue_model_path:
        print("loading from a previously trained model")
        time_esbert = torch.load(args.continue_model_path)
    else:
        print("loading a new model")
        time_esbert = PositionTimeESBert(entity_transformer, time_position_model, fuse_method=args.fuse_method, freeze_time_module=args.freeze_time_module)

    labels = [d['cluster'] for d in train_corpus.documents]
    le = preprocessing.LabelEncoder()
    targets = le.fit_transform(labels)
    for d, target in zip(train_corpus.documents, targets):
        d['cluster_label'] = target
    train_examples = [InputExample(texts=d['full_text'], 
                                label=d['cluster_label'],
                                guid=d['id'], 
                                entities=d['bert_entities'], 
                                times=d['date']
                                ) for d in train_corpus.documents]

    num_epochs = args.num_epochs
    train_batch_size = args.train_batch_size
    margin = args.margin
    max_grad_norm = args.max_grad_norm


    # sampling
    if args.sample_method == "random":
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=train_batch_size)
        train_dataloader.collate_fn = smart_batching_collate
    # elif args.sample_method == "regular": # sample 8 instances of the same class each time
    #     sampler = MyBatchSampler(labels)
    #     train_dataloader = DataLoader(train_examples, sampler=sampler, batch_size=train_batch_size)
    #     train_dataloader.collate_fn = smart_batching_collate
    elif args.sample_method == "offline":
        args.loss_function = "offline"
        with open(args.offline_triplet_data_path, 'rb') as handle:
            train_offline_triplets_idxes = pickle.load(handle)
        train_dev_triplets = triplets_from_offline_sampling(train_examples, train_offline_triplets_idxes, mode="train_triplet")
        train_dataloader = DataLoader(train_dev_triplets, shuffle=True, batch_size=train_batch_size)
        train_dataloader.collate_fn = triplet_batching_collate
    
    # loss function
    if args.loss_function == "BatchHardTripletLoss":
        loss_model = losses.BatchHardTripletLoss(model=time_esbert, 
                                                distance_metric=losses.BatchHardTripletLossDistanceFunction.cosine_distance,
                                                margin=margin)
    elif args.loss_function == "BatchHardSoftMarginTripletLoss":
        loss_model = losses.BatchHardSoftMarginTripletLoss(model=time_esbert, 
                                                distance_metric=losses.BatchHardTripletLossDistanceFunction.cosine_distance)
    elif args.loss_function == "BatchSemiHardTripletLoss":
        loss_model = losses.BatchSemiHardTripletLoss(model=time_esbert, 
                                                distance_metric=losses.BatchHardTripletLossDistanceFunction.cosine_distance,
                                                margin=margin)
    elif args.loss_function == "BatchAllTripletLoss":
        loss_model = losses.BatchAllTripletLoss(model=time_esbert, 
                                                distance_metric=losses.BatchHardTripletLossDistanceFunction.cosine_distance,
                                                margin=margin)
    elif args.loss_function == "offline":
        loss_model = BatchOfflineTripletLoss(time_esbert, distance_metric=cosine_distance, margin=margin)

    warmup_steps = math.ceil(len(train_examples)*num_epochs/train_batch_size*0.1) #10% of train data for warm-up
    # if args.freeze_time_module:
    #     folder_name = "output/{}_ep{}_mgn{}_btch{}_norm{}_max_seq_{}_fuse_{}_{}_sample_{}_time_frozen".format("exp_pos2vec_esbert", num_epochs, margin, train_batch_size, max_grad_norm, args.max_seq_length, args.fuse_method, args.sample_method, args.loss_function)
    # else:
    folder_name = "output/{}_{}_ep{}_mgn{}_btch{}_norm{}_max_seq_{}_fuse_{}_{}_sample_{}".format("exp_pos2vec_esbert", args.dataset_name, num_epochs, margin, train_batch_size, max_grad_norm, args.max_seq_length, args.fuse_method, args.sample_method, args.loss_function)
    if args.continue_model_path:
        folder_name = "output/{}_{}_ep{}_mgn{}_btch{}_norm{}_max_seq_{}_fuse_{}_{}_sample_{}_time_{}_continued_training".format("exp_pos2vec_esbert", args.dataset_name, num_epochs, margin, train_batch_size, max_grad_norm, args.max_seq_length, args.fuse_method, args.sample_method, args.loss_function, args.time_encoding)
    os.makedirs(folder_name, exist_ok=True)
    train(loss_model, 
        train_dataloader,
        epochs=num_epochs, 
        train_batch_size=train_batch_size, 
        warmup_steps=warmup_steps, 
        max_grad_norm=max_grad_norm,
        device=device,
        folder_name=folder_name, esbert_model=time_esbert)


if __name__ == "__main__":
    main()
