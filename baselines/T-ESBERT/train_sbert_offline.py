"""
train sentenceBERT (our model)
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
# from esbert.transformer_entity import BertEntityEmbeddings, EntityBertModel
# from esbert.transformer_entity import EntityTransformer
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
import sys
from datetime import datetime
sys.path.insert(0,"/mas/u/hjian42/tdt-twitter/baselines/T-ESBERT/")
import load_corpora
import clustering
from Model import Date2VecConvert
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
from sentence_transformers.evaluation import TripletEvaluator
from sklearn import preprocessing
from torch.utils.data.sampler import Sampler


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
            # dates = []
            labels = []

            for triplet in batch:
                example = triplet[idx]
                texts.append(example.texts)
                # dates.append(convert_str_to_date_tensor(example.times))

                entity_list = np.array(example.entities)
                entity_list = entity_list[entity_list < 512]
                new_entity_list = np.zeros(512, dtype=int)
                new_entity_list[entity_list] = 1
                entities.append(new_entity_list)

                labels.append(example.label)

            tokenized = model[0].tokenize(texts) # HACK: use the model's internal tokenize() function
#             print(tokenized)
            tokenized['entity_type_ids'] = torch.tensor(entities)
            # tokenized['dates'] = torch.tensor(dates).float()
        
            tokenized_dict[idx] = tokenized
            labels_dict[idx] = labels
        batch_to_device(tokenized_dict, device)
        batch_to_device(labels_dict, device)

        return tokenized_dict, labels_dict


def cosine_distance(embeddings1, embeddings2):
    """
    Compute the 2D matrix of cosine distances (1-cosine_similarity) between all embeddings.
    """
    return 1 - nn.CosineSimilarity(dim=1, eps=1e-6)(embeddings1, embeddings2)


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

        # print(anchor_embed)
        
        anchor_pos_distance = self.distance_metric(anchor_embed, pos_embed)
        anchor_neg_distance = self.distance_metric(anchor_embed, neg_embed)

        tl = anchor_pos_distance - anchor_neg_distance + self.triplet_margin
        tl[tl < 0] = 0
        triplet_loss = tl.mean()
        return triplet_loss


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
            dates.append(convert_str_to_date_tensor(example.times))

            entity_list = np.array(example.entities)
            entity_list = entity_list[entity_list < 512]
            new_entity_list = np.zeros(512, dtype=int)
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

        # save models at certain checkpoints
        # if epoch+1 in set([2, 5, 10, 30]):
        torch.save(esbert_model, "{}/sbert_model_ep{}.pt".format(folder_name, epochs))
        print("saving checkpoint: epoch {}".format(epochs))


# global variable
# entity_transformer = EntityTransformer("/mas/u/hjian42/tdt-twitter/baselines/T-ESBERT/pretrained_bert/0_Transformer/")
model = SentenceTransformer("bert-base-nli-stsb-mean-tokens")

def main():
    random.seed(123)

    parser = argparse.ArgumentParser(description="main training script for word2vec dynamic word embeddings...")
    parser.add_argument("--num_epochs", type=int, default=2, help="num_epochs")
    parser.add_argument("--train_batch_size", type=int, default=64, help="train_batch_size")
    parser.add_argument("--margin", type=float, default=2.0, help="margin")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="max_grad_norm")
    parser.add_argument("--max_seq_length", type=int, default=512, help="max_seq_length")
    parser.add_argument("--fuse_method", type=str, default="selfatt_pool", help="dest dir")
    parser.add_argument("--sample_method", type=str, default="random", help="dest dir")
    parser.add_argument("--loss_function", type=str, default="BatchHardTripletLoss", help="dest dir")
    parser.add_argument("--freeze_time_module", type=int, default=0, help="max_seq_length")
    parser.add_argument("--offline_triplet_data_path", type=str, default="/mas/u/hjian42/tdt-twitter/baselines/T-ESBERT/output/exp_sbert_ep1_mgn2.0_btch64_norm1.0_max_seq_128/train_dev_offline_triplets.pickle", help="dest dir")
    args = parser.parse_args()
    
    with open('/mas/u/hjian42/tdt-twitter/baselines/T-ESBERT/dataset/train_dev.pickle', 'rb') as handle:
        train_corpus = pickle.load(handle)

#     with open('/mas/u/hjian42/tdt-twitter/baselines/news-clustering/entity-bert/test.pickle', 'rb') as handle:
#         test_corpus = pickle.load(handle)
    print("finished loading pickle files")

    # initialize a model
    # entity_transformer = EntityTransformer("/mas/u/hjian42/tdt-twitter/baselines/news-clustering/entity-bert/pretrained/0_Transformer/")
    # entity_transformer.max_seq_length = args.max_seq_length
    
    if args.max_seq_length < 512:
        model.max_seq_length = args.max_seq_length
    # date2vec_model = Date2VecConvert(model_path="/mas/u/hjian42/tdt-twitter/baselines/T-ESBERT/Date2Vec/d2v_model/d2v_98291_17.169918439404636.pth")
    # print("finished loading date2vec")
    # time_esbert = TimeESBert(entity_transformer, date2vec_model, fuse_method=args.fuse_method, freeze_time_module=args.freeze_time_module)
    # train_corpus.documents = train_corpus.documents[:100]

    labels = [d['cluster'] for d in train_corpus.documents]
    le = preprocessing.LabelEncoder()
    targets = le.fit_transform(labels)
    for d, target in zip(train_corpus.documents, targets):
        d['cluster_label'] = target
    train_examples = [InputExample(texts=d['text'], 
                                label=d['cluster_label'],
                                guid=d['id'], 
                                entities=d['bert_entities'], 
                                times=d['date']
                                ) for d in train_corpus.documents]

    num_epochs = args.num_epochs
    train_batch_size = args.train_batch_size
    margin = args.margin
    max_grad_norm = args.max_grad_norm

#     sampled_examples = random.sample(dev_examples, 30)
#     train_trip_examples = triplets_from_labeled_dataset(train_examples)
    
    # sampling
    if args.sample_method == "random":
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=train_batch_size)
        train_dataloader.collate_fn = smart_batching_collate
    elif args.sample_method == "regular": # sample 8 instances of the same class each time
        sampler = MyBatchSampler(labels)
        train_dataloader = DataLoader(train_examples, sampler=sampler, batch_size=train_batch_size)
        train_dataloader.collate_fn = smart_batching_collate
    elif args.sample_method in set(['EPHN_triplets', 'EPEN_triplets', 'HPHN_triplets', 'HPEN_triplets']):
        args.loss_function = "offline_sampling"
        with open(args.offline_triplet_data_path, 'rb') as handle:
            train_offline_triplets_idxes = pickle.load(handle)
        train_dev_triplets = triplets_from_offline_sampling(train_examples, train_offline_triplets_idxes, mode=args.sample_method)
        train_dataloader = DataLoader(train_dev_triplets, shuffle=True, batch_size=train_batch_size)
        train_dataloader.collate_fn = triplet_batching_collate
    
    # loss function
    if args.loss_function == "BatchHardTripletLoss":
        loss_model = losses.BatchHardTripletLoss(model=model, 
                                                distance_metric=losses.BatchHardTripletLossDistanceFunction.cosine_distance,
                                                margin=margin)
    elif args.loss_function == "BatchHardSoftMarginTripletLoss":
        loss_model = losses.BatchHardSoftMarginTripletLoss(model=model, 
                                                distance_metric=losses.BatchHardTripletLossDistanceFunction.cosine_distance)
    elif args.loss_function == "BatchSemiHardTripletLoss":
        loss_model = losses.BatchSemiHardTripletLoss(model=model, 
                                                distance_metric=losses.BatchHardTripletLossDistanceFunction.cosine_distance,
                                                margin=margin)
    elif args.loss_function == "BatchAllTripletLoss":
        loss_model = losses.BatchAllTripletLoss(model=model, 
                                                distance_metric=losses.BatchHardTripletLossDistanceFunction.cosine_distance,
                                                margin=margin)
    elif args.loss_function == "offline_sampling":
        loss_model = BatchOfflineTripletLoss(model, distance_metric=cosine_distance, margin=margin)

    warmup_steps = math.ceil(len(train_examples)*num_epochs/train_batch_size*0.1) #10% of train data for warm-up
    if args.freeze_time_module:
        folder_name = "output/{}_ep{}_mgn{}_btch{}_norm{}_max_seq_{}_fuse_{}_{}_sample_{}_time_frozen".format("exp_time_sentenceBERT", num_epochs, margin, train_batch_size, max_grad_norm, args.max_seq_length, args.fuse_method, args.sample_method, args.loss_function)
    else:
        folder_name = "output/{}_ep{}_mgn{}_btch{}_norm{}_max_seq_{}_fuse_{}_{}_sample_{}".format("exp_time_sentenceBERT", num_epochs, margin, train_batch_size, max_grad_norm, args.max_seq_length, args.fuse_method, args.sample_method, args.loss_function)
    os.makedirs(folder_name, exist_ok=True)
    train(loss_model, 
        train_dataloader,
        epochs=num_epochs, 
        train_batch_size=train_batch_size, 
        warmup_steps=warmup_steps, 
        max_grad_norm=max_grad_norm,
        device=device,
        folder_name=folder_name, esbert_model=model)
    
#     folder_name = "output/{}_ep{}_mgn{}_btch{}_norm{}".format("exp_time_esbert", num_epochs, margin, train_batch_size, max_grad_norm)
#     os.makedirs(folder_name, exist_ok=True)
#     torch.save(esbert, "{}/time_esbert_model.pt".format(folder_name))
    
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