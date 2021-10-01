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
sys.path.insert(0,"/mas/u/hjian42/tdt-twitter/baselines/T-ESBERT/")
import load_corpora
import clustering
from Model import Date2VecConvert
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from sentence_transformers.evaluation import TripletEvaluator
from sklearn import preprocessing
from torch.utils.data.sampler import Sampler
from utils import CorpusClass


def custom_collate_fn(batch):
    """collate for List of InputExamples, not triplet examples"""
    texts = []
    entities = []

    for example in batch:
        texts.append(example.texts)

        entity_list = np.array(example.entities)
        entity_list = entity_list[entity_list < 512]
        new_entity_list = np.zeros(512, dtype=int)
        if len(entity_list) > 0: # sometimes there is no entity in the document
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
                if len(entity_list) > 0: # sometimes there is no entity in the document
                    new_entity_list[entity_list] = 1
                entities.append(new_entity_list)

                labels.append(example.label)

            tokenized = entity_transformer.tokenize(texts) # HACK: use the model's internal tokenize() function
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
        
        anchor_pos_distance = self.distance_metric(anchor_embed, pos_embed)
        anchor_neg_distance = self.distance_metric(anchor_embed, neg_embed)

        tl = anchor_pos_distance - anchor_neg_distance + self.triplet_margin
        tl[tl < 0] = 0
        triplet_loss = tl.mean()
        return triplet_loss


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


# class EntitySBert_old(nn.Module):
#     """entity-aware BERT"""
    
#     def __init__(self, esbert_model, device="cuda"):
#         super(EntitySBert_old, self).__init__()
#         self.esbert_model = esbert_model.to(device)
#         self.pooler = BertPooler(768).to(device)
            
#     def forward(self, features):
                
#         batch_to_device(features, device)
#         bert_features = self.esbert_model(features)
#         cls_embeddings = bert_features['cls_token_embeddings']
#         token_embeddings = bert_features['token_embeddings']
                
#         pooled_features = self.pooler(token_embeddings)
        
#         features.update({"sentence_embedding": pooled_features})
#         return features


class EntitySBert(nn.Module):
    """entity-aware BERT"""
    
    def __init__(self, esbert_model, device="cuda"):
        super(EntitySBert, self).__init__()
        self.esbert_model = esbert_model.to(device)
            
    def forward(self, features):
        
        batch_to_device(features, device)
        bert_features = self.esbert_model(features)
        cls_embeddings = bert_features['cls_token_embeddings']
        token_embeddings = bert_features['token_embeddings']
        attention_mask = bert_features['attention_mask']
        
        output_vectors = []
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1) # tokens not weighted
        output_vectors.append(sum_embeddings / sum_mask)
        output_vector = torch.cat(output_vectors, 1)
        # print(output_vector.shape)

        features.update({"sentence_embedding": output_vector})
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
            if len(entity_list) > 0: # sometimes there is no entity in the document
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
entity_transformer = EntityTransformer("/mas/u/hjian42/tdt-twitter/baselines/T-ESBERT/pretrained_bert/0_Transformer/")

def main():
    random.seed(123)

    parser = argparse.ArgumentParser(description="main training script for word2vec dynamic word embeddings...")
    parser.add_argument("--dataset_name", type=str, default="vaccine", help="dest dir")
    parser.add_argument("--num_epochs", type=int, default=2, help="num_epochs")
    parser.add_argument("--train_batch_size", type=int, default=64, help="train_batch_size")
    parser.add_argument("--margin", type=float, default=2.0, help="margin")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="max_grad_norm")
    parser.add_argument("--max_seq_length", type=int, default=512, help="max_seq_length")
    parser.add_argument("--sample_method", type=str, default="random", help="dest dir")
    parser.add_argument("--loss_function", type=str, default="BatchHardTripletLoss", help="dest dir")
    parser.add_argument("--continue_model_path", type=str, default=None, help="dest dir")
    parser.add_argument("--offline_triplet_data_path", type=str, default="/mas/u/hjian42/tdt-twitter/baselines/T-ESBERT/output/exp_esbert_ep7_mgn2.0_btch64_norm1.0_max_seq_128/train_dev_offline_triplets.pickle", help="dest dir")
#     parser.add_argument("--dest_dir", type=str, default="./output/exp_time_esbert_ep2_m2/", help="dest dir")
#     parser.add_argument('--dim', type=int, default=100, help='Number of dimensions. Default is 100.')
    args = parser.parse_args()
    
    if args.dataset_name == "vaccine":
        with open('/mas/u/hjian42/tdt-twitter/baselines/T-ESBERT/news_data/train_dev_entity.pickle', 'rb') as handle:
            train_corpus = pickle.load(handle)
    else: # default News dataset
        with open('/mas/u/hjian42/tdt-twitter/baselines/T-ESBERT/dataset/train_dev.pickle', 'rb') as handle:
            train_corpus = pickle.load(handle)

#     with open('/mas/u/hjian42/tdt-twitter/baselines/news-clustering/entity-bert/test.pickle', 'rb') as handle:
#         test_corpus = pickle.load(handle)
    print("finished loading pickle files")

    # initialize a model
    # entity_transformer = EntityTransformer("/mas/u/hjian42/tdt-twitter/baselines/news-clustering/entity-bert/pretrained/0_Transformer/")
    entity_transformer.max_seq_length = args.max_seq_length

    if args.continue_model_path:
        print("loading from a previously trained model")
        esbert = torch.load(args.continue_model_path)
    else:
        esbert = EntitySBert(entity_transformer)
    
    # testing on a sample sample whether we can overfit
#     train_corpus.documents = train_corpus.documents[:100]

    labels = [d['cluster'] for d in train_corpus.documents]
    le = preprocessing.LabelEncoder()
    targets = le.fit_transform(labels)
    for d, target in zip(train_corpus.documents, targets):
        d['cluster_label'] = target
    train_examples = [InputExample(texts=d['full_text'], 
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
    # train_dataloader = DataLoader(train_examples, sampler=sampler, batch_size=train_batch_size)

    # choose sampling method
    if args.sample_method == "random":
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=train_batch_size)
        train_dataloader.collate_fn = smart_batching_collate
    elif args.sample_method == "offline":
        args.loss_function = "offline_sampling"
        with open(args.offline_triplet_data_path, 'rb') as handle:
            train_offline_triplets_idxes = pickle.load(handle)
        train_dev_triplets = triplets_from_offline_sampling(train_examples, train_offline_triplets_idxes, mode="train_triplet") # used to be EPHN_triplets
        train_dataloader = DataLoader(train_dev_triplets, shuffle=True, batch_size=train_batch_size)
        train_dataloader.collate_fn = triplet_batching_collate

    # choose loss model
    if args.loss_function == "BatchHardTripletLoss":
        loss_model = losses.BatchHardTripletLoss(model=esbert, 
                                                distance_metric=losses.BatchHardTripletLossDistanceFunction.cosine_distance,
                                                margin=margin)
    elif args.loss_function == "offline_sampling":
        loss_model = BatchOfflineTripletLoss(esbert, distance_metric=cosine_distance, margin=margin)

    warmup_steps = math.ceil(len(train_examples)*num_epochs/train_batch_size*0.1) #10% of train data for warm-up
    if args.continue_model_path:
        folder_name = "output/{}_{}_ep{}_mgn{}_btch{}_norm{}_max_seq_{}_sample_{}_continued_training".format("exp_esbert", args.dataset_name, num_epochs, margin, train_batch_size, max_grad_norm, args.max_seq_length, args.sample_method)
    else:
        folder_name = "output/{}_{}_ep{}_mgn{}_btch{}_norm{}_max_seq_{}_sample_{}".format("exp_esbert", args.dataset_name, num_epochs, margin, train_batch_size, max_grad_norm, args.max_seq_length, args.sample_method)
    os.makedirs(folder_name, exist_ok=True)
    train(loss_model, 
        train_dataloader, 
        epochs=num_epochs, 
        train_batch_size=train_batch_size, 
        warmup_steps=warmup_steps, 
        max_grad_norm=max_grad_norm,
        device=device,
        folder_name=folder_name, esbert_model=esbert)


if __name__ == "__main__":
    main()

"""
#######################
### TRAINING on train triplets
#######################
export CUDA_VISIBLE_DEVICES=1
for epochnum in 1 2 3 4 5 6 7 8 9 10
do
    python train_esbert.py --num_epochs ${epochnum} --max_seq_length 128 \
        --train_batch_size 16 \
        --sample_method offline --offline_triplet_data_path ./news_data/amt_triplets.pickle
done
for epochnum in 1 2 
do
    echo "epochnum", ${epochnum}
    python evaluate_entity_models.py \
        --use_vaccine_saved_triplets \
        --model_path ./output/exp_esbert_vaccine_ep${epochnum}_mgn2.0_btch16_norm1.0_max_seq_128_sample_offline/esbert_model_ep${epochnum}.pt
done

#######################
### Pre-trained News model + TRAINING on train triplets
#######################
export CUDA_VISIBLE_DEVICES=2
for epochnum in 1 2 3 4 5 6 7 8 9 10
do
    python train_esbert.py --num_epochs ${epochnum} --max_seq_length 128 \
        --train_batch_size 16 \
        --sample_method offline --offline_triplet_data_path ./news_data/amt_triplets.pickle \
        --continue_model_path ./output/exp_esbert_ep2_mgn2.0_btch32_norm1.0_max_seq_256_sample_random/esbert_model_ep2.pt
done

for epochnum in 1 2 3
do
    echo "epochnum", ${epochnum}
    python evaluate_entity_models.py \
        --use_vaccine_saved_triplets \
        --model_path ./output/exp_esbert_vaccine_ep${epochnum}_mgn2.0_btch16_norm1.0_max_seq_128_sample_offline_continued_training/esbert_model_ep${epochnum}.pt
done

"""