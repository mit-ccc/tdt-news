import json, os
import numpy as np
import pickle
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
import load_corpora
import clustering

from transformers import BertModel
from transformers import BertPreTrainedModel
from transformers.modeling_bert import *

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from sentence_transformers import SentenceTransformer, SentencesDataset, losses
from sentence_transformers.readers import InputExample
from torch.utils.data import DataLoader
from sentence_transformers.readers import TripletReader
from collections import defaultdict
import math, random

# from transformer_entity import *
# from sbert_entity import *
import argparse


parser = argparse.ArgumentParser(description="main training script for word2vec dynamic word embeddings...")
parser.add_argument("--source_dir", type=str, default="../eventsim_output/baseline-bert-base-nli-stsb-mean-tokens-ep2-b8-m2.0", help="dest dir")
# parser.add_argument("--output_dir", type=str, default="../eventsim_output/esbert-bert-base-nli-stsb-mean-tokens-ep2-b8-m2.0", help="dest dir")
# parser.add_argument('--batch_size', type=int, default=8, help='Batch Size')
# parser.add_argument('--n_epochs', type=int, default=2, help='Number of epochs. Default is 2.')
# parser.add_argument('--triplet_margin', type=float, default=2.0, help='The threshold for configuring which higher-frequency words are randomly downsampled.')

# args 
args = parser.parse_args()

model = SentenceTransformer(args.source_dir)


def custom_collate_fn(batch):
    """collate for List of InputExamples, not triplet examples"""
    texts = []
    # entities = []

    for example in batch:
        texts.append(example.texts)

        # entity_list = np.array(example.entities)
        # entity_list = entity_list[entity_list < 512]
        # new_entity_list = np.zeros(512, dtype=int)
        # new_entity_list[entity_list] = 1
        # entities.append(new_entity_list)

    tokenized = model[0].tokenize(texts) # HACK: use the model's internal tokenize() function
    # tokenized['entity_type_ids'] = torch.tensor(entities)

    return tokenized


def extract_features(dataloader, model):
    sentence_embeddings_list = []
    dataloader.collate_fn = custom_collate_fn
    for batch in iter(dataloader):
        output_features = model(batch)
        sentence_embeddings = output_features['sentence_embedding']
        sentence_embeddings_list.append(sentence_embeddings.cpu().detach().numpy())
    sents_embeds = np.concatenate(sentence_embeddings_list, 0)
    return sents_embeds

def main():
    model_name = args.source_dir.split('/')[-1]
    print(model_name)

    # read data in pickle format
    with open('./train_dev.pickle', 'rb') as handle:
        train_dev_corpus = pickle.load(handle)
    train_examples = [InputExample(texts=d['text'], label=d['cluster'], guid=d['id'], entities=d['bert_entities']) for d in train_dev_corpus.documents]
    train_dataloader = DataLoader(train_examples, shuffle=False, batch_size=8)
    train_features = extract_features(train_dataloader, model)
    torch.save(train_features, "bert_feats_output/{}_train_sent_embeds.pt".format(model_name))
    print("finished saving train features")

    with open('./test.pickle', 'rb') as handle:
        test_corpus = pickle.load(handle)
    print("finished loading pickle files")
    dev_examples = [InputExample(texts=d['text'], label=d['cluster'], guid=d['id'], entities=d['bert_entities']) for d in test_corpus.documents]
    dev_dataloader = DataLoader(dev_examples, shuffle=False, batch_size=8)
    dev_features = extract_features(dev_dataloader, model)
    torch.save(dev_features, "bert_feats_output/{}_dev_sent_embeds.pt".format(model_name))
    print("finished saving dev features")


if __name__ == "__main__":
    main()