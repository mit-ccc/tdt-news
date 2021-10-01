import json
import logging
import os
import shutil
import math, pickle, queue
from collections import OrderedDict
from typing import List, Dict, Tuple, Iterable, Type, Union, Callable
from zipfile import ZipFile
import requests
import numpy as np
from numpy import ndarray

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
from collections import defaultdict
import random
import argparse

logger = logging.getLogger(__name__)

# from transformer_entity import BertEntityEmbeddings, EntityBertModel
# from transformer_entity import EntityTransformer
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
import load_corpora
import clustering


class InputExample:
    """
    Structure for one input example with texts, the label and a unique id
    """
    def __init__(self, guid: str = '', texts: List[str] = None,  label: Union[int, float] = 0):
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
                                    ))

    return triplets

def main():

    parser = argparse.ArgumentParser(description="main training script for word2vec dynamic word embeddings...")
    # parser.add_argument("--dest_dir", type=str, default="./output/english-uk/", help="dest dir")
    parser.add_argument('--batch_size', type=int, default=8, help='Batch Size')
    parser.add_argument('--n_epochs', type=int, default=2, help='Number of epochs. Default is 2.')
    parser.add_argument('--triplet_margin', type=float, default=2.0, help='The threshold for configuring which higher-frequency words are randomly downsampled.')

    # args 
    args = parser.parse_args()


    model_name = 'bert-base-nli-stsb-mean-tokens'
    num_epochs = args.n_epochs
    train_batch_size = args.batch_size
    triplet_margin = args.triplet_margin # default value=2
    # model_save_path = "./eventsim_output/{}-ep{}-b{}-m{}.pt".format(model_name, num_epochs, train_batch_size, triplet_margin)
    model_save_path = "../eventsim_output/baseline-sbert-{}-ep{}-b{}-m{}-exp".format(model_name, num_epochs, train_batch_size, triplet_margin)

    # read data in pickle format
    with open('./train_dev.pickle', 'rb') as handle:
        train_dev_corpus = pickle.load(handle)

    with open('./test.pickle', 'rb') as handle:
        test_corpus = pickle.load(handle)

    # loading models
    model = SentenceTransformer(model_name)
    # model.save("./entity-bert/pretrained")
    # model = EntitySentenceTransformer("./entity-bert/pretrained/")
    # model[0] = EntityTransformer("./entity-bert/pretrained/0_Transformer/")
    # model[0].max_seq_length = 512
    print(model[0].auto_model.embeddings)
    print(model[0].max_seq_length)
    print("finished loading models")
    print("CUDA is available", torch.cuda.is_available())


    # data loader
    train_ones = train_dev_corpus.documents[:]
    train_examples = [InputExample(texts=d['text'], label=d['cluster'], guid=d['id']) for d in train_ones]
    train_trip_examples = triplets_from_labeled_dataset(train_examples)
    train_dataloader = DataLoader(train_trip_examples, shuffle=True, batch_size=train_batch_size)
    train_loss = losses.BatchHardTripletLoss(model=model, 
                                            distance_metric=losses.BatchHardTripletLossDistanceFunction.cosine_distance,
                                            margin=triplet_margin)


    dev_examples = [InputExample(texts=d['text'], label=d['cluster'], guid=d['id']) for d in test_corpus.documents]
    dev_trip_examples = triplets_from_labeled_dataset(dev_examples)
    # dev_dataset = SentencesDataset(dev_trip_examples, model)
    dev_dataloader = DataLoader(dev_trip_examples, shuffle=False, batch_size=train_batch_size)
    # evaluator = TripletEvaluator.from_input_examples(dev_trip_examples)

    warmup_steps = math.ceil(len(train_trip_examples)*num_epochs/train_batch_size*0.1) #10% of train data for warm-up

    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
            evaluator=None,
            epochs=num_epochs,
            optimizer_class=transformers.AdamW,
            optimizer_params={'lr': 2e-5, 'eps': 1e-06},
            evaluation_steps=math.ceil(len(train_trip_examples)/train_batch_size),
            warmup_steps=warmup_steps,
            output_path=model_save_path)

    # TODO: re-write the evaluator function for TripletEvaluator since it does not take entities into account
    # model.evaluate(evaluator)

if __name__ == "__main__":
    main()