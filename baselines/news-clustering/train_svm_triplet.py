"""
Generate the data used for SVM linear with triplet loss
"""
import pickle
import json, load_corpora, clustering, os
from clustering import *
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.nn import functional as F


def get_triplet_pairs(guid2examples):
    """
    returns
        svm_triplet_train_data: (pos_sims, neg_sims)
        m_labels: 1s and 0s (half of them are randomly chosen to be 0)
    """
    svm_triplet_train_data = [] # (pos_sims, neg_sims)
    for guid, batch in guid2examples.items():
        if len(batch) > 1:
            pos_example = batch[0]
            neg_examples = batch[1:]
            for neg_example in neg_examples:
                svm_triplet_train_data.append((pos_example, neg_example))
    m_labels = np.array([1]*len(svm_triplet_train_data))
    indices = np.random.choice(np.arange(m_labels.size), replace=False, size=int(m_labels.size * 0.5))
    m_labels[indices] = 0
    return svm_triplet_train_data, m_labels


def train(X, Y, model, args):
    """
    modify triplet loss to be a classification task
    TODO: try different strategies
        - MSELoss / MarginRankingLoss
        - SGD / Adam
        - use only 1s as labels or randomly zero half of the labels
    """
    X = torch.FloatTensor(X)
    Y = torch.FloatTensor(Y)
    N = len(X)

    optimizer = optim.SGD(model.parameters(), lr=args.lr)
#     optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MarginRankingLoss() 
#     loss_fn = torch.nn.MSELoss() 

    model.train()
    for epoch in range(args.epoch):
        perm = torch.randperm(N)
        sum_loss = 0

        for i in range(0, N, args.batchsize):
            x = X[perm[i : i + args.batchsize]].to(args.device)
            y = Y[perm[i : i + args.batchsize]].to(args.device)

            optimizer.zero_grad()
            
            batch_pos = x[:, 0, :]
            batch_neg = x[:, 1, :]
            weight = model.weight.squeeze()

            output_pos = model(batch_pos).squeeze()
            output_neg = model(batch_neg).squeeze()
            loss = loss_fn(output_pos, output_neg, y)

#             output = (model(batch_pos - batch_neg)).squeeze()
#             loss = loss_fn(output, y)
#             loss = torch.mean(torch.clamp(y-output, min=0))
#             loss = torch.mean(torch.clamp(1 - y * output, min=0))
#             loss += args.c * (weight.t() @ weight) / 2.0

            loss.backward()
            optimizer.step()

            sum_loss += float(loss)

        print("Epoch: {:4d}\tloss: {}".format(epoch, sum_loss / N))


def main():
    # TODO: change path for different training data
    guid_list = []
    with open("./svm_en_data/train_bert_rank_without_bert.dat") as f:
        for line in f:
            guid_list.append(line.split()[1])

    # TODO: fix the svm reading issue (wrong format)
    y, X_without_bert = svm_read_problem('./svm_en_data/train_bert_rank_without_bert.dat')
    X_without_bert = pd.DataFrame(X_without_bert).values

    guid2examples = {}

    for idx, (x_, y_) in enumerate(zip(X_without_bert, y)):
        guid = guid_list[idx]
        if guid in guid2examples:
            guid2examples[guid].append(x_)
        else:
            guid2examples[guid] = [x_]

    svm_triplet_train_data, m_labels = get_triplet_pairs(guid2examples)

    parser = argparse.ArgumentParser()
    parser.add_argument("--c", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--batchsize", type=int, default=32)
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("-f")
    args = parser.parse_args()
    model = nn.Linear(14, 1)
    model.to(args.device)
    # TODO: save the best model 
    train(svm_triplet_train_data, m_labels, model, args)
    # TODO: format the model to .model format

if __name__ == "__main__":
    main()