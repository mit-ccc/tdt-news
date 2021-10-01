"""
Generate the data used for SVM linear with triplet loss
"""
import pickle
import sys
sys.path.insert(0,"../")
import json, load_corpora, clustering, os
from clustering import *
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.nn import functional as F
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--input_folder", type=str, default="./output/exp_time_esbert_ep2_mgn2.0_btch8_norm1.0_max_seq_128/time_esbert_model_ep1.pt", help="input_folder")
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--batchsize", type=int, default=32)
parser.add_argument("--epoch", type=int, default=30)
parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
parser.add_argument("-f")
args = parser.parse_args()

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
    X = torch.FloatTensor(X).to(args.device)
    Y = torch.FloatTensor(Y).to(args.device)
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


def prepare_data(X_, Y_, guid_list):

    guid2examples = {}
    for idx, (x_, y_) in enumerate(zip(X_, Y_)):
        guid = guid_list[idx]
        if guid in guid2examples:
            guid2examples[guid].append(x_)
        else:
            guid2examples[guid] = [x_]
    svm_triplet_train_data, m_labels = get_triplet_pairs(guid2examples)
    return svm_triplet_train_data, m_labels


def save_pytorch_model(model, output_file_path):
    with open(output_file_path, 'w') as out: 
        for i in range(10):
            out.write(' '.join([str(i), "PLACEHOLDER"]))
            out.write("\n")
        # line 10 -> bias
        if model.bias:
            out.write(' '.join([str(model.bias[0].tolist()), "#", "PLACEHOLDER"]))
        else:
            out.write(' '.join(['0', "#", "PLACEHOLDER"]))            
        out.write("\n")
        # line 11 -> weights
        line11 = ['1'] + [':'.join([str(a), str(b)]) for a, b in zip(range(1, model.weight.shape[1]+1), model.weight[0].tolist())] + ['#']
        out.write(' '.join(line11))


def main():

    # extract X an Y
    guid_list = []
    X = []
    y = []
    with open(os.path.join(args.input_folder, "train_svm_rank.dat")) as f:
        for line in f:
    #         print(line.split()[1])
            guid = line.split()[1]
            guid_list.append(guid)
            if line.split()[0] == "1":
                y.append(1)
            else:
                y.append(0)
            X.append([float(a.split(":")[-1]) for a in line.split()[2:]])
    X = np.array(X)
    Y = np.array(y)

    
    # train and save
    Path(os.path.join(args.input_folder, "margin_ranking_weight_models")).mkdir(parents=True, exist_ok=True)
    svm_triplet_train_data, m_labels = prepare_data(X, Y, guid_list)
    # for lr in [0.5, 0.01]:
    for lr in [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 10, 100, 1000, 10000]:
    # for lr in [0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0]:
    # for lr in [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]:
        model = nn.Linear(X.shape[1], 1)
        model.to(args.device)
        args.lr = lr
        train(svm_triplet_train_data, m_labels, model, args)
        save_pytorch_model(model, os.path.join(args.input_folder, "margin_ranking_weight_models", "margin_ranking_lr{}_ep{}.dat".format(lr, args.epoch)))
        print("Fisnihed saving model with lr {}".format(lr))

if __name__ == "__main__":
    main()