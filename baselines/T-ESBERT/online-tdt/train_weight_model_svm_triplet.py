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
from sklearn import svm
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--input_folder", type=str, default="./output/exp_time_esbert_ep2_mgn2.0_btch8_norm1.0_max_seq_128/time_esbert_model_ep1.pt", help="input_folder")
# parser.add_argument("--c", type=float, default=0.01)
# parser.add_argument("--lr", type=float, default=0.1)
# parser.add_argument("--batchsize", type=int, default=32)
# parser.add_argument("--epoch", type=int, default=30)
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
                svm_triplet_train_data.append(np.array(pos_example)-np.array(neg_example)) # # (sim(ra, rp) − sim(ra, rn))
    m_labels = np.array([1]*len(svm_triplet_train_data))
    indices = np.random.choice(np.arange(m_labels.size), replace=False, size=int(m_labels.size * 0.5))
    m_labels[indices] = 0
    return svm_triplet_train_data, m_labels


def prepare_data(X_, Y_, guid_list):

    guid2examples = {}
    for idx, (x_, y_) in enumerate(zip(X_, Y_)):
        guid = guid_list[idx]
        if guid in guid2examples:
            guid2examples[guid].append(x_)
        else:
            guid2examples[guid] = [x_]
    svm_triplet_train_data, m_labels = get_triplet_pairs(guid2examples) # (sim(ra, rp) − sim(ra, rn))
    return svm_triplet_train_data, m_labels


def save_scikit_model(model, output_file_path):
    with open(output_file_path, 'w') as out: 
        for i in range(10):
            out.write(' '.join([str(i), "PLACEHOLDER"]))
            out.write("\n")
        # line 10 -> bias
        if model.intercept_:
            out.write(' '.join([str(model.intercept_[0]), "#", "PLACEHOLDER"]))
        else:
            out.write(' '.join(['0', "#", "PLACEHOLDER"]))            
        out.write("\n")
        # line 11 -> weights
        line11 = ['1'] + [':'.join([str(a), str(b)]) for a, b in zip(range(1, model.coef_.shape[1]+1), model.coef_[0].tolist())] + ['#']
        out.write(' '.join(line11))


def main():

    # extract X and Y from svm_rank data
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
                y.append(-1)
            X.append([float(a.split(":")[-1]) for a in line.split()[2:]])
    X = np.array(X)
    Y = np.array(y)

    # prepare data
    svm_triplet_train_data, m_labels = prepare_data(X, Y, guid_list)
    svm_triplet_train_data, m_labels = svm_triplet_train_data[7:], m_labels[7:]
    
    Path(os.path.join(args.input_folder, "svm_triplet_weight_models")).mkdir(parents=True, exist_ok=True)
    X_train, Y_train = svm_triplet_train_data, m_labels
    for c in [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 10, 100, 1000]:
        clf = svm.SVC(kernel='linear', C=c)
        clf.fit(X_train, Y_train)
        save_scikit_model(clf, os.path.join(args.input_folder, "svm_triplet_weight_models", "SVM_triplet_c{}.dat".format(c)))
        print("Fisnihed saving model with c {}".format(c))


if __name__ == "__main__":
    main()