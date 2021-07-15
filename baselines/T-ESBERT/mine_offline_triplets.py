import re
import argparse
import torch
import os, pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import random

parser = argparse.ArgumentParser(description="main training script for word2vec dynamic word embeddings...")
parser.add_argument("--input_folder", type=str, default="./output/exp_time_esbert_ep2_mgn2.0_btch8_norm1.0_max_seq_128/", help="vector_path")
args = parser.parse_args()

def extract_triplets(train_dev_distance_mat, class2idx, idx2class):
    EPHN_triplets = []
    EPEN_triplets = []
    HPHN_triplets = []
    HPEN_triplets = []

    # threshold = np.percentile(train_dev_distance_mat, 99) # statistical test, this is not the right implementation
    threshold = np.percentile(train_dev_distance_mat, 99, axis=1).reshape(train_dev_distance_mat.shape[0], 1) # according to the paper, we will calculate outliers for each instance

    # cosine disance in the range: [0, 2]
    for idx, embed in enumerate(train_dev_distance_mat):

        if idx % 1000 == 0:
            print("Finsihed {} instances".format(idx))
        
        embed = np.array(embed) # make a copy
        cluster = idx2class[idx]
        
        positive_idx_list = list(class2idx[cluster])
        mask_pos = np.zeros(embed.shape,dtype=bool)
        mask_pos[positive_idx_list] = True
        mask_not_outlier = (embed < threshold[idx])
        negative_idx_list = list((mask_not_outlier * (~mask_pos)).nonzero())[0]
        # print("embed", embed)
        # print("threshold", threshold[idx])
        # print("mask_pos", mask_pos)
        # print("mask_not_outlier", mask_not_outlier)
        # print("negative_idx_list", negative_idx_list)
        
        sorted_pos_idx_list = np.argsort(embed[positive_idx_list], axis=0)
        sorted_neg_idx_list = np.argsort(embed[negative_idx_list], axis=0)
        
        easy_pos = positive_idx_list[sorted_pos_idx_list[1]] # skip the first, itself
        hard_pos = positive_idx_list[sorted_pos_idx_list[-1]]
        easy_neg = negative_idx_list[sorted_neg_idx_list[0]]
        hard_neg = negative_idx_list[sorted_neg_idx_list[-1]]
        
        EPHN_triplets.append((idx, easy_pos, hard_neg))
        EPEN_triplets.append((idx, easy_pos, easy_neg))
        HPHN_triplets.append((idx, hard_pos, hard_neg))
        HPEN_triplets.append((idx, hard_pos, easy_neg))

    return {
            "EPHN_triplets": EPHN_triplets,
            "EPEN_triplets": EPEN_triplets,
            "HPHN_triplets": HPHN_triplets,
            "HPEN_triplets": HPEN_triplets
        }


def extract_triplets_topk(train_dev_distance_mat, class2idx, idx2class):
    EPHN_triplets = []
    EPEN_triplets = []
    HPHN_triplets = []
    HPEN_triplets = []

    # threshold = np.percentile(train_dev_distance_mat, 99) # statistical test, this is not the right implementation
    threshold = np.percentile(train_dev_distance_mat, 99, axis=1).reshape(train_dev_distance_mat.shape[0], 1) # according to the paper, we will calculate outliers for each instance

    # cosine disance in the range: [0, 2]
    for idx, embed in enumerate(train_dev_distance_mat):

        if idx % 1000 == 0:
            print("Finsihed {} instances".format(idx))
        
        embed = np.array(embed) # make a copy
        cluster = idx2class[idx]
        
        positive_idx_list = list(class2idx[cluster])
        mask_pos = np.zeros(embed.shape,dtype=bool)
        mask_pos[positive_idx_list] = True
        mask_not_outlier = (embed < threshold[idx])
        negative_idx_list = list((mask_not_outlier * (~mask_pos)).nonzero())[0]
        # print("embed", embed)
        # print("threshold", threshold[idx])
        # print("mask_pos", mask_pos)
        # print("mask_not_outlier", mask_not_outlier)
        # print("negative_idx_list", negative_idx_list)
        
        sorted_pos_idx_list = np.argsort(embed[positive_idx_list], axis=0)
        sorted_neg_idx_list = np.argsort(embed[negative_idx_list], axis=0)
        
        topk = 5
        easy_pos_list = [positive_idx_list[j] for j in sorted_pos_idx_list[1:topk+1]] # skip the first, itself
        hard_pos_list = [positive_idx_list[j] for j in sorted_pos_idx_list[-topk:]]
        easy_neg_list = [negative_idx_list[j] for j in sorted_neg_idx_list[:topk-1]]
        hard_neg_list = [negative_idx_list[j] for j in sorted_neg_idx_list[-topk:]]
        
        for i in range(topk):
            easy_pos = random.sample(easy_pos_list, 1)[0]
            hard_pos = random.sample(hard_pos_list, 1)[0]
            easy_neg = random.sample(easy_neg_list, 1)[0]
            hard_neg = random.sample(hard_neg_list, 1)[0]

            EPHN_triplets.append((idx, easy_pos, hard_neg))
            EPEN_triplets.append((idx, easy_pos, easy_neg))
            HPHN_triplets.append((idx, hard_pos, hard_neg))
            HPEN_triplets.append((idx, hard_pos, easy_neg))

    return {
            "EPHN_triplets": EPHN_triplets,
            "EPEN_triplets": EPEN_triplets,
            "HPHN_triplets": HPHN_triplets,
            "HPEN_triplets": HPEN_triplets
        }


def main():

    # train
    with open('/mas/u/hjian42/tdt-twitter/baselines/T-ESBERT/dataset/train_dev.pickle', 'rb') as handle:
        train_dev_corpus = pickle.load(handle)
    print("finished loading train pickle files")

    train_dense_feats = torch.load(os.path.join(args.input_folder, "train_sent_embeds.pt"))
    train_dev_distance_mat = 1 - cosine_similarity(train_dense_feats, train_dense_feats)
    # train_dev_distance_mat = train_dev_distance_mat[:10]

    # class2idx and idx2class
    class2idx = {}
    idx2class = {}
    for idx, doc in enumerate(train_dev_corpus.documents):
    #     docid = doc['id']
        cluster = doc['cluster']
        
        idx2class[idx] = cluster
        if cluster in class2idx:
            class2idx[cluster].add(idx)
        else:
            class2idx[cluster] = set([idx])

    # extract triplets
    offline_triplets = extract_triplets_topk(train_dev_distance_mat, class2idx, idx2class)
    print("Finished extracting triplets...")
    
    with open(os.path.join(args.input_folder, "train_dev_offline_triplets_top5.pickle"), 'wb') as handle:
        pickle.dump(offline_triplets, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()