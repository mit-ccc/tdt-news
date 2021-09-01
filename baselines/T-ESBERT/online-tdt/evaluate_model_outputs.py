import bcubed
import pickle
import sys
sys.path.insert(0,"../")
import json, load_corpora, clustering, os
from clustering import *
import numpy as np
from glob import glob
import argparse
from pathlib import Path
from utils import CorpusClass


def evaluate_clusters(test_corpus, outputname="./clustering.eng.out.ranksvm"):
    ldict = {} #docID 2 clusterIDs
    for d in test_corpus.documents:
        doc_id = int(d['id'])
        cluster_id = int(d['cluster'])
        if d['id'] in ldict:
            ldict[doc_id].add(cluster_id)
        else:
            ldict[doc_id] = set([cluster_id])

    cdict = {}
    pred_cluster_set = set([])
    with open(outputname) as f:
        for line in f:
            doc_id, cluster_id = line.strip().split("\t")
            doc_id, cluster_id = int(doc_id), int(cluster_id)
            pred_cluster_set.add(cluster_id)
            if doc_id in cdict:
                cdict[doc_id].add(cluster_id)
            else:
                cdict[doc_id] = set([cluster_id])
    
    precision = bcubed.precision(cdict, ldict)
    recall = bcubed.recall(cdict, ldict)
    fscore = bcubed.fscore(precision, recall)
    
    print("predicted cluster num:", len(pred_cluster_set))
    precision, recall, fscore = 100*float(precision), 100*float(recall), 100*float(fscore)
    print("precision: {:.2f}; recall: {:.2f}; f-1: {:.2f}".format(precision, recall, fscore))
    return precision, recall, fscore


def main():

    parser = argparse.ArgumentParser(description="main training script for word2vec dynamic word embeddings...")
    parser.add_argument("--dataset_name", type=str, default="./svm_en_data/output/xxx", help="dest dir")
    parser.add_argument("--prediction_path", type=str, default="./svm_en_data/output/xxx", help="dest dir")
    parser.add_argument("-f")
    args = parser.parse_args()

    if args.dataset_name == "news2013":
        # with open('../dataset/train_dev.pickle', 'rb') as handle:
        #     train_dev_corpus = pickle.load(handle)
        with open('../dataset/test.pickle', 'rb') as handle:
            test_corpus = pickle.load(handle)
    elif args.dataset_name == "tdt1": # TDT4
        # with open('../tdt_pilot_data/train_dev_final.pickle', 'rb') as handle:
        #     train_dev_corpus = pickle.load(handle)
        with open('../tdt_pilot_data/test_final.pickle', 'rb') as handle:
            test_corpus = pickle.load(handle)
    elif args.dataset_name == "tdt4": # TDT4
        # with open('../tdt4/train_dev_final.pickle', 'rb') as handle:
        #     train_dev_corpus = pickle.load(handle)
        with open('../tdt4/test_final.pickle', 'rb') as handle:
            test_corpus = pickle.load(handle)

    print('testing en',"#docs",len(test_corpus.documents))

    if args.prediction_path[-4:] == ".out":
        print(args.prediction_path)
        evaluate_clusters(test_corpus, outputname=args.prediction_path)
    else:
        for filename_path in glob(os.path.join(args.prediction_path, "*.out")):
            print(filename_path)
            evaluate_clusters(test_corpus, outputname=filename_path)

if __name__ == "__main__":
    main()