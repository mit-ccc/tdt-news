import bcubed
import pickle
import json, load_corpora, clustering, os
from clustering import *
import numpy as np
from glob import glob
import argparse


def evaluate_clusters(test_corpus, outputname="./clustering.eng.out.ranksvm"):
    ldict = {} #docID 2 clusterIDs
    for d in test_corpus.documents:
        doc_id = d['id']
        cluster_id = d['cluster']
        if d['id'] in ldict:
            ldict[doc_id].add(cluster_id)
        else:
            ldict[doc_id] = set([cluster_id])

    cdict = {}
    with open(outputname) as f:
        for line in f:
            doc_id, cluster_id = line.strip().split("\t")
            if doc_id in cdict:
                cdict[doc_id].add(cluster_id)
            else:
                cdict[doc_id] = set([cluster_id])
    
    precision = bcubed.precision(cdict, ldict)
    recall = bcubed.recall(cdict, ldict)
    fscore = bcubed.fscore(precision, recall)
    
    print("precision: {}; recall: {}; f-1: {}".format(precision, recall, fscore))
    return precision, recall, fscore


parser = argparse.ArgumentParser(description="main training script for word2vec dynamic word embeddings...")
parser.add_argument("--output_folder", type=str, default="./svm_en_data/output/xxx", help="dest dir")
args = parser.parse_args()


# evaluate the output files under one directory
with open('/mas/u/hjian42/tdt-twitter/baselines/T-ESBERT/dataset/test.pickle', 'rb') as handle:
    test_corpus = pickle.load(handle)
print('testing en',"#docs",len(test_corpus.documents))

for filename_path in glob(os.path.join(args.output_folder, "predictions/*.out")):
    print(filename_path)
    evaluate_clusters(test_corpus, outputname=filename_path)
