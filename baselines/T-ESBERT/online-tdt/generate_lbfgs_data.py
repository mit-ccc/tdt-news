"""
Generate the data used for SVM-triplet and LBFGS merge
"""
import pickle
import sys
sys.path.insert(0,"../")
import json, load_corpora, clustering, os
from clustering import *
import numpy as np
import random
import argparse
from pathlib import Path


parser = argparse.ArgumentParser(description="main training script for word2vec dynamic word embeddings...")
parser.add_argument("--input_folder", type=str, default="./output/exp_time_esbert_ep2_mgn2.0_btch8_norm1.0_max_seq_128", help="input_folder")
parser.add_argument("--weight_model", type=str, default="xxx", help="input_folder")
parser.add_argument("--weight_model_ii_file", type=str, default="xxx", help="input_folder")
parser.add_argument("--features", type=str, default="tfidf_time", help="input_folder")
parser.add_argument("-f")
args = parser.parse_args()


class GoldenAggregator:
    """In the GoldenAggregator, we use each document's label to cluster them"""
    def __init__(self):
        self.clusters = []
        self.clusterid2idx = {} # idx in the clusters List
    
    def PutDocument(self, document, cluster_id):
        
        bofs = []
        has_cluster_match = True
        
        # calcualte features before adding
        for cluster in self.clusters:
            bof = sim_bof_dc(document, cluster)
            bofs.append(bof)
        
        if cluster_id not in self.clusterid2idx:
            self.clusterid2idx[cluster_id] = len(self.clusters)
            self.clusters.append(Cluster(document))
            has_cluster_match = False
        else:
            clusterIdx = self.clusterid2idx[cluster_id]
            self.clusters[clusterIdx].add_document(document)
        
        # find the idx of the positive cluster
        pos_example_idx = self.clusterid2idx[cluster_id]
        
        return bofs, pos_example_idx, has_cluster_match


def generate_svm_merge_data_amazon(input_corpus, output_path, weight_model, features="tfidf_time"):

    if features == "tfidf":
        if "tdt4" in args.input_folder:
            feature_list = ['Entities_all','Lemmas_all', 'Tokens_all', 'ZZINVCLUSTER_SIZE']
        else:
            feature_list = ['Entities_all', 'Entities_body', 'Entities_title',
                            'Lemmas_all', 'Lemmas_body', 'Lemmas_title', 
                            'Tokens_all', 'Tokens_body', 'Tokens_title', 
                            'ZZINVCLUSTER_SIZE']
    elif features == "tfidf_time":
        if "tdt4" in args.input_folder:
            feature_list = ['Entities_all','Lemmas_all', 'Tokens_all', 'NEWEST_TS', 'OLDEST_TS', 'RELEVANCE_TS', 'ZZINVCLUSTER_SIZE']
        else:
            feature_list = ['Entities_all', 'Entities_body', 'Entities_title',
                            'Lemmas_all', 'Lemmas_body', 'Lemmas_title', 
                            'Tokens_all', 'Tokens_body', 'Tokens_title', 
                            'NEWEST_TS', 'OLDEST_TS', 'RELEVANCE_TS', 'ZZINVCLUSTER_SIZE']
    elif features in set(['tfidf_time_sbert', 'tfidf_time_esbert', 'tfidf_time_sinpe_esbert']):
        if "tdt4" in args.input_folder:
            feature_list = ['Entities_all','Lemmas_all', 'Tokens_all', 'NEWEST_TS', 'OLDEST_TS', 'RELEVANCE_TS', 'ZZINVCLUSTER_SIZE', 'bert_sent_embeds']
        else:
            feature_list = ['Entities_all', 'Entities_body', 'Entities_title',
                            'Lemmas_all', 'Lemmas_body', 'Lemmas_title', 
                            'Tokens_all', 'Tokens_body', 'Tokens_title', 
                            'NEWEST_TS', 'OLDEST_TS', 'RELEVANCE_TS', 'ZZINVCLUSTER_SIZE', 'bert_sent_embeds']
    elif features in "time_sinpe_esbert":
        if "tdt4" in args.input_folder:
            feature_list = ['NEWEST_TS', 'OLDEST_TS', 'RELEVANCE_TS', 'ZZINVCLUSTER_SIZE', 'bert_sent_embeds']
        else:
            feature_list = ['NEWEST_TS', 'OLDEST_TS', 'RELEVANCE_TS', 'ZZINVCLUSTER_SIZE', 'bert_sent_embeds']
    elif features in "sinpe_esbert":
        if "tdt4" in args.input_folder:
            feature_list = ['ZZINVCLUSTER_SIZE', 'bert_sent_embeds']
        else:
            feature_list = ['ZZINVCLUSTER_SIZE', 'bert_sent_embeds']

    clustersAgg = GoldenAggregator()
    with open(output_path, "w") as out:
        # input_corpus is sorted by time
        for i, sort_document in enumerate(input_corpus.documents):
            # add each document to clusters according to their gold cluster labels
            cluster_id = sort_document['cluster']
            bofs, pos_example_idx, has_cluster_match = clustersAgg.PutDocument(Document(sort_document, "???"), cluster_id)
            
            # remove the newly created cluster from the similarities
            if not has_cluster_match: 
#                 bofs.pop(pos_example_idx)
                label = -1 # create a new
            else:
                label = 1 # merge to one
            
            # get the most compatible cluster
#             print(i, has_cluster_match)
            if len(bofs) > 0:
                bof_scores = [model_score(bof, weight_model) for bof in bofs]
                most_compatible_cluster_idx = np.argmax(bof_scores)
                most_compatible_bof = bofs[most_compatible_cluster_idx]
                
#                 print(i, most_compatible_cluster_idx)
                if label == -1:
                    string_items = ['-1']
                else:
                    string_items = ['+1']
                
                for j, feat in enumerate(feature_list):
                    if feat not in most_compatible_bof:
                        most_compatible_bof[feat] = 0.0
                    string_items.append(":".join([str(j+1), str(most_compatible_bof[feat])]))
                line = " ".join(string_items)
                out.write(line)
                out.write("\n")
                print(i)

    label2sents = {
            1: [], # include into a cluster
            -1: [] # create a cluster
        }
    with open(output_path) as f:
        for line in f:
    #         print(line)
            if line[:2] == "-1": 
                label2sents[-1].append(line)
            else:
                label2sents[1].append(line)
        label2sents[1] = random.sample(label2sents[1], min(len(label2sents[-1]), len(label2sents[1])))
    #     label2sents[-1] = random.sample(label2sents[-1], min(len(label2sents[-1]), len(label2sents[1])))

#     sample negative and positive examples to be the same number
    with open(output_path.replace("train_lbfgs_raw", "train_lbfgs_balanced"), 'w') as out:
        lines = label2sents[1] + label2sents[-1]
        random.shuffle(lines)
        for line in lines:
            out.write(line)

def main():
     ##############################
    # data loading
     ##############################
    with open(os.path.join(args.input_folder, "train_dev_bert.pickle"), 'rb') as handle:
        train_dev_corpus = pickle.load(handle)
    with open(os.path.join(args.input_folder, "test_bert.pickle"), 'rb') as handle:
        test_corpus = pickle.load(handle)

    print('training en',"#docs",len(train_dev_corpus.documents))
    print('testing en',"#docs",len(test_corpus.documents))

    args.output_folder = os.path.join(args.input_folder, args.features)
    Path(args.output_folder).mkdir(parents=True, exist_ok=True)
    
    ##############################
    # generate data for NN-lbfgs
    ##############################
    suffix = args.weight_model.split("_")[-1][:-4]
    output_path2 = os.path.join(args.output_folder, "train_lbfgs_raw_{}.dat".format(suffix))
    weight_model = model.Model()
    weight_model.load(args.weight_model, args.weight_model_ii_file)
    generate_svm_merge_data_amazon(train_dev_corpus, output_path2, weight_model, features=args.features)

if __name__ == "__main__":
    main()