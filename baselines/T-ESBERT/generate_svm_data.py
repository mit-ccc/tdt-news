"""
Generate the data used for SVM-rank and SVM-merge
"""
import pickle
import json, load_corpora, clustering, os
from clustering import *
import numpy as np
import random
import argparse

parser = argparse.ArgumentParser(description="main training script for word2vec dynamic word embeddings...")
parser.add_argument("--input_folder", type=str, default="./output/exp_time_esbert_ep2_mgn2.0_btch8_norm1.0_max_seq_128", help="input_folder")
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
        
        if cluster_id not in self.clusterid2idx:
            self.clusterid2idx[cluster_id] = len(self.clusters)
            self.clusters.append(Cluster(document))
            has_cluster_match = False
        else:
            clusterIdx = self.clusterid2idx[cluster_id]
            self.clusters[clusterIdx].add_document(document)
            
        for cluster in self.clusters:
            bof = sim_bof_dc(document, cluster)
            bofs.append(bof)
        
        pos_example_idx = self.clusterid2idx[cluster_id]
        
        return bofs, pos_example_idx, has_cluster_match


def main():
     ##############################
    # data loading
     ##############################
    with open(os.path.join(args.input_folder, "train_dev_data.pickle"), 'rb') as handle:
        train_dev_corpus = pickle.load(handle)
    with open(os.path.join(args.input_folder, "test_data.pickle"), 'rb') as handle:
        test_corpus = pickle.load(handle)

    print('training en',"#docs",len(train_dev_corpus.documents))
    print('testing en',"#docs",len(test_corpus.documents))

     ##############################
    # generate data for SVM-rank, save to train_bert_rank.dat
     ##############################
    clustersAgg = GoldenAggregator()
    with open(os.path.join(args.input_folder, "train_svm_rank.dat"), "w") as out:
        # train_dev_corpus is sorted by time
        for i, sort_document in enumerate(train_dev_corpus.documents):
            # add each document to clusters according to their gold cluster labels
            cluster_id = sort_document['cluster']
            bofs, pos_example_idx, has_cluster_match = clustersAgg.PutDocument(Document(sort_document, "???"), cluster_id)
            bofs_tokens_all = [b["bert_sent_embeds"] for b in bofs]
            idx_top20 = list(np.argsort(bofs_tokens_all)[-21:][::-1]) # pick top 21
            
            # negative candidates (removing the gold one)
            if pos_example_idx in set(idx_top20):
                idx_top20.remove(pos_example_idx)
            negative_idx_top20 = idx_top20[:20]
            negative_bofs = np.array(bofs)[negative_idx_top20]
            
            def _concatenate_items(bof, target=0, qid=1):
                out_items = [str(target), ':'.join(['qid', str(qid)])]
                for j, key in enumerate(['Entities_all', 'Entities_body', 'Entities_title',
                            'Lemmas_all', 'Lemmas_body', 'Lemmas_title', 
                            'NEWEST_TS', 'OLDEST_TS', 'RELEVANCE_TS',
                            'Tokens_all', 'Tokens_body', 'Tokens_title', 'bert_sent_embeds'
                        ]):
                    if key in bof:
                        out_items.append(":".join([str(j+1), str(bof[key])]))
                    else:
                        out_items.append(":".join([str(j+1), str(0)]))
                return " ".join(out_items)

            # write out the faetures
            qid = i+1
            pos_bof = bofs[pos_example_idx]
            out.write(_concatenate_items(pos_bof, target=len(negative_bofs)+1, qid=qid))
            out.write("\n")
            for j, neg_bof in enumerate(negative_bofs):
                out.write(_concatenate_items(neg_bof, target=len(negative_bofs)-j, qid=qid))
                out.write("\n")
            print(i)
    
    # format training data version without bert feature
    with open(os.path.join(args.input_folder, "train_svm_rank.dat")) as f, \
        open(os.path.join(args.input_folder, "train_svm_rank_without_bert.dat"), 'w') as out:
        for line in f:
            out.write(" ".join(line.split()[:-1]) + "\n")    
    
    ##############################
    # generate data for SVM-merge
    ##############################
    clustersAgg = GoldenAggregator()
    with open(os.path.join(args.input_folder, "train_svmlib0.dat"), "w") as out:
        # train_dev_corpus is sorted by time
        for i, sort_document in enumerate(train_dev_corpus.documents):
            # add each document to clusters according to their gold cluster labels
            cluster_id = sort_document['cluster']
            bofs, pos_example_idx, has_cluster_match = clustersAgg.PutDocument(Document(sort_document, "???"), cluster_id)
                    
            # remove the newly created cluster from the similarities
            if not has_cluster_match: 
                bofs.pop(pos_example_idx)
                label = -1 # create a new
            else:
                label = 1 # merge to one
                
            def _get_sim_matrix(bofs, target=-1):            
                sim_mat = []
                for bof in bofs:
                    current_row = []
                    for j, key in enumerate(['Entities_all', 'Entities_body', 'Entities_title',
                                            'Lemmas_all', 'Lemmas_body', 'Lemmas_title', 
                                            'NEWEST_TS', 'OLDEST_TS', 'RELEVANCE_TS', 'ZZINVCLUSTER_SIZE', 'ZINV_POOL_SIZE',
                                            'Tokens_all', 'Tokens_body', 'Tokens_title', 
                                            'bert_sent_embeds']):
                        if key in bof:
                            current_row.append(bof[key])
                        else:
                            current_row.append(0)
                    sim_mat.append(current_row)
    #             print(sim_mat)
                return sim_mat

            def _format_features(max_features, target=-1):
                if target == -1:
                    string_items = ['-1']
                else:
                    string_items = ['+1']
                max_features = list(max_features)
                for j, feat in enumerate(max_features):
                    string_items.append(":".join([str(j+1), str(feat)]))
                return " ".join(string_items)
                    
            sim_mat = _get_sim_matrix(bofs, target=-1)
            if len(sim_mat) > 0:
                max_features = np.amax(np.array(sim_mat), axis=0)
                line = _format_features(max_features, target=label)
                out.write(line)
                out.write("\n")
            print(i)

    label2sents = {
            1: [], # include into a cluster
            -1: [] # create a cluster
        }
    with open(os.path.join(args.input_folder, "train_svmlib0.dat")) as f:
        for line in f:
    #         print(line)
            if line[:2] == "-1": 
                label2sents[-1].append(line)
            else:
                label2sents[1].append(line)
        label2sents[1] = random.sample(label2sents[1], min(len(label2sents[-1]), len(label2sents[1])))
    #     label2sents[-1] = random.sample(label2sents[-1], min(len(label2sents[-1]), len(label2sents[1])))

    # sample negative and positive examples to be the same number
    with open(os.path.join(args.input_folder, "train_svmlib1.dat"), 'w') as out:
        lines = label2sents[1] + label2sents[-1]
        random.shuffle(lines)
        for line in lines:
            out.write(line)
    
    # format training data version without bert feature
    with open(os.path.join(args.input_folder, "train_svmlib1.dat")) as f, \
        open(os.path.join(args.input_folder, "train_svmlib1_without_bert.dat"), 'w') as out:
        for line in f:
            out.write(" ".join(line.split()[:-1]) + "\n")


if __name__ == "__main__":
    main()