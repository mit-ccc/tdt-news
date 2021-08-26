import hdbscan
import os, sys
sys.path.insert(0,'..')
import bcubed
import torch
import pickle
import json, load_corpora
# import clustering
# from clustering import *
import numpy as np
from glob import glob
import argparse
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.feature_extraction import DictVectorizer
from pandas.io.json._normalize import nested_to_record    
from sklearn.preprocessing import Normalizer
import umap.umap_ as umap

parser = argparse.ArgumentParser()
parser.add_argument("--features", type=str, default="bert")
parser.add_argument("--input_folder", type=str, default="./output/exp_pos2vec_esbert_ep2_mgn2.0_btch32_norm1.0_max_seq_230_fuse_additive_random_sample_BatchHardTripletLoss", help="input_folder")
parser.add_argument("--cluster_algorithm", type=str, default="kmeans", help="input_folder")
parser.add_argument("--algorithm", type=str, default="best", help="input_folder")
parser.add_argument("--cluster_selection_epsilon", type=float, default=0.0, help="input_folder")
parser.add_argument("--min_cluster_size", type=int, default=5)
parser.add_argument("--min_samples", type=int, default=None)
parser.add_argument("--random_seed", type=int, default=0)
parser.add_argument("--allow_single_cluster", type=int, default=0)
parser.add_argument("--reduced_dim_size", type=int, default=10000)
parser.add_argument("--gold_cluster_num", type=int, default=222)

parser.add_argument("-f")
args = parser.parse_args()


def evaluate_clusters(documents, doc2predcluster):
    ldict = {} #docID 2 clusterIDs
    cdict = {}
    for i, d in enumerate(documents):
        doc_id = d['id']
        cluster_id = d['cluster']
        if d['id'] in ldict:
            ldict[doc_id].add(cluster_id)
            cdict[doc_id].add(doc2predcluster[i])
        else:
            ldict[doc_id] = set([cluster_id])
            cdict[doc_id] = set([doc2predcluster[i]])

    precision = bcubed.precision(cdict, ldict)
    recall = bcubed.recall(cdict, ldict)
    fscore = bcubed.fscore(precision, recall)
    
    precision, recall, fscore = 100*float(precision), 100*float(recall), 100*float(fscore)
    print("precision: {:.2f}; recall: {:.2f}; f-1: {:.2f}".format(precision, recall, fscore))
    return precision, recall, fscore

test_data = os.path.join(args.input_folder, "test_bert.pickle")
with open(test_data, 'rb') as handle:
    test_corpus = pickle.load(handle)
vectorizer = DictVectorizer(sparse=True)

if args.features == "bert":
    test_vectors = torch.load(os.path.join(args.input_folder, "test_sent_embeds.pt"))
    X = test_vectors
    length = np.sqrt((X**2).sum(axis=1))[:,None]
    X = X / length
elif args.features == "tfidf":
    test_dict = [dict(d['features']) for d in test_corpus.documents]
    for d in test_dict:
        del d['bert_sent_embeds']
    test_dict = [nested_to_record(d, sep='_') for d in test_dict]
    X = vectorizer.fit_transform(test_dict)
    X = Normalizer(norm="l2").fit_transform(X).toarray()
    # X = Normalizer(norm="l2").fit_transform(X)
    # X = umap.UMAP(metric='cosine', n_components=args.reduced_dim_size).fit_transform(X)

elif args.features == "bert_tfidf":
    test_dict = [dict(d['features']) for d in test_corpus.documents]
    test_dict = [nested_to_record(d, sep='_') for d in test_dict]
    X = vectorizer.fit_transform(test_dict)
    X = Normalizer(norm="l2").fit_transform(X).toarray()
    # X = umap.UMAP(metric='cosine', n_components=args.reduced_dim_size).fit_transform(X)


if args.cluster_algorithm == "kmeans":
    results = KMeans(n_clusters=33, random_state=args.random_seed).fit(X).labels_
elif args.cluster_algorithm == "agg_ward":
    results = AgglomerativeClustering(n_clusters=args.gold_cluster_num, linkage="ward").fit(X).labels_
elif args.cluster_algorithm == "agg_complete":
    results = AgglomerativeClustering(n_clusters=args.gold_cluster_num, linkage="complete").fit(X).labels_
elif args.cluster_algorithm == "agg_single":
    results = AgglomerativeClustering(n_clusters=args.gold_cluster_num, linkage="single").fit(X).labels_
elif args.cluster_algorithm == "agg_average":
    results = AgglomerativeClustering(n_clusters=args.gold_cluster_num, linkage="average").fit(X).labels_
elif args.cluster_algorithm == "hdbscan":
    test_clusterer = hdbscan.HDBSCAN(min_cluster_size=args.min_cluster_size, 
                                    min_samples=args.min_samples,
                                    algorithm=args.algorithm,
                                    cluster_selection_epsilon=args.cluster_selection_epsilon,
                                    allow_single_cluster=args.allow_single_cluster,
                                    prediction_data=True)
    test_cluster_labels = test_clusterer.fit_predict(X).tolist()
    soft_clusters = [np.argmax(x) for x in hdbscan.all_points_membership_vectors(test_clusterer)]
    soft_cluster_probs = [np.max(x) for x in hdbscan.all_points_membership_vectors(test_clusterer)]
    results = []
    for i, e in enumerate(X):
        # If hdbscan couldn't find a cluster for this point, map it ot the closest one
        if test_cluster_labels[i] == -1 and float(soft_cluster_probs[i]) > 0:
            pred_cluster = soft_clusters[i]
        else:
            pred_cluster = test_cluster_labels[i]
        results.append(pred_cluster)

print("Number of predicted clusters", len(set([v for v in results])))
precision, recall, fscore = evaluate_clusters(test_corpus.documents, results)


"""
docker exec -it hjian42-tdt-1 bash
######### SBERT
for algm in hdbscan
do
    for epochnum in 1 2 3 4 5 6 7 8 9 10
    do
        echo "SBERT", ${epochnum}
        python run_retrospective_clustering.py --cluster_algorithm ${algm} \
            --min_cluster_size 7 --min_samples 3 \
            --input_folder ./output/exp_sbert_news2013_ep${epochnum}_mgn2.0_btch32_norm1.0_max_seq_230_sample_random
    done
done

######### ESBERT
for algm in hdbscan
do
    for epochnum in 1 2 3 4 5 6 7 8 9 10
    do
        echo "E-SBERT", ${epochnum}
        python run_retrospective_clustering.py --cluster_algorithm ${algm} \
            --min_cluster_size 7 --min_samples 3 \
            --input_folder ./output/exp_esbert_news2013_ep${epochnum}_mgn2.0_btch32_norm1.0_max_seq_230_sample_random
    done
done

#### Time
for algm in hdbscan
do
    for epochnum in 1 2 3 4 5 6 7 8 9 10
    do
        echo "T-E-SBERT", ${epochnum}
        python run_retrospective_clustering.py --cluster_algorithm ${algm} \
            --min_cluster_size 7 --min_samples 3 \
            --input_folder ../output/exp_pos2vec_esbert_news2013_ep${epochnum}_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss
    done
done



############################################################
#################### Pos2Vec  ##############################
############################################################

######### Pos2Vec + BERT (additive)
for algm in hdbscan
do
    for iter in 1 2 3 4 5
    do
        echo ${iter}
        python run_retrospective_clustering.py --cluster_algorithm ${algm} --min_cluster_size ${iter} --input_folder ./output/exp_pos2vec_esbert_ep2_mgn2.0_btch32_norm1.0_max_seq_230_fuse_additive_random_sample_BatchHardTripletLoss
    done
done

######### Pos2Vec + BERT (additive-selfatt)
for algm in hdbscan
do
    for iter in 2 3 4 5 6 7 8 9 10 20
    do
        echo ${iter}
        python run_retrospective_clustering.py --cluster_algorithm ${algm} --min_cluster_size ${iter} --input_folder ./output/exp_pos2vec_esbert_ep1_mgn2.0_btch32_norm1.0_max_seq_230_fuse_additive_selfatt_pool_random_sample_BatchHardTripletLoss
    done
done

######### Pos2Vec + BERT (concat-selfatt)
for algm in hdbscan
do
    for iter in 2 3 4 5 6 7 8 9 10 20
    do
        echo "min_cluster_size", ${iter}
        python run_retrospective_clustering.py --cluster_algorithm ${algm} --min_cluster_size ${iter} --input_folder ./output/exp_pos2vec_esbert_ep2_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss
    done
done

for algm in hdbscan
do
    for ms in 2 3 4 5 6 7 8 9 10 20
    do
        echo "min_sample", ${ms}
        python run_retrospective_clustering.py --cluster_algorithm ${algm} --min_cluster_size 7 --min_samples ${ms} --input_folder ./output/exp_pos2vec_esbert_ep2_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss
    done
done

for algm in hdbscan
do
    for algorithm in best generic prims_kdtree prims_balltree boruvka_kdtree boruvka_balltree
    do
        echo "algorithm", ${algorithm}
        python run_retrospective_clustering.py --cluster_algorithm ${algm} --algorithm ${algorithm} --min_cluster_size 7 --min_samples 3 --input_folder ./output/exp_pos2vec_esbert_ep2_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss
    done
done

for algm in hdbscan
do
    for cluster_selection_epsilon in 0.1 0.2 0.3 0.4 0.5
    do
        echo "cluster_selection_epsilon", ${algorithm}
        python run_retrospective_clustering.py --cluster_algorithm ${algm} --cluster_selection_epsilon ${cluster_selection_epsilon} --algorithm boruvka_kdtree --min_cluster_size 7 --min_samples 3 --input_folder ./output/exp_pos2vec_esbert_ep2_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss
    done
done


python run_retrospective_clustering.py --features bert --cluster_algorithm hdbscan --min_cluster_size 7 --min_samples 3 --input_folder ./output/exp_pos2vec_esbert_ep2_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss
python run_retrospective_clustering.py --features bert --cluster_algorithm hdbscan --min_cluster_size 7 --min_samples 3 --input_folder ./output/exp_sbert_ep3_mgn2.0_btch32_norm1.0_max_seq_256
python run_retrospective_clustering.py --features bert --cluster_algorithm hdbscan --min_cluster_size 7 --min_samples 3 --input_folder ./output/exp_esbert_ep2_mgn2.0_btch32_norm1.0_max_seq_256_sample_random
python run_retrospective_clustering.py --features tfidf --cluster_algorithm hdbscan --min_cluster_size 7 --min_samples 3 --input_folder ./output/exp_pos2vec_esbert_ep2_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss
python run_retrospective_clustering.py --features bert_tfidf --cluster_algorithm hdbscan --min_cluster_size 7 --min_samples 3 --input_folder ./output/exp_pos2vec_esbert_ep2_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss
python run_retrospective_clustering.py --features bert_tfidf --cluster_algorithm hdbscan --min_cluster_size 7 --min_samples 3 --input_folder ./output/exp_sbert_ep3_mgn2.0_btch32_norm1.0_max_seq_256
python run_retrospective_clustering.py --features bert_tfidf --cluster_algorithm hdbscan --min_cluster_size 7 --min_samples 3 --input_folder ./output/exp_esbert_ep2_mgn2.0_btch32_norm1.0_max_seq_256_sample_random

############################################################
#################### Learned Pos2Vec  ######################
############################################################

python run_retrospective_clustering.py --features bert --cluster_algorithm hdbscan --min_cluster_size 7 --min_samples 3 --input_folder ./output/exp_learned_pos2vec_esbert_ep8_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss
python run_retrospective_clustering.py --features bert --cluster_algorithm hdbscan --min_cluster_size 7 --min_samples 3 --input_folder ./output/exp_learned_pos2vec_esbert_ep8_mgn2.0_btch32_norm1.0_max_seq_230_fuse_additive_random_sample_BatchHardTripletLoss
python run_retrospective_clustering.py --features bert --cluster_algorithm hdbscan --min_cluster_size 7 --min_samples 3 --input_folder ./output/exp_learned_pos2vec_esbert_ep8_mgn2.0_btch32_norm1.0_max_seq_230_fuse_additive_selfatt_pool_random_sample_BatchHardTripletLoss
python run_retrospective_clustering.py --features bert --cluster_algorithm hdbscan --min_cluster_size 7 --min_samples 3 --input_folder ./output/exp_learned_pos2vec_esbert_ep8_mgn2.0_btch32_norm1.0_max_seq_230_fuse_additive_concat_selfatt_pool_random_sample_BatchHardTripletLoss



"""




"""HDBSCAN experiments

for fuse_method in selfatt_pool
do
    for num_epoch in 1 2 3 4 5 6 7 8 9 10
    do
        echo ${iter}
        python run_retrospective_clustering.py --cluster_algorithm hdbscan \
        --min_cluster_size 7 --min_samples 3 \
        --input_folder ./output/exp_pos2vec_esbert_ep2_mgn2.0_btch32_norm1.0_max_seq_230_fuse_${fuse_method}_random_sample_BatchHardTripletLoss
    done
done


for fuse_method in selfatt_pool additive additive_selfatt_pool additive_concat_selfatt_pool
do
    # python extract_features.py --model_path ./output/exp_pos2vec_esbert_ep2_mgn2.0_btch32_norm1.0_max_seq_230_fuse_${fuse_method}_random_sample_BatchHardTripletLoss_time_hour/time_esbert_model_ep2.pt
    for algm in hdbscan
    do
        for iter in 3 4 5 6 7 8 9 10
        do
            echo ${iter}
            python run_retrospective_clustering.py --cluster_algorithm ${algm} --min_cluster_size ${iter} --min_samples 3 --input_folder ./output/exp_pos2vec_esbert_ep2_mgn2.0_btch32_norm1.0_max_seq_230_fuse_${fuse_method}_random_sample_BatchHardTripletLoss_time_hour
        done
    done
done


python run_retrospective_clustering.py --cluster_algorithm hdbscan --min_cluster_size 7 --min_samples 3 --input_folder ./output/exp_pos2vec_esbert_ep2_mgn2.0_btch32_norm1.0_max_seq_230_fuse_additive_concat_selfatt_pool_random_sample_BatchHardTripletLoss

"""


"""KMeans experiments
- order: SBERT, ESBERT, additive, additive-selfatt, concat-selfatt

for iter in 1 2 3 4 5
do
    echo "SBERT"
    python run_retrospective_clustering.py --random_seed ${iter} --input_folder ./output/exp_sbert_ep3_mgn2.0_btch32_norm1.0_max_seq_256
done

for iter in 1 2 3 4 5
do
    echo "ESBERT"
    python run_retrospective_clustering.py --random_seed ${iter} --input_folder ./output/exp_esbert_ep2_mgn2.0_btch32_norm1.0_max_seq_256_sample_random
done

for iter in 1 2 3 4 5
do
    echo "additive SBERT"
    python run_retrospective_clustering.py --random_seed ${iter} --input_folder ./output/exp_pos2vec_esbert_ep2_mgn2.0_btch32_norm1.0_max_seq_230_fuse_additive_random_sample_BatchHardTripletLoss
done

for iter in 1 2 3 4 5
do
    echo "additive selfatt SBERT"
    python run_retrospective_clustering.py --random_seed ${iter} --input_folder ./output/exp_pos2vec_esbert_ep1_mgn2.0_btch32_norm1.0_max_seq_230_fuse_additive_selfatt_pool_random_sample_BatchHardTripletLoss
done

# USE ONLY THIS ONE
for iter in 1 2 3 4 5
do
    echo "concat selfatt SBERT"
    python run_retrospective_clustering.py --random_seed ${iter} --input_folder ./output/exp_pos2vec_esbert_ep2_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss
done


for iter in 1 2 3 4 5
do
    echo "additive concat selfatt SBERT"
    python run_retrospective_clustering.py --random_seed ${iter} --input_folder ./output/exp_pos2vec_esbert_ep2_mgn2.0_btch32_norm1.0_max_seq_230_fuse_additive_concat_selfatt_pool_random_sample_BatchHardTripletLoss
done


for iter in 1 2 3 4 5
do
    echo "kmeans-tfidf", ${iter}
    python run_retrospective_clustering.py --features tfidf --random_seed ${iter} --input_folder ./output/exp_pos2vec_esbert_ep2_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss
done

for iter in 1 2 3 4 5
do
    echo "kmeans-bert-tfidf concat-selfatt", ${iter}
    python run_retrospective_clustering.py --features bert_tfidf --random_seed ${iter} --input_folder ./output/exp_pos2vec_esbert_ep2_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss
done

for iter in 1 2 3 4 5
do
    echo "kmeans-bert-tfidf additive", ${iter}
    python run_retrospective_clustering.py --features bert_tfidf --random_seed ${iter} --input_folder ./output/exp_pos2vec_esbert_ep2_mgn2.0_btch32_norm1.0_max_seq_230_fuse_additive_random_sample_BatchHardTripletLoss
done

for iter in 1 2 3 4 5
do
    echo "kmeans-bert-tfidf additive-selfatt", ${iter}
    python run_retrospective_clustering.py --features bert_tfidf --random_seed ${iter} --input_folder ./output/exp_pos2vec_esbert_ep1_mgn2.0_btch32_norm1.0_max_seq_230_fuse_additive_selfatt_pool_random_sample_BatchHardTripletLoss
done


for iter in 1 2 3 4 5
do
    echo "kmeans-bert-tfidf concat-additive-selfatt", ${iter}
    python run_retrospective_clustering.py --features bert_tfidf --random_seed ${iter} --input_folder ./output/exp_pos2vec_esbert_ep2_mgn2.0_btch32_norm1.0_max_seq_230_fuse_additive_concat_selfatt_pool_random_sample_BatchHardTripletLoss
done
"""


"""agglomerative algorithm
python run_retrospective_clustering.py --cluster_algorithm agg_average --input_folder ./output/exp_sbert_ep3_mgn2.0_btch32_norm1.0_max_seq_256

python run_retrospective_clustering.py --cluster_algorithm agg_average --input_folder ./output/exp_esbert_ep2_mgn2.0_btch32_norm1.0_max_seq_256_sample_random

python run_retrospective_clustering.py --cluster_algorithm agg_average --input_folder ./output/exp_pos2vec_esbert_ep2_mgn2.0_btch32_norm1.0_max_seq_230_fuse_additive_random_sample_BatchHardTripletLoss

python run_retrospective_clustering.py --cluster_algorithm agg_average --input_folder ./output/exp_pos2vec_esbert_ep1_mgn2.0_btch32_norm1.0_max_seq_230_fuse_additive_selfatt_pool_random_sample_BatchHardTripletLoss

python run_retrospective_clustering.py --cluster_algorithm agg_average --input_folder ./output/exp_pos2vec_esbert_ep2_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss

python run_retrospective_clustering.py --cluster_algorithm agg_average --input_folder ./output/exp_pos2vec_esbert_ep2_mgn2.0_btch32_norm1.0_max_seq_230_fuse_additive_concat_selfatt_pool_random_sample_BatchHardTripletLoss

#NEW ones
python run_retrospective_clustering.py --cluster_algorithm agg_average --input_folder ./output/exp_pos2vec_esbert_ep2_mgn2.0_btch32_norm1.0_max_seq_230_fuse_additive_concat_selfatt_pool_random_sample_BatchHardTripletLoss

python run_retrospective_clustering.py --features tfidf --cluster_algorithm agg_average --input_folder ./output/exp_pos2vec_esbert_ep2_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss

python run_retrospective_clustering.py --features bert_tfidf --cluster_algorithm agg_average --input_folder ./output/exp_pos2vec_esbert_ep2_mgn2.0_btch32_norm1.0_max_seq_230_fuse_selfatt_pool_random_sample_BatchHardTripletLoss

python run_retrospective_clustering.py --features bert_tfidf --cluster_algorithm agg_average --input_folder ./output/exp_sbert_ep3_mgn2.0_btch32_norm1.0_max_seq_256

python run_retrospective_clustering.py --features bert_tfidf --cluster_algorithm agg_average --input_folder ./output/exp_esbert_ep2_mgn2.0_btch32_norm1.0_max_seq_256_sample_random

"""