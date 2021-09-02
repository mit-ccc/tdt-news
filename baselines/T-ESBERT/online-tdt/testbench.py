# The 3-Clause BSD License
# For Priberam Clustering Software
# Copyright 2018 by PRIBERAM INFORMÁTICA, S.A. ("PRIBERAM") (www.priberam.com)
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder (PRIBERAM) nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


# python testbench.py
# python eval.py clustering.out  E:\Corpora\clustering\processed_clusters\dataset.test.json -f

import sys
sys.path.insert(0,"../")
import model
import clustering
import load_corpora
import json
import os
import argparse
import pickle 
from utils import CorpusClass
from evaluate_model_outputs import evaluate_clusters
import pandas as pd
import glob
from sklearn.neural_network import MLPClassifier


parser = argparse.ArgumentParser(description="main training script for word2vec dynamic word embeddings...")
parser.add_argument("--use_cross_validation", type=int, default=0, help="dest dir")
parser.add_argument("--weight_model_dir", type=str, default="models/en/4_1491902620.876421_10000.0.model", help="source dir")
parser.add_argument("--merge_model_dir", type=str, default="models/en/md_3", help="dest dir")
parser.add_argument("--data_path", type=str, default="./output/exp_pos2vec_esbert_ep2_mgn2.0_btch32_norm1.0_max_seq_230_fuse_additive_random_sample_BatchHardTripletLoss/test_data.pickle", help="dest dir")
parser.add_argument("--output_filename", type=str, default="./svm_en_data/output/xxx", help="dest dir")
parser.add_argument("--weight_model_ii_file", type=str, default="./dataset/svm_rank.ii", help="dest dir")
parser.add_argument("--sklearn_model_specs", type=str, default=None, help="dest dir")
parser.add_argument("--numdays_stddev", type=int, default=3, help="dest dir")
args = parser.parse_args()
    

def test(corpus, lang, thr, model_path, model_path_ii, merge_model_path=None, output_filename=None, sklearn_model_specs=None, numdays_stddev=3):
    # corpus = load_corpora.load(r"dataset/dataset.test.json",
    #                            r"dataset/clustering.test.json", set([lang]))
    # with open(args.data_path, "rb") as handle:
    #     corpus = pickle.load(handle)
    print(lang,"#docs",len(corpus.documents))
    clustering_model = model.Model()
    clustering_model.load(model_path, model_path_ii)

    merge_model = None
    if merge_model_path:
        if "nn_lbfgs" in merge_model_path:
            with open(merge_model_path, 'rb') as f:
                merge_model = pickle.load(f)
        else:
            merge_model = model.Model()
            merge_model.load_raw(merge_model_path)

    if "nn_lbfgs" in merge_model_path:
        aggregator = clustering.Aggregator(clustering_model, thr, merge_model, sklearn_model_specs=sklearn_model_specs, numdays_stddev=numdays_stddev) 
    else:
        aggregator = clustering.Aggregator(clustering_model, thr, merge_model, numdays_stddev=numdays_stddev)

    for i, d in enumerate(corpus.documents):
        print("\r", i, "/", len(corpus.documents),
              " | #c= ", len(aggregator.clusters), end="")
        # # early stop
        if len(aggregator.clusters) > 1500:
            break
        aggregator.PutDocument(clustering.Document(d, "???"))

    # early stop
    if len(aggregator.clusters) > 1500:
        return

    with open(output_filename+lang+".out", "w") as fo:
        ci = 0
        for c in aggregator.clusters:
            for d in c.ids:
                fo.write(str(d))
                fo.write("\t")
                fo.write(str(ci))
                fo.write("\n")
            ci += 1


def show_suggested_configurations(args):
    # suggest the best configuration
    print("==SUGGEST CONFIGURATION==")
    if "cross_validations" not in args.output_filename:
        cv_dir = os.path.dirname(args.output_filename) + "/cross_validations"
    else:
        cv_dir = os.path.dirname(args.output_filename)
    rows = []
    for filename in glob.glob(cv_dir+"/*.csv"):
        df = pd.read_csv(filename)
        rows.append([os.path.basename(filename)] + df.mean().values[1:].tolist())
    df = pd.DataFrame(rows, columns=["filename", "precision", "recall", "fscore"])
    df = df.sort_values(by=["fscore"], ascending=False)
    print(df.head())


def main():

    print("Running... ")
    print(args.weight_model_dir)
    print(args.merge_model_dir)
    
    if args.use_cross_validation:
        if "tdt4" in args.data_path:
            with open('../tdt4/train_dev_final.pickle', 'rb') as handle:
                train_dev_corpus = pickle.load(handle)
            with open("../tdt4/tdt4_cv5.pickle", 'rb') as handle:
                cv_splits = pickle.load(handle)
        elif "tdt1" in args.data_path:
            with open('../tdt_pilot_data/train_dev_final.pickle', 'rb') as handle:
                train_dev_corpus = pickle.load(handle)
            with open("../tdt_pilot_data/tdt1_cv5.pickle", 'rb') as handle:
                cv_splits = pickle.load(handle)
        elif "news2013" in args.data_path:
            with open('../dataset/train_dev.pickle', 'rb') as handle:
                train_dev_corpus = pickle.load(handle)
            with open("../dataset/news2013_cv5.pickle", 'rb') as handle:
                cv_splits = pickle.load(handle)
        
        # run cross validation
        rows = []
        for split_idx, (train_index, val_index) in enumerate(cv_splits):
            val_documents = [d for i, d in enumerate(train_dev_corpus.documents) if i in set(val_index)]  # sorted already
            val_corpus = CorpusClass(val_documents)
            split_filename = args.output_filename + "_split{}".format(split_idx)
            test(val_corpus, 'eng', 0.0, 
                args.weight_model_dir, 
                args.weight_model_ii_file, 
                merge_model_path=args.merge_model_dir, 
                output_filename=split_filename, 
                sklearn_model_specs=args.sklearn_model_specs,
                numdays_stddev=args.numdays_stddev)
            precision, recall, fscore = evaluate_clusters(val_corpus, split_filename + "eng" + ".out")
            rows.append([precision, recall, fscore])
        df_runs = pd.DataFrame(rows, columns=['precision', 'recall', 'fscore'])
        df_runs.to_csv(args.output_filename+".csv")
        print(args.output_filename)
        print(df_runs)
        print(df_runs.mean())

        show_suggested_configurations(args)
        
    else:
        if "tdt4" in args.data_path:
            with open('../tdt4/test_final.pickle', 'rb') as handle:
                test_corpus = pickle.load(handle)
        elif "tdt1" in args.data_path:
            with open('../tdt_pilot_data/test_final.pickle', 'rb') as handle:
                test_corpus = pickle.load(handle)
        elif "news2013" in args.data_path:
            with open('../dataset/test.pickle', 'rb') as handle:
                test_corpus = pickle.load(handle)

        show_suggested_configurations(args)
        test(test_corpus, 'eng', 0.0, 
            args.weight_model_dir,
            args.weight_model_ii_file, 
            merge_model_path=args.merge_model_dir, 
            output_filename=args.output_filename, 
            sklearn_model_specs=args.sklearn_model_specs,
            numdays_stddev=args.numdays_stddev)

if __name__ == "__main__":
    main()