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

import model
import clustering
import load_corpora
import json
import os
import argparse
import pickle 

parser = argparse.ArgumentParser(description="main training script for word2vec dynamic word embeddings...")
parser.add_argument("--weight_model_dir", type=str, default="models/en/4_1491902620.876421_10000.0.model", help="source dir")
parser.add_argument("--merge_model_dir", type=str, default="models/en/md_3", help="dest dir")
parser.add_argument("--data_path", type=str, default="./output/exp_pos2vec_esbert_ep2_mgn2.0_btch32_norm1.0_max_seq_230_fuse_additive_random_sample_BatchHardTripletLoss/test_data.pickle", help="dest dir")
parser.add_argument("--output_filename", type=str, default="./svm_en_data/output/xxx", help="dest dir")
parser.add_argument("--weight_model_ii_file", type=str, default="./dataset/svm_rank.ii", help="dest dir")
args = parser.parse_args()
    

def test(lang, thr, model_path, model_path_ii, merge_model_path=None, output_filename=None):
    # corpus = load_corpora.load(r"dataset/dataset.test.json",
    #                            r"dataset/clustering.test.json", set([lang]))
    with open(args.data_path, "rb") as handle:
        corpus = pickle.load(handle)
    print(lang,"#docs",len(corpus.documents))
    clustering_model = model.Model()
    clustering_model.load(model_path, model_path_ii)

    merge_model = None
    if merge_model_path:
        merge_model = model.Model()
        merge_model.load_raw(merge_model_path)

    if "lbfgs" in merge_model_path:
        aggregator = clustering.Aggregator(clustering_model, thr, merge_model, is_logreg=True) 
    else:
        aggregator = clustering.Aggregator(clustering_model, thr, merge_model)

    for i, d in enumerate(corpus.documents):
        print("\r", i, "/", len(corpus.documents),
              " | #c= ", len(aggregator.clusters), end="")
        # early stop
        if len(aggregator.clusters) > 800:
            break
        aggregator.PutDocument(clustering.Document(d, "???"))

    # early stop
    if len(aggregator.clusters) > 800:
        return

    with open(output_filename+lang+".out", "w") as fo:
        ci = 0
        for c in aggregator.clusters:
            for d in c.ids:
                fo.write(d)
                fo.write("\t")
                fo.write(str(ci))
                fo.write("\n")
            ci += 1

# test('eng', 0.0, r'models/en/4_1491902620.876421_10000.0.model',
#      r'models/en/example_2017-04-10T193850.536289.ii', r'models/en/md_3')
def main():
    
    test('eng', 0.0, args.weight_model_dir,
        args.weight_model_ii_file, merge_model_path=args.merge_model_dir, output_filename=args.output_filename)

    # test('spa', 8.18067, r'models/es/2_1492035151.291134_100.0.model',
    #      r'models/es/example_2017-04-12T215308.030747.ii')

    # test('deu', 8.1175, r'models/de/2_1499938269.299021_100.0.model',
    #      r'models/de/example_2017-07-13T085725.498310.ii')

if __name__ == "__main__":
    main()