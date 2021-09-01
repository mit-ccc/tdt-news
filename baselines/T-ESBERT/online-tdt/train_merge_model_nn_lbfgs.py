from liblinear.liblinearutil import *
import pandas as pd 
import smote_variants as sv
import imbalanced_databases as imbd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from pathlib import Path
import argparse, os
import pickle


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./output/exp_time_esbert_ep2_mgn2.0_btch8_norm1.0_max_seq_128/tfidf_time", help="input_folder")
    # parser.add_argument("--features", type=str, default="tfidf_time", help="input_folder")  
    # parser.add_argument("--use_smote", type=int, default=0, help="input_folder")
    parser.add_argument("-f")
    args = parser.parse_args()

    # args.data_path = "../output/exp_sbert_tdt4_ep5_mgn2.0_btch32_norm1.0_max_seq_230_sample_random/tfidf_time/train_lbfgs_raw.dat"

    # oversampling
    y, X = svm_read_problem(args.data_path)
    X = pd.DataFrame(X).values
    oversampler= sv.SVM_balance()
    X_samp, y_samp = oversampler.sample(X, y)
    print("Before oversampling", len(X))
    print("After oversampling", len(X_samp))

    args.input_folder = os.path.dirname(args.data_path)

    Path(os.path.join(args.input_folder, "nn_lbfgs_merge_models")).mkdir(parents=True, exist_ok=True)
    for lr in [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 10, 100, 1000, 10000]:
        clf = MLPClassifier(random_state=0, solver='lbfgs', learning_rate_init=lr, max_iter=1000).fit(X_samp, y_samp)
        model_out = os.path.join(args.input_folder, "nn_lbfgs_merge_models", "nn_lbfgs_merge_lr{}.md".format(lr))
        with open('{}.pkl'.format(model_out),'wb') as f:
            pickle.dump(clf, f)

if __name__ == "__main__":
    main()