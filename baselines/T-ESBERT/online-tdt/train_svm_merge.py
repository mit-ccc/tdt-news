"""
train merge models with SVM liblinear
"""
from liblinear.liblinearutil import *
import pandas as pd 
import smote_variants as sv
import imbalanced_databases as imbd
import argparse, os
from pathlib import Path


def convert_model_format(model_in, model_out, features="tfidf_time"):

    if features == "tfidf_time":
        feature_list = ['Entities_all','Lemmas_all', 'Tokens_all', 'NEWEST_TS', 'OLDEST_TS', 'RELEVANCE_TS', 'ZZINVCLUSTER_SIZE', 'ZINV_POOL_SIZE']
    
    with open(model_in) as f:
        num_lines = len(f.readlines())-1 # excluding the extra one in the end
    
    with open(model_in) as f, open(model_out, "w") as out:
        featIdx = 0
        for i, line in enumerate(f):
            if i == 4:
                out.write(line.split()[-1])
                out.write("\n")
            if (i < num_lines) and (i >= 6):
#                 print(featIdx)
                line = feature_list[featIdx] + "\t" + line
                featIdx +=1
                out.write(line)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, default="./output/exp_time_esbert_ep2_mgn2.0_btch8_norm1.0_max_seq_128/tfidf_time", help="input_folder")
    parser.add_argument("--features", type=str, default="tfidf_time", help="input_folder")  
    parser.add_argument("--use_smote", type=int, default=0, help="input_folder")
    parser.add_argument("-f")
    args = parser.parse_args()

    y, X = svm_read_problem(os.path.join(args.input_folder, 'train_svmlib_raw.dat'))
    X = pd.DataFrame(X).values
    if args.use_smote:
        print("using oversampled data")
        oversampler= sv.SVM_balance()
        X_samp, y_samp = oversampler.sample(X, y)
        args.output_folder = os.path.join(args.input_folder, "svm_merge_models_smote")
    else:
        print("using original data")
        X_samp, y_samp = X, y
        args.output_folder = os.path.join(args.input_folder, "svm_merge_models")
    
    Path(args.output_folder).mkdir(parents=True, exist_ok=True)
   
    # for c in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
    for c in [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 10, 100, 1000, 10000, 100000, 1000000]:
        m = train(y_samp, X_samp, '-c {} -B 0.0'.format(c))
        model_in = os.path.join(args.output_folder, 'libSVM_c{}_b0.model'.format(c))
        model_out = os.path.join(args.output_folder, 'libSVM_c{}_b0.md'.format(c))
        save_model(model_in, m)
        print("saving model {}".format(model_out))
        convert_model_format(model_in, model_out, features=args.features)
    
    # for c in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
    for c in [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 10, 100, 1000, 10000, 100000, 1000000]:
        m = train(y_samp, X_samp, '-c {} -B 1.0'.format(c))
        model_in = os.path.join(args.output_folder, 'libSVM_c{}_b1.model'.format(c))
        model_out = os.path.join(args.output_folder, 'libSVM_c{}_b1.md'.format(c))
        save_model(model_in, m)
        print("saving model {}".format(model_out))
        convert_model_format(model_in, model_out)


if __name__ == "__main__":
    main()