"""
train merge models with SVM liblinear
"""
from liblinear.liblinearutil import *
import pandas as pd 
import smote_variants as sv
import imbalanced_databases as imbd
import argparse, os

def convert_model_format(model_in, model_out):

    features = ['Entities_all', 'Entities_body', 'Entities_title',
                'Lemmas_all', 'Lemmas_body', 'Lemmas_title', 
                'NEWEST_TS', 'OLDEST_TS', 'RELEVANCE_TS', 'ZZINVCLUSTER_SIZE', 'ZINV_POOL_SIZE',
                'Tokens_all', 'Tokens_body', 'Tokens_title', 'bert_sent_embeds']
    
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
                line = features[featIdx] + "\t" + line
                featIdx +=1
                out.write(line)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, default="./output/exp_time_esbert_ep2_mgn2.0_btch8_norm1.0_max_seq_128/time_esbert_model_ep1.pt", help="input_folder")
    parser.add_argument("-f")
    args = parser.parse_args()
    ###########
    # svm-merge
    ############
#     train models with only TFIDF features
    # y, x = svm_read_problem(os.path.join(args.input_folder, 'train_svmlib1_without_bert.dat'))
    # for c in [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 10, 100]:
    #     m = train(y, x, '-c {} -B 0.0'.format(c)) # NOTICE: bias=0 not 1 since the best model uses this parameter
    #     model_in = os.path.join(args.input_folder, "models", 'liblinearSVM_tfidf_c{}.model'.format(c))
    #     model_out = os.path.join(args.input_folder, "models", 'merge_model_tfidf_c{}.md'.format(c))
    #     save_model(model_in, m)
    #     print("saving model {}".format(model_in))
    #     convert_model_format(model_in, model_out)
    
    # # # train models with TFIDF + BERT features
    # y, x = svm_read_problem(os.path.join(args.input_folder, 'train_svmlib1.dat'))
    # for c in [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 10, 100]:
    #     m = train(y, x, '-c {} -B 0.0'.format(c)) # NOTICE: bias=0 not 1 since the best model uses this parameter
    #     model_in = os.path.join(args.input_folder, "models", 'liblinearSVM_tfidf_bert_c{}.model'.format(c))
    #     model_out = os.path.join(args.input_folder, "models", 'merge_model_tfidf_bert_c{}.md'.format(c))
    #     save_model(model_in, m)
    #     print("saving model {}".format(model_in))
    #     convert_model_format(model_in, model_out)

    ###########
    # svm-merge + oversampling (SVM SMOTE)
        # models using SMOTE do not contain suffix
        # models without SMOTE contain merge_model_c{}_unbalanced
    ############
    y, X = svm_read_problem(os.path.join(args.input_folder, 'train_svmlib0.dat'))
    X = pd.DataFrame(X).values
    oversampler= sv.SVM_balance()
    X_samp, y_samp = oversampler.sample(X, y)
    # X_samp, y_samp = X, y # not doing any sampling
    print("Before oversampling", len(X))
    print("After oversampling", len(X_samp))

    # train models with only TFIDF features + oversampling
    # X_samp_without_bert = X_samp[:, :-1]
    # y, x = y_samp, X_samp_without_bert
    # # for c in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
    # for c in [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 1.0, 10, 100, 1000, 10000, 100000, 1000000]:
    #     m = train(y, x, '-c {} -B 0.0'.format(c))
    #     model_in = os.path.join(args.input_folder, "models", 'liblinearSVM_tfidf_smote_c{}_b0.model'.format(c))
    #     model_out = os.path.join(args.input_folder, "models", 'merge_model_tfidf_smote_c{}_b0.md'.format(c))
    #     save_model(model_in, m)
    #     print("saving model {}".format(model_out))
    #     convert_model_format(model_in, model_out)
    
    # for c in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
    # for c in [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 1.0, 10, 100, 1000, 10000, 100000, 1000000]:
    #     m = train(y, x, '-c {} -B 1.0'.format(c))
    #     model_in = os.path.join(args.input_folder, "models", 'liblinearSVM_tfidf_smote_c{}_b1.model'.format(c))
    #     model_out = os.path.join(args.input_folder, "models", 'merge_model_tfidf_smote_c{}_b1.md'.format(c))
    #     save_model(model_in, m)
    #     print("saving model {}".format(model_out))
    #     convert_model_format(model_in, model_out)

    # train models with TFIDF + BERT features + oversampling
    y, x = y_samp, X_samp
    # for c in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
    for c in [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 1.0, 10, 100, 1000, 10000, 100000, 1000000]:
        m = train(y, x, '-c {} -B 0.0'.format(c))
        model_in = os.path.join(args.input_folder, "models", 'liblinearSVM_tfidf_bert_smote_c{}_b0.model'.format(c))
        model_out = os.path.join(args.input_folder, "models", 'merge_model_tfidf_bert_smote_c{}_b0.md'.format(c))
        save_model(model_in, m)
        print("saving model {}".format(model_out))
        convert_model_format(model_in, model_out)
    
    y, x = y_samp, X_samp
    # for c in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
    for c in [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 10, 100, 1000, 10000, 100000, 1000000]:
        m = train(y, x, '-c {} -B 1.0'.format(c))
        model_in = os.path.join(args.input_folder, "models", 'liblinearSVM_tfidf_bert_smote_c{}_b1.model'.format(c))
        model_out = os.path.join(args.input_folder, "models", 'merge_model_tfidf_bert_smote_c{}_b1.md'.format(c))
        save_model(model_in, m)
        print("saving model {}".format(model_out))
        convert_model_format(model_in, model_out)


    # use only BERT feature
    # y, x = y_samp, X_samp
    # # for c in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
    # for c in [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 1.0, 10, 100, 1000, 10000, 100000, 1000000]:
    #     m = train(y, x, '-c {} -B 0.0'.format(c))
    #     model_in = os.path.join(args.input_folder, "models", 'liblinearSVM_pure_bert_smote_c{}_b0.model'.format(c))
    #     model_out = os.path.join(args.input_folder, "models", 'merge_model_pure_bert_smote_c{}_b0.md'.format(c))
    #     save_model(model_in, m)
    #     print("saving model {}".format(model_out))
    #     convert_model_format(model_in, model_out)
    
    # y, x = y_samp, X_samp
    # # for c in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
    # for c in [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 1.0, 10, 100, 1000, 10000, 100000, 1000000]:
    #     m = train(y, x, '-c {} -B 1.0'.format(c))
    #     model_in = os.path.join(args.input_folder, "models", 'liblinearSVM_pure_bert_smote_c{}_b1.model'.format(c))
    #     model_out = os.path.join(args.input_folder, "models", 'merge_model_pure_bert_smote_c{}_b1.md'.format(c))
    #     save_model(model_in, m)
    #     print("saving model {}".format(model_out))
    #     convert_model_format(model_in, model_out)

if __name__ == "__main__":
    main()