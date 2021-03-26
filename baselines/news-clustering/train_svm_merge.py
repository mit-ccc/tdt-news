"""

"""
from liblinear.liblinearutil import *
import pandas as pd 
import smote_variants as sv
import imbalanced_databases as imbd

def convert_model_format(model_in, model_out):

    features = ['Entities_all', 'Entities_body', 'Entities_title',
                'Lemmas_all', 'Lemmas_body', 'Lemmas_title', 
                'NEWEST_TS', 'OLDEST_TS', 'RELEVANCE_TS', 'ZZINVCLUSTER_SIZE', 'ZINV_POOL_SIZE',
                'Tokens_all', 'Tokens_body', 'Tokens_title', 'bert_sent_embeds']

    with open(model_in) as f, open(model_out, "w") as out:
        featIdx = 0
        for i, line in enumerate(f):
            if i == 4:
                out.write(line.split()[-1])
                out.write("\n")
            if (i < 20) and (i > 5):
#                 print(featIdx)
                line = features[featIdx] + "\t" + line
                featIdx +=1
                out.write(line)

def main():
    ###########
    # svm-merge
    ############
    # train models with only TFIDF features
    y, x = svm_read_problem('./svm_en_data/train_svmlib1_without_bert.dat')
    for c in [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 10, 100]:
        m = train(y, x, '-c {} -B 0.0'.format(c)) # NOTICE: bias=0 not 1 since the best model uses this parameter
        model_in = './svm_en_data/merge_models/tfidf/liblinearSVM_c{}.model'.format(c)
        model_out = './svm_en_data/merge_models/tfidf/merge_model_c{}.md'.format(c)
        save_model(model_in, m)
        print("saving model {}".format(model_in))
        convert_model_format(model_in, model_out)
    
    # train models with TFIDF + BERT features
    y, x = svm_read_problem('./svm_en_data/train_svmlib1.dat')
    for c in [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 10, 100]:
        m = train(y, x, '-c {} -B 0.0'.format(c)) # NOTICE: bias=0 not 1 since the best model uses this parameter
        model_in = './svm_en_data/merge_models/tfidf_bert/liblinearSVM_c{}.model'.format(c)
        model_out = './svm_en_data/merge_models/tfidf_bert/merge_model_c{}.md'.format(c)
        save_model(model_in, m)
        print("saving model {}".format(model_in))
        convert_model_format(model_in, model_out)

    ###########
    # svm-merge + oversampling (SVM SMOTE)
    ############
    y, X = svm_read_problem('./svm_en_data/train_svmlib0.dat')
    X = pd.DataFrame(X).values
    oversampler= sv.SVM_balance()
    X_samp, y_samp = oversampler.sample(X, y)
    print("Before oversampling", len(X))
    print("After oversampling", len(X_samp))

    # train models with only TFIDF features + oversampling
    X_samp_without_bert = X_samp[:, :-1]
    y, x = y_samp, X_samp_without_bert
    for c in [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 10, 100]:
        m = train(y, x, '-c {} -B 0.0'.format(c))
        model_in = './svm_en_data/merge_models/tfidf_smote/liblinearSVM_c{}.model'.format(c)
        model_out = './svm_en_data/merge_models/tfidf_smote/merge_model_c{}.md'.format(c)
        save_model(model_in, m)
        print("saving model {}".format(model_in))
        convert_model_format(model_in, model_out)

    # train models with TFIDF + BERT features + oversampling
    y, x = y_samp, X_samp
    for c in [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 10, 100]:
        m = train(y, x, '-c {} -B 0.0'.format(c))
        model_in = './svm_en_data/merge_models/tfidf_bert_smote/liblinearSVM_c{}.model'.format(c)
        model_out = './svm_en_data/merge_models/tfidf_bert_smote/merge_model_c{}.md'.format(c)
        save_model(model_in, m)
        print("saving model {}".format(model_in))
        convert_model_format(model_in, model_out)

if __name__ == "__main__":
    main()