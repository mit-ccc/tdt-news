from liblinear.liblinearutil import *
import pandas as pd 
import smote_variants as sv
import imbalanced_databases as imbd
from sklearn.linear_model import LogisticRegression


def save_sklearn_model(model_in, model_out):
    features = ['Entities_all', 'Entities_body', 'Entities_title',
                'Lemmas_all', 'Lemmas_body', 'Lemmas_title', 
                'NEWEST_TS', 'OLDEST_TS', 'RELEVANCE_TS', 'ZZINVCLUSTER_SIZE', 'ZINV_POOL_SIZE',
                'Tokens_all', 'Tokens_body', 'Tokens_title', 'bert_sent_embeds']
    
    with open(model_out, "w") as out:
        featIdx = 0
        # write down bias
        out.write(str(model_in.intercept_[0]))
        out.write("\n")

        # write down weights
        for weight in model_in.coef_[0]:
            line = features[featIdx] + "\t" + str(weight) + "\n"
            featIdx +=1
            out.write(line)

def main():
    # oversampling
    y, X = svm_read_problem('./svm_en_data/train_svmlib0.dat')
    X = pd.DataFrame(X).values
    oversampler= sv.SVM_balance()
    X_samp, y_samp = oversampler.sample(X, y)
    print("Before oversampling", len(X))
    print("After oversampling", len(X_samp))

    # train without BERT features
    X_samp_without_bert = X_samp[:, :-1]
    y, x = y_samp, X_samp_without_bert
    for c in [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 10, 100]:
        clf = LogisticRegression(random_state=0, C=1.0, solver='lbfgs', max_iter=100).fit(x, y)
        model_in = clf
        model_out = './svm_en_data/merge_models/tfidf_smote_lbfgs/merge_model_c{}.md'.format(c)
        save_sklearn_model(model_in, model_out)
        
    # train with BERT features
    y, x = y_samp, X_samp
    for c in [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 10, 100]:
        clf = LogisticRegression(random_state=0, C=1.0, solver='lbfgs', max_iter=100).fit(x, y)
        model_in = clf
        model_out = './svm_en_data/merge_models/tfidf_bert_smote_lbfgs/merge_model_c{}.md'.format(c)
        save_sklearn_model(model_in, model_out)

if __name__ == "__main__":
    main()