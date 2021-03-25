# pip install -U liblinear-official
from liblinear.liblinearutil import *

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
    # convert models for the models without bert
    y, x = svm_read_problem('./svm_en_data/train_lib1_without_bert.dat')
    for c in [0.0005, 0.005, 0.05, 0.5]:
        m = train(y, x, '-c {} -B 0.0'.format(c))
        # m = train(y, x, '-c 1.0')
        model_in = './svm_en_data/merge_models/tfidf/liblinearSVM_tfidf_c{}.model'.format(c)
        model_out = './svm_en_data/merge_models/tfidf/merge_model_tfidf_c{}.md'.format(c)
        save_model(model_in, m)
        print("saving model {}".format(model_in))
        convert_model_format(model_in, model_out)

if __name__ == "__main__":
    main()