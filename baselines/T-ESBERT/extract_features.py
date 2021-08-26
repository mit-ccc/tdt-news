import re, argparse, sys
from utils import CorpusClass, InputExample

parser = argparse.ArgumentParser(description="main training script for word2vec dynamic word embeddings...")
parser.add_argument("--dataset_name", type=str, default="news2013", help="model_path")
parser.add_argument("--model_path", type=str, default="./output/exp_time_esbert_ep2_mgn2.0_btch8_norm1.0_max_seq_128/time_esbert_model_ep1.pt", help="model_path")
args = parser.parse_args()

if "exp_vaccine_pos2vec_esbert" in args.model_path:
    model_type = 'vaccine_pos2vec_esbert'
    from train_pos2vec_esbert_vaccine import *
    model = torch.load(args.model_path)
    input_folder = os.path.dirname(args.model_path) 
elif "exp_pos2vec_esbert" in args.model_path:
    model_type = 'pos2vec_esbert'
    from train_pos2vec_esbert import *
    model = torch.load(args.model_path)
    input_folder = os.path.dirname(args.model_path)
    if "time_hour" in args.model_path:
        entity_transformer.time_encoding = "hour"
    elif "time_2day" in args.model_path:
        entity_transformer.time_encoding = "2day"
    elif "time_3day" in args.model_path:
        entity_transformer.time_encoding = "3day"
    elif "time_4day" in args.model_path:
        entity_transformer.time_encoding = "4day"
    elif "time_week" in args.model_path:
        entity_transformer.time_encoding = "week"
    elif "time_month" in args.model_path:
        entity_transformer.time_encoding = "month"
    elif "time_40day" in args.model_path:
        entity_transformer.time_encoding = "40day"
    elif "time_2month" in args.model_path:
        entity_transformer.time_encoding = "2month"
    elif "time_90day" in args.model_path:
        entity_transformer.time_encoding = "90day"
    elif "time_180day" in args.model_path:
        entity_transformer.time_encoding = "180day"
    elif "time_year" in args.model_path:
        entity_transformer.time_encoding = "year"
elif "exp_learned_pe_esbert" in args.model_path:
    model_type = 'learned_pos2vec_esbert'
    from train_pos2vec_esbert import *
    model = torch.load(args.model_path)
    input_folder = os.path.dirname(args.model_path)
    if "time_hour" in args.model_path:
        entity_transformer.time_encoding = "hour"
elif "date2vec" in args.model_path:
    sys.path.insert(0,"./legacy/")
    model_type = 'tesbert' 
    from legacy.train_date2vec_esbert import *
    model = torch.load(args.model_path)
    input_folder = os.path.dirname(args.model_path)
elif 'exp_esbert' in args.model_path:
    model_type = "esbert"
    from train_entity_sbert_models import *
    model = torch.load(args.model_path)
    input_folder = os.path.dirname(args.model_path)
elif 'exp_sbert' in args.model_path: # "exp_sbert"
    model_type = 'sbert'
    # sbert training module does not have the entity_transformer part by default
    # entity_transformer = EntityTransformer("/mas/u/hjian42/tdt-twitter/baselines/news-clustering/entity-bert/pretrained/0_Transformer/")
    # from train_sbert import *
    from train_entity_sbert_models import *
    model = torch.load(args.model_path)
    input_folder = os.path.dirname(args.model_path)
    def custom_collate_fn(batch):
        """collate for List of InputExamples, not triplet examples
        entity_transformer is used only after loaded globally
        """
        texts = []

        for example in batch:
            texts.append(example.texts) # [0], different from esbert and time_esbert

        tokenized = entity_transformer.tokenize(texts) # HACK: use the model's internal tokenize() function

        return tokenized
else:
    # use pre-trained BERT, e.g. specify "pretrained_max_seq_128"
    from train_sbert import *
    model = SentenceTransformer("bert-base-nli-stsb-mean-tokens")
    entity_transformer = model[0]
    
    def custom_collate_fn(batch):
        """collate for List of InputExamples, not triplet examples
        entity_transformer is used only after loaded globally
        """
        texts = []

        for example in batch:
            texts.append(example.texts) # [0], different from esbert and time_esbert

        tokenized = entity_transformer.tokenize(texts) # HACK: use the model's internal tokenize() function

        return tokenized
    input_folder = args.model_path # differ from the other two


# # mimic the corpus class as the News dataset
# class CorpusClass:
#     def __init__(self, documents):
#         self.documents = documents


def extract_features(dataloader, model):
    sentence_embeddings_list = []
    dataloader.collate_fn = custom_collate_fn
    for batch in iter(dataloader):
        # print(batch['input_ids'], batch['input_ids'].shape)
        output_features = model(batch)
        sentence_embeddings = output_features['sentence_embedding']
        sentence_embeddings_list.append(sentence_embeddings.cpu().detach().numpy())
    sents_embeds = np.concatenate(sentence_embeddings_list, 0)
    return sents_embeds


def add_bert_features(corpus, dense_features):
    for i, document in enumerate(corpus.documents):
        vector = list(dense_features[i])
        if "vaccine" in args.model_path:
            document['bert_sent_embeds'] = {'b_'+str(j): v for j, v in enumerate(vector)}
        else:
            document['features']['bert_sent_embeds'] = {'b_'+str(j): v for j, v in enumerate(vector)}
    return corpus


def main():

    # turn on the evaluation mode
    model.eval()

    # intialize the model
    max_seq_length = int(re.search(r"max\_seq\_(\d*)", args.model_path).group(1))
    entity_transformer.max_seq_length = max_seq_length # entity_transformer is only for tokenization
    # time_esbert = torch.load("./output/exp_time_esbert_ep10_m2/model_t_esbert_selfatt_pool.pt")


    # train
    # with open('/mas/u/hjian42/tdt-twitter/baselines/T-ESBERT/dataset/train_dev.pickle', 'rb') as handle:
    #     train_dev_corpus = pickle.load(handle)
    # print("finished loading train pickle files")
    # train_examples = [InputExample(texts=d['full_text'], 
    #                         label=d['cluster'],
    #                         guid=d['id'], 
    #                         entities=d['bert_entities'], 
    #                         times=d['date']) for d in train_dev_corpus.documents]
    # train_dataloader = DataLoader(train_examples, shuffle=False, batch_size=16)
    # train_features = extract_features(train_dataloader, model)
    # torch.save(train_features, os.path.join(input_folder, "train_sent_embeds.pt"))
    # print("finished saving train features")
    # train_dense_feats = torch.load(os.path.join(input_folder, "train_sent_embeds.pt"))
    # print('train_dense_feats', train_dense_feats.shape)
    # train_dev_corpus = add_bert_features(train_dev_corpus, train_dense_feats)
    # with open(os.path.join(input_folder, "train_dev_data.pickle"), 'wb') as handle:
    #     pickle.dump(train_dev_corpus, handle, protocol=pickle.HIGHEST_PROTOCOL)


    # test
    # with open('/mas/u/hjian42/tdt-twitter/baselines/T-ESBERT/dataset/test.pickle', 'rb') as handle:
    #     test_corpus = pickle.load(handle)
    # print("Running with split {}; time_encoding {}".format(entity_transformer.split, entity_transformer.time_encoding))
    # print("finished loading test pickle files")
    # test_examples = [InputExample(texts=d['full_text'], 
    #                             label=d['cluster'],
    #                             guid=d['id'], 
    #                             entities=d['bert_entities'], 
    #                             times=d['date']) for d in test_corpus.documents]
    # test_dataloader = DataLoader(test_examples, shuffle=False, batch_size=16)
    # sents_embeds = extract_features(test_dataloader, model)
    # torch.save(sents_embeds, os.path.join(input_folder, "test_sent_embeds.pt"))
    # print("finished saving test features")
    # test_dense_feats = torch.load(os.path.join(input_folder, "test_sent_embeds.pt"))
    # print('test_dense_feats', test_dense_feats.shape)
    # test_corpus = add_bert_features(test_corpus, test_dense_feats)
    # with open(os.path.join(input_folder, "test_data.pickle"), 'wb') as handle:
    #     pickle.dump(test_corpus, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # # ALTERNATIVE test
    # with open('/mas/u/hjian42/tdt-twitter/baselines/T-ESBERT/dataset/test.pickle', 'rb') as handle:
    #     test_corpus = pickle.load(handle)
    # entity_transformer.split = "test" #HACK
    # print("Running with split {}; time_encoding {}".format(entity_transformer.split, entity_transformer.time_encoding))
    # print("finished loading test pickle files")
    # test_examples = [InputExample(texts=d['full_text'], 
    #                             label=d['cluster'],
    #                             guid=d['id'], 
    #                             entities=d['bert_entities'], 
    #                             times=d['date']) for d in test_corpus.documents]
    # test_dataloader = DataLoader(test_examples, shuffle=False, batch_size=16)
    # sents_embeds = extract_features(test_dataloader, model)
    # torch.save(sents_embeds, os.path.join(input_folder, "test_sent_embeds_reanchoring.pt"))
    # print("finished saving test features")
    # test_dense_feats = torch.load(os.path.join(input_folder, "test_sent_embeds_reanchoring.pt"))
    # print('test_dense_feats', test_dense_feats.shape)
    # test_corpus = add_bert_features(test_corpus, test_dense_feats)
    # with open(os.path.join(input_folder, "test_data_reanchoring.pickle"), 'wb') as handle:
    #     pickle.dump(test_corpus, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # test
    if args.dataset_name == "news2013":
        with open('./dataset/train_dev.pickle', 'rb') as handle:
            train_dev_corpus = pickle.load(handle)
        with open('./dataset/test.pickle', 'rb') as handle:
            test_corpus = pickle.load(handle)
    elif args.dataset_name == "tdt1": # TDT4
        with open('./tdt_pilot_data/train_dev_final.pickle', 'rb') as handle:
            train_dev_corpus = pickle.load(handle)
        with open('./tdt_pilot_data/test_final.pickle', 'rb') as handle:
            test_corpus = pickle.load(handle)
    elif args.dataset_name == "tdt4": # TDT4
        with open('./tdt4/train_dev_final.pickle', 'rb') as handle:
            train_dev_corpus = pickle.load(handle)
        with open('./tdt4/test_final.pickle', 'rb') as handle:
            test_corpus = pickle.load(handle)
    elif args.dataset_name == "vaccine":
        # vaccine data do not have a test split
        with open('./news_data/train_dev_entity.pickle', 'rb') as handle:
            test_corpus = pickle.load(handle)

    # train 
    entity_transformer.split = "train" #HACK: use the earliest time in train files as the anchoring time
    print("finished loading training pickle files")
    train_examples = [InputExample(texts=d['full_text'], 
                            label=d['cluster'],
                            guid=d['id'], 
                            entities=d['bert_entities'], 
                            times=d['date']) for d in train_dev_corpus.documents]
    train_dataloader = DataLoader(train_examples, shuffle=False, batch_size=16)
    train_features = extract_features(train_dataloader, model)
    torch.save(train_features, os.path.join(input_folder, "train_sent_embeds.pt"))
    print("finished saving train features")
    train_dense_feats = torch.load(os.path.join(input_folder, "train_sent_embeds.pt"))
    print('train_dense_feats', train_dense_feats.shape)
    train_dev_corpus = add_bert_features(train_dev_corpus, train_dense_feats)
    with open(os.path.join(input_folder, "train_dev_bert.pickle"), 'wb') as handle:
        pickle.dump(train_dev_corpus, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # test
    entity_transformer.split = "test" #HACK: use the earliest time in test files as the anchoring time
    print("finished loading test pickle files")
    test_examples = [InputExample(texts=d['full_text'], 
                                label=d['cluster'],
                                guid=d['id'], 
                                entities=d['bert_entities'], 
                                times=d['date']) for d in test_corpus.documents]
    test_dataloader = DataLoader(test_examples, shuffle=False, batch_size=16)
    sents_embeds = extract_features(test_dataloader, model)
    torch.save(sents_embeds, os.path.join(input_folder, "test_sent_embeds.pt"))
    print("finished saving test features")
    test_dense_feats = torch.load(os.path.join(input_folder, "test_sent_embeds.pt"))
    print('test_dense_feats', test_dense_feats.shape)
    test_corpus = add_bert_features(test_corpus, test_dense_feats)
    with open(os.path.join(input_folder, "test_bert.pickle"), 'wb') as handle:
        pickle.dump(test_corpus, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()