import re
import argparse

parser = argparse.ArgumentParser(description="main training script for word2vec dynamic word embeddings...")
parser.add_argument("--model_path", type=str, default="./output/exp_time_esbert_ep2_mgn2.0_btch8_norm1.0_max_seq_128/time_esbert_model_ep1.pt", help="model_path")
args = parser.parse_args()

if "exp_time_esbert" in args.model_path:
    model_type = 'tesbert' 
    from train_time_esbert import *
    model = torch.load(args.model_path)
    input_folder = os.path.dirname(args.model_path)
elif 'exp_esbert' in args.model_path:
    model_type = "esbert"
    from train_esbert import *
    model = torch.load(args.model_path)
    input_folder = os.path.dirname(args.model_path)
else: # "exp_sbert"
    model_type = 'sbert'
    # sbert training module does not have the entity_transformer part by default
    # entity_transformer = EntityTransformer("/mas/u/hjian42/tdt-twitter/baselines/news-clustering/entity-bert/pretrained/0_Transformer/")
    from train_sbert import *
    model = SentenceTransformer(args.model_path)
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
        document['features']['bert_sent_embeds'] = {'b_'+str(j): v for j, v in enumerate(vector)}
    return corpus


def main():

    # intialize the model
    max_seq_length = int(re.search(r"max\_seq\_(\d*)", args.model_path).group(1))
    entity_transformer.max_seq_length = max_seq_length # entity_transformer is only for tokenization
    # time_esbert = torch.load("./output/exp_time_esbert_ep10_m2/model_t_esbert_selfatt_pool.pt")


    # train
    with open('/mas/u/hjian42/tdt-twitter/baselines/news-clustering/entity-bert/train_dev.pickle', 'rb') as handle:
        train_dev_corpus = pickle.load(handle)
    print("finished loading train pickle files")
    # train_examples = [InputExample(texts=d['text'], 
    #                         label=d['cluster'],
    #                         guid=d['id'], 
    #                         entities=d['bert_entities'], 
    #                         times=d['date']) for d in train_dev_corpus.documents]
    # train_dataloader = DataLoader(train_examples, shuffle=False, batch_size=32)
    # train_features = extract_features(train_dataloader, model)
    # torch.save(train_features, os.path.join(input_folder, "train_sent_embeds.pt"))
    # print("finished saving train features")
    train_dense_feats = torch.load(os.path.join(input_folder, "train_sent_embeds.pt"))
    print('train_dense_feats', train_dense_feats.shape)
    train_dev_corpus = add_bert_features(train_dev_corpus, train_dense_feats)
    with open(os.path.join(input_folder, "train_dev_data.pickle"), 'wb') as handle:
        pickle.dump(train_dev_corpus, handle, protocol=pickle.HIGHEST_PROTOCOL)


    # dev
    with open('/mas/u/hjian42/tdt-twitter/baselines/news-clustering/entity-bert/test.pickle', 'rb') as handle:
        test_corpus = pickle.load(handle)
    print("finished loading test pickle files")
    # dev_examples = [InputExample(texts=d['text'], 
    #                             label=d['cluster'],
    #                             guid=d['id'], 
    #                             entities=d['bert_entities'], 
    #                             times=d['date']) for d in test_corpus.documents]
    # dev_dataloader = DataLoader(dev_examples, shuffle=False, batch_size=32)
    # sents_embeds = extract_features(dev_dataloader, model)
    # torch.save(sents_embeds, os.path.join(input_folder, "test_sent_embeds.pt"))
    # print("finished saving test features")
    test_dense_feats = torch.load(os.path.join(input_folder, "test_sent_embeds.pt"))
    print('test_dense_feats', test_dense_feats.shape)
    test_dev_corpus = add_bert_features(test_corpus, test_dense_feats)
    with open(os.path.join(input_folder, "test_data.pickle"), 'wb') as handle:
        pickle.dump(test_dev_corpus, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()