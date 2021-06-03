from train_time_esbert import *
entity_transformer.max_seq_length = 512

def extract_features(dataloader, model):
    sentence_embeddings_list = []
    dataloader.collate_fn = custom_collate_fn
    for batch in iter(dataloader):
        output_features = model(batch)
        sentence_embeddings = output_features['sentence_embedding']
        sentence_embeddings_list.append(sentence_embeddings.cpu().detach().numpy())
    sents_embeds = np.concatenate(sentence_embeddings_list, 0)
    return sents_embeds


time_esbert = torch.load("./output/exp_time_esbert_ep10_m2/model_t_esbert_selfatt_pool.pt")

# train
with open('/mas/u/hjian42/tdt-twitter/baselines/news-clustering/entity-bert/train_dev.pickle', 'rb') as handle:
    train_dev_corpus = pickle.load(handle)
train_examples = [InputExample(texts=d['text'], 
                         label=d['cluster'],
                         guid=d['id'], 
                         entities=d['bert_entities'], 
                         times=d['date']) for d in train_dev_corpus.documents]
train_dataloader = DataLoader(train_examples, shuffle=False, batch_size=4)
train_features = extract_features(train_dataloader, time_esbert)
torch.save(train_features, "/mas/u/hjian42/tdt-twitter/baselines/T-ESBERT/output/exp_time_esbert_ep10_m2/model_t_esbert_selfatt_pool_features/train_sent_embeds.pt")
print("finished saving train features")


# dev
with open('/mas/u/hjian42/tdt-twitter/baselines/news-clustering/entity-bert/test.pickle', 'rb') as handle:
    test_corpus = pickle.load(handle)
print("finished loading pickle files")
dev_examples = [InputExample(texts=d['text'], 
                             label=d['cluster'],
                             guid=d['id'], 
                             entities=d['bert_entities'], 
                             times=d['date']) for d in test_corpus.documents]
dev_dataloader = DataLoader(dev_examples, shuffle=False, batch_size=4)
sents_embeds = extract_features(dev_dataloader, time_esbert)
torch.save(sents_embeds, "/mas/u/hjian42/tdt-twitter/baselines/T-ESBERT/output/exp_time_esbert_ep10_m2/model_t_esbert_selfatt_pool_features/dev_sent_embeds.pt")
print("finished saving dev features")

