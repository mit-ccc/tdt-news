import re
import argparse
import pickle
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances

parser = argparse.ArgumentParser(description="main training script for word2vec dynamic word embeddings...")
parser.add_argument("--model_path", type=str, default="./output/exp_time_esbert_ep2_mgn2.0_btch8_norm1.0_max_seq_128/time_esbert_model_ep1.pt", help="model_path")
parser.add_argument("--use_saved_triplets", dest='use_saved_triplets', action='store_true') # default is false
parser.add_argument("--use_vaccine_saved_triplets", dest='use_vaccine_saved_triplets', action='store_true') # default is false
args = parser.parse_args()

if "exp_time_esbert" in args.model_path:
    model_type = 'tesbert'
    from train_time_esbert import *
elif "exp_pos2vec_esbert" in args.model_path:
    model_type = 'pos2vec_esbert'
    from train_pos2vec_esbert import *
    # from train_pos2vec_esbert_vaccine import * #HACK
    entity_transformer.split = "test"
elif "exp_vaccine_pos2vec_esbert" in args.model_path:
    model_type = 'pos2vec_esbert_vaccine'
    from train_pos2vec_esbert_vaccine import *
    entity_transformer.split = "test"
elif "exp_learned_pos2vec_esbert" in args.model_path:
    model_type = 'learned_pos2vec_esbert'
    from train_learned_pos2vec_esbert import *
    entity_transformer.split = "test"
elif "exp_esbert" in args.model_path:
    model_type = 'esbert'
    from train_entity_sbert_models import *
elif "exp_sbert" in args.model_path:
    model_type = "sbert"
    from train_entity_sbert_models import *


# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


def get_examples_labels(selected_corpus):
    labels = [d['cluster'] for d in selected_corpus.documents]
    le = preprocessing.LabelEncoder()
    targets = le.fit_transform(labels)
    for d, target in zip(selected_corpus.documents, targets):
        d['cluster_label'] = target
    examples = [InputExample( texts=d['full_text'], # just need to add [] for the default sentenceBERT training pipeline
                                    label=d['cluster_label'],
                                    guid=d['id'], 
                                    entities=d['bert_entities'], 
                                    times=d['date']
                                    ) for d in selected_corpus.documents]
    return examples, targets


def triplets_from_labeled_dataset(input_examples):
    # Create triplets for a [(label, sentence), (label, sentence)...] dataset
    # by using each example as an anchor and selecting randomly a
    # positive instance with the same label and a negative instance with a different label
    triplets = []
    label2sentence = defaultdict(list)
    for inp_example in input_examples:
        label2sentence[inp_example.label].append(inp_example)

    for inp_example in input_examples:
        anchor = inp_example

        if len(label2sentence[inp_example.label]) < 2: #We need at least 2 examples per label to create a triplet
            continue

        positive = None
        while positive is None or positive.guid == anchor.guid:
            positive = random.choice(label2sentence[inp_example.label])

        negative = None
        while negative is None or negative.label == anchor.label:
            negative = random.choice(input_examples)

        triplets.append(InputExample(texts=[anchor.texts, positive.texts, negative.texts],
                                     entities=[anchor.entities, positive.entities, negative.entities],
                                     times=[anchor.times, positive.times, negative.times]
                                    ))
    
    return triplets


def custom_collate_fn_triplets(batch):
    """collate for triplet examples
    
    return a list of batches (anchor, positive, negative)
    """
    texts_list = [[] for i in range(3)]
    dates_list = [[] for i in range(3)]
    entities_list = [[] for i in range(3)]

    for example in batch:
        for idx in range(3): # iterate three instances per triplet
            texts_list[idx].append(example.texts[idx]) # example.texts is a list strings with length 3: [text1, text2, text3]
            if model_type == "tesbert":
                dates_list[idx].append(convert_str_to_date_tensor(example.times[idx]))
            elif "pos2vec_esbert" in model_type:
                dates_list[idx].append(compute_time_stamp(example.times[idx]))
            
            entities_np = np.array(example.entities[idx]) 
            entities_np = entities_np[entities_np < 512]
            new_entities_tensor = np.zeros(512, dtype=int)
            if len(entities_np) > 0: # sometimes there is no entity in the document
                new_entities_tensor[entities_np] = 1
            entities_list[idx].append(new_entities_tensor)
    
    tokenized_list = []
    for idx, (text, dates, entities) in enumerate(zip(texts_list, dates_list, entities_list)):
        # print("SEE: ", text, dates, entities)
        tokenized = entity_transformer.tokenize(text)
        tokenized['dates'] = torch.tensor(dates).float()
        tokenized['entity_type_ids'] = torch.tensor(entities)
        tokenized_list.append(tokenized)
    return tokenized_list


def evaluate_model(model, test_dataloader):
    """use cosine distance"""
    num_triplets = 0
    num_correct_cos_triplets = 0
    with torch.no_grad():
        sentence_embeddings_list = []
        test_dataloader.collate_fn = custom_collate_fn_triplets
        for batch in iter(test_dataloader):
            anchor, positive, negative = batch
    #         print(anchor, positive, negative)
            anchor_features = model.forward(anchor)['sentence_embedding'].cpu().detach().numpy()
            pos_features = model.forward(positive)['sentence_embedding'].cpu().detach().numpy()
            neg_features = model.forward(negative)['sentence_embedding'].cpu().detach().numpy()
            pos_cos_distance = paired_cosine_distances(anchor_features, pos_features)
            neg_cos_distance = paired_cosine_distances(anchor_features, neg_features)

            for idx in range(len(pos_cos_distance)):
                num_triplets += 1

                if pos_cos_distance[idx] < neg_cos_distance[idx]:
                    num_correct_cos_triplets += 1
        accuracy_cos = num_correct_cos_triplets / num_triplets
        print("Accuracy Cosine Distance:   \t{:.2f}".format(accuracy_cos*100))
    return accuracy_cos

def main():

    if args.use_vaccine_saved_triplets:
        print("using pre-extracted vaccine triplets...")
        with open('/mas/u/hjian42/tdt-twitter/baselines/T-ESBERT/news_data/test_eventsim_triplets.pickle', 'rb') as handle:
            test_triplets = pickle.load(handle)
    elif args.use_saved_triplets:
        print("using pre-extracted triplets...")
        with open('/mas/u/hjian42/tdt-twitter/baselines/T-ESBERT/dataset/amt_test_triples.pickle', 'rb') as handle:
            test_triplets = pickle.load(handle)
    else:
        print("using newly-generated triplets...")
        with open('/mas/u/hjian42/tdt-twitter/baselines/T-ESBERT/dataset/test.pickle', 'rb') as handle:
            test_corpus = pickle.load(handle)
        
        random.seed(42)
        test_corpus.documents = test_corpus.documents[:]
        test_examples, test_labels = get_examples_labels(test_corpus)
        test_triplets = triplets_from_labeled_dataset(test_examples)
    
    test_dataloader = DataLoader(test_triplets, shuffle=False, batch_size=8)
    
    max_seq_length = int(re.search(r"max\_seq\_(\d*)", args.model_path).group(1))
    if args.use_vaccine_saved_triplets:
        max_seq_length = 128
    entity_transformer.max_seq_length = max_seq_length # entity_transformer is only for tokenization
    model = torch.load(args.model_path)
    model.eval() # switch to evaluation mode, to ensure reproducibility during test time

    acc = evaluate_model(model, test_dataloader)

if __name__ == "__main__":
    main()

"""
#######################
### SBERT : pretrained / finetuned on Vaccine
#######################
python evaluate_entity_models.py \
    --use_vaccine_saved_triplets \
    --model_path ./pretrained_bert/exp_sbert_pretrained_max_seq_128/SBERT-base-nli-stsb-mean-tokens.pt

python evaluate_entity_models.py \
    --use_vaccine_saved_triplets \
    --model_path ./output/exp_sbert_ep2_mgn2.0_btch32_norm1.0_max_seq_256/sbert.pt

python evaluate_entity_models.py \
    --use_vaccine_saved_triplets \
    --model_path ./pretrained_bert/exp_sbert_pretrained_max_seq_128/SBERT-base-nli-stsb-mean-tokens.pt

#######################
### SBERT : pretrained / finetuned on News2013
#######################
python evaluate_entity_models.py \
    --use_saved_triplets \
    --model_path ./pretrained_bert/exp_sbert_pretrained_max_seq_128/SBERT-base-nli-stsb-mean-tokens.pt

"""