from train_sbert import *
from sentence_transformers.evaluation import TripletEvaluator
import re

class InputExample:
    """
    Structure for one input example with texts, the label and a unique id
    """
    def __init__(self, 
                 guid: str = '', 
                 texts: List[str] = None,  
                 label: Union[int, float] = 0, 
                 entities: List = None,
                 times: List = None
                ):
        """
        Creates one InputExample with the given texts, guid and label
        :param guid
            id for the example
        :param texts
            the texts for the example. Note, str.strip() is called on the texts
        :param label
            the label for the example
        """
        self.guid = guid
        self.texts = texts
        self.label = label
        self.entities = entities
        self.times = times

    def __str__(self):
        return "<InputExample> label: {}, texts: {}".format(str(self.label), "; ".join(self.texts[:10]))


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

        triplets.append(InputExample(texts=[anchor.texts[0], positive.texts[0], negative.texts[0]]))

    return triplets

def get_examples_labels(selected_corpus):
    labels = [d['cluster'] for d in selected_corpus.documents]
    le = preprocessing.LabelEncoder()
    targets = le.fit_transform(labels)
    for d, target in zip(selected_corpus.documents, targets):
        d['cluster_label'] = target
    examples = [InputExample( texts=[d['text']], # just need to add [] for the default sentenceBERT training pipeline
                                    label=d['cluster_label'],
                                    guid=d['id'], 
                                    entities=d['bert_entities'], 
                                    times=d['date']
                                    ) for d in selected_corpus.documents]
    return examples, targets


def main():

    parser = argparse.ArgumentParser(description="main training script for word2vec dynamic word embeddings...")
    parser.add_argument("--model_path", type=str, default="./output/exp_sbert_ep2_mgn2.0_btch8_norm1.0_max_seq_512", help="model_path")
    args = parser.parse_args()

    with open('/mas/u/hjian42/tdt-twitter/baselines/T-ESBERT/dataset/test.pickle', 'rb') as handle:
        test_corpus = pickle.load(handle)
    print("finished loading test set")

    random.seed(42)
    test_corpus.documents = test_corpus.documents[:]
    test_examples, test_labels = get_examples_labels(test_corpus)
    test_triplets = triplets_from_labeled_dataset(test_examples)

    test_evaluator = TripletEvaluator.from_input_examples(test_triplets, name='eventsim-test')
    
    model = SentenceTransformer(args.model_path)
    max_seq_length = int(re.search(r"max\_seq\_(\d*)", args.model_path).group(1))
    model.max_seq_length = max_seq_length

    acc = model.evaluate(test_evaluator)

    print("{}: \n accuracy is {}".format(args.model_path, acc))


if __name__ == "__main__":
    main()