import pickle
import spacy
from spacy.lang.en.examples import sentences 
import sys
import tokenizations 
from transformers import BertTokenizer

class CorpusClass:
    def __init__(self, documents):
        self.documents = documents

sys.path.insert(0,"/mas/u/hjian42/tdt-twitter/baselines/T-ESBERT/")

with open('./train_dev.pickle', 'rb') as handle:
    train_dev_corpus = pickle.load(handle)

with open('./test.pickle', 'rb') as handle:
    test_corpus = pickle.load(handle)

nlp = spacy.load("en_core_web_md", disable=["lemmatizer"])
bert_tokenizer = BertTokenizer.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")

def align_tokenizations(corpus):
    for example in corpus.documents[:]:
        spacy_tokenized = example['spacy_tokens']
        bert_tokenized = example['bert_tokens']
        sidx, bidx = tokenizations.get_alignments([t for t in spacy_tokenized], bert_tokenized)
        entidx = example['spacy_entities']
        bert_entidx = []
        for idx in entidx:
            bert_entidx.extend(sidx[idx])
        example['bert_entities'] = bert_entidx
    return corpus


def bert_tokenize(bert_tokenizer, corpus):
    for example in corpus.documents:
        example['bert_tokens'] = bert_tokenizer.tokenize(example['full_text'])
    return corpus


def annotate_ner(nlp_model, corpus):

    texts = []
    for example in corpus.documents:
        # example['full_text'] = example['title'] + " . " + example['text']
        texts.append(example['full_text'])
    docs = list(nlp_model.pipe(texts, disable=["tagger", "parser"]))

    for i, doc in enumerate(docs):

        example = corpus.documents[i]

        doc_entity_indices = []
        for ent in doc.ents:
            if ent.label_ in set(['PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE']):
                doc_entity_indices.extend(list(range(ent.start, ent.end)))
#             else:
#                 print(ent, ent.label_)

        example['spacy_tokens'] = [w.text for w in doc]
        example['spacy_entities'] = doc_entity_indices
    return corpus


def remove_spacy_entries(corpus):
    for example in corpus.documents:
        del example['spacy_tokens']
    return corpus


train_dev_corpus = annotate_ner(nlp, train_dev_corpus)
train_dev_corpus = bert_tokenize(bert_tokenizer, train_dev_corpus)
train_dev_corpus = align_tokenizations(train_dev_corpus)

with open('./train_dev_entities.pickle', 'wb') as handle:
    pickle.dump(train_dev_corpus, handle, protocol=pickle.HIGHEST_PROTOCOL)

test_corpus = annotate_ner(nlp, test_corpus)
test_corpus = bert_tokenize(bert_tokenizer, test_corpus)
test_corpus = align_tokenizations(test_corpus)

with open('./test_entities.pickle', 'wb') as handle:
    pickle.dump(test_corpus, handle, protocol=pickle.HIGHEST_PROTOCOL)
