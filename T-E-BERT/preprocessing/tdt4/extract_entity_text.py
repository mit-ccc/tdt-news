import pickle
import spacy
from spacy.lang.en.examples import sentences 
import sys
import tokenizations 
from transformers import BertTokenizer
import pandas as pd
import numpy as np


class CorpusClass:
    def __init__(self, documents):
        self.documents = documents

sys.path.insert(0,"/mas/u/hjian42/tdt-twitter/baselines/T-ESBERT/")

nlp = spacy.load("en_core_web_md", disable=["lemmatizer"])

def annotate_ner(nlp_model, df_corpus):

    docs = list(nlp_model.pipe(df_corpus.lemma, disable=["tagger", "parser"]))
    lemma_entity_list = []
    for i, doc in enumerate(docs):

        doc_entity_indices = []
        for ent in doc.ents:
            if ent.label_ in set(['PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE']):
                doc_entity_indices.extend(list(range(ent.start, ent.end)))
#             else:
#                 print(ent, ent.label_)

        spacy_tokens = np.array([w.text for w in doc])
        name_entities = " ".join(spacy_tokens[doc_entity_indices])
        lemma_entity_list.append(name_entities)
    df_corpus['entity'] = lemma_entity_list
    return df_corpus


df_texts = pd.read_csv("df_texts_large.csv")
print(df_texts.text.head())
df_corpus = annotate_ner(nlp, df_texts)
df_corpus.to_csv("df_texts_large_entities.csv")