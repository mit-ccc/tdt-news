from bs4 import BeautifulSoup
import sys,glob
import os
import pandas as pd
import pickle
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

class CorpusClass:
    def __init__(self, documents):
        self.documents = documents

with open('./train_dev.pickle', 'rb') as handle:
    train_corpus = pickle.load(handle)

with open('./test.pickle', 'rb') as handle:
    test_corpus = pickle.load(handle)

df_en_news = pd.read_csv("processed_tdt4.csv")
df_texts = pd.read_csv("df_texts_large.csv")
print(df_texts.text.head())

nlp = spacy.load('en_core_web_md', disable=['parser', 'ner'])

lemma_rows = []
for doc in nlp.pipe(df_texts.text, batch_size=100, n_process=6):
    lemma_doc = " ".join([token.lemma_ for token in doc])
    lemma_rows.append(lemma_doc)
df_texts['lemma'] = lemma_rows
df_texts.to_csv("df_texts_large.csv")

vectorizer = TfidfVectorizer(stop_words={"English"}, max_features=50000)
tfidf_matrix = vectorizer.fit_transform(df_texts['lemma'])
feature_names = vectorizer.get_feature_names()

doc2tfidf = {} # docno, tfidf_dict
docno_set = set(df_en_news.docno)

for doc in range(len(df_texts)):
#     print(doc)
    docno = df_texts.docno.values[doc]
    if docno in docno_set:
        feature_index = tfidf_matrix[doc,:].nonzero()[1]
        tfidf_scores = zip(feature_index, [tfidf_matrix[doc, x] for x in feature_index])
        tf_idf_dict = {"l_"+feature_names[i]: s for (i, s) in tfidf_scores}
        doc2tfidf[docno] = tf_idf_dict

for doc in train_corpus.documents:
    doc['features'] = {"Lemmas_all": doc2tfidf[doc['docno']]}
for doc in test_corpus.documents:
    doc['features'] = {"Lemmas_all": doc2tfidf[doc['docno']]}

with open('./train_dev.pickle', 'wb') as handle:
    pickle.dump(train_corpus, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('./test.pickle', 'wb') as handle:
    pickle.dump(test_corpus, handle, protocol=pickle.HIGHEST_PROTOCOL)
