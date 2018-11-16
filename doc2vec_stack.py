import numpy as np
import pandas as pd
import nltk
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from gensim.models import Doc2Vec
from sklearn import utils
from sklearn.model_selection import train_test_split
import gensim
from sklearn.linear_model import LogisticRegression
from gensim.models.doc2vec import TaggedDocument
import re
import os
import spacy
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import multiprocessing
cores = multiprocessing.cpu_count()
print("cores = ",cores)

nltk.download('stopwords')
stop = set(stopwords.words('english'))
exclude = set(string.punctuation) 
nltk.download('wordnet')
lemma = WordNetLemmatizer()
nltk.download('punkt')

df_train = pd.read_pickle("stack_data.pkl")
df_test = pd.read_pickle("test.pkl")

print(df_train.info())
print(df_test.info()) 

df_train.index = range(350000)
print( "vocab train",df_train['Body'].apply(lambda x: len(x.split(' '))).sum())

df_test.index = range(99999)
print("vocb test ",df_test['Body'].apply(lambda x: len(x.split(' '))).sum())


def clean_text(feature):
    #doc_out=[]
    #features = list(df["post"])
    if isinstance(feature, str):
        #num_free = re.sub(r'\d+', '', feature)
        stop_free = " ".join([i for i in feature.lower().split() if i not in stop])
        #punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
        normalized = " ".join(lemma.lemmatize(word) for word in stop_free.split())
        #normalized = punc_free
    else:
            pass
    return normalized

x_train = list(df_train['Body'].iloc[0:50000].apply(clean_text))
x_test = list(df_test['Body'].iloc[0:10000].apply(clean_text))

tagged_data_train = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(x_train)]
tagged_data_test = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(x_test)]

model_dbow = Doc2Vec(dm=0, vector_size=100, negative=5, hs=0, min_count=20, sample = 0, workers=cores)

model_dbow.build_vocab(tagged_data_train)

for epoch in range(10):
    print('iteration {0}'.format(epoch))
    model_dbow.train(tagged_data_train,
                total_examples=model_dbow.corpus_count,
                epochs=model_dbow.iter)
    # decrease the learning rate
    model_dbow.alpha -= 0.002
    # fix the learning rate, no 
    model_dbow.min_alpha = model_dbow.alpha

model_dbow.save("d2v.model")
print("Model Saved")


from gensim.models.doc2vec import Doc2Vec

model = Doc2Vec.load("d2v.model")
#print(model.shape)
print(model.docvecs['1'])





'''
def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word.lower())
    return tokens

x = list(df_train['Body'].iloc[0:5].apply(tokenize_text))
y = list(df_test['Body'].iloc[0:5].apply(tokenize_text))

print(x)
print(y)

tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(x)]
print(tagged_data[0])

import multiprocessing
cores = multiprocessing.cpu_count()
print("cores = ",cores)

#model_dbow = Doc2Vec(dm=0, vector_size=100, negative=5, hs=0, min_count=20, sample = 0, workers=cores)
#model_dbow.build_vocab([a for a in tqdm(list(x))])

'''
