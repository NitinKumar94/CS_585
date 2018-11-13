#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 14:43:30 2018

@author: nehachoudhary
"""

import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
from gensim import corpora
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import models
from nltk.stem.porter import PorterStemmer
import spacy
import os
import numpy as np
import pandas as pd
import chardet
import codecs
import re
from sklearn import preprocessing
import math
from sklearn.model_selection import train_test_split
from collections import Counter


nlp = spacy.load('en_core_web_md')
nltk.download('stopwords')
stop = set(stopwords.words('english'))
exclude = set(string.punctuation) 
nltk.download('wordnet')
lemma = WordNetLemmatizer()

pathname = os.getcwd()
url =  pathname + '/train_data.csv'
url1 =  pathname + '/test_data.csv'

def read_data(url):
    with open(url, 'rb') as f:
        result = chardet.detect(f.read())  # or readline if the file is large
        data = pd.read_csv(url, encoding = result['encoding'])
    return data

def combine_test_train(df_train, df_test):
    df_train["filt"] = "train"
    df_test["filt"] = "test"
    data = pd.concat([df_train, df_test], ignore_index=True)
    return data

def get_vocab(posts):
    full = ' '.join(posts)
    full = full.lower()
    Vocab = re.findall(r"[A-Za-z]+|\S", full)
    V = Counter(Vocab)
    return V    
    
def get_missing(df, percent):
    print("Percentage missing by columns =")
    print(df.isnull().mean())
    return df

def get_tag_summary(tags):
    all_tags = []
    for t in tags:
        try:
            all_tags += t.split("|")
        except (AttributeError):
            pass
    unique_tags = list(set(all_tags))
    print("The total number of unique tags =", len(unique_tags))
    dist = Counter(all_tags)
    return dist
    

def test_split(size,X,Y):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=size, random_state=42, stratify=Y)
    return  x_train, x_test, y_train, y_test

def clean_text(df):
    doc_out=[]
    features = list(df["post"])
    for feature in features:
        if isinstance(feature, str):
            #num_free = re.sub(r'\d+', '', feature)
            stop_free = " ".join([i for i in feature.lower().split() if i not in stop])
            #punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
            normalized = " ".join(lemma.lemmatize(word) for word in stop_free.split())
            #normalized = punc_free
            doc_out.append(normalized)
        else:
            pass
    return doc_out

def get_tfidf_feature(clean_data, top_n = 20):
    vectorizer = TfidfVectorizer(#min_df=.0025,\
                                 #min_df=6,\
                                 #max_df=.5,\
                                 #norm='l2',\
                                 use_idf = 1,\
                                 #smooth_idf=True,\
                                 #tokenizer=tokens,\
                                 lowercase=True,\
                                 stop_words='english',\
                                 sublinear_tf=True,\
                                 strip_accents='unicode',\
                                 analyzer='word',\
                                 #token_pattern=r'\w{1,}',\
                                 #token_pattern=r'\b[^_\d\W]+\b',\
                                 ngram_range=(1,1),
                                 #dtype=np.float32,\
                                 max_features=10000)
    vector = vectorizer.fit_transform(clean_data)
    return vector.toarray(), vectorizer

##########################################################
if __name__ == '__main__':
    print("Reading data...")
    stack_data = read_data(url)
    stack_data_test = read_data(url1)
    stack_data.columns = ["title", "post", "tag"]
    stack_data_test.columns = ["title", "post", "tag"]

    stack_data["filt"] = "train"
    stack_data_test["filt"] = "test"

    tags = list(stack_data['tag'])
    posts = list(stack_data['post'])    

    print("Cleaning Train data...")
    clean_data = clean_text(stack_data)
    tf_idf_train, vec_tf_train = get_tfidf_feature(clean_data)
    
    print("Cleaning Test data...")
    clean_data = clean_text(stack_data_test)
    tf_idf_test, vec_tf_test = get_tfidf_feature(clean_data)
    
    np.save('X_train.npy',tf_idf_train) 
    np.save('X_test.npy',tf_idf_test) 
    np.save('y_train.npy', stack_data['tag'].as_matrix()) 
    np.save('y_test.npy', stack_data_test['tag'].as_matrix()) 

