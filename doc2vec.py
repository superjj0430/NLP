# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 11:45:21 2019

@author: Barry
"""

import pandas as pd
import numpy as np
import gensim
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import re
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn import utils

#train data 匯入
df = pd.read_csv("multi_all.csv")

#查找類別
def print_Task(index):
    example = df[df.index == index][['Abstract', 'Task']].values[0]
    if len(example) > 0:
        print(example[0])
        print('Task:', example[1])
#print_Task(90)



#切分訓練資料和驗證資料
train, test = train_test_split(df, test_size=0.3, random_state=42)

import nltk
from nltk.corpus import stopwords

def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word.lower())
    return tokens

train_tagged = train.apply(
        lambda r: TaggedDocument(words=tokenize_text(r['Abstract']),
                                 tags=[r.Task]), axis=1)
test_tagged = test.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['Abstract']), tags=[r.Task]), axis=1)

#print(train_tagged.values[30])

import multiprocessing
cores = multiprocessing.cpu_count()


#建字典
model_dbow = Doc2Vec(dm=0, vector_size=1000, negative=5, 
                     hs=0, min_count=2, sample = 0, workers=cores)
model_dbow.build_vocab([x for x in tqdm(train_tagged.values)])

#訓練30次
for epoch in range(30):
    model_dbow.train(utils.shuffle([x for x in tqdm(train_tagged.values)]), total_examples=len(train_tagged.values), epochs=1)
    model_dbow.alpha -= 0.002
    model_dbow.min_alpha = model_dbow.alpha


#建final feature for classifier
def vec_for_learning(model, tagged_docs):
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, regressors


#訓練邏輯回歸
from sklearn.linear_model import LogisticRegression
y_train, X_train = vec_for_learning(model_dbow, train_tagged)
y_test, X_test = vec_for_learning(model_dbow, test_tagged)
logreg = LogisticRegression(n_jobs=1, C=1e5)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

from sklearn.metrics import accuracy_score, f1_score

print('Testing accuracy %s' % accuracy_score(y_test, y_pred))
print('Testing F1 score: {}'.format(f1_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred))))












