#! python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from gensim.models import KeyedVectors


# data from https://www.cs.umb.edu/~smimarog/textmining/datasets/
train = pd.read_csv('./data/r8-train-all-terms.txt', header=None, sep='\t')
test = pd.read_csv('./data/r8-test-all-terms.txt', header=None, sep='\t')
train.columns = ['label', 'content']
test.columns = ['label', 'content']


class GloveVectorizer:
    def __init__(self):
        print('loading...')
        word2vec = {}
        embedding = []
        idx2word = []
        with open('./big_data/glove.6B.50d.txt') as f:
            # space separated text file.
            # word v0 v1 ... v49
            for line in f:
                values = line.split()
                word = values[0]
                vec = np.asarray(values[1:], dtype='float32')
                word2vec[word] = vec
                embedding.append(vec)
                idx2word.append(word)
            print('Found %s word vectors.' % len(word2vec))

        self.word2vec = word2vec
        self.embedding = np.array(embedding)
        self.word2idx = {v:k for k,v in enumerate(idx2word)}
        self.V, self.D = self.embedding.shape

    def fit(self, data):
        pass

    def transform(self, data):
        # each line of data, is -> vec,vec,vec; then run a mean()
        X = np.zeros((len(data), self.D))
        n = 0
        emptycount = 0
        for sentence in data:
            tokens = sentence.lower().split()
            vecs = []
            for word in tokens:
                if word in self.word2vec:
                    vec = self.word2vec[word]
                    vecs.append(vec)
            if len(vecs) > 0:
                vecs = np.array(vecs)
                X[n] = vecs.mean(axis=0)
            else:
                emptycount += 1
            n += 1
        print('Number of samples with 0 words found: %s / %s' % (emptycount, len(data)))
        return X

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)


class Word2VecVectorizer:
    def __init__(self):
        print('loading')
        self.word_vectors = \
            KeyedVectors.load_word2vec_format('./big_data/GoogleNews-vectors-negative300.bin',
                            binary=True)
        # get D
        v = self.word_vectors.get_vector('king')
        self.D = v.shape[0]
        print('finished loading')

    def fit(self, data):
        pass

    def transform(self, data):
        X = np.zeros((len(data), self.D))
        n = 0

        emptycount = 0
        for sentence in data:
            tokens = sentence.split()
            vecs = []
            m = 0
            for word in tokens:
                try:
                    # throws KeyError if word not found
                    vec = self.word_vectors.get_vector(word)
                    vecs.append(vec)
                    m += 1
                except KeyError:
                    pass
            if len(vecs) > 0:
                vecs = np.array(vecs)
                X[n] = vecs.mean(axis=0)
            else:
                emptycount += 1
            n += 1
        print("Numer of samples with no words found: %s / %s" % (emptycount, len(data)))
        return X

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)


# vectorizer = Word2VecVectorizer()
vectorizer = GloveVectorizer()
Xtrain = vectorizer.fit_transform(train.content)
Ytrain = train.label

Xtest = vectorizer.fit_transform(test.content)
Ytest = test.label

model = RandomForestClassifier(n_estimators=200)
model.fit(Xtrain, Ytrain)
print('train score:', model.score(Xtrain, Ytrain))
print('test score:', model.score(Xtest, Ytest))
