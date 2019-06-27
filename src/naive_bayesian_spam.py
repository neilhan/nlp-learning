#! python
# -*- coding: utf-8 -*-

from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np

data = pd.read_csv('./data/spambase.data').values  # returns numpy.ndarray
np.random.shuffle(data)

X = data[:, :48]
Y = data[:, -1]

Xtrain = X[:-100,]
Ytrain = Y[:-100,]
Xtest = X[-100:,]
Ytest = Y[-100:,]

model = MultinomialNB()
model.fit(Xtrain, Ytrain)
print('Classification rate for MultinomialNB:', model.score(Xtest, Ytest))

# AdaBoostClassifier
from sklearn.ensemble import AdaBoostClassifier

model = AdaBoostClassifier()
model.fit(Xtrain, Ytrain)
print('Classification rate for AdaBoostClassifier:', model.score(Xtest, Ytest))
