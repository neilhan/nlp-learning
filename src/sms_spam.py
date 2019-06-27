#! python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

from wordcloud import WordCloud


df = pd.read_csv('./data/sms_spam.csv', encoding='ISO-8859-1')
# remove unnamed col2,3
df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
# rename columes v1, v2
df.columns = ['labels', 'data']

df['b_labels'] = df['labels'].map({'ham': 0, 'spam': 1})
Y = df['b_labels'].values

# # feature extraction tfidf
# tfidf = TfidfVectorizer(decode_error='ignore')
# X = tfidf.fit_transform(df['data'])

# feature extraction count vectorizer
count_vectorizer = CountVectorizer(decode_error='ignore')
X = count_vectorizer.fit_transform(df['data'])

# split the train/test data
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.33)

# model, train, score
model = MultinomialNB()
model.fit(Xtrain, Ytrain)
print('Train set score:', model.score(Xtrain, Ytrain))
print('Test set score:', model.score(Xtest, Ytest))

# visualize the data
def visualize(label):
    words = ''
    for msg in df[df['labels'] == label]['data']:
        msg = msg.lower()
        words += msg + ' '
    wordcloud = WordCloud(width=600, height=400).generate(words)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()

visualize('spam')
visualize('ham')


# find what's wrong
df['predictions'] = model.predict(X)

print('Should be Spam:')
# should be spam
sneaky_spam = df[(df['predictions'] ==0) & (df['b_labels'] == 1)]['data']
for msg in sneaky_spam:
    print(msg)
print('----------------------------')
print('Should be ham')
not_spam = df[(df['predictions'] == 1) & (df['b_labels'] == 0)]['data']
for msg in not_spam:
    print(msg)
