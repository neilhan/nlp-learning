#! python
# -*- coding: utf-8 -*-

import nltk
from nltk.stem import WordNetLemmatizer

import numpy as np

from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression

from bs4 import BeautifulSoup


# words to base mode
wordnet_lemmatizer = WordNetLemmatizer()
stopwords = set(w.rstrip() for w in open('stopwords.txt'))

# load reviews
positive_reviews = BeautifulSoup(open('./data/electronics/positive.review').read())
positive_reviews = positive_reviews.findAll('review_text')
negative_reviews = BeautifulSoup(open('./data/electronics/negative.review').read())
negative_reviews = negative_reviews.findAll('review_text')

# # positive, negative are different in size
# np.random.shuffle(positive_reviews)
# positive_reviews = positive_reviews[:len(negative_reviews)]

# over sample negative reviews
diff = len(positive_reviews) - len(negative_reviews)
if diff > 0:
    idxs = np.random.choice(len(negative_reviews), size=diff)
    extra = [negative_reviews[i] for i in idxs]
    negative_reviews += extra

def my_tokenizer(s):
    s = s.lower()
    tokens = nltk.tokenize.word_tokenize(s)
    tokens = [t for t in tokens if len(t) > 2]
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens]
    tokens = [t for t in tokens if t not in stopwords]
    return tokens

word_index_map = {}
positive_tokenized = []
negative_tokenized = []
orig_reviews = []

for review in positive_reviews:
    orig_reviews.append(review.text)
    tokens = my_tokenizer(review.text)
    positive_tokenized.append(tokens)
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token] = len(word_index_map)

for review in negative_reviews:
    orig_reviews.append(review.text)
    tokens = my_tokenizer(review.text)
    negative_tokenized.append(tokens)
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token] = len(word_index_map)

print('size of word_index_map:', len(word_index_map))

# input matrices, a vector, each feature is a count of that word
# then normalized by / total_words_count
def tokens_to_vector(tokens, label):
    x = np.zeros(len(word_index_map) + 1)
    for t in tokens:
        i = word_index_map[t]
        x[i] += 1
    x = x / x.sum()
    x[-1] = label
    return x

N = len(positive_tokenized) + len(negative_tokenized)
data = np.zeros((N, len(word_index_map) + 1))
i = 0
# positive label=1
for tokens in positive_tokenized:
    xy = tokens_to_vector(tokens, 1)
    data[i,:] = xy
    i += 1

for tokens in negative_tokenized:
    xy = tokens_to_vector(tokens, 0)
    data[i,:] = xy
    i += 1

# shuffle, split data
orig_reviews, data = shuffle(orig_reviews, data)

X = data[:, :-1]
Y = data[:, -1]

# 100 for testing
Xtrain = X[:-100,]
Ytrain = Y[:-100,]
Xtest = X[-100:,]
Ytest = Y[-100:,]

# -------------------------
model = LogisticRegression()
model.fit(Xtrain, Ytrain)
print('Train accuracy:', model.score(Xtrain, Ytrain))
print('Test accuracy:', model.score(Xtest, Ytest))

# ---------------------------------
# check the weights for words
threshold = 0.5
for word, index in word_index_map.items():
    weight = model.coef_[0][index]
    if weight > threshold or weight < -threshold:
        print(word, weight)

# misclassified examples
preds = model.predict(X)
P = model.predict_proba(X)[:, 1]

# since there are many, just print the "most" wrong samples
minP_whenYis1 = 1
maxP_whenYis0 = 0
wrong_positive_review = None
wrong_negative_review = None
wrong_positive_prediction = None
wrong_negative_prediction = None
for i in range(N):
    p = P[i]
    y = Y[i]
    if y == 1 and p < 0.5:
        if p < minP_whenYis1:
            wrong_positive_review = orig_reviews[i]
            wrong_positive_prediction = preds[i]
            minP_whenYis1 = p
    elif y == 0 and p > 0.5:
        if p > maxP_whenYis0:
            wrong_negative_review = orig_reviews[i]
            wrong_negative_prediction = preds[i]
            maxP_whenYis0 = p

print("Most wrong positive review (prob = %s, pred = %s):" % (minP_whenYis1, wrong_positive_prediction))
print(wrong_positive_review)
print("Most wrong negative review (prob = %s, pred = %s):" % (maxP_whenYis0, wrong_negative_prediction))
print(wrong_negative_review)
