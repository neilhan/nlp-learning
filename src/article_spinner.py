#! python
# -*- coding: utf-8 -*-

import nltk
import random
import numpy as np

from bs4 import BeautifulSoup

positive_reviews = BeautifulSoup(open('./data/electronics/positive.review').read())
positive_reviews = positive_reviews.findAll('review_text')

# trigram
# (w1, w3) => [w2, w2, w2] is the entry
trigrams = {}
for review in positive_reviews:
    s = review.text.lower()
    tokens = nltk.tokenize.word_tokenize(s)
    for i in range(len(tokens) - 2):
        k = (tokens[i], tokens[i+2])
        if k not in trigrams:
            trigrams[k] =[]
        trigrams[k].append(tokens[i+1])
# turn trigrams [w2, w2, w2, w2] to probability vector

trigram_probabilities = {}
for k, words in trigrams.items():
    # word => count
    if len(set(words)) > 1:
        d = {}
        n = 0
        for w in words:
            if w not in d:
                d[w] = 0
            d[w] += 1
            n += 1
        for w, c in d.items():
            d[w] = float(c) / n
        trigram_probabilities[k] = d

# random sample ?
def random_sample(d):
    r = random.random()
    cumulative = 0
    for w, p in d.items():
        cumulative += p
        if r < cumulative:
            return w

# try spinner
def test_spinner():
    review = random.choice(positive_reviews)
    s = review.text.lower()
    print('Original:', s)
    tokens = nltk.tokenize.word_tokenize(s)
    for i in range(len(tokens) - 2):
        if random.random() < 0.2:
            # 20% chance of replacing
            k = (tokens[i], tokens[i+2])
            if k in trigram_probabilities:
                w = random_sample(trigram_probabilities[k])
                tokens[i+1] = w
    print('Spun result:')
    print(" ".join(tokens)\
          .replace(" .", ".")\
          .replace(" '", "'")\
          .replace(" ,", ",")\
          .replace("$ ", "$")\
          .replace(" !", "!"))

if __name__ == '__main__':
    test_spinner()
    print('-------------')
    test_spinner()
    print('-------------')
    test_spinner()
    print('-------------')
    test_spinner()
    print('-------------')
    test_spinner()
    print('-------------')
