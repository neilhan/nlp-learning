#! python
# -*- coding: utf-8 -*-

import nltk
import numpy as np
import matplotlib.pyplot as plt

from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import TruncatedSVD


lemmatizer = WordNetLemmatizer()
titles = [line.rstrip() for line in open('./data/all_book_titles.txt')]
stopwords = set(w.rstrip() for w in open('stopwords.txt'))
stopwords = stopwords.union({
    'introduction', 'edition', 'series', 'application',
    'approach', 'card', 'access', 'package', 'plus', 'etext',
    'brief', 'vol', 'fundamental', 'guide', 'essential', 'printed',
    'third', 'second', 'fourth',})

def my_tokenizer(s):
    s = s.lower()
    tokens = nltk.tokenize.word_tokenize(s) # split string into words (tokens)
    tokens = [t for t in tokens if len(t) > 2] # remove short words, they're probably not useful
    tokens = [lemmatizer.lemmatize(t) for t in tokens] # put words into base form
    tokens = [t for t in tokens if t not in stopwords] # remove stopwords
    # remove any digits, i.e. "3rd edition"
    tokens = [t for t in tokens if not any(c.isdigit() for c in t)]
    return tokens

# word index map
word_index_map = {}
all_tokens = []
all_titles = []
index_word_map = []
error_count = 0
for title in titles:
    try:
        title = title.encode('ascii', 'ignore').decode('utf-8') # throw exception for bad characters
        all_titles.append(title)
        tokens = my_tokenizer(title)
        all_tokens.append(tokens)
        for token in tokens:
            if token not in word_index_map:
                word_index_map[token] = len(word_index_map)
                index_word_map.append(token)
    except Exception as e:
        print(e)
        print(title)
        error_count += 1

# print error titles
print('number of errors:', error_count, 'total title:', len(titles))

# token to vector. 1 if a word is in the title
def tokens_to_vector(tokens):
    x = np.zeros(len(word_index_map))
    for t in tokens:
        i = word_index_map[t]
        x[i] = 1
    return x

N = len(all_titles)
D = len(word_index_map)
X = np.zeros((D, N))  # vector size D. Title is the N.
i = 0
for tokens in all_tokens:
    X[:,i] = tokens_to_vector(tokens)
    i += 1

def main():
    svd = TruncatedSVD()
    Z = svd.fit_transform(X)
    print('TruncatedSVD Z.shape:', Z.shape)
    plt.scatter(Z[:,0], Z[:, 1])
    for i in range(D):
        plt.annotate(s=index_word_map[i], xy=(Z[i,0], Z[i, 1]))
    plt.show()

if __name__ == '__main__':
    main()
