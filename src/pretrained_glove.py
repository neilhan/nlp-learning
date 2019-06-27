#! python
# -*- coding: utf-8 -*-

# WHERE TO GET THE VECTORS:
# GloVe: https://nlp.stanford.edu/projects/glove/
# Direct link: http://nlp.stanford.edu/data/glove.6B.zip

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances


def dist_euclidean(a, b):
    return np.linalg.norm(a - b)

def dist_cosine(a, b):
    return 1 - a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b))


# Pick one
# dist, metric = dist_euclidean, 'euclidean'
dist, metric = dist_cosine, 'cosine'


# find analogies, a->b, c->d, gaven a,b,c, find d
def find_analogies(word2vec, embedding, idx2word, V, D, w1, w2, w3):
    '''
    V - number of words, vocabulary size
    D - feature size
    '''
    for w in (w1, w2, w3):
        if w not in word2vec:
            print('%s not in dictionary' %w)

    king = word2vec[w1]
    man = word2vec[w2]
    woman = word2vec[w3]
    v = king - man + woman

    distances = pairwise_distances(v.reshape(1, D), embedding, metric=metric).reshape(V)
    idxs = distances.argsort()[:4]
    for idx in idxs:
        word = idx2word[idx]
        if word not in (w1, w2, w3):
            best_word = word
            break
    print(w1, '-', w2, '=', best_word, '-', w3)
    return best_word

def nearest_neighbors(word2vec, embedding, idx2word, V, D, w, n=5):
    if w not in word2vec:
        print('%s not in dictionary.' % w)
        return

    v = word2vec[w]
    distances = pairwise_distances(v.reshape(1, D), embedding, metric=metric).reshape(V)
    idxs = distances.argsort()[1:n+1]

    print('neighbors of %s' % w)
    for idx in idxs:
        print('\t%s' % idx2word[idx])


def main():
    word2vec = {}
    embedding = []
    idx2word = []

    print('loading...')
    with open('./big_data/glove.6B.50d.txt', encoding='utf-8') as f:
        # space-separated file: word v1 v2 v3... v50
        for line in f:
            values = line.split()
            word = values[0]
            vec = np.asarray(values[1:], dtype='float32')
            word2vec[word] = vec
            embedding.append(vec)
            idx2word.append(word)
    print('done loading.')
    print('%s word vectors.' % len(word2vec))

    embedding = np.array(embedding)
    V, D = embedding.shape

    find_analogies(word2vec, embedding, idx2word, V, D, 'king', 'man', 'woman')
    find_analogies(word2vec, embedding, idx2word, V, D, 'france', 'paris', 'london')
    find_analogies(word2vec, embedding, idx2word, V, D, 'france', 'paris', 'rome')
    find_analogies(word2vec, embedding, idx2word, V, D, 'paris', 'france', 'italy')
    find_analogies(word2vec, embedding, idx2word, V, D, 'france', 'french', 'english')
    find_analogies(word2vec, embedding, idx2word, V, D, 'japan', 'japanese', 'chinese')
    find_analogies(word2vec, embedding, idx2word, V, D, 'japan', 'japanese', 'italian')
    find_analogies(word2vec, embedding, idx2word, V, D, 'japan', 'japanese', 'australian')
    find_analogies(word2vec, embedding, idx2word, V, D, 'december', 'november', 'june')
    find_analogies(word2vec, embedding, idx2word, V, D, 'miami', 'florida', 'texas')
    find_analogies(word2vec, embedding, idx2word, V, D, 'einstein', 'scientist', 'painter')
    find_analogies(word2vec, embedding, idx2word, V, D, 'china', 'rice', 'bread')
    find_analogies(word2vec, embedding, idx2word, V, D, 'man', 'woman', 'she')
    find_analogies(word2vec, embedding, idx2word, V, D, 'man', 'woman', 'aunt')
    find_analogies(word2vec, embedding, idx2word, V, D, 'man', 'woman', 'sister')
    find_analogies(word2vec, embedding, idx2word, V, D, 'man', 'woman', 'wife')
    find_analogies(word2vec, embedding, idx2word, V, D, 'man', 'woman', 'actress')
    find_analogies(word2vec, embedding, idx2word, V, D, 'man', 'woman', 'mother')
    find_analogies(word2vec, embedding, idx2word, V, D, 'heir', 'heiress', 'princess')
    find_analogies(word2vec, embedding, idx2word, V, D, 'nephew', 'niece', 'aunt')
    find_analogies(word2vec, embedding, idx2word, V, D, 'france', 'paris', 'tokyo')
    find_analogies(word2vec, embedding, idx2word, V, D, 'france', 'paris', 'beijing')
    find_analogies(word2vec, embedding, idx2word, V, D, 'february', 'january', 'november')
    find_analogies(word2vec, embedding, idx2word, V, D, 'france', 'paris', 'rome')
    find_analogies(word2vec, embedding, idx2word, V, D, 'paris', 'france', 'italy')

    nearest_neighbors(word2vec, embedding, idx2word, V, D, 'king')
    nearest_neighbors(word2vec, embedding, idx2word, V, D, 'france')
    nearest_neighbors(word2vec, embedding, idx2word, V, D, 'japan')
    nearest_neighbors(word2vec, embedding, idx2word, V, D, 'einstein')
    nearest_neighbors(word2vec, embedding, idx2word, V, D, 'woman')
    nearest_neighbors(word2vec, embedding, idx2word, V, D, 'nephew')
    nearest_neighbors(word2vec, embedding, idx2word, V, D, 'february')
    nearest_neighbors(word2vec, embedding, idx2word, V, D, 'rome')


if __name__ == '__main__':
    main()
