#! python
# -*- coding: utf-8 -*-

from gensim.models import KeyedVectors


word_vectors = KeyedVectors.load_word2vec_format('./big_data/GoogleNews-vectors-negative300.bin', binary=True)

def find_analogies(word_vectors, w1, w2, w3):
    r = word_vectors.most_similar(positive=[w1, w3], negative=[w2])
    print('%s - %s = %s - %s' %(w1, w2, r[0][0], w3))

def nearest_neighbors(word_vectors, w):
    r = word_vectors.most_similar(positive=[w])
    print('neighbors of %s' % w)
    for word, score in r:
        print('\t%s' % word)


find_analogies(word_vectors, 'king', 'man', 'woman')
find_analogies(word_vectors, 'france', 'paris', 'london')
find_analogies(word_vectors, 'france', 'paris', 'rome')
find_analogies(word_vectors, 'paris', 'france', 'italy')
find_analogies(word_vectors, 'france', 'french', 'english')
find_analogies(word_vectors, 'japan', 'japanese', 'chinese')
find_analogies(word_vectors, 'japan', 'japanese', 'italian')
find_analogies(word_vectors, 'japan', 'japanese', 'australian')
find_analogies(word_vectors, 'december', 'november', 'june')
find_analogies(word_vectors, 'miami', 'florida', 'texas')
find_analogies(word_vectors, 'einstein', 'scientist', 'painter')
find_analogies(word_vectors, 'china', 'rice', 'bread')
find_analogies(word_vectors, 'man', 'woman', 'she')
find_analogies(word_vectors, 'man', 'woman', 'aunt')
find_analogies(word_vectors, 'man', 'woman', 'sister')
find_analogies(word_vectors, 'man', 'woman', 'wife')
find_analogies(word_vectors, 'man', 'woman', 'actress')
find_analogies(word_vectors, 'man', 'woman', 'mother')
find_analogies(word_vectors, 'heir', 'heiress', 'princess')
find_analogies(word_vectors, 'nephew', 'niece', 'aunt')
find_analogies(word_vectors, 'france', 'paris', 'tokyo')
find_analogies(word_vectors, 'france', 'paris', 'beijing')
find_analogies(word_vectors, 'february', 'january', 'november')
find_analogies(word_vectors, 'france', 'paris', 'rome')
find_analogies(word_vectors, 'paris', 'france', 'italy')

nearest_neighbors(word_vectors, 'king')
nearest_neighbors(word_vectors, 'france')
nearest_neighbors(word_vectors, 'japan')
nearest_neighbors(word_vectors, 'einstein')
nearest_neighbors(word_vectors, 'woman')
nearest_neighbors(word_vectors, 'nephew')
nearest_neighbors(word_vectors, 'february')
nearest_neighbors(word_vectors, 'rome')
