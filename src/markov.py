#! python
# -*- coding: utf-8 -*-

import os
import sys

import numpy as np
import operator

# sys.path.append(os.path.abspath('..'))
from nltk.corpus import brown

KEEP_WORDS = set([
  'king', 'man', 'queen', 'woman',
  'italy', 'rome', 'france', 'paris',
  'london', 'britain', 'england',
])


def get_sentences():
  # returns 57340 of the Brown corpus
  # each sentence is represented as a list of individual string tokens
  return brown.sents()


def get_sentences_with_word2idx():
  sentences = get_sentences()
  indexed_sentences = []

  i = 2
  word2idx = {'START': 0, 'END': 1}
  for sentence in sentences:
    indexed_sentence = []
    for token in sentence:
      token = token.lower()
      if token not in word2idx:
        word2idx[token] = i
        i += 1

      indexed_sentence.append(word2idx[token])
    indexed_sentences.append(indexed_sentence)

  print("Vocab size:", i)
  return indexed_sentences, word2idx


def get_sentences_with_word2idx_limit_vocab(n_vocab=2000, keep_words=KEEP_WORDS):
  sentences = get_sentences()
  indexed_sentences = []

  i = 2
  word2idx = {'START': 0, 'END': 1}
  idx2word = ['START', 'END']

  word_idx_count = {
    0: float('inf'),
    1: float('inf'),
  }

  for sentence in sentences:
    indexed_sentence = []
    for token in sentence:
      token = token.lower()
      if token not in word2idx:
        idx2word.append(token)
        word2idx[token] = i
        i += 1

      # keep track of counts for later sorting
      idx = word2idx[token]
      word_idx_count[idx] = word_idx_count.get(idx, 0) + 1

      indexed_sentence.append(idx)
    indexed_sentences.append(indexed_sentence)

  # restrict vocab size

  # set all the words I want to keep to infinity
  # so that they are included when I pick the most
  # common words
  for word in keep_words:
    word_idx_count[word2idx[word]] = float('inf')

  sorted_word_idx_count = sorted(word_idx_count.items(), key=operator.itemgetter(1), reverse=True)
  word2idx_small = {}
  new_idx = 0
  idx_new_idx_map = {}
  for idx, count in sorted_word_idx_count[:n_vocab]:
    word = idx2word[idx]
    print(word, count)
    word2idx_small[word] = new_idx
    idx_new_idx_map[idx] = new_idx
    new_idx += 1
  # let 'unknown' be the last token
  word2idx_small['UNKNOWN'] = new_idx
  unknown = new_idx

  assert('START' in word2idx_small)
  assert('END' in word2idx_small)
  for word in keep_words:
    assert(word in word2idx_small)

  # map old idx to new idx
  sentences_small = []
  for sentence in indexed_sentences:
    if len(sentence) > 1:
      new_sentence = [idx_new_idx_map[idx] if idx in idx_new_idx_map else unknown for idx in sentence]
      sentences_small.append(new_sentence)

  return sentences_small, word2idx_small


def get_bigram_probs(sentences, V, start_idx, end_idx, smoothing=1):
    # (last_word, current_word) -> probability
    # smoothing: add-1
    # ignore END token
    # matrix V * V
    bigram_probs = np.ones((V, V)) * smoothing
    for sentence in sentences:
        for i in range(len(sentence)):
            if i == 0:
                # begin
                bigram_probs[start_idx, sentence[i]] += 1
            else:
                # words
                bigram_probs[sentence[i-1], sentence[i]] += 1
            # at final word
            # last -> current, current-> END.
            if i == len(sentence) -1:
                bigram_probs[sentence[i], end_idx] += 1
        # normalize, / sum along the rows, to get the probabilities
        bigram_probs /= bigram_probs.sum(axis=1, keepdims=True)
        return bigram_probs

def get_score_norm_log(sentence, bigram_probs, start_idx, end_idx):
    score = 0
    for i in range(len(sentence)):
        if i == 0:
            # begin-word
            score += np.log(bigram_probs[start_idx, sentence[i]])
        else:
            # middle-word
            score += np.log(bigram_probs[sentence[i-1], sentence[i]])
        # END-word
        score += np.log(bigram_probs[sentence[-1], end_idx])
    return score / (len(sentence) + 1)

def get_words_idx2word(sentence, idx2word):
    return ' '.join(idx2word[i] for i in sentence)

def main():
    sentences, word2idx = get_sentences_with_word2idx_limit_vocab(20000)
    # sentences, word2idx = get_sentences_with_word2idx()

    V = len(word2idx)
    print('Vocab size:', V)

    # START -> word, w->END, also treated as bigram
    start_idx = word2idx['START']
    end_idx = word2idx['END']
    # bigram_probs - matrix, row is last word, col = current word
    bigram_probs = get_bigram_probs(sentences, V, start_idx, end_idx, smoothing=0.1)

    idx2word = dict((v, k) for k, v in word2idx.items())

    # test
    # start token or end token
    sample_probs = np.ones(V)
    sample_probs[start_idx] = 0
    sample_probs[end_idx] = 0
    sample_probs /= sample_probs.sum()

    # test our model on real and fake sentences
    while True:
        # real sentence
        real_idx = np.random.choice(len(sentences))
        real = sentences[real_idx]

        # fake sentence
        fake = np.random.choice(V, size=len(real), p=sample_probs)

        print("REAL:", get_words_idx2word(real, idx2word), "SCORE:", get_score_norm_log(real, bigram_probs, start_idx, end_idx))
        print("FAKE:", get_words_idx2word(fake, idx2word), "SCORE:", get_score_norm_log(fake, bigram_probs, start_idx, end_idx))

        # input your own sentence
        custom = input("Enter your own sentence:\n")
        custom = custom.lower().split()

        # check that all tokens exist in word2idx (otherwise, we can't get score)
        bad_sentence = False
        for token in custom:
            if token not in word2idx:
                bad_sentence = True

        if bad_sentence:
            print("Sorry, you entered words that are not in the vocabulary")
        else:
        # convert sentence into list of indexes
            custom = [word2idx[token] for token in custom]
            print("SCORE:", get_score_norm_log(custom, bigram_probs, start_idx, end_idx))


        cont = input("Continue? [Y/n]")
        if cont and cont.lower() in ('N', 'n'):
            break

if __name__ == '__main__':
    main()
