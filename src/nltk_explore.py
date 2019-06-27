#! python
# -*- coding: utf-8 -*-

import nltk

tags = nltk.pos_tag('This course is great.'.split())
print(tags)

# stem
from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()

print(porter_stemmer.stem('geese'))
print(porter_stemmer.stem('wolves'))

# another stem
from nltk.stem import WordNetLemmatizer
wnLemmatizer = WordNetLemmatizer()

print(wnLemmatizer.lemmatize('geese'))
print(wnLemmatizer.lemmatize('wolves'))

def tag_chunk(s):
    tags = nltk.pos_tag(s.split())
    print(tags)
    chunk = nltk.ne_chunk(tags)
    print(chunk)
    chunk.draw()

# Named entity Recognition
tag_chunk('Albert Einstein was born on March 14, 1879')
tag_chunk('Steve Jobs was the CEO of Apple Corp.')
