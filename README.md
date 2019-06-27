# Natural Language Process learning

- src/naive_bayesian_spam.py - Naive Bayesian spam detection
- src/sms_spam.py - sms_spam data related
- src/sentiment.py - word count, logistic regression
- src/latent_semantic_analysis.py - Latent Semantic Analysis with SVD. Singular-value Decomposition.
  Scatter plot the Z matrix.
- src/article_spinner.py - generate articles. trigram probability.

NLP - class2
- src/pretrained_glove.py - explore GloVe
- src/pretrained_word2vec.py - explore word2vec, using Gensim library. Wrapper for word2vec
- src/bow_classifier  - text classification
- src/markov.py - bigram, statistics


## downloads
- https://dumps.wikimedia.org/enwiki/20190620/enwiki-20190620-pages-articles-multistream1.xml-p10p30302.bz2

# Notes
## create venv "env"
```
# install virtualenv if needed
pip install virtualenv
# to create the venv "env"
python3 -m venv env
# to activate the "env"
source env/bin/activate
# to deactivate the "env"
deactivate

# create packages into requirements.txt
pip freeze > requirements.txt
# to install from requirements.txt
pip install -r requirements.txt
```

## VirtualEnvWrapper

```
# virtualenvwrapper, organize, easy create/delete/copy env, switch env
pip install virtualenvwrapper

# edit .bashrc, add
export WORKON_HOME=$HOME/.virtualenvs   # Optional
export PROJECT_HOME=$HOME/projects      # Optional
source /usr/local/bin/virtualenvwrapper.sh
```
VirtualEnvWrapper provides following new commands:
- workon
- deactivate
- mkvirtualenv
- cdvirtualenv
- rmvirtualenv
