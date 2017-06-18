# this is inspired by:
# https://www.youtube.com/watch?v=ogrJaOIuBx4


import json
from collections import Counter
from spacy.en import English
import multiprocessing
from gensim.models import word2vec as w2v
import numpy as np
import _pickle as pickle
from itertools import chain
import keras

# what file are we working on
# samllsample or sample-1M
filename = 'mediumsample'


# Step 1 - load data
# 1.1 load json file
temp = open('C:\\Users\\Rey\\Projects\\PersonalDevelopment\\Tutorials\\RNN\\TextSummerization\\%s.json' %filename)
json_data = json.load(temp)
temp.close()
heads = [item['head'] for item in json_data]
heads = [h.lower() for h in heads]
texts = [item['desc'] for item in json_data]
texts = [h.lower() for h in texts]
# split sentences to words
heads_sentences= [headline.split() for headline in heads]
texts_sentences= [t.split() for t in texts]


# Step 2 - tokenize
lst = heads+texts
vocabcount = Counter(w for txt in lst for w in txt.split())
vocab = list(map(lambda x: x[0], sorted(vocabcount.items(), key=lambda x: -x[1])))

# Step 3 - index words
empty = 0 # RNN mask of no data
eos = 1  # end of sentence
start_idx = eos+1 # first real word
word2idx = dict((word, idx+start_idx) for idx,word in enumerate(vocab))
word2idx['<empty>'] = empty
word2idx['<eos>'] = eos
idx2word = dict((idx,word) for word,idx in word2idx.items())
# save sentences as sequences of indexed words
Y = [[word2idx[token] for token in headline.split()] for headline in heads]
X = [[word2idx[token] for token in t.split()] for t in texts]

# Step 4 - parse sentences using spaCy
parser = English()
multiSentence = '. '.join(heads+texts)
parsedData = parser(multiSentence)

# now some words already have vector represenataions
# Step 5 - build vector representations for all words
# 5.1 setup parameters for model
embedding_dim = 300
min_word_count = 3
num_workers = multiprocessing.cpu_count()
context_size = 7
downsampling = 1e-3
seed = 1
# 5.2 define model
articles2vec = w2v.Word2Vec(
    sg=1,
    seed=seed,
    workers=num_workers,
    size=embedding_dim,
    min_count=min_word_count,
    window=context_size,
    sample=downsampling
)
# 5.3 create embeddings
articles2vec.build_vocab(heads_sentences + texts_sentences)
articles2vec.train(heads_sentences + texts_sentences,
        total_examples=articles2vec.corpus_count, epochs=5)
# save trained model
articles2vec.save('C:\\Users\\Rey\\Projects\\PersonalDevelopment\\Tutorials\\RNN\\TextSummerization\\%s.w2v'%filename)
# save most common words in and embedding matrix
vocab_size = min(40000, len(articles2vec.wv.vocab))
embedding_shape = (vocab_size, embedding_dim)
embedding = np.empty(embedding_shape)
for i in range(2, vocab_size):
    embedding[i, :] = articles2vec.wv.word_vec(idx2word[i])
# save embeddings
with open('C:\\Users\\Rey\\Projects\\PersonalDevelopment\\Tutorials\\RNN\\TextSummerization\\%s.pkl'%filename,'wb') as fp:
    pickle.dump((embedding, idx2word, word2idx),fp,-1)
with open('C:\\Users\\Rey\\Projects\\PersonalDevelopment\\Tutorials\\RNN\\TextSummerization\\%s.pkl'%filename,'wb') as fp:
    pickle.dump((X,Y),fp,-1)



# Step 6 -
# continue from here
# https://github.com/llSourcell/How_to_make_a_text_summarizer/blob/master/train.ipynb