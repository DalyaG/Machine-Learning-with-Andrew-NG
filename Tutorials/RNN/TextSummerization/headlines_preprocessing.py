# toy example for an RNN that takes sequence as input and a list of numbers as output

import json
from collections import Counter
import multiprocessing
from gensim.models import word2vec as w2v
import numpy as np
import _pickle as pickle

# what file are we working on
# samllsample or mediumsample
filename = 'mediumsample'
preprocess = False

# part A - preprocess input and output
if preprocess:
    # Step 1 - load data
    # 1.1 load json file
    temp = open('C:\\Users\\Rey\\Projects\\PersonalDevelopment\\Tutorials\\RNN\\TextSummerization\\%s.json' %filename)
    json_data = json.load(temp)
    temp.close()
    heads = [item['head'] for item in json_data]
    heads = [h.lower() for h in heads]
    # split sentences to words
    heads_sentences= [headline.split() for headline in heads]
    # extract some random numeric features
    numeric_feature = [len(h) for h in heads_sentences]
    #
    # Step 2 - tokenize
    vocabcount = Counter(w for txt in heads for w in txt.split())
    vocab = list(map(lambda x: x[0], sorted(vocabcount.items(), key=lambda x: -x[1])))
    #
    # Step 3 - index words
    empty = 0 # RNN mask of no data
    word2idx = dict((word, idx+1) for idx,word in enumerate(vocab))
    word2idx['<empty>'] = empty
    idx2word = dict((idx,word) for word,idx in word2idx.items())
    # save sentences as sequences of indexed words
    X = [[word2idx[token] for token in headline.split()] for headline in heads]
    Y = numeric_feature
    #
    # Step 4 - build vector representations for all words
    # 4.1 setup parameters for model
    embedding_dim = 100
    min_word_count = 3
    num_workers = multiprocessing.cpu_count()
    context_size = 5
    downsampling = 1e-3
    seed = 1
    # 4.2 define model
    headlines2vec = w2v.Word2Vec(
        sg=1,
        seed=seed,
        workers=num_workers,
        size=embedding_dim,
        min_count=min_word_count,
        window=context_size,
        sample=downsampling
    )
    # 4.3 create embeddings
    headlines2vec.build_vocab(heads_sentences)
    headlines2vec.train(heads_sentences,
                        total_examples=headlines2vec.corpus_count, epochs=10)
    # save trained model
    headlines2vec.save('C:\\Users\\Rey\\Projects\\PersonalDevelopment\\Tutorials\\RNN\\TextSummerization\\%sTrained.w2v'%filename)
    # save most common words in and embedding matrix
    vocab_size = min(10000, len(headlines2vec.wv.vocab))
    embedding_shape = (vocab_size, embedding_dim)
    embedding = np.zeros(embedding_shape)
    for i in range(1, vocab_size):
        embedding[i, :] = headlines2vec.wv.word_vec(idx2word[i])
    # save embeddings
    with open('C:\\Users\\Rey\\Projects\\PersonalDevelopment\\Tutorials\\RNN\\TextSummerization\\%sembedding.pkl'%filename,'wb') as fp:
        pickle.dump((embedding, idx2word, word2idx),fp,-1)
    # save input and output
    with open('C:\\Users\\Rey\\Projects\\PersonalDevelopment\\Tutorials\\RNN\\TextSummerization\\%sXY.pkl'%filename,'wb') as fp:
        pickle.dump((X,Y),fp,-1)
else:
    with open('C:\\Users\\Rey\\Projects\\PersonalDevelopment\\Tutorials\\RNN\\TextSummerization\\%sembedding.pkl'%filename, 'rb') as fp:
        embedding, idx2word, word2idx = pickle.load(fp)
        vocab_size, embedding_size = embedding.shape
    with open('C:\\Users\\Rey\\Projects\\PersonalDevelopment\\Tutorials\\RNN\\TextSummerization\\%sXY.pkl'%filename, 'rb') as fp:
        X, Y = pickle.load(fp)
