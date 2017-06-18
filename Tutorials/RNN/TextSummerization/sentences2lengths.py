# toy example for an RNN that takes sequence as input and a list of numbers as output

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
import _pickle as pickle
import numpy as np


# what file are we working on
# the code to produce this is under EnderNLP folder
with open('C:\\Users\\Rey\\Projects\\PersonalDevelopment\\Tutorials\\RNN\\TextSummerization\\EnderSentences.pkl',
          'rb') as fp:
    idx2word, word2idx, embedding, sentences, sequences, numeric_feature = pickle.load(fp)

# get some parameters and arrange data
vocab_size, embedding_dim = embedding.shape
n_inputs = len(sequences)
maxlen = np.max(numeric_feature)
shape_X = [n_inputs, maxlen]
X = np.zeros(shape_X)
for i in range(n_inputs):
    X[i,:len(sequences[i])] = sequences[i]

# split to train and test sets
seed = 42
n_test = int(np.rint(n_inputs*0.25))
n_train = n_inputs - n_test
Y = numeric_feature
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=n_test, random_state=seed)

# build model
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim,
                    weights=[embedding],
                    embeddings_regularizer=None,
                    activity_regularizer=None, embeddings_constraint=None,
                    mask_zero=False, input_length=None))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=16, epochs=10)
predictions = model.predict(X_test)
print(np.transpose(predictions[-10:]))
print(np.transpose(Y_test[-10:]))

score = model.evaluate(X_test, Y_test, batch_size=16)







