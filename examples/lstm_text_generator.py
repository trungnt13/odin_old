''' Replicate example from keras
Original work Copyright (c) 2014-2015 keras contributors
Modified work Copyright 2016-2017 TrungNT
-------
Example script to generate text from shakespear's writings.

At least 20 epochs are required before the generated text
starts sounding coherent.

It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.
'''

from __future__ import print_function, division

from six.moves import zip, range
import numpy as np
import random
import sys

import os
os.environ['ODIN'] = 'theano,float32,verbose30,cpu'
import odin
from odin import nnet
from odin import tensor as T


# ===========================================================================
# Helper function
# ===========================================================================
def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

# ===========================================================================
# Test
# ===========================================================================
path = odin.ie.get_file('shakespear.txt',
        origin='https://s3.amazonaws.com/ai-datasets/shakespear.txt')
text = open(path).read()
chars = set(text)
print('total chars:', len(chars))
print('text length:', len(text))

char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 20
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    # for example, for every step=3 chars, predict the next char given
    # maxlen=20 previous chars.
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
print('X shape:', X.shape)
print('y shape:', y.shape)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

# build the model: 2 stacked LSTM
print('Build model...')
f = odin.nnet.LSTM((None, maxlen, len(chars)), num_units=512)
f = odin.nnet.Dropout(f, p=0.2)
f = odin.nnet.LSTM(f, num_units=512, only_return_final=True)
f = odin.nnet.Dense(f, num_units=len(chars), nonlinearity=T.softmax)

print('Input variables:', f.input_var)
print('Output variables:', f.output_var)
cost, updates = f.get_optimization(
    objective=odin.objectives.mean_categorical_crossentropy,
    optimizer=odin.optimizers.rmsprop)
print('Build training function...')
f_train = T.function(
    inputs=f.input_var + f.output_var,
    outputs=cost,
    updates=updates)
f_pred = T.function(
    inputs=f.input_var,
    outputs=f(training=False)[0]
)

# train the model, output generated text after each iteration
for iteration in range(1, 60):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    seed = T.get_random_magic_seed()
    cost = []
    niter = X.shape[0] // 128
    for k, (i, j) in enumerate(
        zip(odin.batch(arrays=X).iter(128, shuffle=True, seed=seed),
            odin.batch(arrays=y).iter(128, shuffle=True, seed=seed))):
        cost.append(f_train(i, j))
        odin.logger.progress(k, niter, title='Cost:%.5f' % cost[-1])
    print()
    odin.visual.print_bar(cost, bincount=20)
    print()

    start_index = random.randint(0, len(text) - maxlen - 1)
    for diversity in [0.2, 1.0, 1.8]:
        print()
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.

            preds = f_pred(x).ravel()
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()
