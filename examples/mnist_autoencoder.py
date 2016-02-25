from __future__ import print_function, division

import os
os.environ['ODIN'] = 'float32,theano,verbose'
import odin
from odin import tensor as T
import numpy as np
from matplotlib import pyplot as plt

ds = odin.dataset.load_mnist()
print(ds)

def test_dA(): # AutoEncoder
    dA = odin.nnet.AutoEncoder((None, 28, 28), num_units=512, denoising=0.3)
    sgd = lambda x, y: odin.optimizers.sgd(x, y, learning_rate=0.01)
    cost, updates = dA.get_optimization(
        objective=odin.objectives.categorical_crossentropy,
        optimizer=sgd,
        globals=True)
    f_train = T.function(
        inputs=dA.input_var,
        outputs=cost,
        updates=updates)

    cost = []
    niter = ds['X_train'].iter_len() / 64
    choices = None
    for _ in xrange(3):
        for i, x in enumerate(ds['X_train'].iter(64)):
            cost.append(f_train(x))
            odin.logger.progress(i, niter, title=str(cost[-1]))
        print()
        odin.visual.print_bar(cost, bincount=20)
        W = T.get_value(dA.get_params(False)[0]).T.reshape(-1, 28, 28)
        if choices is None:
            choices = np.random.choice(
                np.arange(W.shape[0]), size=16, replace=False)
        W = W[choices]
        odin.visual.plot_images(W)
        plt.show(block=False)
        raw_input('<enter>')

    f_pred = T.function(
        inputs=dA.input_var,
        outputs=dA(reconstructed=True))
    for i in xrange(3):
        t = np.random.randint(ds['X_test'].shape[0] - 16)
        X = ds['X_test'][t:t + 16]
        X_pred = f_pred(X)[0]
        odin.visual.plot_images(X)
        odin.visual.plot_images(X_pred)
        plt.show(block=False)
        raw_input('<enter>')
        plt.close('all')

def test_aED(): #AutoEncoderDecoder
    Wa = T.variable(T.np_glorot_uniform(shape=(784, 256)), name='W')
    Wb = T.variable(T.np_glorot_uniform(shape=(256, 128)), name='W')

    d1a = odin.nnet.Dense((None, 28, 28), num_units=256, W=Wa, name='d1a',
        nonlinearity=T.sigmoid)
    d1b = odin.nnet.Dense(d1a, num_units=128, W=Wb, name='d1b',
        nonlinearity=T.sigmoid)

    d2a = odin.nnet.Dense(d1b, num_units=256, W=Wb.T, name='d2a',
        nonlinearity=T.sigmoid)
    d2b = odin.nnet.Dense(d2a, num_units=784, W=Wa.T, name='d2b',
        nonlinearity=T.sigmoid)

    aED = odin.nnet.AutoEncoderDecoder(d1b, d2b)
    sgd = lambda x, y: odin.optimizers.sgd(x, y, learning_rate=0.01)
    cost, updates = aED.get_optimization(
        objective=odin.objectives.categorical_crossentropy,
        optimizer=sgd,
        globals=True)
    f_train = T.function(
        inputs=aED.input_var,
        outputs=cost,
        updates=updates)

    cost = []
    niter = ds['X_train'].iter_len() / 64
    choices = None
    for _ in xrange(3):
        for i, x in enumerate(ds['X_train'].iter(64)):
            cost.append(f_train(x))
            odin.logger.progress(i, niter, title=str(cost[-1]))
        print()
        odin.visual.print_bar([i for i in cost if i == i], bincount=20)
        W = T.get_value(aED.get_params(True)[0]).T.reshape(-1, 28, 28)
        if choices is None:
            choices = np.random.choice(
                np.arange(W.shape[0]), size=16, replace=False)
        W = W[choices]
        odin.visual.plot_images(W)
        plt.show(block=False)
        raw_input('<enter>')

    # ====== Output reconstruction ====== #
    f_pred = T.function(
        inputs=aED.input_var,
        outputs=aED(reconstructed=True))

    for i in xrange(3):
        t = np.random.randint(ds['X_test'].shape[0] - 16)
        X = ds['X_test'][t:t + 16]
        X_pred = f_pred(X)[0].reshape(-1, 28, 28)
        odin.visual.plot_images(X)
        odin.visual.plot_images(X_pred)
        plt.show(block=False)
        raw_input('<enter>')
        plt.close('all')

    # ====== OUtput hidden activation ====== #
    f_pred = T.function(
        inputs=aED.input_var,
        outputs=aED())
    X = ds['X_test'][t:t + 16]
    print(f_pred(X)[0].shape)

# test_dA()
test_aED()
