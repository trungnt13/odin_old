from __future__ import print_function, division

import os
os.environ['ODIN'] = 'float32,theano,verbose,cpu,graphic'
import odin
from odin import tensor as T
import numpy as np
from matplotlib import pyplot as plt

ds = odin.dataset.load_mnist()
print(ds)


def test_dA(): # AutoEncoder
    dA = odin.nnet.AutoEncoder((None, 28, 28),
        num_units=512, denoising=0.3, contractive=False,
        nonlinearity=T.sigmoid)
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
        outputs=dA.set_reconstruction_mode(True)())
    for i in xrange(3):
        t = np.random.randint(ds['X_test'].shape[0] - 16)
        X = ds['X_test'][t:t + 16]
        X_pred = f_pred(X)[0]
        odin.visual.plot_images(X)
        odin.visual.plot_images(X_pred)
        odin.visual.plot_show()


def test_aED(): #AutoEncoderDecoder
    Wa = T.variable(T.np_glorot_uniform(shape=(784, 256)), name='W')
    Wb = T.variable(T.np_glorot_uniform(shape=(256, 128)), name='W')

    d1a = odin.nnet.Dense((None, 28, 28), num_units=256, W=Wa, name='d1a',
        nonlinearity=T.sigmoid)
    d1a = odin.nnet.Flatten(d1a, 2)
    d1b = odin.nnet.Dense(d1a, num_units=128, W=Wb, name='d1b',
        nonlinearity=T.sigmoid)

    # or d1b, (None, 128) as incoming
    d2a = odin.nnet.Dense((None, 128), num_units=256, W=Wb.T, name='d2a',
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
        outputs=aED.set_reconstruction_mode(True)())

    for i in xrange(3):
        t = np.random.randint(ds['X_test'].shape[0] - 16)
        X = ds['X_test'][t:t + 16]
        X_pred = f_pred(X)[0].reshape(-1, 28, 28)
        odin.visual.plot_images(X)
        odin.visual.plot_images(X_pred)
        odin.visual.plot_show()

    # ====== OUtput hidden activation ====== #
    f_pred = T.function(
        inputs=aED.input_var,
        outputs=aED())
    X = ds['X_test'][t:t + 16]
    print(f_pred(X)[0].shape)


def test_vae():
    ds = odin.dataset.load_mnist()

    W = T.variable(T.np_glorot_uniform(shape=(784, 512)), name='W')
    WT = T.transpose(W)
    encoder = odin.nnet.Dense((None, 28, 28), num_units=512, W=W, name='encoder')
    encoder = odin.nnet.Flatten(encoder, 2)
    # decoder = odin.nnet.Dense((None, 256), num_units=512, name='decoder1')
    decoder = odin.nnet.Dense((None, 512), num_units=784, W=WT, name='decoder2')

    vae = odin.nnet.VariationalEncoderDecoder(encoder=encoder, decoder=decoder,
        prior_logsigma=1.7, batch_size=64)

    # ====== prediction ====== #
    x = ds['X_train'][:16]

    f = T.function(inputs=vae.input_var, outputs=vae(training=False))
    print("Predictions:", f(x)[0].shape)

    f = T.function(
        inputs=vae.input_var,
        outputs=vae.set_reconstruction_mode(True)(training=False))
    y = f(x)[0].reshape(-1, 28, 28)
    print("Predictions:", y.shape)

    odin.visual.plot_images(x)
    odin.visual.plot_images(y)
    odin.visual.plot_show()

    print('Params:', [p.name for p in vae.get_params(False)])
    print('Params(globals):', [p.name for p in vae.get_params(True)])
    # ====== Optimizer ====== #
    cost, updates = vae.get_optimization(
        objective=odin.objectives.categorical_crossentropy,
        optimizer=lambda x, y: odin.optimizers.sgd(x, y, learning_rate=0.01),
        globals=True)

    f = T.function(inputs=vae.input_var, outputs=cost, updates=updates)
    cost = []
    niter = ds['X_train'].iter_len() / 64
    for j in xrange(2):
        for i, x in enumerate(ds['X_train'].iter(64)):
            if x.shape[0] != 64: continue
            cost.append(f(x))
            odin.logger.progress(i, niter, str(cost[-1]))
    odin.visual.print_bar(cost)

    # ====== reconstruc ====== #
    f = T.function(
        inputs=vae.input_var,
        outputs=vae.set_reconstruction_mode(True)(training=False))
    X_test = ds['X_test'][:16]
    X_reco = f(X_test)[0].reshape(-1, 28, 28)
    odin.visual.plot_images(X_test)
    odin.visual.plot_images(X_reco)
    odin.visual.plot_show()

test_dA()
test_aED()
test_vae()
