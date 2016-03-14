from __future__ import print_function, division

import os
os.environ['ODIN'] = 'float32,theano,verbose,graphic'
import odin
from odin import tensor as T
import numpy
from matplotlib import pyplot as plt

ds = odin.dataset.load_mnist()
print(ds)


def test_rbm():
    batch_size = 32
    persistent_chain = T.variable(numpy.zeros((batch_size, 500)))
    input_ = odin.nnet.Flatten((None, 28, 28), 2)
    input_ = odin.nnet.Dense(input_, num_units=784)
    input_ = (None, 28, 28)
    rbm = odin.nnet.RBM(input_, 500, persistent=persistent_chain)
    print('Input variables:', rbm.input_var)
    print('Output variables:', rbm.output_var)

    sgd = lambda x, y: odin.optimizers.sgd(x, y, learning_rate=0.01)
    cost, updates = rbm.get_optimization(
        optimizer=sgd, globals=True, objective=odin.objectives.contrastive_divergence)
    print('Building functions...')
    train_rbm = T.function(
        inputs=rbm.input_var,
        outputs=cost,
        updates=updates
    )

    cost = []
    niter = ds['X_train'].iter_len() / batch_size
    for i, x in enumerate(ds['X_train'].iter(batch_size, seed=13)):
        if x.shape[0] != batch_size: continue
        # x = x.astype(int) # this one can mess up the whole training process
        cost.append(train_rbm(x))
        odin.logger.progress(i, niter, title='%.5f' % cost[-1])
    odin.visual.print_bar(cost, bincount=20)

    vis_mfc = rbm.set_sampling_steps(1)(reconstruction=True)
    print('Building functions...')
    sample_rbm = T.function(
        inputs=rbm.input_var,
        outputs=vis_mfc,
        updates=updates)

    test_x = ds['X_test'].value
    for i in xrange(3):
        t = numpy.random.randint(test_x.shape[0] - 16)
        x = test_x[t:t + 16]

        x_mean = sample_rbm(x)[0]
        odin.visual.plot_images(x)
        odin.visual.plot_images(x_mean)
        plt.show(block=False)
        raw_input('<Enter>')
        plt.close('all')


def test_rbm_multiple():
    persistent_chain = T.variable(numpy.zeros((20, 500)))
    input_ = odin.nnet.Flatten((None, 28, 28), 2)
    input_ = odin.nnet.Dense(input_, num_units=784)
    input_ = [(None, 28, 28), (None, 28, 28)]
    rbm = odin.nnet.RBM(input_, 500, persistent=persistent_chain)
    print('Input variables:', rbm.input_var)
    print('Output variables:', rbm.output_var)

    sgd = lambda x, y: odin.optimizers.sgd(x, y, learning_rate=0.01)
    cost, updates = rbm.get_optimization(optimizer=sgd, globals=True, objective=odin.objectives.contrastive_divergence)
    print('Building functions...')
    train_rbm = T.function(
        inputs=rbm.input_var,
        outputs=cost,
        updates=updates
    )

    cost = []
    batch_size = 64
    niter = ds['X_train'].iter_len() / batch_size
    for i, x in enumerate(ds['X_train'].iter(batch_size, seed=13)):
        # x = x.astype(int) # this one can mess up the whole training process
        cost.append(train_rbm(x, x))
        odin.logger.progress(i, niter, title='%.5f' % cost[-1])
    odin.visual.print_bar(cost, bincount=20)

    vis_mfc = rbm.set_sampling_steps(1)(reconstruction=True)
    print('Building functions...')
    sample_rbm = T.function(
        inputs=rbm.input_var,
        outputs=vis_mfc,
        updates=updates)

    test_x = ds['X_test'].value
    for i in xrange(3):
        t = numpy.random.randint(test_x.shape[0] - 16)
        x = test_x[t:t + 16]

        x_mean = sample_rbm(x, x)[0]
        odin.visual.plot_images(x)
        odin.visual.plot_images(x_mean)
        plt.show(block=False)
        raw_input('<Enter>')
        plt.close('all')

test_rbm()
# test_rbm_multiple()
