from __future__ import print_function, division

import os
os.environ['ODIN'] = 'float32,theano,verbose'
import odin
from odin import tensor as T
import numpy
from matplotlib import pyplot as plt

ds = odin.dataset.load_mnist()
print(ds)

def test_rbm():
    persistent_chain = T.variable(numpy.zeros((20, 500)))
    rbm = odin.funcs.RBM((None, 28, 28), 500, persistent=persistent_chain)
    print(rbm.get_params(True))
    sgd = lambda x, y: odin.optimizers.sgd(x, y, learning_rate=0.01)
    cost, updates = rbm.get_optimization(optimizer=sgd)
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
        cost.append(train_rbm(x))
        odin.logger.progress(i, niter, title='%.5f' % cost[-1])
    odin.visual.print_bar(cost, bincount=20)

    ([vis_mfc, vis_samples], updates) = rbm(gibbs_steps=100)
    print('Building functions...')
    sample_rbm = T.function(
        inputs=rbm.input_var,
        outputs=[vis_mfc, vis_samples],
        updates=updates)

    test_x = ds['X_test'].value
    for i in xrange(3):
        t = numpy.random.randint(test_x.shape[0] - 20)

        x_mean, x_sample = sample_rbm(test_x[t:t + 20])
        x_mean = x_mean.reshape(-1, 28, 28)
        x_sample = x_sample.reshape(-1, 28, 28)

        odin.visual.plot_images(test_x[:20])
        odin.visual.plot_images(x_mean)
        odin.visual.plot_images(x_sample)
        plt.show(block=False)
        raw_input('<Enter>')
        plt.close('all')

# test_rbm()
