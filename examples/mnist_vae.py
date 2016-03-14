from __future__ import print_function, division
import numpy as np
import os
os.environ['ODIN'] = 'theano,float32,cpu,verbose'
import odin
from odin import tensor as T

ds = odin.dataset.load_mnist()
batch_size = 128
epoch = 5

# ====== Weights is shared between encoder and decoder ====== #
W = T.variable(T.np_glorot_uniform(shape=(784, 512)))
WT = T.transpose(W)

encoder = odin.nnet.Dense(odin.nnet.Flatten((None, 28, 28), 2), W=W,
    num_units=512,
    nonlinearity=T.sigmoid, name='encoder')

decoder = odin.nnet.Dense((None, 512), W=WT,
    num_units=784,
    nonlinearity=T.sigmoid, name='decoder')

vae = odin.nnet.VariationalEncoderDecoder(encoder, decoder,
    nonlinearity=T.sigmoid)
print('Params:', vae.get_params(True))

# ====== Create optimization ====== #
cost, updates = vae.get_optimization(
    objective=odin.objectives.categorical_crossentropy,
    optimizer=odin.optimizers.rmsprop,
    globals=True,
    training=True
)
f_train = T.function(
    inputs=vae.input_var + vae.output_var,
    outputs=cost,
    updates=updates)
f_pred = T.function(
    inputs=vae.input_var,
    outputs=vae.set_reconstruction_mode(True)(False)[0])
# ====== Training ====== #
cost = []
niter_train = ds['X_train'].iter_len() // batch_size
for _ in range(epoch):
    # make sure iteration has the same seed, so the order of X and y is
    # the same
    seed = T.get_random_magic_seed()
    for i, x in enumerate(ds['X_train'].iter(batch_size, shuffle=True, seed=seed)):
        # reshape to match input_shape and output_shape
        cost.append(f_train(x))
        odin.logger.progress(i, niter_train, 'Train:%.4f' % cost[-1])
    # fastest plot in terminal to monitor training costs
    print()
    odin.visual.print_bar(cost, bincount=20)
    # ====== Visual results ====== #
    w = vae.get_params_value(True)[0].T.reshape(-1, 28, 28)
    x = ds['X_valid'][:16]
    x_ = f_pred(x).reshape(-1, 28, 28)
    odin.visual.plot_images([w, x, x_],
        title=['Weights', 'Original', 'Reconstructed'])
    odin.visual.plot_save('/Users/trungnt13/tmp/vae.pdf', dpi=600)
