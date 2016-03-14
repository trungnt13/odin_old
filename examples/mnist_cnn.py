''' Replicate example from keras
Original work Copyright (c) 2014-2015 keras contributors
Modified work Copyright 2016-2017 TrungNT
-------
Train a simple convnet on the MNIST dataset.

Get to 99.25% test accuracy after 12 epochs (there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function, division
import numpy as np

from six.moves import range, zip
import os
# if you want to configure the library, add this line before any odin import
os.environ['ODIN'] = 'verbose30,theano,gpu,graphic'
# if you want to see the full footprint of nested funtions, set only: verbose
import odin
from odin import tensor as T

odin.tensor.set_magic_seed(1337) # for reproducibility

batch_size = 128
nb_classes = 10
nb_epoch = 12

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

# the data, shuffled and split between tran and test sets
ds = odin.dataset.load_mnist()
# print summary about the dataset
print(ds)

f = odin.nnet.Conv2D((None, 1, 28, 28),
    num_filters=nb_filters,
    filter_size=(nb_conv, nb_conv),
    pad='valid', nonlinearity=T.relu)
f = odin.nnet.Conv2D(f,
    num_filters=nb_filters,
    filter_size=(nb_conv, nb_conv),
    nonlinearity=T.relu)
f = odin.nnet.MaxPool2D(f,
    pool_size=(nb_pool, nb_pool))
f = odin.nnet.Dropout(f, p=0.25, rescale=True)

f = odin.nnet.Flatten(f, 2)
f = odin.nnet.Dense(f,
    num_units=128,
    nonlinearity=T.relu)
f = odin.nnet.Dropout(f, p=0.5)
f = odin.nnet.Dense(f, num_units=nb_classes, nonlinearity=T.softmax)

print('Input variables:', f.input_var)
print('Outputs variables:', f.output_var)
# ====== create cost and updates based on given objectives and optimizers ====== #
print('\nBuilding training functions ...')
cost, updates = f.get_optimization(
    objective=odin.objectives.categorical_crossentropy,
    optimizer=odin.optimizers.adadelta,
    training=True)
f_train = T.function(
    inputs=f.input_var + f.output_var,
    outputs=cost,
    updates=updates)

# if you don't specify th eoptimizer only calculate the objectives
print('\nBuilding monitoring functions ...')
monitor_cost, _ = f.get_optimization(
    objective=odin.objectives.mean_categorical_accuracy,
    optimizer=None,
    training=False)
f_monitor = T.function(
    inputs=f.input_var + f.output_var,
    outputs=monitor_cost)

# ====== Start the trianing process ====== #
cost = []
niter_train = ds['X_train'].iter_len() // batch_size
niter_valid = ds['X_valid'].iter_len() // batch_size
for _ in range(nb_epoch):
    # make sure iteration has the same seed, so the order of X and y is
    # the same
    seed = T.get_random_magic_seed()
    for i, (x, y) in enumerate(zip(
        ds['X_train'].iter(batch_size, shuffle=True, seed=seed),
        ds['y_train'].iter(batch_size, shuffle=True, seed=seed))):
        # reshape to match input_shape and output_shape
        x = x.reshape(-1, 1, img_rows, img_cols)
        y = T.np_one_hot(y, n_classes=nb_classes)
        cost.append(f_train(x, y))
        odin.logger.progress(i, niter_train, 'Train:%.4f' % cost[-1], idx='train')
    # fastest plot in terminal to monitor training costs
    print()
    odin.visual.print_bar(cost, bincount=20)

    # validation process
    valid_cost = []
    for i, (x, y) in enumerate(zip(
        ds['X_valid'].iter(batch_size, shuffle=False),
        ds['y_valid'].iter(batch_size, shuffle=False))):
        # reshape to match input_shape and output_shape
        x = x.reshape(-1, 1, img_rows, img_cols)
        y = T.np_one_hot(y, n_classes=nb_classes)
        valid_cost.append(f_monitor(x, y))
        odin.logger.progress(i, niter_valid, 'Valid:%.4f' % np.mean(valid_cost), idx='valid')
    print()
    print('Validation accuracy:', np.mean(valid_cost))
    print()

# ====== Visualize the prediction ====== #
f_pred = T.function(
    inputs=f.input_var,
    outputs=f()[0]) # remeber function always return a list of outputs
x = ds['X_train'][:16]
y = np.argmax(f_pred(x.reshape(-1, 1, 28, 28)), axis=-1).tolist()
odin.visual.plot_images(x, title=','.join([str(i) for i in y]))
odin.visual.plot_show()
# odin.visual.plot_save('tmp.pdf') # if you want to save figure to pdf file
