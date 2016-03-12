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

# ====== The encoder ====== #
f1 = odin.nnet.Conv2D((None, 1, 28, 28),
    num_filters=nb_filters,
    filter_size=(nb_conv, nb_conv),
    pad='valid', nonlinearity=T.sigmoid)
f2 = odin.nnet.Conv2D(f1,
    num_filters=nb_filters,
    filter_size=(nb_conv, nb_conv),
    nonlinearity=T.sigmoid)
f3 = odin.nnet.MaxPool2D(f2,
    pool_size=(nb_pool, nb_pool))
f4 = odin.nnet.Dropout(f3, p=0.25, rescale=True)
f5 = odin.nnet.Dense(f4,
    num_units=128,
    nonlinearity=T.sigmoid)

# ====== The decoder ====== #
f5_de = f5.get_inv(None)
f4_de = odin.nnet.Ops(
    odin.nnet.Dropout(f5_de, p=0.25, rescale=True),
    ops=lambda x: T.reshape(x, shape=f4.output_shape[0]))
f3_de = odin.nnet.UpSampling2D(f4_de,
    scale_factor=(nb_pool, nb_pool))
f2_de = f2.get_inv(f3_de)
f1_de = f1.get_inv(f2_de)

# ====== The AutoEncoderDecoder ====== #
print('Encoder input_shape:', f1.input_shape)
print('Dencoder output_shape:', f1_de.output_shape)
ae = odin.nnet.AutoEncoderDecoder(encoder=f5, decoder=f1_de)

# ====== create cost and updates based on given objectives and optimizers ====== #
print('\nBuilding training functions ...')
cost, updates = ae.get_optimization(
    objective=odin.objectives.categorical_crossentropy,
    optimizer=odin.optimizers.adadelta)
f_train = T.function(
    inputs=ae.input_var,
    outputs=cost,
    updates=updates)

# if you don't specify th eoptimizer only calculate the objectives
print('\nBuilding monitoring functions ...')
monitor_cost = ae.get_cost(odin.objectives.mean_categorical_crossentropy)
f_monitor = T.function(
    inputs=ae.input_var,
    outputs=monitor_cost)

# ====== Start the trianing process ====== #
cost = []


def batch_start(t):
    X = t.data[0].reshape(-1, 1, 28, 28)
    t.data = (X,)


def batch_end(t):
    if t.task == 'train':
        cost.append(np.mean(t.output))


def epoch_end(t):
    global cost
    if t.task == 'train':
        odin.visual.print_bar(cost, bincount=20)
        cost = []
    elif t.task == 'valid':
        print()
        print('Validation accuracy:', np.mean(t.output))


trainer = odin.trainer()
trainer.set_callback(
    batch_start=batch_start, batch_end=batch_end, epoch_end=epoch_end)

trainer.add_data('train', [ds['X_train']])
trainer.add_data('valid', [ds['X_valid']])

trainer.add_task('train', f_train, data='train', epoch=nb_epoch,
    bs=batch_size, shuffle=True, mode=1)
trainer.add_subtask('valid', f_monitor, data='valid', single_run=False, freq=1.0,
    bs=batch_size, shuffle=False, mode=0)
trainer.run(progress=True)

# ====== Visualize the prediction ====== #
reconstruction = ae.set_reconstruction_mode(True)(False)
f_pred = T.function(
    inputs=ae.input_var,
    outputs=reconstruction) # remeber function always return a list of outputs
x = ds['X_train'][:16]
y = f_pred(x.reshape(-1, 1, 28, 28))[0].reshape(-1, 28, 28)

odin.visual.plot_images(x)
odin.visual.plot_images(y)
odin.visual.plot_show()
# odin.visual.plot_save('tmp.pdf') # if you want to save figure to pdf file
