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

# the output_shape from previous function is 3D, but no need to reshape
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
    optimizer=odin.optimizers.adadelta)
f_train = T.function(
    inputs=f.input_var + f.output_var,
    outputs=cost,
    updates=updates)

# if you don't specify th eoptimizer only calculate the objectives
print('\nBuilding monitoring functions ...')
monitor_cost = f.get_cost(odin.objectives.mean_categorical_accuracy)
f_monitor = T.function(
    inputs=f.input_var + f.output_var,
    outputs=monitor_cost)

# ====== Start the trianing process ====== #
cost = []


def batch_start(t):
    X = t.data[0].reshape(-1, 1, 28, 28)
    y = T.np_one_hot(t.data[1], n_classes=nb_classes)
    t.data = (X, y)


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
trainer.add_data('train', [ds['X_train'], ds['y_train']])
trainer.add_data('valid', [ds['X_valid'], ds['y_valid']])
trainer.add_task('train', f_train, data='train', epoch=nb_epoch,
    bs=batch_size, shuffle=True, mode=1)
trainer.add_subtask('valid', f_monitor, data='valid', single_run=False, freq=0.5,
    bs=batch_size, shuffle=False, mode=0)
trainer.run(progress=True)

# ====== Visualize the prediction ====== #
f_pred = T.function(
    inputs=f.input_var,
    outputs=f()[0]) # remeber function always return a list of outputs
x = ds['X_train'][:16]
y = np.argmax(f_pred(x.reshape(-1, 1, 28, 28)), axis=-1).tolist()
odin.visual.plot_images(x, title=','.join([str(i) for i in y]))
odin.visual.plot_show()
# odin.visual.plot_save('tmp.pdf') # if you want to save figure to pdf file
