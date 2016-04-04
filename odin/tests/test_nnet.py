# ======================================================================
# Author: TrungNT
# ======================================================================
from __future__ import print_function, division

from .. import nnet
from .. import tensor as T
from .. import logger
from .. import objectives
from .. import optimizers
from .. import config

import unittest

import os
import numpy as np

from six.moves import zip, range
import time


# ===========================================================================
# Main Test
# ===========================================================================
class FunctionsTest(unittest.TestCase):

    def setUp(self):
        logger.set_enable(False)

    def tearDown(self):
        logger.set_enable(True)

    def test_ops(self):
        f = nnet.Flatten((None, 8, 12, 13), outdim=2)
        f = f.get_inv(f)
        f = T.function(f.input_var, f())

        np.random.seed(1208251813)
        x = np.random.rand(128, 8, 12, 13)
        tmp = np.sum(np.abs(f(x)[0] - x))
        self.assertLessEqual(tmp, 0.005)

    def test_inv(self):
        f0 = nnet.Dense([(None, 12), (None, 12)], num_units=8)
        f1 = nnet.Dense([(None, 30), (None, 30)], num_units=8)
        f2 = nnet.Dense([f0, f1], num_units=12)
        f3 = nnet.Dense(f2, num_units=12)

        f3_de = nnet.Inverse(f3, f2)
        f = T.function(f3_de.input_var, f3_de())
        x = f(np.random.rand(16, 12), np.random.rand(20, 12),
              np.random.rand(24, 30), np.random.rand(28, 30))
        for i in x:
            self.assertEqual(i.shape[1], 8)

    def test_batch_norm(self):
        np.random.seed(12)
        X1 = np.random.rand(10, 8)
        X2 = np.random.rand(10, 8)

        b = nnet.BatchNormalization([(None, 8)])

        f_train = T.function(b.input_var,
                             outputs=b(True)[0].astype(config.floatX()))
        mean1 = b.get_params_value(True)[2]
        for i in xrange(30):
            f_train(X1)
            f_train(X2)
        mean2 = b.get_params_value(True)[2]
        diff = mean1 - mean2
        val = np.asarray([-0.41652647, -0.54181147, -0.48397034, -0.53209579,
            -0.5128513, -0.63138282, -0.45040295, -0.3329283])
        if config.floatX() == 'float16':
            self.assertLessEqual(np.sum(np.abs(diff - val)), 0.008)
        else:
            self.assertLessEqual(np.sum(np.abs(diff - val)), 0.00005)

    def test_cnn(self):
        np.random.seed(13)
        f = nnet.Conv2D((None, 3, 28, 28),
            num_filters=32, filter_size=(3, 3), stride=(1, 1), pad='same')
        f = T.function(inputs=f.input_var, outputs=f())
        self.assertEqual(f(np.random.rand(32, 3, 28, 28))[0].shape,
                        (32, 32, 28, 28))

    def test_deconv(self):
        filter_size = (3, 3)
        stride = (3, 5)
        pad = 'valid'
        # ====== test ====== #
        x = np.random.rand(16, 3, 28, 28)
        W = T.variable(np.random.rand(32, 3, 3, 3))

        f = nnet.Conv2D((None, 3, 28, 28),
            num_filters=32,
            filter_size=filter_size,
            stride=stride,
            pad=pad,
            W=W,
            b=None,
            nonlinearity=T.linear)
        f1 = nnet.Inverse(f, f)
        f1 = T.function(inputs=f1.input_var, outputs=f1())
        x1 = f1(x)[0]

        # ====== Input ====== #
        f2 = nnet.Deconv2D(f,
            img_shape=f.input_shape[0],
            filter_size=filter_size,
            stride=stride,
            pad=pad,
            W=W,
            b=None,
            nonlinearity=T.linear)
        f2 = T.function(inputs=f2.input_var, outputs=f2())
        x2 = f2(x)[0]

        self.assertEqual((np.abs(np.sum(x1 - x2))), 0.)
        self.assertEqual(x1.shape, x2.shape)

    def test_dense_func(self):
        threshold = 10e-4 if config.floatX() == 'float16' else 10e-5

        d3 = nnet.Dense((None, 10), num_units=5, nonlinearity=T.linear)
        f_pred = T.function(d3.input_var, d3())

        x = np.random.rand(16, 10)
        y = np.random.rand(16, 5)

        p = d3.get_params_value(True)[0]
        pred1 = np.round(f_pred(x)[0], 6)
        pred2 = np.round(np.dot(x, p), 6)
        self.assertLessEqual(np.sum(np.abs(pred1 - pred2)), threshold)

        # ====== only cost ====== #
        cost, _ = d3.get_optimization(
            objective=objectives.squared_loss)
        f_cost = T.function(d3.input_var + d3.output_var, cost)

        cost1 = np.round(np.mean(f_cost(x, y)), 6)
        cost2 = np.round(np.mean((np.dot(x, p) - y)**2), 6)
        self.assertLessEqual(np.sum(np.abs(cost1 - cost2)), threshold)

        # ====== optimization ====== #
        cost, updates = d3.get_optimization(
            objective=objectives.mean_squared_loss,
            optimizer=optimizers.sgd)
        f_updates = T.function(
            inputs=d3.input_var + d3.output_var,
            outputs=cost,
            updates=updates)
        cost = []
        for i in range(10):
            cost.append(f_updates(x, y))
        self.assertGreater(cost[:-1], cost[1:])

    def test_summation_merge(self):
        d1 = nnet.Dense((None, 10), num_units=5, nonlinearity=T.linear)
        d2 = nnet.Dense((None, 20), num_units=5, nonlinearity=T.linear)
        d3 = nnet.Summation((d1, d2))

        params = d3.get_params_value(True)
        p1 = params[0]
        p2 = params[2]

        f_pred = T.function(d3.input_var, d3()[0])

        x1 = np.random.rand(16, 10)
        x2 = np.random.rand(16, 20)

        pred1 = np.round(f_pred(x1, x2), 6)
        pred2 = np.round(np.dot(x1, p1) + np.dot(x2, p2), 6)

        self.assertLessEqual(np.sum(np.abs(pred1 - pred2)), 10e-5)

    def test_get_roots_and_children(self):
        d1a = nnet.Dense((None, 28, 28), num_units=256, name='d1a')
        d1b = nnet.Dense(d1a, num_units=128, name='d1b')
        d1c = nnet.Dense(d1b, num_units=128, name='d1c')
        d1d = nnet.Summation([(None, 128), d1c], name='Summation')

        self.assertEqual(d1d.input_shape, [(None, 128), (None, 28, 128)])
        self.assertEqual([T.ndim(i) for i in d1d.input_var], [2, 3])
        self.assertEqual(d1d.get_roots(), [d1d, d1a])
        self.assertEqual(d1d.get_children(include_self=False), [d1c, d1b, d1a])

    def test_noise(self):
        np.random.seed(12082518)
        x1 = np.ones((16, 5, 8))
        x2 = np.ones((16, 5, 8))
        f = nnet.Dropout([(None, 5, 8), (None, 5, 8)],
            p=0.5, rescale=True, noise_dims=1, seed=13, consistent=True)
        f = nnet.Ops(f, ops=lambda x: x + 0.)
        f = T.function(inputs=f.input_var, outputs=f(True))
        y = f(x1, x2)
        y = y[0] - y[1]
        self.assertEqual(y.ravel().tolist(), [0.] * len(y.ravel()))

        f = nnet.Noise([(None, 5, 8), (None, 5, 8)],
            sigma=0.5, noise_dims=(1, 2), uniform=True, seed=13, consistent=True)
        f = nnet.Ops(f, ops=lambda x: x + 0.)
        f = T.function(inputs=f.input_var, outputs=f(True))
        y = f(x1, x2)
        y = y[0] - y[1]
        self.assertEqual(y.ravel().tolist(), [0.] * len(y.ravel()))

    def test_function_as_weights(self):
        dense_in = nnet.Dense((10, 30), num_units=20, name='dense_in')

        dense = nnet.Dense((None, 28), num_units=10, name='dense1')
        dense = nnet.Dense(dense, num_units=20,
            W=dense_in()[0], name='dense2')
        f = T.function(
            inputs=dense.input_var + dense_in.input_var,
            outputs= dense())
        shape = f(np.random.rand(13, 28),
              np.random.rand(10, 30))[0].shape
        self.assertEqual(shape, (13, 20))

    def test_rnn(self):
        return

    def test_lstm(self):
        try:
            import lasagne
        except:
            print('\n This test require lasagne.')
            return
        np.random.seed(12082518)
        X = np.random.rand(128, 28, 13)
        hid_init = T.variable(T.np_constant((1, 12), val=1.), name='hid_init')
        cell_init = T.variable(T.np_constant((1, 12), val=2.), name='cell_init')

        W_in_forget = T.np_glorot_normal((13, 12))
        W_hid_forget = T.np_glorot_normal((12, 12))
        W_cell_forget = T.np_glorot_normal((12,))
        b_forget = T.np_constant((12,))

        W_in_input = T.np_glorot_normal((13, 12))
        W_hid_input = T.np_glorot_normal((12, 12))
        W_cell_input = T.np_glorot_normal((12,))
        b_input = T.np_constant((12,))

        W_in_cell = T.np_glorot_normal((13, 12))
        W_hid_cell = T.np_glorot_normal((12, 12))
        b_cell = T.np_constant((12,))

        W_in_output = T.np_glorot_normal((13, 12))
        W_hid_output = T.np_glorot_normal((12, 12))
        W_cell_output = T.np_glorot_normal((12,))
        b_output = T.np_constant((12,))

        forgetgate = nnet.Gate(W_in=W_in_forget, W_hid=W_hid_forget, b=b_forget,
            W_cell=W_cell_forget, nonlinearity=T.sigmoid)
        ingate = nnet.Gate(W_in=W_in_input, W_hid=W_hid_input, b=b_input,
            W_cell=W_cell_input, nonlinearity=T.sigmoid)
        cell = nnet.Gate(W_in=W_in_cell, W_hid=W_hid_cell, b=b_cell,
            W_cell=None, nonlinearity=T.tanh)
        outgate = nnet.Gate(W_in=W_in_output, W_hid=W_hid_output, b=b_output,
            W_cell=W_cell_output, nonlinearity=T.sigmoid)

        g = nnet.LSTM((None, 28, 13), hidden_info=hid_init,
            ingate=ingate,
            forgetgate=forgetgate,
            cell=cell,
            outgate=outgate,
            cell_init=cell_init,
            unroll_scan=False)
        f1 = T.function(inputs=g.input_var, outputs=g()[0])
        cost, updates = g.get_optimization(
            objective=objectives.mean_squared_loss,
            optimizer=optimizers.rmsprop,
            globals=True)

        forgetgate = lasagne.layers.Gate(W_in=W_in_forget, W_hid=W_hid_forget, b=b_forget,
            W_cell=W_cell_forget, nonlinearity=T.sigmoid)
        ingate = lasagne.layers.Gate(W_in=W_in_input, W_hid=W_hid_input, b=b_input,
            W_cell=W_cell_input, nonlinearity=T.sigmoid)
        cell = lasagne.layers.Gate(W_in=W_in_cell, W_hid=W_hid_cell, b=b_cell,
            W_cell=None, nonlinearity=T.tanh)
        outgate = lasagne.layers.Gate(W_in=W_in_output, W_hid=W_hid_output, b=b_output,
            W_cell=W_cell_output, nonlinearity=T.sigmoid)
        l_in = lasagne.layers.InputLayer(shape=(None, 28, 13))
        l = lasagne.layers.LSTMLayer(l_in, num_units=12,
            ingate=ingate,
            forgetgate=forgetgate,
            cell=cell,
            outgate=outgate,
            cell_init=cell_init,
            unroll_scan=False,
            hid_init=hid_init, learn_init=False)
        f2 = T.function(inputs=[l_in.input_var], outputs=lasagne.layers.get_output(l))

        y1 = f1(X)
        y2 = f2(X)
        self.assertEqual(y1.shape, y2.shape)
        self.assertLessEqual(np.sum(np.abs(y1 - y2)), 0.005)

        print()
        start = time.time()
        for i in xrange(12):
            f1(X)
        print('Odin LSTM speed:', (time.time() - start) / 12)
        start = time.time()
        for i in xrange(12):
            f2(X)
        print('Lasagne LSTM speed:', (time.time() - start) / 12)

    def test_gru_benchmark(self):
        try:
            import lasagne
            from keras.layers.recurrent import GRU
        except:
            print('\n This test require lasagne and keras.')
            return
        if config.backend() == 'theano': # keras will mess up with our floatX
            import theano
            theano.config.floatX = config.floatX()

        np.random.seed(12082518)
        X = np.random.rand(32, 12, 13)
        g1 = nnet.GRU((None, 12, 13), hidden_info=8,
            resetgate=nnet.Gate(),
            updategate=nnet.Gate(),
            hidden_update=nnet.Gate(nonlinearity=T.tanh),
            batch_norm=False,
            dropoutW=None, dropoutU=None)

        f1 = T.function(g1.input_var, outputs=g1(True))
        x1 = f1(X)[0]

        g2 = GRU(output_dim=8, input_shape=(12, 13),
                activation=T.tanh, inner_activation=T.sigmoid,
                dropout_W=None, dropout_U=None,
                return_sequences=True)
        g2.set_weights(g1.get_params_value(True, True))
        f2 = T.function([g2.get_input(True)],
            outputs=g2.get_output(True))
        x2 = f2(X)

        l_in = lasagne.layers.InputLayer(shape=(None, 12, 13))
        l = lasagne.layers.GRULayer(l_in, num_units=8)
        lasagne.layers.set_all_param_values(l,
            g1.get_params_value(True, True) + [T.np_constant((1, 8))])
        f3 = T.function([l_in.input_var],
            outputs=lasagne.layers.get_output(l, deterministic=False))
        x3 = f3(X)
        print()
        print('Odin - Keras:   ', np.sum(np.abs(x1 - x2)))
        print('Odin - Lasagne: ', np.sum(np.abs(x1 - x3)))
        print('Keras - Lasagne:', np.sum(np.abs(x2 - x3)))
        self.assertAlmostEqual(round(np.sum(np.abs(x1 - x3)), 3),
                               0.)
        # print(g1.get_params(True, True))
        # p1 = g1.get_params_value(True, True)
        # print(g2.get_params()[0])
        # p2 = [T.get_value(i) for i in g2.get_params()[0]]
        # print([np.sum(np.abs(i - j)) for i, j in zip(p1, p2)])

        print()
        time.sleep(1)

        start = time.time()
        for i in xrange(12):
            f2(X)
        print('Keras GRU speed:', (time.time() - start) / 12)
        time.sleep(1)

        start = time.time()
        for i in xrange(12):
            f1(X)
        print('Odin GRU speed:', (time.time() - start) / 12)
        time.sleep(1)

        start = time.time()
        for i in xrange(12):
            f3(X)
        print('Lasagne GRU speed:', (time.time() - start) / 12)
        time.sleep(1)

# ===========================================================================
# Main
# ===========================================================================
if __name__ == '__main__':
    print(' Use nosetests to run these tests ')
