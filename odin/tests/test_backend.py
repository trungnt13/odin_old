# ======================================================================
# Author: TrungNT
# ======================================================================
from __future__ import print_function, division

from .. import nnet
from .. import tensor as T
from .. import logger, config
from .. import objectives
from .. import optimizers

import unittest

import os
import numpy as np

from six.moves import zip, range


# ===========================================================================
# Main Test
# ===========================================================================
class BackendTest(unittest.TestCase):

    def setUp(self):
        logger.set_enable(False)

    def tearDown(self):
        logger.set_enable(True)

    def test_set_subtensor(self):
        x = T.variable(np.zeros((10, 10)))
        y = T.variable(np.ones((10, 10)))
        z = T.eval(T.set_subtensor(x[:,:], y[:,:]))

        self.assertEqual(z.ravel().tolist(), [1.] * 100)

    def test_gradients(self):
        a = T.variable(1.2)
        b = T.variable(1.3)
        x = a * b

        y = T.variable(2.)
        z = a + b

        true = T.variable(3.)
        pred = x * y * z
        loss = T.pow((true - pred), 2)

        G = T.gradients(loss, [a, b, y],
            consider_constant=[x], known_grads={a: T.ones_like(a)})
        G = [T.eval(g) for g in G]
        G = [g.tolist() if isinstance(g, np.ndarray) else g for g in G]
        G = [np.asarray(g).tolist() for g in G]
        for i, j in zip(G, [1.0, 29.951998, 37.439999]):
            self.assertAlmostEqual(round(i), round(j))

    def test_loop(self):
        def numpy_loop():
            o1 = []
            o2 = []

            seq1 = np.arange(10)
            seq2 = np.arange(10, 15)
            nonseq1 = 2.
            nonseq2 = 3.
            output1 = np.zeros((2, 2)) + 1
            output2 = np.zeros((2, 2)) + 2

            for i in xrange(5):
                s1 = seq1[i]
                s2 = seq2[i]
                output1 = output1 * s1 + nonseq1
                output2 = output2 * s2 + nonseq2
                o1.append(np.copy(output1).reshape(1, 2, 2))
                o2.append(np.copy(output2).reshape(1, 2, 2))
            return np.vstack(o1), np.vstack(o2)

        r_np = numpy_loop()

        # ====== odin loop ====== #
        seq1 = T.variable(np.arange(10))
        seq2 = T.variable(np.arange(10, 15))
        nonseq1 = T.variable(2.)
        nonseq2 = T.variable(3.)
        output1 = T.zeros((2, 2), dtype='float32') + 1
        output2 = T.zeros((2, 2), dtype='float32') + 2

        def step_fn(s1, s2, o1, o2, ns1, ns2):
            return (T.cast(o1 * s1 + ns1, 'float32'),
                    T.cast(o2 * s2 + ns2, 'float32'))

        r = T.loop(step_fn,
            sequences=[seq1, seq2],
            outputs_info=[output1, output2],
            non_sequences=[nonseq1, nonseq2],
            n_steps=5,
            go_backwards=False)
        f = T.function(
            inputs=[],
            outputs=r)
        r_loop = f()

        # ====== odin scan ====== #
        r = T.scan(step_fn,
            sequences=[seq1, seq2],
            outputs_info=[output1, output2],
            non_sequences=[nonseq1, nonseq2],
            n_steps=5,
            go_backwards=False)[0]
        f = T.function(
            inputs=[],
            outputs=r)
        r_scan = f()

        # print()
        # print(r_scan)
        # print(r_loop)
        # print(r_np)
        self.assertAlmostEqual(np.sum(np.abs(r_loop[0] - r_np[0])), 0.)
        self.assertAlmostEqual(np.sum(np.abs(r_loop[1] - r_np[1])), 0.)
        self.assertAlmostEqual(np.sum(np.abs(r_loop[0] - r_scan[0])), 0.)
        self.assertAlmostEqual(np.sum(np.abs(r_loop[1] - r_scan[1])), 0.)

    def test_scan(self):
        def step(s1, s2, s3, o1, o2, n1, n2):
            return o1, o2

        seq1 = T.variable(np.arange(10))
        seq2 = T.variable(np.arange(20))
        seq3 = T.variable(np.arange(5))

        nonseq1 = T.variable(1.)
        nonseq2 = T.variable(2.)

        ([o1, o2], updates) = T.scan(step,
            sequences=[seq1, seq2, seq3],
            outputs_info=[T.zeros((2, 2)), T.ones((2, 2))],
            non_sequences=[nonseq1, nonseq2],
            n_steps=5)
        f2 = T.function(
            inputs=[],
            outputs=[o1, o2],
            updates=updates)
        a, b = f2()

    def test_regularize(self):
        np.random.seed(T.get_magic_seed())
        x = T.variable(np.random.rand(10, 10))
        mean = T.variable(np.random.rand(32, 16))
        logsigma = T.variable(np.random.rand(32, 16))

        round_value = 1
        if config.floatX() == 'float32':
            round_value = 5
        elif config.floatX() == 'float64':
            round_value = 5

        self.assertAlmostEqual(round(T.eval(T.l1_regularize(x)), round_value),
                               round(50.150932312, round_value))
        self.assertAlmostEqual(round(T.eval(T.l2_regularize(x)), round_value),
                               round(33.7269096375, round_value))
        self.assertAlmostEqual(
            round(T.eval(T.mean(T.kl_gaussian(mean, logsigma))), round_value),
            round(0.29442, round_value))
        self.assertAlmostEqual(round(T.eval(T.correntropy_regularize(x)), round_value),
                               round(-5.86702489853, round_value))

        np.random.seed(12082518)
        x = T.variable(np.random.rand(16, 10))
        y = T.variable(np.random.rand(32, 10))
        self.assertAlmostEqual(
            round(T.eval(T.jacobian_regularize(x, y)), round_value),
            round(3.89396, round_value))

# ===========================================================================
# Main
# ===========================================================================
if __name__ == '__main__':
    print(' Use nosetests to run these tests ')
