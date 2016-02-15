# ======================================================================
# Author: TrungNT
# ======================================================================
from __future__ import print_function, division

from .. import logger
from ..trainer import _data, _task, trainer
from ..dataset import dataset
from ..model import model
from .. import tensor
import unittest
import os
from collections import defaultdict

import numpy as np
import h5py
# ===========================================================================
# Main Tests
# ===========================================================================
def model_func():
    import lasagne
    l = lasagne.layers.InputLayer(shape=(None, 10))
    l = lasagne.layers.DenseLayer(l, num_units=64)
    l = lasagne.layers.DenseLayer(l, num_units=3,
        nonlinearity=lasagne.nonlinearities.linear)
    return l

class ModelTest(unittest.TestCase):

    def setUp(self):
        logger.set_enable(False)
        f = h5py.File('tmp.h5', 'w')
        f['X_train'] = np.random.rand(1024, 10)
        f['y_train'] = np.random.rand(1024, 3)

        f['X_test'] = np.random.rand(512, 10)
        f['y_test'] = np.random.rand(512, 3)

        f['X_valid'] = np.random.rand(512, 10)
        f['y_valid'] = np.random.rand(512, 3)

    def tearDown(self):
        logger.set_enable(True)
        if os.path.exists('tmp.h5'):
            os.remove('tmp.h5')

    def test_data(self):
        ds = dataset('tmp.h5', 'r')
        d = _data()
        d.set(['X_train', 'y_train', ds['X_valid'], ds['y_valid'],
               np.random.rand(1024, 13)])
        self.assertEqual(len(d._batches), 3)

        d.set_dataset(ds)
        self.assertEqual(len(d._batches), 5)

        it = d.create_iter(32, 0., 1., shuffle=True, seed=13, mode=0)
        count = defaultdict(int)
        for a, b, c, d, e in it:
            count['a'] += a.shape[0]
            count['b'] += b.shape[0]
            count['c'] += c.shape[0]
            count['d'] += d.shape[0]
            count['e'] += e.shape[0]
        self.assertEqual(count.values(), [512] * 5)
        self.assertEqual(a.shape[1], 10)
        self.assertEqual(b.shape[1], 3)
        self.assertEqual(e.shape[1], 13)

    def test_task(self):
        ds = dataset('tmp.h5', 'r')
        global niter
        niter = 0

        def task_func(*X):
            global nargs, niter
            nargs = len(X)
            niter += 1

        t = _task('task', task_func, _data(ds).set(['X_train', ds['y_train']]),
                  epoch=2, p=1., seed=13)
        t.set_iter(128, 0., 1., shuffle=True, mode=0)
        run_it = t.run_iter()
        while run_it.next() is not None:
            pass
        self.assertEqual(nargs, 2)
        self.assertEqual(niter, 16)

        t._epoch = float('inf') # infinite run
        niter = 0
        run_it = t.run_iter()
        for i in xrange(1000):
            run_it.next()
        self.assertEqual(niter, 1000)

    def test_training(self):
        import lasagne
        # ====== create model ====== #
        m = model()
        m.set_model(model_func, 'lasagne')

        f_cost = m.create_cost(
            lambda y_pred, y_true: tensor.mean(tensor.square(y_pred - y_true), axis=-1))
        f_update = m.create_updates(
            lambda y_pred, y_true: tensor.mean(tensor.square(y_pred - y_true), axis=-1),
            lasagne.updates.rmsprop)

        # ====== create trainer ====== #
        global i, j, k
        i, j, k = 0, 0, 0

        def train_func(*X):
            global i
            i += 1
            # print('Train', i)

        def valid_func(*X):
            global j
            j += 1
            # print('Valid', j)

        def test_func(*X):
            global k
            k += 1

        flag = True

        def task_start_end(trainer):
            self.assertEqual(
                trainer.task == 'train' or trainer.task == 'test' or
                trainer.task == 'realtrain' or trainer.task == 'realtest', True)

        def batch_start(trainer):
            global flag
            # print(trainer.task, trainer.iter)
            if trainer.task == 'train' and train.iter == 100 and flag:
                trainer.restart()
                flag = False
            elif trainer.task == 'test' and train.iter == 50:
                trainer.stop()

        train = trainer()
        train.set_callback(
            batch_start=batch_start, task_start=task_start_end,
            task_end=task_start_end)
        train.add_data('valid', ['X_valid', 'y_valid'])
        train.add_data('test', ['X_test', 'y_test'])

        train.add_task('train', train_func, ['X_train', 'y_train'], 'tmp.h5',
            epoch=2, seed=13)
        train.add_subtask(valid_func, 'valid', freq=0.58)
        train.add_subtask(test_func, 'test', single_run=True, epoch=-1, p=0.1)

        train.add_task('test', test_func, 'test')
        while not train.step(): pass
        self.assertEqual(train.run(), True)
        self.assertEqual(train.step(), True)

        self.assertEqual(i, 16) # 2 epochs, 8 iter each
        self.assertEqual(j, 12) # 3 epochs, 4 iter each
        self.assertEqual(k, 4) # 10% activated

        # ====== Main training ====== #
        def batch_end(trainer):
            pass

        def epoch_end(trainer):
            if trainer.task == 'realtrain_subtask[0]':
                print('Valid:', np.mean(trainer.output))
            elif trainer.task == 'realtrain':
                print('Train:', np.mean(trainer.output))

        ds = dataset('tmp.h5', 'r')
        cost1 = f_cost(ds['X_train'].value, ds['y_train'].value).mean()
        w1 = m.get_weights()

        train.set_callback(batch_end=batch_end, epoch_end=epoch_end)
        train.add_task('realtrain', f_update, ['X_train', 'y_train'], 'tmp.h5',
            epoch=10, seed=13)
        train.add_subtask(f_cost, 'valid', freq=0.6)
        train.add_task('realtest1', f_cost, 'test', 'tmp.h5', epoch=1, seed=13)
        train.add_task('realtest2', f_cost, 'test', 'tmp.h5', epoch=1, seed=13)
        self.assertEqual(train.run(), True)

        w2 = m.get_weights()
        cost2 = f_cost(ds['X_train'].value, ds['y_train'].value).mean()
        for i, j in zip(w1, w2):
            self.assertNotEqual(i.tolist(), j.tolist())
        self.assertEqual(cost1 > cost2, True)

# ===========================================================================
# Main
# ===========================================================================
if __name__ == '__main__':
    print(' Use nosetests to run these tests ')
