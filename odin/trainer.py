from __future__ import print_function, division

from .base import OdinObject
from .dataset import dataset, batch
from .utils import get_magic_seed
from .logger import progress

import random
import numpy as np
from numpy.random import RandomState

import itertools.tee

__all__ = [
    'trainer'
]
# ======================================================================
# Trainer
# ======================================================================
class _iterator_wrapper(object):

    '''Fake class with iter function like dnntoolkit.batch'''

    def __init__(self, creator):
        super(_iterator_wrapper, self).__init__()
        self.creator = creator

    def iter(self, batch, shuffle, seed, mode, *args, **kwargs):
        ''' Create and return an iterator'''
        creator = self.creator
        if hasattr(creator, '__call__'):
            return creator(batch, shuffle, seed)
        elif hasattr(creator, 'next'):
            creator, news = itertools.tee(creator)
            self.creator = creator
            return news
        else:
            raise ValueError(
                'Creator of data for trainer must be a function or iterator')

def _callback(trainer):
    pass

def _parse_data_config(task, data):
    '''return train,valid,test'''
    train = None
    test = None
    valid = None
    if type(data) in (tuple, list):
        # only specified train data
        if type(data[0]) not in (tuple, list):
            if 'train' in task: train = data
            elif 'test' in task: test = data
            elif 'valid' in task: valid = data
        else: # also specified train and valid
            if len(data) == 1:
                if 'train' in task: train = data[0]
                elif 'test' in task: test = data[0]
                elif 'valid' in task: valid = data[0]
            if len(data) == 2:
                if 'train' in task: train = data[0]; valid = data[1]
                elif 'test' in task: test = data[0]
                elif 'valid' in task: valid = data[0]
            elif len(data) == 3:
                train = data[0]
                test = data[1]
                valid = data[2]
    elif type(data) == dict:
        if 'train' in data: train = data['train']
        if 'test' in data: test = data['test']
        if 'valid' in data: valid = data['valid']
    elif data is not None:
        if 'train' in task: train = [data]
        if 'test' in task: test = [data]
        if 'valid' in task: valid = [data]
    return train, valid, test

class trainer(OdinObject):

    """
    TODO:
     - custom data (not instance of dataset),
     - cross training 2 dataset,
     - custom action trigger under certain condition
     - layers configuration: ['dropout':0.5, 'noise':'0.075']
     - default ArgumentsParser
     - Add iter_mode, start, end to set_strategy
     - Add prediction task
    Value can be queried on callback:
     - idx(int): current run idx in the strategies, start from 0
     - cost: current training, testing, validating cost
     - iter(int): number of iteration, start from 0
     - data: current data (batch_start)
     - epoch(int): current epoch, start from 0
     - task(str): current task 'train', 'test', 'valid'
    Command can be triggered when running:
     - stop()
     - valid()
     - restart()
    """

    def __init__(self):
        super(trainer, self).__init__()
        self._seed = RandomState(get_magic_seed())
        self._strategy = []

        self._train_data = None
        self._valid_data = None
        self._test_data = None

        self.idx = 0 # index in strategy
        self.cost = None
        self.iter = None
        self.data = None
        self.epoch = 0
        self.task = None

        # callback
        self._epoch_start = _callback
        self._epoch_end = _callback
        self._batch_start = _callback
        self._batch_end = _callback
        self._train_start = _callback
        self._train_end = _callback
        self._valid_start = _callback
        self._valid_end = _callback
        self._test_start = _callback
        self._test_end = _callback

        self._stop = False
        self._valid_now = False
        self._restart_now = False

        self._log_enable = True
        self._log_newline = False

        self._cross_data = None
        self._pcross = 0.3

        self._iter_mode = 1

    # ==================== Trigger Command ==================== #
    def stop(self):
        ''' Stop current activity of this trainer immediatelly '''
        self._stop = True

    def valid(self):
        ''' Trigger validation immediatelly, asap '''
        self._valid_now = True

    def restart(self):
        ''' Trigger restart current process immediatelly '''
        self._restart_now = True

    # ==================== Setter ==================== #
    def set_action(self, name, action,
                   epoch_start=False, epoch_end=False,
                   batch_start=False, batch_end=False,
                   train_start=False, train_end=False,
                   valid_start=False, valid_end=False,
                   test_start=False, test_end=False):
        pass

    def set_log(self, enable=True, newline=False):
        self._log_enable = enable
        self._log_newline = newline

    def set_iter_mode(self, mode):
        '''
        ONly for training, for validation and testing mode = 0
        mode : 0, 1, 2
            0 - default, sequentially read each dataset
            1 - parallel read: proportionately for each dataset (e.g. batch_size=512,
                dataset1_size=1000, dataset2_size=500 => ds1=341, ds2=170)
            2 - parallel read: make all dataset equal size by over-sampling
                smaller dataset (e.g. batch_size=512, there are 5 dataset
                => each dataset 102 samples) (only work if batch size <<
                dataset size)
            3 - parallel read: make all dataset equal size by under-sampling
                smaller dataset (e.g. batch_size=512, there are 5 dataset
                => each dataset 102 samples) (only work if batch size <<
                dataset size)        '''
        self._iter_mode = mode

    def set_dataset(self, data, train=None, valid=None,
        test=None, cross=None, pcross=None):
        ''' Set dataset for trainer.

        Parameters
        ----------
        data : dnntoolkit.dataset
            dataset instance which contain all your data
        train : str, list(str), np.ndarray, dnntoolkit.batch, iter, func(batch, shuffle, seed, mode)-return iter
            list of dataset used for training
        valid : str, list(str), np.ndarray, dnntoolkit.batch, iter, func(batch, shuffle, seed, mode)-return iter
            list of dataset used for validation
        test : str, list(str), np.ndarray, dnntoolkit.batch, iter, func(batch, shuffle, seed, mode)-return iter
            list of dataset used for testing
        cross : str, list(str), np.ndarray, dnntoolkit.batch, iter, func(batch, shuffle, seed, mode)-return iter
            list of dataset used for cross training
        pcross : float (0.0-1.0)
            probablity of doing cross training when training, None=default=0.3

        Returns
        -------
        return : trainer
            for chaining method calling

        Note
        ----
        the order of train, valid, test must be the same in model function
        any None input will be ignored
        '''
        if isinstance(data, str):
            data = dataset(data, mode='r')
        if not isinstance(data, dataset):
            raise ValueError('[data] must be instance of dataset')

        self._dataset = data

        if train is not None:
            if type(train) not in (tuple, list):
                train = [train]
            self._train_data = train

        if valid is not None:
            if type(valid) not in (tuple, list):
                valid = [valid]
            self._valid_data = valid

        if test is not None:
            if type(test) not in (tuple, list):
                test = [test]
            self._test_data = test

        if cross is not None:
            if type(cross) not in (tuple, list):
                cross = [cross]
            self._cross_data = cross
        if self._pcross:
            self._pcross = pcross
        return self

    def set_model(self, cost_func=None, updates_func=None):
        ''' Set main function for this trainer to manipulate your model.

        Parameters
        ----------
        cost_func : theano.Function, function
            cost function: inputs=[X,y]
                           return: cost
        updates_func : theano.Function, function
            updates parameters function: inputs=[X,y]
                                         updates: parameters
                                         return: cost while training

        Returns
        -------
        return : trainer
            for chaining method calling
        '''
        if cost_func is not None and not hasattr(cost_func, '__call__'):
           raise ValueError('cost_func must be function')
        if updates_func is not None and not hasattr(updates_func, '__call__'):
           raise ValueError('updates_func must be function')

        self._cost_func = cost_func
        self._updates_func = updates_func
        return self

    def set_callback(self, epoch_start=_callback, epoch_end=_callback,
                     batch_start=_callback, batch_end=_callback,
                     train_start=_callback, train_end=_callback,
                     valid_start=_callback, valid_end=_callback,
                     test_start=_callback, test_end=_callback):
        ''' Set Callback while training, validating or testing the model.

        Parameters
        ----------
            all arguments is in form of:
                def function(trainer): ...
        Returns
        -------
        return : trainer
            for chaining method calling
        '''
        self._epoch_start = epoch_start
        self._epoch_end = epoch_end
        self._batch_start = batch_start
        self._batch_end = batch_end

        self._train_start = train_start
        self._valid_start = valid_start
        self._test_start = test_start

        self._train_end = train_end
        self._valid_end = valid_end
        self._test_end = test_end
        return self

    def set_strategy(self, task=None, data=None,
                     epoch=1, batch=512, validfreq=0.4,
                     shuffle=True, seed=None, yaml=None,
                     cross=None, pcross=None):
        ''' Set strategy for training.

        Parameters
        ----------
        task : str
            train, valid, or test
        data : str, list(str), map
            for training, data contain all training data and validation data names
            example: [['X_train','y_train'],['X_valid','y_valid']]
                  or {'train':['X_train','y_train'],'valid':['X_valid','y_valid']}
            In case of missing data, strategy will take default data names form
            set_dataset method
        epoch : int
            number of epoch for training (NO need for valid and test)
        batch : int, 'auto'
            number of samples for each batch
        validfreq : int(number of iteration), float(0.-1.)
            validation frequency when training, when float, it mean percentage
            of dataset
        shuffle : boolean
            shuffle dataset while training
        seed : int
            set seed for shuffle so the result is reproducible
        yaml : str
            path to yaml strategy file. When specify this arguments,
            all other arguments are ignored
        cross : str, list(str), numpy.ndarray
            list of dataset used for cross training
        pcross : float (0.0-1.0)
            probablity of doing cross training when training

        Returns
        -------
        return : trainer
            for chaining method calling
        '''
        if yaml is not None:
            import yaml as yaml_
            f = open(yaml, 'r')
            strategy = yaml_.load(f)
            f.close()
            for s in strategy:
                if 'dataset' in s:
                    self._dataset = dataset(s['dataset'])
                    continue
                if 'validfreq' not in s: s['validfreq'] = validfreq
                if 'batch' not in s: s['batch'] = batch
                if 'epoch' not in s: s['epoch'] = epoch
                if 'shuffle' not in s: s['shuffle'] = shuffle
                if 'data' not in s: s['data'] = data
                if 'cross' not in s: s['cross'] = cross
                if 'pcross' not in s: s['pcross'] = pcross
                if 'seed' in s: self._seed = RandomState(seed)
                self._strategy.append(s)
            return

        if task is None:
            raise ValueError('Must specify both [task] and [data] arguments')

        self._strategy.append({
            'task': task,
            'data': data,
            'epoch': epoch,
            'batch': batch,
            'shuffle': shuffle,
            'validfreq': validfreq,
            'cross': cross,
            'pcross': pcross
        })
        if seed is not None:
            self._seed = RandomState(seed)
        return self

    # ==================== Helper function ==================== #
    def _early_stop(self):
        # just a function reset stop flag and return its value
        tmp = self._stop
        self._stop = False
        return tmp

    def _early_valid(self):
        # just a function reset valid flag and return its value
        tmp = self._valid_now
        self._valid_now = False
        return tmp

    def _early_restart(self):
        # just a function reset valid flag and return its value
        tmp = self._restart_now
        self._restart_now = False
        return tmp

    def _get_str_datalist(self, datalist):
        if not datalist:
            return 'None'
        return ', '.join(['<Array: ' + str(i.shape) + '>'
                          if isinstance(i, np.ndarray) else str(i)
                          for i in datalist])

    def _check_dataset(self, data):
        ''' this function convert all pair of:
        [dataset, dataset_name] -> [batch_object]
        '''
        batches = []
        for i in data:
            if (type(i) in (tuple, list) and isinstance(i[0], np.ndarray)) or \
                isinstance(i, np.ndarray):
                batches.append(batch(arrays=i))
            elif isinstance(i, batch):
                batches.append(i)
            elif hasattr(i, '__call__') or hasattr(i, 'next'):
                batches.append(_iterator_wrapper(i))
            else:
                batches.append(self._dataset[i])
        return batches

    def _create_iter(self, data, batch, shuffle, mode, cross=None, pcross=0.3):
        ''' data: is [dnntoolkit.batch] instance, always a list
            cross: is [dnntoolkit.batch] instance, always a list
        '''
        seed = self._seed.randint(0, 10e8)
        data = [i.iter(batch, shuffle=shuffle, seed=seed, mode=mode)
                for i in data]
        # handle case that 1 batch return all data
        if len(data) == 1:
            iter_data = data[0]
        else:
            iter_data = zip(*data)
        if cross: # enable cross training
            seed = self._seed.randint(0, 10e8)
            if len(cross) == 1: # create cross iteration
                cross_it = cross[0].iter(batch, shuffle=shuffle,
                                  seed=seed, mode=mode)
            else:
                cross_it = zip(*[i.iter(batch, shuffle=shuffle,
                                    seed=seed, mode=mode)
                                 for i in cross])
            for d in iter_data:
                if random.random() < pcross:
                    try:
                        yield cross_it.next()
                    except StopIteration:
                        seed = self._seed.randint(0, 10e8)
                        if len(cross) == 1: # recreate cross iteration
                            cross_it = cross[0].iter(batch, shuffle=shuffle,
                                              seed=seed, mode=mode)
                        else:
                            cross_it = zip(*[i.iter(batch, shuffle=shuffle,
                                                seed=seed, mode=mode)
                                             for i in cross])
                yield d
        else: # only normal training
            for d in iter_data:
                yield d

    def _finish_train(self, train_cost, restart=False):
        self.cost = train_cost
        self._train_end(self) # callback
        self.cost = None
        self.task = None
        self.data = None
        self.it = 0
        return not restart

    # ==================== Main workflow ==================== #
    def _cost(self, task, valid_data, batch):
        '''
        Return
        ------
        True: finished the task
        False: restart the task
        '''
        self.task = task
        self.iter = 0
        if task == 'valid':
            self._valid_start(self)
        elif task == 'test':
            self._test_start(self)

        # convert name and ndarray to [dnntoolkit.batch] object
        valid_data = self._check_dataset(valid_data)
        # find n_samples
        n_samples = [len(i) for i in valid_data if hasattr(i, '__len__')]
        if len(n_samples) == 0: n_samples = 10e8
        else: n_samples = max(n_samples)
        # init some things
        valid_cost = []
        n = 0
        it = 0
        for data in self._create_iter(valid_data, batch, False, 0):
            # batch start
            it += 1
            n += data[0].shape[0]
            self.data = data
            self.iter = it
            self._batch_start(self)

            # main cost
            cost = self._cost_func(*self.data)
            if len(cost.shape) == 0:
                valid_cost.append(cost.tolist())
            else:
                valid_cost += cost.tolist()

            if self._log_enable:
                progress(n, max_val=n_samples,
                    title='%s:Cost:%.4f' % (task, np.mean(cost)),
                    newline=self._log_newline, idx=task)

            # batch end
            self.cost = cost
            self.iter = it
            self._batch_end(self)
            self.data = None
            self.cost = None
        # ====== statistic of validation ====== #
        self.cost = valid_cost
        self.log('\n => %s Stats: Mean:%.4f Var:%.2f Med:%.2f Min:%.2f Max:%.2f' %
                (task, np.mean(self.cost), np.var(self.cost), np.median(self.cost),
                np.percentile(self.cost, 5), np.percentile(self.cost, 95)), 20)
        # ====== callback ====== #
        if task == 'valid':
            self._valid_end(self) # callback
        else:
            self._test_end(self)
        # ====== reset all flag ====== #
        self.cost = None
        self.task = None
        self.iter = 0
        return True

    def _train(self, train_data, valid_data, epoch, batch, validfreq, shuffle,
               cross=None, pcross=0.3):
        '''
        Return
        ------
        True: finished the task
        False: restart the task
        '''
        self.task = 'train'
        self.iter = 0
        self._train_start(self)
        it = 0
        # convert name and ndarray to [dnntoolkit.batch] object
        train_data = self._check_dataset(train_data)
        if cross:
            cross = self._check_dataset(cross)
        # get n_samples in training
        ntrain = [i.iter_len(self._iter_mode) for i in train_data
                  if hasattr(i, 'iter_len')]
        if len(ntrain) == 0: ntrain = 20 * batch
        else: ntrain = ntrain[0]
        # validfreq_it: number of iterations after each validation
        validfreq_it = 1
        if validfreq > 1.0:
            validfreq_it = int(validfreq)
        # ====== start ====== #
        train_cost = []
        for i in xrange(epoch):
            self.epoch = i
            self.iter = it
            self._epoch_start(self) # callback
            if self._early_stop(): # earlystop
                return self._finish_train(train_cost, self._early_restart())
            epoch_cost = []
            n = 0
            # ====== start batches ====== #
            for data in self._create_iter(train_data, batch, shuffle,
                                          self._iter_mode,
                                          cross, pcross):
                # start batch
                n += data[0].shape[0]
                # update ntrain constantly, if iter_mode = 1, no idea how many
                # data point in the dataset because of upsampling
                ntrain = max(ntrain, n)
                self.data = data
                self.iter = it
                self._batch_start(self) # callback
                if self._early_stop(): # earlystop
                    return self._finish_train(train_cost, self._early_restart())

                # main updates
                cost = self._updates_func(*self.data)

                # log
                epoch_cost.append(cost)
                train_cost.append(cost)
                if self._log_enable:
                    progress(n, max_val=ntrain,
                        title='Epoch:%d,Iter:%d,Cost:%.4f' % (i + 1, it, cost),
                        newline=self._log_newline, idx='train')

                # end batch
                self.cost = cost
                self.iter = it
                self._batch_end(self)  # callback
                self.data = None
                self.cost = None
                if self._early_stop(): # earlystop
                    return self._finish_train(train_cost, self._early_restart())

                # validation, must update validfreq_it because ntrain updated also
                if validfreq <= 1.0:
                    validfreq_it = int(max(validfreq * ntrain / batch, 1))
                it += 1 # finish 1 iteration
                if (it % validfreq_it == 0) or self._early_valid():
                    if valid_data is not None:
                        self._cost('valid', valid_data, batch)
                        if self._early_stop(): # earlystop
                            return self._finish_train(train_cost, self._early_restart())
                    self.task = 'train' # restart flag back to train
            # ====== end epoch: statistic of epoch cost ====== #
            self.cost = epoch_cost
            self.iter = it
            self.log('\n => Epoch Stats: Mean:%.4f Var:%.2f Med:%.2f Min:%.2f Max:%.2f' %
                    (np.mean(self.cost), np.var(self.cost), np.median(self.cost),
                    np.percentile(self.cost, 5), np.percentile(self.cost, 95)))

            self._epoch_end(self) # callback
            self.cost = None
            if self._early_stop(): # earlystop
                return self._finish_train(train_cost, self._early_restart())

        # end training
        return self._finish_train(train_cost, self._early_restart())

    def debug(self):
        raise NotImplementedError()

    def run(self):
        ''' run specified strategies
        Returns
        -------
        return : bool
            if exception raised, return False, otherwise return True
        '''
        try:
            while self.idx < len(self._strategy):
                config = self._strategy[self.idx]
                task = config['task']
                train, valid, test = _parse_data_config(task, config['data'])
                if train is None: train = self._train_data
                if test is None: test = self._test_data
                if valid is None: valid = self._valid_data
                cross = config['cross']
                pcross = config['pcross']
                if pcross is None: pcross = self._pcross
                if cross is None: cross = self._cross_data
                elif not hasattr(cross, '__len__'):
                    cross = [cross]

                epoch = config['epoch']
                batch = config['batch']
                validfreq = config['validfreq']
                shuffle = config['shuffle']

                if self._log_enable:
                    self.log('\n******* %d-th run, with configuration: *******' % self.idx, 0)
                    self.log(' - Task:%s' % task, 0)
                    self.log(' - Train data:%s' % self._get_str_datalist(train), 0)
                    self.log(' - Valid data:%s' % self._get_str_datalist(valid), 0)
                    self.log(' - Test data:%s' % self._get_str_datalist(test), 0)
                    self.log(' - Cross data:%s' % self._get_str_datalist(cross), 0)
                    self.log(' - Cross prob:%s' % str(pcross), 0)
                    self.log(' - Epoch:%d' % epoch, 0)
                    self.log(' - Batch:%d' % batch, 0)
                    self.log(' - Validfreq:%d' % validfreq, 0)
                    self.log(' - Shuffle:%s' % str(shuffle), 0)
                    self.log('**********************************************', 0)

                if 'train' in task:
                    if train is None:
                        self.log('*** no TRAIN data found, ignored **', 30)
                    else:
                        while (not self._train(
                                train, valid, epoch, batch, validfreq, shuffle,
                                cross, pcross)):
                            pass
                elif 'valid' in task:
                    if valid is None:
                        self.log('*** no VALID data found, ignored **', 30)
                    else:
                        while (not self._cost('valid', valid, batch)):
                            pass
                elif 'test' in task:
                    if test is None:
                        self.log('*** no TEST data found, ignored **', 30)
                    else:
                        while (not self._cost('test', test, batch)):
                            pass
                # only increase idx after finish the task
                self.idx += 1
        except Exception, e:
            self.log(str(e), 40)
            import traceback; traceback.print_exc();
            return False
        return True

    # ==================== Debug ==================== #
    def __str__(self):
        s = '\n'
        s += 'Dataset:' + str(self._dataset) + '\n'
        s += 'Current run:%d' % self.idx + '\n'
        s += '============ \n'
        s += 'defTrain:' + self._get_str_datalist(self._train_data) + '\n'
        s += 'defValid:' + self._get_str_datalist(self._valid_data) + '\n'
        s += 'defTest:' + self._get_str_datalist(self._test_data) + '\n'
        s += 'defCross:' + self._get_str_datalist(self._cross_data) + '\n'
        s += 'pCross:' + str(self._pcross) + '\n'
        s += '============ \n'
        s += 'Cost_func:' + str(self._cost_func) + '\n'
        s += 'Updates_func:' + str(self._updates_func) + '\n'
        s += '============ \n'
        s += 'Epoch start:' + str(self._epoch_start) + '\n'
        s += 'Epoch end:' + str(self._epoch_end) + '\n'
        s += 'Batch start:' + str(self._batch_start) + '\n'
        s += 'Batch end:' + str(self._batch_end) + '\n'
        s += 'Train start:' + str(self._train_start) + '\n'
        s += 'Train end:' + str(self._train_end) + '\n'
        s += 'Valid start:' + str(self._valid_start) + '\n'
        s += 'Valid end:' + str(self._valid_end) + '\n'
        s += 'Test start:' + str(self._test_start) + '\n'
        s += 'Test end:' + str(self._test_end) + '\n'

        for i, st in enumerate(self._strategy):
            train, valid, test = _parse_data_config(st['task'], st['data'])
            if train is None: train = self._train_data
            if test is None: test = self._test_data
            if valid is None: valid = self._valid_data
            cross = st['cross']
            pcross = st['pcross']
            if cross and not hasattr(cross, '__len__'):
                cross = [cross]

            s += '====== Strategy %d-th ======\n' % i
            s += ' - Task:%s' % st['task'] + '\n'
            s += ' - Train:%s' % self._get_str_datalist(train) + '\n'
            s += ' - Valid:%s' % self._get_str_datalist(valid) + '\n'
            s += ' - Test:%s' % self._get_str_datalist(test) + '\n'
            s += ' - Cross:%s' % self._get_str_datalist(cross) + '\n'
            s += ' - pCross:%s' % str(pcross) + '\n'
            s += ' - Epoch:%d' % st['epoch'] + '\n'
            s += ' - Batch:%d' % st['batch'] + '\n'
            s += ' - Shuffle:%s' % st['shuffle'] + '\n'

        return s
