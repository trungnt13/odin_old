from __future__ import print_function, division

from .base import OdinObject
from .dataset import dataset, batch
from .utils import get_magic_seed, seed_generate
from .logger import progress

import random
import numpy as np
from numpy.random import RandomState

from itertools import tee
from six.moves import range, zip

__all__ = [
    '_data',
    '_task',
    'trainer'
]
# ===========================================================================
# Helper
# ===========================================================================
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
            creator, news = tee(creator)
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

# ===========================================================================
# Task
# ===========================================================================
class _data(OdinObject):

    """
    Object store information that create data for trainer
    - ds: dataset object (optional)
    - batches: list of batch objects, can be create from
        - str: key name in given dataset
        - array: np.ndarray
        - function: function that return an iterator
        - iterator: duplicate a given iterator
    - start: start position of a batch when iterate
    - end: end position of a batch when iterate
    """

    def __init__(self, ds=None):
        super(_data, self).__init__()
        self._batches = []
        self._ds = ds
        self._data = []

    def _get_data_str(self, d):
        if not d: return 'None'
        if isinstance(d, np.ndarray):
            return '<Array: ' + str(d.shape) + '>'
        return str(d)

    def set_dataset(self, ds):
        self._ds = ds
        self.set(self._data)

    def set(self, data):
        self._batches = []
        if type(data) not in (tuple, list):
            data = [data]

        self._data = data
        for i in data:
            if isinstance(i, batch):
                self._batches.append(i)
            elif self._ds and isinstance(i, str):
                self._batches.append(self._ds[i])
            elif hasattr(i, '__call__') or hasattr(i, 'next'):
                self._batches.append(_iterator_wrapper(i))
            elif isinstance(i, np.ndarray):
                self._batches.append(batch(arrays=i))
        return self

    def create_iter(self, batch, start, end, shuffle, seed, mode):
        ''' data: is [dnntoolkit.batch] instance, always a list
            cross: is [dnntoolkit.batch] instance, always a list
        '''
        data = [i.iter(batch, start=start, end=end,
                       shuffle=shuffle, seed=seed, mode=mode)
                for i in self._batches]
        # handle case that 1 batch return all data
        if len(data) == 1:
            iter_data = data[0]
        else:
            iter_data = zip(*data)
        return iter_data

    def __str__(self):
        ds_str = ','.join(self._ds.get_path()) if self._ds else 'None'
        s = 'Data: - ds: %s\n' % ds_str
        for i, j in enumerate(self._data):
            j = self._get_data_str(j)
            s += '      - batch[%d]: %s\n' % (i, j)
        return s[:-1]

class _task(object):

    """
    An executable task:
    - name: str
    - func: function execute on task data
    - data: _data instance, or function, iterator, str, batch
    - p: probability will be executed after each batch
    - seed: seed for RandomState of task
    """

    def __init__(self, name, func, data, ds=None,
        epoch=1, p=1., seed=None):
        super(_task, self).__init__()
        self._name = name
        self._func = func
        if isinstance(data, _data):
            self._data = data
        else:
            self._data = _data(ds=ds).set(data)
        self._p = p
        if not seed:
            seed = get_magic_seed()
        self._seed = seed
        self._rand = np.random.RandomState(seed)

        self._batch_start = lambda x: x
        self._batch_end = lambda x: x

        if not epoch or epoch <= 0:
            epoch = float('inf')
        self._epoch = epoch
        self._batch = 128
        self._start = 0.0
        self._end = 1.0
        self._shuffle = True
        self._mode = 1

    def set_dataset(self, ds):
        self._data.set_dataset(ds)

    def set_callback(self, trainer):
        self._batch_start = trainer._batch_start
        self._batch_end = trainer._batch_end

    def set_iter(self, batch, start, end, shuffle, mode):
        self._batch = batch
        self._start = start
        self._end = end
        self._shuffle = shuffle
        self._mode = mode

    def run_iter(self):
        '''
        return True if epoch ended, otherwise return False
        return None at the end of iteration
        '''
        i = 0
        while i < self._epoch:
            for dat in self._data.create_iter(
                self._batch, self._start, self._end,
                self._shuffle, self._rand.randint(0, 10e8), self._mode):
                if self._rand.rand() < self._p:
                    dat = self._batch_start(dat)
                    res = self._func(*dat)
                    self._batch_end(res)
                yield False
            yield True
            i += 1

        while True:
            yield None

    def __str__(self):
        s = ''
        s += 'Task: %s\n' % str(self._name)
        s += '  - Func: %s\n' % str(self._func)
        s += '  - p: %s\n' % str(self._p)
        s += '  - seed: %s\n' % str(self._seed)
        s += '\n'.join(['  ' + i for i in str(self._data).split('\n')])
        return s

# ======================================================================
# Trainer
# ======================================================================

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
     - output: current training, testing, validating output
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

        # ====== dataset ====== #
        self._data_map = {}
        self._loaded_dataset = {}

        # ====== tasks ====== #
        self._task_list = []
        self._subtask_map = {}

        # ====== query ====== #
        self.idx = 0 # index in strategy
        self.output = None
        self.iter = None
        self.data = None
        self.epoch = 0
        self.task = None

        # ====== callback ====== #
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

        # ====== command ====== #
        self._stop = False
        self._valid_now = False
        self._restart_now = False

        self._iter_mode = 1

    # ==================== Helper ==================== #
    def _batch_start(self, dat):
        self.data = dat
        self._batch_start(self)
        return trainer.data

    def _batch_end(self, result):
        self.output = result
        self._batch_end(self)

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
        for i in self._task_list:
            i._mode = mode

    def add_dataset(self, name, data, ds=None):
        ''' Set dataset for trainer.

        Parameters
        ----------
        name : str
            name of given dataset
        data : str, list(str), np.ndarray, dnntoolkit.batch, iter, func(batch, shuffle, seed, mode)-return iter
            list of data used for given task name
        ds : ``odin.dataset`` (optional)
            dataset instance that conatin all given str in data

        Returns
        -------
        return : trainer
            for method chaining

        '''
        if ds:
            if not isinstance(ds, dataset):
                ds = dataset(ds, 'r')
            self._loaded_dataset[ds.get_path()] = ds

        self._data_map[name] = _data(ds=ds).set(data)
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

    def _validate_data(self, data, ds):
        if isinstance(data, str):
            if data in self._data_map:
                return self._data_map[data]
        if ds and not isinstance(ds, dataset):
            ds = dataset(ds, 'r')
            self._loaded_dataset[ds.get_path()] = ds
        return _data(ds=ds).set(data)

    def add_task(self, name, func, data, ds=None,
                 epoch=1, p=1.,
                 batch=128, shuffle=True, seed=None,
                 start=0., end=1.):
        data = self._validate_data(data, ds)
        task = _task(name, func, data, epoch=epoch, p=p, seed=seed)
        task.set_iter(batch, start, end, shuffle, self._iter_mode)
        self._task_list.append(task)

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
        self.output = train_cost
        self._train_end(self) # callback
        self.output = None
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
            self.output = cost
            self.iter = it
            self._batch_end(self)
            self.data = None
            self.output = None
        # ====== statistic of validation ====== #
        self.output = valid_cost
        self.log('\n => %s Stats: Mean:%.4f Var:%.2f Med:%.2f Min:%.2f Max:%.2f' %
                (task, np.mean(self.output), np.var(self.output), np.median(self.output),
                np.percentile(self.output, 5), np.percentile(self.output, 95)), 20)
        # ====== callback ====== #
        if task == 'valid':
            self._valid_end(self) # callback
        else:
            self._test_end(self)
        # ====== reset all flag ====== #
        self.output = None
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
        for i in range(epoch):
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
                self.output = cost
                self.iter = it
                self._batch_end(self)  # callback
                self.data = None
                self.output = None
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
            self.output = epoch_cost
            self.iter = it
            self.log('\n => Epoch Stats: Mean:%.4f Var:%.2f Med:%.2f Min:%.2f Max:%.2f' %
                    (np.mean(self.output), np.var(self.output), np.median(self.output),
                    np.percentile(self.output, 5), np.percentile(self.output, 95)))

            self._epoch_end(self) # callback
            self.output = None
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
