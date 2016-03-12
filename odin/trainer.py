from __future__ import print_function, division, absolute_import

from .base import OdinObject
from .dataset import dataset, batch
from .tensor import get_magic_seed
from . import logger

import numpy as np
from numpy.random import RandomState

from itertools import tee
from collections import defaultdict
from six.moves import range, zip

__all__ = [
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
        if isinstance(d, np.ndarray):
            return '<Array: ' + str(d.shape) + '>'
        if d is None:
            return 'None'
        elif isinstance(d, str):
            return d + ' (str)'
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

    def est_niter(self, bs, start, end, mode):
        '''estimate number of iteration for 1 epoch'''
        if len(self._batches) == 0:
            return float('inf')
        niter = []
        for i in self._batches:
            if hasattr(i, 'iter_len'):
                n = end - start if end - start > 1 \
                    else i.iter_len(mode) * (end - start)
                niter.append(int(np.ceil(n / bs)))
            elif hasattr(i, 'shape'):
                niter.append(int(np.ceil(i.shape[0] / bs)))
            elif hasattr(i, 'len'):
                niter.append(int(np.ceil(len(i) / bs)))
            else:
                niter.append(float('inf'))
        return min(niter)

    def create_iter(self, bs, start, end, shuffle, seed, mode):
        data = [i.iter(bs, start=start, end=end,
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
        return s


class _task(object):

    """
    An executable task:
    - name: str
    - func: function execute on task data
    - data: _data instance, or function, iterator, str, batch
    - p: probability will be executed after each batch
    - seed: seed for RandomState of task
    """

    def __init__(self, name, func, data,
                 epoch=1, p=1., seed=None,
                 mode=1):
        super(_task, self).__init__()
        self.name = str(name)
        self._func = func
        if not isinstance(data, _data):
            raise ValueError('Only support _data for _task')
        self._data = data
        self._p = p
        if not seed:
            seed = get_magic_seed()
        self.seed = seed
        self._rand = np.random.RandomState(seed)

        self._batch_start = lambda t, x, y, z: z
        self._batch_end = lambda t, x, y, z: None
        self._epoch_start = lambda t, x, y: None
        self._epoch_end = lambda t, x, y, z: None

        if not epoch or epoch <= 0:
            epoch = float('inf')
        self._epoch = epoch
        self._batch = 128
        self._start = 0.0
        self._end = 1.0
        self._shuffle = True
        self._mode = mode

        # ====== information for child tasks ====== #
        self.ds = data._ds # store ds, so the subtask can find parents' dataset

    def set_dataset(self, ds):
        self.ds = ds
        self._data.set_dataset(ds)

    def set_callback(self, trainer):
        self._batch_start = trainer._batch_start_callback
        self._batch_end = trainer._batch_end_callback
        self._epoch_start = trainer._epoch_start_callback
        self._epoch_end = trainer._epoch_end_callback

    def set_iter(self, bs, start, end, shuffle, mode):
        self._batch = bs
        self._start = start
        self._end = end
        self._shuffle = shuffle
        self._mode = mode

    def est_niter(self):
        return self._data.est_niter(
            self._batch, self._start, self._end, self._mode)

    def run_iter(self):
        '''
        return True if epoch ended, otherwise return False
        return None at the end of iteration
        '''
        i = 0
        it = 0
        just_finish_epoch = False
        while i < self._epoch:
            epoch_results = []
            self._epoch_start(self.name, i, it)

            for dat in self._data.create_iter(
                self._batch, self._start, self._end,
                self._shuffle, self._rand.randint(0, 10e8), self._mode):
                if self._rand.rand() < self._p:
                    it += 1
                    dat = self._batch_start(self.name, i, it, dat)
                    res = self._func(*dat)
                    epoch_results.append(res)
                    self._batch_end(self.name, i, it, res)
                yield just_finish_epoch
                just_finish_epoch = False

            self._epoch_end(self.name, i, it, epoch_results)
            just_finish_epoch = True
            i += 1

        while True:
            yield None

    def __str__(self):
        s = ''
        s += 'Task: %s\n' % str(self.name)
        s += '  - Func: %s\n' % str(self._func)
        s += '  - p: %s\n' % str(self._p)
        s += '  - seed: %s\n' % str(self.seed)
        s += '  - epoch: %s\n' % str(self._epoch)
        s += '  - batch: %s\n' % str(self._batch)
        s += '  - start: %s\n' % str(self._start)
        s += '  - end: %s\n' % str(self._end)
        s += '  - shuffle: %s\n' % str(self._shuffle)
        s += '  - mode: %s\n' % str(self._mode)
        s += '  - batch_start: %s\n' % str(self._batch_start.__name__)
        s += '  - batch_end:   %s\n' % str(self._batch_end.__name__)
        s += '  - epoch_start: %s\n' % str(self._epoch_start.__name__)
        s += '  - epoch_end:   %s\n' % str(self._epoch_end.__name__)
        s += '\n'.join(['  ' + i for i in str(self._data).split('\n')])
        return s[:-2]

# ======================================================================
# Trainer
# ======================================================================


class trainer(OdinObject):

    """
    TODO:
     - layers configuration: ['dropout':0.5, 'noise':'0.075']
     - default ArgumentsParser
     - Add iter_mode, start, end to set_strategy
     - Add prediction task
    Value can be queried on callback:
     - idx: (int) current run idx in the strategies, start from 0
     - output: current training, testing, validating output
     - iter: (int) number of iteration, start from 0
     - data: current data (batch_start)
     - epoch: (int) current epoch, start from 0
     - task: (str) current task name:'train', 'test', 'valid'
    Command can be triggered when running:
     - stop()
     - valid()
     - restart()

    Example
    -------
    >>> from odin import trainer
    ...
    >>> train = trainer()
    >>> train.set_callback(
    ...     batch_start=batch_start, task_start=task_start_end,
    ...     task_end=task_start_end)
    ... # add data alias
    >>> train.add_data('valid', ['X_valid', 'y_valid'])
    >>> train.add_data('test', ['X_test', 'y_test'])
    ...
    ... # add task
    >>> train.add_task('train', train_func, ['X_train', 'y_train'], 'tmp.h5',
    >>>     epoch=2, seed=13, mode=1)
    >>> train.add_subtask('subvalid', valid_func, 'valid', freq=0.58)
    >>> train.add_subtask('subtest', test_func, 'test', single_run=True, epoch=-1, p=0.1)
    ...
    >>> train.add_task('test', test_func, 'test', mode=0)

    """

    def __init__(self):
        super(trainer, self).__init__()
        self._seed = RandomState(get_magic_seed())

        # ====== dataset ====== #
        self._data_map = {} # name: _data
        self._loaded_dataset = {} # dataset_path: dataset
        self._del_able_dataset = []

        # ====== tasks ====== #
        self._task_list = []
        self._subtask_map = defaultdict(list)
        self._last_task = None
        self._subtask_single_run = {} #_task: is_subtask_single_run (bool)
        self._subtask_freq = {} #_task: freq (int, float)

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
        self._task_start = _callback
        self._task_end = _callback

        # ====== command ====== #
        self._debug_run = None
        self._stop_now = False
        self._restart_now = False
        self._iter_mode = 1

    # =============== Callback interface for task =============== #
    def _batch_start_callback(self, task, nepoch, niter, dat):
        self.task = task
        self.epoch = nepoch
        self.iter = niter
        self.data = dat
        self._batch_start(self)
        return self.data

    def _batch_end_callback(self, task, nepoch, niter, result):
        self.task = task
        self.epoch = nepoch
        self.iter = niter
        self.output = result
        self._batch_end(self)

    def _epoch_start_callback(self, task, nepoch, niter):
        self.task = task
        self.epoch = nepoch
        self.iter = niter
        self._epoch_start(self)

    def _epoch_end_callback(self, task, nepoch, niter, results):
        self.task = task
        self.epoch = nepoch
        self.niter = niter
        self.output = results
        self._epoch_end(self)

    # ==================== Trigger Command ==================== #
    def stop(self):
        ''' Stop current activity of this trainer immediatelly '''
        self._stop_now = True

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

    def _create_dataset(self, ds):
        if isinstance(ds, dataset):
            self._loaded_dataset[','.join(ds.get_path())] = ds
            return ds

        if type(ds) not in (tuple, list):
            ds = [ds]
        ds_key = ','.join(ds)
        if ds_key in self._loaded_dataset:
            return self._loaded_dataset[ds_key]

        ds = dataset(ds, 'r')
        self._del_able_dataset.append(ds)
        self._loaded_dataset[','.join(ds.get_path())] = ds
        return ds

    def _validate_data(self, data, ds, parent_ds=None):
        if isinstance(data, str) and data in self._data_map:
            data = self._data_map[data]
            if data._ds is None and parent_ds is not None:
                data.set_dataset(parent_ds)
            return data
        elif type(data) not in (tuple, list):
            data = [data]

        if ds:
            ds = self._create_dataset(ds)
        return _data(ds=ds).set(data)

    def add_data(self, name, data, ds=None):
        ''' Set dataset for trainer.

        Parameters
        ----------
        name : str
            name of given dataset
        data : str, list(str), ``ndarray``, ``batch``, iter,
               func(batch, shuffle, seed, mode)->return iter
            list of data used for given task name
        ds : ``odin.dataset`` (optional)
            dataset instance that conatin all given str in data

        Returns
        -------
        return : trainer
            for method chaining
        '''
        if ds: ds = self._create_dataset(ds)
        self._data_map[name] = _data(ds=ds).set(data)
        return self

    def set_callback(self, epoch_start=_callback, epoch_end=_callback,
                     batch_start=_callback, batch_end=_callback,
                     task_start=_callback, task_end=_callback):
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
        self._task_start = task_start
        self._task_end = task_end
        return self

    def add_task(self, name, func, data, ds=None,
                 epoch=1, p=1.,
                 bs=128, shuffle=True, seed=None,
                 start=0., end=1.,
                 mode=1):
        ''' Set task for running.
        A task is a function operates on a list of ``odin.dataset.batch``, the
        batch return an iterator with add necessary data for the function.

        Parameters
        ----------
        name : str
            identification of a task
        func : function
            for training, data contain all training data and validation data names
            example: [['X_train','y_train'],['X_valid','y_valid']]
                  or {'train':['X_train','y_train'],'valid':['X_valid','y_valid']}
            In case of missing data, strategy will take default data names form
            set_dataset method
        data : str, list(str), ``ndarray``, ``batch``, iter,
               func(batch, shuffle, seed, mode)->return iter
            list of data used for given task name
        ds : ``odin.dataset.dataset``, paths
            if ds is a paths (strings), we create a dataset to handle all string
            specified in ``data``
        epoch : int
            number of epoch to run this batch
        p : float (0.-1.)
            probability this task will be execute during its epoches
        bs : int
            batch size, number of samples for each batch
        shuffle : boolean
            shuffle dataset while training
        seed : int
            set seed for shuffle so the result is reproducible
        start : int, float(0.-1.)
            starting point within each ``odin.dataset.batch``
        end : int, float(0.-1.)
            ending point within each ``odin.dataset.batch``
        mode : 0, 1, 2
            0 - default, sequentially read each dataset
            1 - parallel read: proportionately for each dataset (e.g.
                batch_size=512, dataset1_size=1000, dataset2_size=500
                => ds1=341, ds2=170)
            2 - parallel read (upsampling): upsampling smaller dataset
                (e.g. batch_size=512, there are 5 dataset => each dataset
                102 samples) (only work if batch size <<
                dataset size)
            3 - parallel read (downsampling): downsampling larger dataset
                (e.g. batch_size=512, there are 5 dataset => each dataset
                102 samples) (only work if batch size <<
                dataset size)

        Returns
        -------
        return : trainer
            for chaining method calling, subtasks can be add after calling this
            function
        '''
        if not seed:
            seed = get_magic_seed()
        data = self._validate_data(data, ds)
        # ====== create task ====== #
        task = _task(name, func, data, epoch=epoch, p=p, seed=seed, mode=mode)
        task.set_iter(bs, start, end, shuffle, self._iter_mode)
        task.set_callback(self)
        # ====== store the task ====== #
        self._task_list.append(task)
        self._last_task = task # store last task for adding subtask
        return self

    def add_subtask(self, name, func, data, ds=None,
                 single_run=False, freq=0.,
                 epoch=1, p=1.,
                 bs=128, shuffle=True,
                 start=0., end=1.,
                 mode=1):
        ''' Set task for running.
        A task is a function operates on a list of ``odin.dataset.batch``, the
        batch return an iterator with add necessary data for the function.

        Subtask will inherit name, seed and dataset from its parents

        Parameters
        ----------
        func : function
            for training, data contain all training data and validation data names
            example: [['X_train','y_train'],['X_valid','y_valid']]
                  or {'train':['X_train','y_train'],'valid':['X_valid','y_valid']}
            In case of missing data, strategy will take default data names form
            set_dataset method
        data : str, ``odin.dataset.batch``, iterator, function
            check document of ``add_data``
        ds : ``odin.dataset.dataset``, paths
            if ds is a paths (strings), we create a dataset to handle all string
            specified in ``data``
        single_run : bool
            only run one time after each main task batch, or run until finished
        freq : int, float(0.-1.)
            after fixed amount of iteration, execute this subtask
        epoch : int
            number of epoch to run this batch
        p : float (0.-1.)
            probability this task will be execute during its epoches
        bs : int
            batch size, number of samples for each batch
        shuffle : boolean
            shuffle dataset while training
        start : int, float(0.-1.)
            starting point within each ``odin.dataset.batch``
        end : int, float(0.-1.)
            ending point within each ``odin.dataset.batch``
        mode : 0, 1, 2
            0 - default, sequentially read each dataset
            1 - parallel read: proportionately for each dataset (e.g.
                batch_size=512, dataset1_size=1000, dataset2_size=500
                => ds1=341, ds2=170)
            2 - parallel read (upsampling): upsampling smaller dataset
                (e.g. batch_size=512, there are 5 dataset => each dataset
                102 samples) (only work if batch size <<
                dataset size)
            3 - parallel read (downsampling): downsampling larger dataset
                (e.g. batch_size=512, there are 5 dataset => each dataset
                102 samples) (only work if batch size <<
                dataset size)

        Returns
        -------
        return : trainer
            for chaining method calling, subtasks can be add after calling this
            function
        '''
        if self._last_task is None:
            raise ValueError('Call add_task first to set parents task')
        # n = len(self._subtask_map[self._last_task])
        # name = self._last_task.name + '_' + 'subtask[%d]' % n
        data = self._validate_data(data, ds, self._last_task.ds)
        # ====== create task ====== #
        task = _task(name, func, data, epoch=epoch, p=p,
            seed=self._last_task.seed, mode=mode)
        task.set_iter(bs, start, end, shuffle, self._iter_mode)
        task.set_callback(self)
        # ====== add subtask ====== #
        self._subtask_map[self._last_task].append(task)
        self._subtask_single_run[task] = single_run
        self._subtask_freq[task] = freq
        return self

    # ==================== Helper function ==================== #
    def _early_stop(self):
        # just a function reset stop flag and return its value
        tmp = self._stop_now
        self._stop_now = False
        return tmp

    def _early_restart(self):
        # just a function reset valid flag and return its value
        tmp = self._restart_now
        self._restart_now = False
        return tmp

    # ==================== Main workflow ==================== #
    def _run(self, iterate=False, progress=False):
        if self.idx < len(self._task_list):
            run_idx = range(self.idx, len(self._task_list))
            for i in run_idx:
                task = self._task_list[i]
                niter = task.est_niter()
                subtasks = self._subtask_map[task]
                task_it = task.run_iter()
                sub_it = {i: i.run_iter() for i in subtasks}
                # task_start
                self.task = task.name
                self.idx = i + 1  # id of next task
                self.iter = 0
                self._task_start(self)
                while True:
                    # ====== run task ====== #
                    run_signal = task_it.next()
                    current_it = self.iter
                    if run_signal: # update exact niter
                        niter = current_it / self.epoch
                    # ====== print progress ====== #
                    if progress:
                        logger.progress(current_it % niter, niter,
                                        title=task.name + ', epoch:%d' % self.epoch)
                    # ====== run subtasks ====== #
                    for i, j in sub_it.iteritems():
                        single_run = self._subtask_single_run[i]
                        freq = self._subtask_freq[i]
                        freq = freq if freq > 1 else int(max(1, round(freq * niter)))
                        if current_it % freq == 0:
                            if single_run:
                                j.next()
                            else:
                                maxit = i.est_niter()
                                while j.next() is not None:
                                    if progress:
                                        logger.progress(self.iter, maxit,
                                                        title=i.name)
                                sub_it[i] = i.run_iter() # new run_iter
                    # ====== check signal ====== #
                    if run_signal is None or self._early_stop():
                        break
                    elif self._early_restart():
                        task_it = task.run_iter()
                    if iterate: # step mode
                        yield False
                # task_end
                if progress: print()
                self.task = task.name
                self._task_end(self)
            # step mode
            if iterate:
                while True:
                    yield True
        yield True

    def step(self, restart=False):
        ''' run the training process step-by-step
        Return
        ------
        True : the tasks are finished
        False : if the tasks are running
        '''
        if self._debug_run is None or restart:
            self._debug_run = self._run(iterate=True, progress=False)
        run_signal = self._debug_run.next()
        if run_signal:
            self._debug_run = None
        return run_signal

    def run(self, progress=False):
        '''
        Return
        ------
        True : success finish the tasks
        False : some error happened
        '''
        try:
            if self._debug_run is not None:
                raise ValueError(
                    'Cannot run while haven\'t finished the step function')
            self._run(iterate=False, progress=progress).next()
        except Exception, e:
            self.log(str(e), 40)
            import traceback; traceback.print_exc()
            return False
        return True

    # ==================== Debug ==================== #
    def __del__(self):
        for i in self._del_able_dataset:
            i.close()

    def __str__(self):
        s = '\n'
        s += '=============== Current run:%d \n' % self.idx
        s += 'Epoch start:' + str(self._epoch_start.__name__) + '\n'
        s += 'Epoch end:' + str(self._epoch_end.__name__) + '\n'
        s += 'Batch start:' + str(self._batch_start.__name__) + '\n'
        s += 'Batch end:' + str(self._batch_end.__name__) + '\n'
        s += 'Task start:' + str(self._task_start.__name__) + '\n'
        s += 'Task end:' + str(self._task_end.__name__) + '\n'
        for i, j in enumerate(self._task_list):
            s += '=============== Task:%d \n' % i
            s += str(j)
            for k, n in enumerate(self._subtask_map[j]):
                s += ' ======= Subtask: %d\n' % k
                s += 'Single run: %s\n' % self._subtask_single_run[n]
                s += 'Freq: %s\n' % self._subtask_freq[n]
                s += str(n)
        return s
