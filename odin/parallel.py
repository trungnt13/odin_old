from __future__ import print_function, division, absolute_import

import numpy as np
from .base import OdinObject
from .utils import segment_list

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
npro = comm.Get_size()

__all__ = [
    'MapReduce'
]
# ===========================================================================
# MPI MapReduce
# ===========================================================================
class MapReduce(OdinObject):

    """
    Example
    -------
    >>> a = MapReduce()
    >>> a.set_log_point(10)
    >>> a.set_cache_size(30)
    >>>
    >>> def map_func(v, job):
    >>>     return (v['tmp'], job)
    >>> def reduce_func(v, results):
    >>>     v['test'] += results
    >>> def root_func(v):
    >>>     v['test'] = []
    >>> def global_func(v):
    >>>     v['tmp'] = v['rank']
    >>>
    >>> a.set_init(root_init=root_func, global_init=global_func)
    >>> a.add_task(range(100),
    >>>     map_func,
    >>>     reduce_func).add_task(range(100, 200),
    >>>     map_func,
    >>>     reduce_func).add_task(range(200, 300),
    >>>     map_func,
    >>>     reduce_func)
    >>> a()
    >>> print(a.get_results('test'))
    """

    def __init__(self):
        super(MapReduce, self).__init__()
        self._cache = 30
        self._log = 50
        self._tasks = []
        self._global_vars = {'rank': rank, 'size': npro}
        self._root_rank = 0

    def _check_init_func(self, init_func):
        if init_func:
           if not hasattr(init_func, '__call__') or \
              init_func.func_code.co_argcount != 1:
              raise ValueError('Init function must be callable and has 1 arg')

    def _check_mapreduce_func(self, func):
        if not func or \
           not hasattr(func, '__call__') or \
           func.func_code.co_argcount != 2:
           raise ValueError('Map/Reduce function must be callable and has 2 arg')

    def set_cache_size(self, cache):
        '''
        cache : int
            maximum number of cache for each process before gathering the data
        '''
        self._cache = cache
        return self

    def set_log_point(self, log):
        '''
        log : int
            after this amount of preprocessed data, print log
        '''
        self._log = log
        return self

    def set_root(self, rank):
        self._root_rank = rank

    def set_init(self, root_init=None, rank_init=None, global_init=None):
        '''
        root_init: function(map)
            store variables in root's vars
        rank_init: function(map)
            store variables in all processes except root
        global_init: function(map)
            store variables in all processes
        '''
        self._global_vars = {'rank': rank, 'size': npro}

        if root_init:
            self._check_init_func(root_init)
            if rank == self._root_rank:
                root_init(self._global_vars)

        if rank_init:
            self._check_init_func(rank_init)
            if rank != self._root_rank:
                rank_init(self._global_vars)

        if global_init:
            self._check_init_func(global_init)
            global_init(self._global_vars)
        return self

    def get_results(self, key):
        if key in self._global_vars:
            return self._global_vars[key]
        return None

    def add_task(self, jobs, map_func, reduce_func, init_func=None, name=None):
        ''' Wrapped preprocessing procedure in MPI.
                    root
                / / / | \ \ \
                 mapping_func
                \ \ \ | / / /
                 reduce_func
            * NO need call Barrier at the end of this methods

        Parameters
        ----------
        jobs : list
            [data_concern_job_1, job_2, ....]
        map_func : function(job_i)
            function object to extract feature from each job
        reduce_func : function([job_i,...])
            transfer all data to process 0 as a list for saving to disk

        Notes
        -----
        Any None return by features_func will be ignored

        Example
        -------
        >>> jobs = range(1, 110)
        >>> if rank == 0:
        >>>     f = h5py.File('tmp.hdf5', 'w')
        >>>     idx = 0
        >>> def feature_extract(j):
        >>>     return rank
        >>> def save(j):
        >>>     global idx
        >>>     f[str(idx)] = str(j)
        >>>     idx += 1
        >>> dnntoolkit.mpi.preprocess_mpi(jobs, feature_extract, save, n_cache=5)
        >>> if rank == 0:
        >>>     f['idx'] = idx
        >>>     f.close()
        '''
        self._check_init_func(init_func)
        self._check_mapreduce_func(map_func)
        self._check_mapreduce_func(reduce_func)
        self._tasks.append([jobs, map_func, reduce_func, init_func, name])
        return self

    def _run_mpi(self, jobs_list, init_func, map_func, reduce_func):
        root = self._root_rank
        # create variables
        global_vars = self._global_vars.copy()
        local_vars = {}
        if init_func:
            init_func(local_vars)
        for i, j in local_vars.iteritems():
            global_vars[i] = j

        #####################################
        # 1. Scatter jobs for all process.
        if rank == self._root_rank:
            self.log('Process %d found %d jobs' % (root, len(jobs_list)))
            jobs = segment_list(jobs_list, npro)
            n_loop = max([len(i) for i in jobs])
        else:
            jobs = None
            n_loop = 0
            # self.log('Process %d waiting for Process %d!' % (rank, root))

        jobs = comm.scatter(jobs, root=root)
        n_loop = comm.bcast(n_loop, root=root)
        self.log('[Received] Process %d: %d jobs' % (rank, len(jobs)))
        comm.Barrier()

        #####################################
        # 2. Start preprocessing.
        data = []

        for i in xrange(n_loop):
            if i % self._cache == 0 and i > 0:
                all_data = comm.gather(data, root=root)
                if rank == root:
                    self.log('Reduce all data to process %d' % self._root_rank)
                    all_data = [k for j in all_data for k in j]
                    if len(all_data) > 0:
                        reduce_func(global_vars, all_data)
                data = []

            if i >= len(jobs): continue
            feature = map_func(global_vars, jobs[i])
            if feature is not None:
                data.append(feature)

            if i > 0 and i % self._log == 0:
                self.log(' - [Map] Process %d finished %d files!' % (rank, i))

        #####################################
        # 3. Reduce task
        all_data = comm.gather(data, root=root)
        if rank == root:
            self.log('Finished MapReduce task !!!!')
            all_data = [k for j in all_data for k in j]
            if len(all_data) > 0:
                reduce_func(global_vars, all_data)

    def __call__(self):
        for k, t in enumerate(self._tasks):
            j, m, r, i, n = t[0], t[1], t[2], t[3], t[4]
            if rank == self._root_rank:
                self.log('****** Start %dth task, name:%s ******' % (k, str(n)), 20)
            self._run_mpi(j, i, m, r)
            comm.Barrier()
