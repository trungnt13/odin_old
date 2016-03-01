from __future__ import print_function, division, absolute_import

import numpy as np
from .base import OdinObject
from .utils import segment_list, queue

__all__ = [
    'MapReduce'
]
# ===========================================================================
# MPI MapReduce
# ===========================================================================
class MapReduce(OdinObject):

    """ This class manage all MapReduce task by callback function:

    map_function : argmuents(static_data, job)
        static_data: dictionary, which initialized right after you set the
        init_funciotn
        job: is a single job that automatically scheduled to each MPI process

    reduce_function : arguments(static_data, results, finnished)
        static_data: dictionary, which initialized right after you set the
        init_funciotn
        results: list, of returned result from each map_function (None returned
        will be ignored)
        finnished: bool, whether this function is called at the end of MapReduce
        task

    Default information contained in the data dictionary:
        rank : rank of current process
        npro : number of processor used
        task : name of current task
    * You shouldn't override these values

    Example
    -------
    >>> mr = odin.parallel.MapReduce()
    >>>
    >>> mr.set_log_point(10)
    >>> mr.set_cache_size(20)
    >>>
    >>> def global_init(data):
    >>>     data['global'] = 'global'
    >>> mr.set_init({'root': 'root'}, {'shit': 'shit'}, global_init)
    >>>
    >>> def map_func(data, j):
    >>>     return str(j) + '_' + str(data['rank']) + '_' + str(data['init'])
    >>>
    >>> def reduce_func(data, r, finnished):
    >>>     data['root'] = data['task']
    >>>     for i in r:
    >>>         data['results'].append(i)
    >>>
    >>>     if finnished:
    >>>         data['results'].append('finnished')
    >>>         print(data['results'], len(data['results']))
    >>>
    >>> mr.add_task(range(100), map_func, reduce_func, init={'init': 'st1', 'results': []},
    >>>     update_data=True, name='task1')
    >>> mr.add_task(range(200), map_func, reduce_func, init={'init': 'st2', 'results': []},
    >>>     update_data=False, name='task2')
    >>> mr()
    >>> print(mr._global_vars)
    """

    def __init__(self):
        super(MapReduce, self).__init__()
        # MPI information
        from mpi4py import MPI
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.npro = self.comm.Get_size()
        if self.rank == 0:
            self.log('MPI started with %d processes' % self.npro)
        # variables
        self._cache = 30
        self._log = 50
        self._tasks = queue()
        self._global_vars = {'rank': self.rank, 'size': self.npro}
        self._root_rank = 0

    # ==================== Helper ==================== #
    def _check_init(self, init):
        if init:
            if isinstance(init, dict) or \
                (hasattr(init, '__call__') and init.func_code.co_argcount == 1):
                return
            raise ValueError('Init can be a function or a dictionary. If a'
                             ' function is given, it must be callable '
                             'and has only 1 arg')

    def _check_mapreduce_func(self, func):
        if func is not None and \
           hasattr(func, '__call__') and \
           (func.func_code.co_argcount == 2 or func.func_code.co_argcount == 3):
           return
        raise ValueError('Map/Reduce function must be callable and '
                         'has 2 or 3 args')

    # ==================== Get & set ==================== #
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

    def set(self, key, value):
        self._global_vars[key] = value
        return self

    def get(self, key):
        if key in self._global_vars:
            return self._global_vars[key]
        return None

    def set_init(self, root_init=None, rank_init=None, global_init=None):
        '''
        root_init: function(dictionary), dictionary
            store variables in root's vars. If dictionary is given, update all
            data dictionary according to given dictionary
        rank_init: function(map), dictionary
            store variables in all processes except root. If dictionary is given,
            update all data dictionary according to given dictionary
        global_init: function(map), dictionary
            store variables in all processes. If dictionary is given, update all
            data dictionary according to given dictionary
        '''
        self._global_vars = {'rank': self.rank, 'size': self.npro}

        if root_init:
            self._check_init(root_init)
            if self.rank == self._root_rank:
                if hasattr(root_init, '__call__'):
                    root_init(self._global_vars)
                elif isinstance(root_init, dict):
                    for i, j in root_init.iteritems():
                        self._global_vars[i] = j

        if rank_init:
            self._check_init(rank_init)
            if self.rank != self._root_rank:
                if hasattr(rank_init, '__call__'):
                    rank_init(self._global_vars)
                elif isinstance(rank_init, dict):
                    for i, j in rank_init.iteritems():
                        self._global_vars[i] = j

        if global_init:
            self._check_init(global_init)
            if hasattr(global_init, '__call__'):
                global_init(self._global_vars)
            elif isinstance(global_init, dict):
                for i, j in global_init.iteritems():
                    self._global_vars[i] = j
        return self

    def add_task(self, jobs, map_func, reduce_func,
        init=None, update_data=True, name=None):
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

        map_func : function(dict, job_i)
            function object to extract feature from each job, the dictionary
            will contain all static data initilized from set_init function

        reduce_func : function(dict, [job_i,...], finnished)
            transfer all data to process 0 as a list for saving to disk, the
            dictionary will contain all static data initilized from set_init
            function

        update_data : bool
            if True all the data dictionary will be updated according to the
            changes made during MapReduce

        Notes
        -----
        Any None return by features_func will be ignored

        '''
        self._check_init(init)
        self._check_mapreduce_func(map_func)
        self._check_mapreduce_func(reduce_func)
        self._tasks.append([jobs, map_func, reduce_func, init, update_data, name])
        return self

    def _run_mpi(self, jobs_list, init, map_func, reduce_func, update, name):
        root = self._root_rank
        # create variables
        global_vars = self._global_vars.copy()
        global_vars['task'] = name
        if init:
            if hasattr(init, '__call__'):
                init(global_vars)
            elif isinstance(init, dict):
                for i, j in init.iteritems():
                    global_vars[i] = j

        #####################################
        # 1. Scatter jobs for all process.
        if self.rank == self._root_rank:
            self.log('Process %d found %d jobs' % (root, len(jobs_list)))
            jobs = segment_list(jobs_list, self.npro)
            n_loop = max([len(i) for i in jobs])
        else:
            jobs = None
            n_loop = 0
            # self.log('Process %d waiting for Process %d!' % (rank, root))

        jobs = self.comm.scatter(jobs, root=root)
        n_loop = self.comm.bcast(n_loop, root=root)
        self.log('[Received] Process %d: %d jobs' % (self.rank, len(jobs)))
        self.comm.Barrier()

        #####################################
        # 2. Start preprocessing.
        data = []

        for i in xrange(n_loop):
            if i % self._cache == 0 and i > 0:
                all_data = self.comm.gather(data, root=root)
                if self.rank == root:
                    self.log('Reduce all data to process %d' % self._root_rank)
                    all_data = [k for j in all_data for k in j]
                    if len(all_data) > 0:
                        reduce_func(global_vars, all_data, False)
                data = []

            if i >= len(jobs): continue
            feature = map_func(global_vars, jobs[i])
            if feature is not None:
                data.append(feature)

            if i > 0 and i % self._log == 0:
                self.log(' - [Map] Process %d finished %d files!' % (self.rank, i))

        #####################################
        # 3. Reduce task
        all_data = self.comm.gather(data, root=root)
        if self.rank == root:
            self.log('Finished MapReduce task !!!!')
            all_data = [k for j in all_data for k in j]
            if len(all_data) > 0:
                reduce_func(global_vars, all_data, True)

        #####################################
        # 4. Update global variables.
        if update:
            for i, j in global_vars.iteritems():
                if i in self._global_vars:
                    self._global_vars[i] = j

    def __call__(self):
        while not self._tasks.empty():
            t = self._tasks.get()
            j, m, r, i, u, n = t[0], t[1], t[2], t[3], t[4], t[5]
            if self.rank == self._root_rank:
                self.log('****** Start new task, name:%s ******' % (str(n)), 20)
            self._run_mpi(j, i, m, r, u, n)
            self.comm.Barrier()
