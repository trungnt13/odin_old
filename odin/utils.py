from __future__ import print_function, division

from .base import OdinObject
import numpy as np
import types
import time
from collections import defaultdict

# ===========================================================================
# RandomStates
# ===========================================================================
_MAGIC_SEED = 12082518
_SEED_GENERATOR = np.random.RandomState(_MAGIC_SEED)

def set_magic_seed(seed):
    global _MAGIC_SEED
    _MAGIC_SEED = seed

def get_magic_seed():
    return _MAGIC_SEED

def set_seed_generator(seed):
    global _SEED_GENERATOR
    _SEED_GENERATOR = np.random.RandomState(seed)

def seed_generate():
    return _SEED_GENERATOR.randint(0, 10e8)

# ===========================================================================
# DAA
# ===========================================================================
def _get_ms_time():
    return int(round(time.time() * 1000)) # in ms

def _create_comparator(t):
    return lambda x: x == t

def _is_tags_match(func, tags, absolute=False):
    '''
    Example
    -------
        > tags = [1, 2, 3]
        > func = [lambda x: x == 1]
        > func1 = [lambda x: x == 1, lambda x: x == 2, lambda x: x == 3]
        > _is_tags_match(func, tags, absolute=False) # True
        > _is_tags_match(func, tags, absolute=True) # False
        > _is_tags_match(func1, tags, absolute=True) # True
    '''
    for f in func:
        match = False
        for t in tags:
            match |= f(t)
        if not match: return False
    if absolute and len(func) != len(tags):
        return False
    return True

class frame(OdinObject):

    """
    Simple object to record data row in form:
        [time, [tags], values]

    Notes
    -----
    Should store primitive data
    """

    def __init__(self, name=None, description=None):
        super(frame, self).__init__()
        self._name = name
        self._description = description
        self._records = []
        self._init_time = _get_ms_time()

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return self._description

    @property
    def time(self):
        return self._init_time

    def __len__(self):
        return len(self._records)

    # ==================== multiple history ==================== #
    def merge(self, *history):
        # input is a list
        if len(history) > 0 and type(history[0]) in (tuple, list):
            history = history[0]

        h = frame()
        all_names = [self.name] + [i.name for i in history]
        h._name = 'Merge:<' + ','.join([str(i) for i in all_names]) + '>'
        data = []
        data += self._records
        for i in history:
            data += i._records
        data = sorted(data, key=lambda x: x[0])
        h._records = data
        return h

    # ==================== History manager ==================== #
    def clear(self):
        self._records = []

    def record(self, values, *tags):
        curr_time = _get_ms_time()

        if not isinstance(tags, list) and not isinstance(tags, tuple):
            tags = [tags]
        tags = set(tags)

        # timestamp must never equal
        if len(self._records) > 0 and self._records[-1][0] >= curr_time:
            curr_time = self._records[-1][0] + 1

        self._records.append([curr_time, tags, values])

    def update(self, tags, new, after=None, before=None, n=None, absolute=False):
        ''' Apply a funciton to all selected value

        Parameters
        ----------
        tags : list, str, filter function or any comparable object
            get all values contain given tags
        new : function, object, values
            update new values
        after, before : time constraint (in millisecond)
            after < t < before
        n : int
            number of record will be update
        filter_value : function
            function to filter each value found
        absolute : boolean
            whether required the same set of tags or just contain

        '''
        # ====== preprocess arguments ====== #
        history = self._records
        if not isinstance(tags, list) and not isinstance(tags, tuple):
            tags = [tags]
        tags = set(tags)
        tags = [t if hasattr(t, '__call__') else _create_comparator(t) for t in tags]

        if len(history) == 0:
            return []
        if not hasattr(tags, '__len__'):
            tags = [tags]
        if after is None:
            after = history[0][0]
        if before is None:
            before = history[-1][0]
        if n is None or n < 0:
            n = len(history)

        # ====== searching ====== #
        count = 0
        for row in history:
            if count > n:
                break

            # check time
            time = row[0]
            if time < after or time > before:
                continue
            # check tags
            if not _is_tags_match(tags, row[1], absolute):
                continue
            # check value
            if hasattr(new, '__call__'):
                row[2] = new(row[2])
            else:
                row[2] = new
            count += 1

    def select(self, tags, default=None, after=None, before=None,
               filter_value=None, absolute=False, newest=False,
               return_time=False):
        ''' Query in history, the results is returned in order of oldest first

        Parameters
        ----------
        tags : list, str, filter function or any comparable object
            get all values contain given tags
        default : object
            default return value of found nothing
        after, before : time constraint (in millisecond)
            after < t < before
        filter_value : function(value)
            function to filter each value found
        absolute : boolean
            whether required the same set of tags or just contain
        newest : boolean
            returning order (newest first, default is False)
        time : boolean
            whether return time tags

        Returns
        -------
        return : list
            always return list, in case of no value, return empty list
        '''
        # ====== preprocess arguments ====== #
        history = self._records
        if not isinstance(tags, list) and not isinstance(tags, tuple):
            tags = [tags]
        tags = set(tags)
        tags = [t if hasattr(t, '__call__') else _create_comparator(t) for t in tags]

        if len(history) == 0:
            return []
        if after is None:
            after = history[0][0]
        if before is None:
            before = history[-1][0]

        # ====== searching ====== #
        res = []
        for row in history:
            # check time
            time = row[0]
            if time < after or time > before:
                continue
            # check tags
            if not _is_tags_match(tags, row[1], absolute):
                continue
            # check value
            val = row[2]
            if filter_value is not None and not filter_value(val):
                continue
            # ok, found it!
            res.append((time, val))

        # ====== return results ====== #
        if not return_time:
            res = [i[1] for i in res]

        if newest:
            return list(reversed(res))

        if len(res) == 0 and default is not None:
            return default
        return res

    # ==================== pretty print ==================== #
    def __str__(self):
        fmt = '|%13s | %40s | %20s|'
        sep = ('-' * 13, '-' * 40, '-' * 20)
        records = ''
        # header
        records += fmt % sep + '\n'
        records += fmt % ('Time', 'Tags', 'Values') + '\n'
        records += fmt % sep + '\n'
        # content
        unique_tags = defaultdict(int)
        for row in self._records:
            for j in row[1]:
                unique_tags[j] += 1
            records += fmt % tuple([str(i) for i in row]) + '\n'

        s = ''
        s += '======== Frame Records ========' + '\n'
        s += records
        s += '======== Frame Statistics ========' + '\n'
        s += ' - Name: %s' % self._name + '\n'
        s += ' - Description: %s' % self._description + '\n'
        s += ' - Time: %s' % time.ctime(int(self._init_time / 1000)) + '\n'
        s += ' - Statistic:' + '\n'
        s += '   - Len:%d' % len(self._records) + '\n'
        for k, v in unique_tags.iteritems():
            s += '   - tags: %-10s  -  freq:%d' % (k, v) + '\n'
        return s

class queue(object):

    """ FIFO, fast, NO thread-safe queue """

    def __init__(self):
        super(queue, self).__init__()
        self._data = []
        self._idx = 0

    def put(self, value):
        self._data.append(value)

    def append(self, value):
        self._data.append(value)

    def pop(self):
        if self._idx == len(self._data):
            raise ValueError('Queue is empty')
        self._idx += 1
        return self._data[self._idx - 1]

    def get(self):
        if self._idx == len(self._data):
            raise ValueError('Queue is empty')
        self._idx += 1
        return self._data[self._idx - 1]

    def empty(self):
        if self._idx == len(self._data):
            return True
        return False

    def clear(self):
        del self._data
        self._data = []
        self._idx = 0

    def __len__(self):
        return len(self._data) - self._idx

# ===========================================================================
# Utilities functions
# ===========================================================================
def segment_list(l, n_seg):
    '''
    Example
    -------
    >>> segment_list([1,2,3,4,5],2)
    >>> [[1, 2, 3], [4, 5]]
    >>> segment_list([1,2,3,4,5],4)
    >>> [[1], [2], [3], [4, 5]]
    '''
    # by floor, make sure and process has it own job
    size = int(np.ceil(len(l) / float(n_seg)))
    if size * n_seg - len(l) > size:
        size = int(np.floor(len(l) / float(n_seg)))
    # start segmenting
    segments = []
    for i in xrange(n_seg):
        start = i * size
        if i < n_seg - 1:
            end = start + size
        else:
            end = max(start + size, len(l))
        segments.append(l[start:end])
    return segments

def create_batch(n_samples, batch_size,
    start=None, end=None, prng=None, upsample=None, keep_size=False):
    '''
    No gaurantee that this methods will return the extract batch_size

    Parameters
    ----------
    n_samples : int
        size of original full dataset (not count start and end)
    prng : numpy.random.RandomState
        if prng != None, the upsampling process will be randomized
    upsample : int
        upsample > n_samples, batch will be sampled from original data to make
        the same total number of sample
        if [start] and [end] are specified, upsample will be rescaled according
        to original n_samples

    Example
    -------
    >>> from numpy.random import RandomState
    >>> create_batch(100, 17, start=0.0, end=1.0)
    >>> [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
    >>> create_batch(100, 17, start=0.0, end=1.0, upsample=130)
    >>> [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100), (0, 20), (20, 37)]
    >>> create_batch(100, 17, start=0.0, end=1.0, prng=RandomState(12082518), upsample=130)
    >>> [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100), (20, 40), (80, 90)]

    Notes
    -----
    If you want to generate similar batch everytime, set the same seed before
    call this methods
    For odd number of batch and block, a goal of Maximize number of n_block and
    n_batch are applied
    '''
    #####################################
    # 1. Validate arguments.
    if start is None or start >= n_samples or start < 0:
        start = 0
    if end is None or end > n_samples:
        end = n_samples
    if end < start: #swap
        tmp = start
        start = end
        end = tmp

    if start < 1.0:
        start = int(start * n_samples)
    if end <= 1.0:
        end = int(end * n_samples)
    orig_n_samples = n_samples
    n_samples = end - start

    if upsample is None:
        upsample = n_samples
    else: # rescale
        upsample = int(upsample * float(n_samples) / orig_n_samples)
    #####################################
    # 2. Init.
    jobs = []
    n_batch = float(n_samples / batch_size)
    if n_batch < 1 and keep_size:
        raise ValueError('Cannot keep size when number of data < batch size')
    i = -1
    for i in xrange(int(n_batch)):
        jobs.append((start + i * batch_size, start + (i + 1) * batch_size))
    if not n_batch.is_integer():
        if keep_size:
            jobs.append((end - batch_size, end))
        else:
            jobs.append((start + (i + 1) * batch_size, end))

    #####################################
    # 3. Upsample jobs.
    upsample_mode = True if upsample >= n_samples else False
    upsample_jobs = []
    n = n_samples if upsample_mode else 0
    i = 0
    while n < upsample:
        # pick a package
        # ===========================================================================
        # DAA
        # ===========================================================================
        if prng is None:
            added_job = jobs[i % len(jobs)]
            i += 1
        elif prng is not None:
            added_job = jobs[prng.randint(0, len(jobs))]
        tmp = added_job[1] - added_job[0]
        if not keep_size: # only remove redundant size if not keep_size
            if n + tmp > upsample:
                tmp = n + tmp - upsample
                added_job = (added_job[0], added_job[1] - tmp)
        n += added_job[1] - added_job[0]
        # done
        upsample_jobs.append(added_job)

    if upsample_mode:
        return jobs + upsample_jobs
    else:
        return upsample_jobs

# ===========================================================================
# Python utilities
# ===========================================================================
def serialize_sandbox(environment):
    '''environment, dictionary (e.g. globals(), locals())
    Returns
    -------
    dictionary : cPickle dumps-able dictionary to store as text
    '''
    import re
    sys_module = re.compile('__\w+__')
    primitive = (bool, int, float, str,
                 tuple, list, dict, type, types.ModuleType)
    ignore_key = ['__name__', '__file__']
    sandbox = {}
    for k, v in environment.iteritems():
        if k in ignore_key: continue
        if type(v) in primitive and sys_module.match(k) is None:
            if isinstance(v, types.ModuleType):
                v = {'name': v.__name__, '__module': True}
            sandbox[k] = v

    return sandbox

def deserialize_sandbox(sandbox):
    '''
    environment : dictionary
        create by `serialize_sandbox`
    '''
    if not isinstance(sandbox, dict):
        raise ValueError(
            '[environment] must be dictionary created by serialize_sandbox')
    import importlib
    primitive = (bool, int, float, str,
                 tuple, list, dict, type, types.ModuleType)
    environment = {}
    for k, v in sandbox.iteritems():
        if type(v) in primitive:
            if isinstance(v, dict) and '__module' in v:
                v = importlib.import_module(v['name'])
            environment[k] = v
    return environment

class function(OdinObject):

    """ Class handles save and load a function with its arguments
    Note
    ----
    This class does not support nested functions
    All the complex objects must be created in the function
    """

    def __init__(self, func, *args, **kwargs):
        super(function, self).__init__()
        self._function = func
        self._function_name = func.func_name
        self._function_args = args
        self._function_kwargs = kwargs
        self._sandbox = serialize_sandbox(func.func_globals)

    def __call__(self):
        return self._function(*self._function_args, **self._function_kwargs)

    def __str__(self):
        import inspect
        s = 'Name:%s\n' % self._function_name
        s += 'args:%s\n' % str(self._function_args)
        s += 'kwargs:%s\n' % str(self._function_kwargs)
        s += 'Sandbox:%s\n' % str(self._sandbox)
        s += inspect.getsource(self._function)
        return s

    def __eq__(self, other):
        if self._function == other._function and \
           self._function_args == other._function_args and \
           self._function_kwargs == other._function_kwargs:
           return True
        return False

    def get_config(self):
        config = super(function, self).get_config()
        import marshal
        from array import array

        model_func = marshal.dumps(self._function.func_code)
        b = array("B", model_func)
        config['func'] = b
        config['args'] = self._function_args
        config['kwargs'] = self._function_kwargs
        config['name'] = self._function_name
        config['sandbox'] = self._sandbox
        return config

    @staticmethod
    def parse_config(config):
        import marshal

        b = config['func']
        func = marshal.loads(b.tostring())
        func_name = config['name']
        func_args = config['args']
        func_kwargs = config['kwargs']

        sandbox = globals().copy() # create sandbox
        sandbox.update(deserialize_sandbox(config['sandbox']))
        func = types.FunctionType(func, sandbox, func_name)
        return function(func, *func_args, **func_kwargs)

