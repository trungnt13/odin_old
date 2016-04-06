# -*- coding: utf-8 -*-
# ===========================================================================
# Adapted cache mechanism for both function and method
# Some modification from: https://wiki.python.org/moin/PythonDecoratorLibrary
# Copyright 2016-2017 TrungNT
# ===========================================================================
from __future__ import print_function, division

import sys
from collections import OrderedDict, defaultdict
from functools import wraps, partial, WRAPPER_UPDATES, WRAPPER_ASSIGNMENTS
import inspect
from six.moves import zip, zip_longest
import types

__all__ = [
    'cache',
    'typecheck'
]

# ===========================================================================
# Cache
# ===========================================================================
_CACHE = defaultdict(lambda: ([], [])) #KEY_ARGS, RET_VALUE


def cache(func, *attrs):
    '''Decorator. Caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned
    (not reevaluated).

    Parameters
    ----------
    args : str or list(str)
        list of object attributes in comparation for selecting cache value

    Example
    -------
    >>> class ClassName(object):
    >>>     def __init__(self, arg):
    >>>         super(ClassName, self).__init__()
    >>>         self.arg = arg
    >>>     @cache('arg')
    >>>     def abcd(self, a):
    >>>         return np.random.rand(*a)
    >>>     def e(self):
    >>>         pass
    >>> x = c.abcd((10000, 10000))
    >>> x = c.abcd((10000, 10000)) # return cached value
    >>> c.arg = 'test'
    >>> x = c.abcd((10000, 10000)) # return new value
    '''
    if not inspect.ismethod(func) and not inspect.isfunction(func):
        attrs = (func,) + attrs
        func = None

    if any(not isinstance(i, str) for i in attrs):
        raise ValueError('Tracking attribute must be string represented name of'
                         ' attribute, but given attributes have types: {}'
                         ''.format(tuple(map(type, attrs))))

    def wrap_function(func):
        # ====== fetch arguments order ====== #
        _ = inspect.getargspec(func)
        args_name = _.args
        # reversed 2 time so everything in the right order
        if _.defaults is not None:
            args_defaults = OrderedDict(reversed([(i, j)
                for i, j in zip(reversed(_.args), reversed(_.defaults))]))
        else:
            args_defaults = OrderedDict()

        @wraps(func)
        def wrapper(*args, **kwargs):
            input_args = list(args)
            excluded = {i: j for i, j in zip(args_name, input_args)}
            # check default kwargs
            for i, j in args_defaults.iteritems():
                if i in excluded: # already input as positional argument
                    continue
                if i in kwargs: # specified value
                    input_args.append(kwargs[i])
                else: # default value
                    input_args.append(j)
            # ====== create cache_key ====== #
            object_vars = {k: getattr(args[0], k) for k in attrs
                           if hasattr(args[0], k)}
            cache_key = (input_args, object_vars)
            # ====== check cache ====== #
            key_list = _CACHE[id(func)][0]
            value_list = _CACHE[id(func)][1]
            if cache_key in key_list:
                idx = key_list.index(cache_key)
                return value_list[idx]
            else:
                value = func(*args, **kwargs)
                key_list.append(cache_key)
                value_list.append(value)
                return value
        return wrapper

    # return wrapped function
    if func is None:
        return wrap_function
    return wrap_function(func)


# ===========================================================================
# Type enforcement
# ===========================================================================
def _info(fname, expected, actual, flag):
    '''Convenience function outputs nicely formatted error/warning msg.'''
    format = lambda typess: ', '.join([str(t).split("'")[1] for t in typess])
    expected, actual = format(expected), format(actual)
    ftype = 'method'
    msg = "'{}' {} ".format(fname, ftype)\
          + ("inputs", "outputs")[flag] + " ({}), but ".format(expected)\
          + ("was given", "result is")[flag] + " ({})".format(actual)
    return msg


def typecheck(inputs=None, outputs=None, debug=2):
    '''Function/Method decorator. Checks decorated function's arguments are
    of the expected types.

    Parameters
    ----------
    inputs : types
        The expected types of the inputs to the decorated function.
        Must specify type for each parameter.
    outputs : types
        The expected type of the decorated function's return value.
        Must specify type for each parameter.
    debug : int, str
        Optional specification of 'debug' level:
        0:'ignore', 1:'warn', 2:'raise'

    Examples
    --------
    >>> # Function typecheck
    >>> @typecheck(inputs=(int, str, float), outputs=(str))
    >>> def function(a, b, c):
    ...     return b
    >>> function(1, '1', 1.) # no error
    >>> function(1, '1', 1) # error, final argument must be float
    ...
    >>> # method typecheck
    >>> class ClassName(object):
    ...     @typecheck(inputs=(str, int), outputs=int)
    ...     def method(self, a, b):
    ...         return b
    >>> x = ClassName()
    >>> x.method('1', 1) # no error
    >>> x.method(1, '1') # error

    '''
    # ====== parse debug ====== #
    if isinstance(debug, str):
        debug_str = debug.lower()
        if 'raise' in debug_str:
            debug = 2
        elif 'warn' in debug_str:
            debug = 1
        else:
            debug = 0
    elif debug not in (0, 1, 2):
        debug = 2
    # ====== check types ====== #
    if inputs is not None and not isinstance(inputs, (tuple, list)):
        inputs = (inputs,)
    if outputs is not None and not isinstance(outputs, (tuple, list)):
        outputs = (outputs,)

    def wrap_function(func):
        # ====== fetch arguments order ====== #
        _ = inspect.getargspec(func)
        args_name = _.args
        # reversed 2 time so everything in the right order
        if _.defaults is not None:
            args_defaults = OrderedDict(reversed([(i, j)
                for i, j in zip(reversed(_.args), reversed(_.defaults))]))
        else:
            args_defaults = OrderedDict()

        @wraps(func)
        def wrapper(*args, **kwargs):
            input_args = list(args)
            excluded = {i: j for i, j in zip(args_name, input_args)}
            # check default kwargs
            for i, j in args_defaults.iteritems():
                if i in excluded: # already input as positional argument
                    continue
                if i in kwargs: # specified value
                    input_args.append(kwargs[i])
                else: # default value
                    input_args.append(j)
            ### main logic
            if debug is 0: # ignore
                return func(*args, **kwargs)
            ### Check inputs
            if inputs is not None:
                # main logic
                length = int(min(len(input_args), len(inputs)))
                argtypes = tuple(map(type, input_args))
                # TODO: smarter way to check argtypes for methods
                if argtypes[:length] != inputs[:length] and \
                   argtypes[1:length + 1] != inputs[:length]: # wrong types
                    msg = _info(func.__name__, inputs, argtypes, 0)
                    if debug is 1:
                        print('TypeWarning:', msg)
                    elif debug is 2:
                        raise TypeError(msg)
            ### get results
            results = func(*args, **kwargs)
            ### Check outputs
            if outputs is not None:
                res_types = ((type(results),)
                             if not isinstance(results, (tuple, list))
                             else tuple(map(type, results)))
                length = min(len(res_types), len(outputs))
                if len(outputs) > len(res_types) or \
                   res_types[:length] != outputs[:length]:
                    msg = _info(func.__name__, outputs, res_types, 1)
                    if debug is 1:
                        print('TypeWarning: ', msg)
                    elif debug is 2:
                        raise TypeError(msg)
            ### finally everything ok
            return results
        return wrapper
    return wrap_function

# Override the module's __call__ attribute
# sys.modules[__name__] = cache
