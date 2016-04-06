# -*- coding: utf-8 -*-
# ===========================================================================
# Adapted cache mechanism for both function and method
# Some modification from: https://wiki.python.org/moin/PythonDecoratorLibrary
# Copyright 2016-2017 TrungNT
# ===========================================================================
from __future__ import print_function, division

import sys
from collections import OrderedDict
from functools import wraps, partial, WRAPPER_UPDATES, WRAPPER_ASSIGNMENTS
import inspect
from six.moves import zip, zip_longest
import types

__all__ = [
    'cache',
    'typecheck'
]


class func_decorator(object):

    '''Top class for all function decorator object'''

    def __init__(self, func):
        if not hasattr(func, '__call__'):
            raise ValueError('func must callable')
        # ====== adapted code from functools.wraps ====== #
        for attr in WRAPPER_ASSIGNMENTS:
            setattr(self, attr, getattr(func, attr))
        for attr in WRAPPER_UPDATES:
            getattr(self, attr).update(getattr(func, attr, {}))
        # self.__name__ = func.__name__
        # self.__module__ = func.__module__
        # self.__doc__ = func.__doc__
        # s1 = set(dir(func))
        # s2 = set(dir(self))
        # print(s2 - s1)
        # print(s1 - s2)
        # print(hasattr(func, '__isabstractmethod__'))

        self.func = func
        self.get_func = None
        self._is_method = False
        # ====== setup some info about the function ====== #
        # if a function decorator, just copy necessary information
        if isinstance(func, func_decorator):
            self.args_name = func.args_name
            self.args_defaults = func.args_defaults
        else: # store argspec
            _ = inspect.getargspec(self.func)
            self.args_name = _.args
            # reversed 2 time so everything in the right order
            if _.defaults is not None:
                self.args_defaults = OrderedDict(reversed([(i, j)
                    for i, j in zip(reversed(_.args), reversed(_.defaults))]))
            else:
                self.args_defaults = OrderedDict()

    def __str__(self):
        return str(self.func)

    # def __repr__(self):
        # '''Return the function's docstring.'''
        # return self.func.__doc__

    def __set__(self, obj, value):
        raise ValueError('Cached function cannot be assigned to new value!')

    def __get__(self, instance, owner):
        '''Support instance methods.'''
        # cache partial object so only 1 object returned
        if self.get_func is None:
            self.get_func = partial(self.__call__, instance)
            if hasattr(self, '__isabstractmethod__'):
                self.get_func.__isabstractmethod__ = self.__isabstractmethod__
            self._is_method = True
        # if instance is not None:
        #     func = types.MethodType(self.__call__, instance, owner)
        # else:
        #     func = types.MethodType(self.__call__, None, owner)
        return self.get_func


# ===========================================================================
# Cache
# ===========================================================================
class _call_memory(func_decorator):

    '''Decorator. Caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned
    (not reevaluated).

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

    def __init__(self, func):
        super(_call_memory, self).__init__(func)
        self._tracking_vars = []
        self.cache_key = []
        self.cache_value = []

    def set_tracking_vars(self, key):
        if key not in self._tracking_vars:
            self._tracking_vars.append(key)

    def __call__(self, *args, **kwargs):
        input_args = list(args)
        # check default kwargs
        for i, j in self.args_defaults.iteritems():
            if i in kwargs: # specified value
                input_args.append(kwargs[i])
            else: # default value
                input_args.append(j)

        # ====== create cache_key ====== #
        object_vars = {}
        if self._is_method:
            for k in self._tracking_vars:
                if hasattr(args[0], k):
                    object_vars[k] = getattr(args[0], k)
        cache_key = (input_args, object_vars)
        # ====== check cache ====== #
        if cache_key in self.cache_key:
            idx = self.cache_key.index(cache_key)
            return self.cache_value[idx]
        else:
            value = self.func(*args, **kwargs)
            self.cache_key.append(cache_key)
            self.cache_value.append(value)
            return value

    def __delete__(self, instance):
        del self.cache_key
        del self.cache_value
        self.cache_key = []
        self.cache_value = []


def cache(*args):
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
    if not args or \
        (not hasattr(args[0], '__call__') and
         any(not isinstance(i, str) for i in args)):
        raise ValueError("Not a valid decorator, only callable or string accepted"
                         ". In case staticmethod, you should place decorator"
                         " in following order: @staticmethod:@cache")

    if len(args) == 1 and hasattr(args[0], '__call__'):
        application_function, = args
        memory = _call_memory(application_function)
        return memory
    else:
        def wrap_application(application_function):
            memory = _call_memory(application_function)
            for i in args:
                memory.set_tracking_vars(i)
            return memory
        return wrap_application


# ===========================================================================
# Type enforcement
# ===========================================================================
def _info(fname, expected, actual, flag, is_method):
    '''Convenience function outputs nicely formatted error/warning msg.'''
    format = lambda typess: ', '.join([str(t).split("'")[1] for t in typess])
    expected, actual = format(expected), format(actual)
    ftype = 'method' if is_method else 'function'
    msg = "'{}' {} ".format(fname, ftype)\
          + ("accepts", "outputs")[flag] + " ({}), but ".format(expected)\
          + ("was given", "result is")[flag] + " ({})".format(actual)
    return msg


class _typecheck(func_decorator):

    """docstring for _typecheck"""

    def __init__(self, func):
        super(_typecheck, self).__init__(func)
        self.inputs = None
        self.outputs = None
        self.debug = 2

    def set_types(self, inputs=None, outputs=None, debug=2):
        # ====== parse debug ====== #
        if isinstance(debug, str):
            if 'raise' in debug.lower():
                debug = 2
            elif 'warn' in debug.lower():
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
        # ====== assign ====== #
        self.inputs = inputs
        self.outputs = outputs
        self.debug = debug

    def __call__(self, *args, **kwargs):
        if self.debug is 0: # ignore
            return self.func(*args)
        ### Check inputs
        if self.inputs is not None:
            # little hack to deal with classmethod
            if self._is_method or '<class ' in str(type(args[0])): # ignore self
                input_args = list(args[1:])
            else: # full args
                input_args = list(args)
            # check default kwargs
            for i, j in self.args_defaults.iteritems():
                if i in kwargs: # specified value
                    input_args.append(kwargs[i])
                else: # default value
                    input_args.append(j)
            # main logic
            length = int(min(len(input_args), len(self.inputs)))
            argtypes = tuple(map(type, input_args))
            if argtypes[:length] != self.inputs[:length]: # wrong types
                msg = _info(self.__name__, self.inputs, argtypes, 0,
                            self._is_method)
                if self.debug is 1:
                    print('TypeWarning:', msg)
                elif self.debug is 2:
                    raise TypeError(msg)
        ### get results
        results = self.func(*args, **kwargs)
        ### Check outputs
        if self.outputs is not None:
            res_types = ((type(results),)
                         if not isinstance(results, (tuple, list))
                         else tuple(map(type, results)))
            length = min(len(res_types), len(self.outputs))
            if len(self.outputs) > len(res_types) or \
               res_types[:length] != self.outputs[:length]:
                msg = _info(self.__name__, self.outputs, res_types, 1,
                            self._is_method)
                if self.debug is 1:
                    print('TypeWarning: ', msg)
                elif self.debug is 2:
                    raise TypeError(msg)
        ### finally everything ok
        return results


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
    def wrap_function(func):
        f = _typecheck(func)
        f.set_types(inputs=inputs, outputs=outputs, debug=debug)
        return f
    return wrap_function

# Override the module's __call__ attribute
# sys.modules[__name__] = cache
