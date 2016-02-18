# ===========================================================================
# Some functions in this module adpats the idea from: Lasagne library
# Original idea Copyright (c) 2014-2015 Lasagne contributors
# Modified work Copyright 2016-2017 TrungNT
# ===========================================================================

from __future__ import print_function, division, absolute_import

from collections import OrderedDict
import numpy as np
from six.moves import zip, range

from . import logger
from . import tensor as T
from .utils import get_object_api

# ===========================================================================
# Based class design
# ===========================================================================
from abc import ABCMeta, abstractmethod, abstractproperty

class OdinObject(object):
    __metaclass__ = ABCMeta
    _logging = True

    def get_config(self):
        ''' Always return as pickle-able dictionary '''
        config = OrderedDict()
        config['class'] = self.__class__.__name__
        return config

    @staticmethod
    def parse_config(config):
        raise NotImplementedError()

    def set_logging(self, enable):
        self._logging = enable

    def log(self, msg, level=20):
        '''
        VERBOSITY level:
         - CRITICAL: 50
         - ERROR   : 40
         - WARNING : 30
         - INFO    : 20
         - DEBUG   : 10
         - UNSET   : 0
        '''
        if not self._logging:
            return
        msg = '[%s]: %s' % (self.__class__.__name__, str(msg))
        if level == 10:
            logger.debug(msg)
        elif level == 20:
            logger.info(msg)
        elif level == 30:
            logger.warning(msg)
        elif level == 40:
            logger.error(msg)
        elif level == 50:
            logger.critical(msg)
        else:
            logger.log(msg)

class OdinFunction(OdinObject):
    __metaclass__ = ABCMeta

    '''
    Properties
    ----------
    input_shape : list(shape_tuple)
        always a list of shape tuple
    input_function : list
        a list of theano, tensorflow expression or placeholder
    input_var : list
        list of placeholders for input of this functions
    output_var : list
        list of placeholders for output of this functions

    Parameters
    ----------
    incoming : a :class:`OdinFunction` instance or a tuple
        The layer feeding into this layer, or the expected input shape.
    tags : a string, None or list of string
        An optional identifiers to attach to this layer.

    '''

    def __init__(self, incoming, tags=None):
        super(OdinFunction, self).__init__()

        if isinstance(incoming, (tuple, list)) and \
           isinstance(incoming[-1], int):
            self.input_shape = [tuple(incoming)]
            self.input_function = [None]
        else:
            if not isinstance(incoming, (tuple, list)):
                incoming = [incoming]
            self.input_function = incoming
            self.input_shape = [i.output_shape for i in incoming]

        if not isinstance(tags, (tuple, list)):
            self.tags = [tags]

        self.params = OrderedDict()
        self.params_tags = OrderedDict()

        self._input_var = None
        self._output_var = None

    # ==================== Abstract methods ==================== #
    @abstractproperty
    def output_shape(self):
        raise NotImplementedError

    @abstractmethod
    def get_cost(self, objectives=None, training=True):
        raise NotImplementedError

    @abstractmethod
    def __call__(self, training=False):
        raise NotImplementedError

    # ==================== Built-in ==================== #
    @property
    def input_var(self):
        '''
        Return
        ------
        placeholder : list
            list of placeholder for input of this function
        '''

        if self._input_var is None:
            self._input_var = []
            for i, j in zip(self.input_function, self.input_shape):
                if i is None:
                    self._input_var.append(T.placeholder(shape=j))
                elif T.is_placeholder(i):
                    self._input_var.append(i)
                else:
                    api = get_object_api(i)
                    if api == 'lasagne':
                        import lasagne
                        self._input_var += [l.input_var
                            for l in lasagne.layers.get_all_layers(i)
                            if hasattr(l, 'input_var')]
                    elif api == 'keras':
                        tmp = i.get_input(train=True)
                        if hasattr(tmp, 'len'):
                            self._input_var += tmp
                        else:
                            self._input_var.append(tmp)
                    elif api == 'odin':
                        self._input_var += i.input_var
        return self._input_var

    @property
    def output_var(self):
        if self._output_var is None:
            outshape = self.output_shape
            if not isinstance(outshape[0], (tuple, list)):
                outshape = [outshape]
            self._output_var = [T.placeholder(ndim=len(i)) for i in outshape]
        return self._output_var

    def get_inputs(self, training=True):
        '''
        Parameters
        ----------
        training : bool
            if True, return the intermediate input (output from previous
            function). If False, return the placeholder variables can be used
            as input for the whole graph.

        Return
        ------
        input : list(expression)
            theano or tensorflow expression which represent intermediate input
            to this function.
        '''
        inputs = []

        for i in self.input_function:
            # this is InputLayer
            if i is None or T.is_placeholder(i):
                return self.input_var
            # this is expression
            else:
                api = get_object_api(i)
                if api == 'lasagne':
                    import lasagne
                    inputs.append(lasagne.layers.get_output(i, deterministic=(not training)))
                elif api == 'keras':
                    inputs.append(i.get_output(train=training))
                elif api == 'odin':
                    inputs.append(i(training=training))
                elif T.is_variable(self.input_function):
                    inputs.append(i)
        return inputs

    def get_params(self, globals, trainable=None, regularizable=None):
        params = []
        if globals:
            for i in self.input_function:
                if i is not None:
                    params += i.get_params(globals, trainable, regularizable)

        cond_trainable = [True, False]
        if trainable is True:
            cond_trainable = [True]
        elif trainable is False:
            cond_trainable = [False]

        cond_regularizable = [True, False]
        if regularizable is True:
            cond_regularizable = [True]
        elif regularizable is False:
            cond_regularizable = [False]

        cond = lambda x, y: x in cond_trainable and y in cond_regularizable
        local_params = [j for i, j in self.params.iteritems()
                        if cond(self.params_tags[i + '_trainable'],
                                self.params_tags[i + '_regularizable'])
                        ]

        return params + local_params

    def get_params_value(self, globals, trainable=None, regularizable=None):
        return [T.get_value(x) for x in
        self.get_params(globals, trainable, regularizable)]

    def create_params(self, spec, shape, name, regularizable, trainable):
        shape = tuple(shape)  # convert to tuple if needed
        if any(d <= 0 for d in shape):
            raise ValueError((
                "Cannot create param with a non-positive shape dimension. "
                "Tried to create param with shape=%r, name=%r") % (shape, name))

        if T.is_variable(spec):
            # We cannot check the shape here, Theano expressions (even shared
            # variables) do not have a fixed compile-time shape. We can check the
            # dimensionality though.
            # Note that we cannot assign a name here. We could assign to the
            # `name` attribute of the variable, but the user may have already
            # named the variable and we don't want to override this.
            if T.ndim(spec) != len(shape):
                raise RuntimeError("parameter variable has %d dimensions, "
                                   "should be %d" % (spec.ndim, len(shape)))
        elif isinstance(spec, np.ndarray):
            if spec.shape != shape:
                raise RuntimeError("parameter array has shape %s, should be "
                                   "%s" % (spec.shape, shape))
            spec = T.variable(spec, name=name)
        elif hasattr(spec, '__call__'):
            arr = spec(shape)
            if T.is_variable(arr):
                spec = arr
            else:
                if T.is_placeholder(arr):
                    # we do not support expression as params
                    arr = T.eval(arr)
                if arr.shape != shape:
                    raise RuntimeError("cannot initialize parameters: the "
                                       "provided callable did not return a value "
                                       "with the correct shape")
                spec = T.variable(arr, name=name)
        else:
            raise RuntimeError("cannot initialize parameters: 'spec' is not "
                               "a numpy array, a Theano expression, or a "
                               "callable")
        self.params[name] = spec
        self.params_tags[name + '_regularizable'] = regularizable
        self.params_tags[name + '_trainable'] = trainable
        return spec
