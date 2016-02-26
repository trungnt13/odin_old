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
from .utils import api as API
from abc import ABCMeta, abstractmethod, abstractproperty

# ===========================================================================
# Constants
# ===========================================================================
_FOOT_PRINT = '''
%s is activated with: training=%s \n
 - input_shape:  %-15s   - inputs:  %s
 - output_shape: %-15s   - outputs: %s
'''

# ===========================================================================
# Based class design
# ===========================================================================

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
        return self

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

    def raise_arguments(self, msg):
        raise ValueError('[%s] ' % self.__class__.__name__ + msg)


class OdinFunction(OdinObject):
    __metaclass__ = ABCMeta

    '''
    Properties
    ----------
    input_shape : list
        always a list of shape tuples
    output_shape : list
        always a list of shape tuples
    input_function : list
        can be one of these: shape_tuple, lasagne_layers, keras_model,
        odin_funciton, shared_variable
    input_var : list
        list of placeholders for input of this functions
    output_var : list
        list of placeholders for output of this functions, None if the function
        is unsupervised function

    Parameters
    ----------
    incoming : a :class:`OdinFunction`, Lasagne :class:`Layer` instance,
               keras :class:`Models` instance, or a tuple
        The layer feeding into this layer, or the expected input shape.
    unsupervised : bool
        whether or not this is unsupervised model, this affect the output_var
        will be the same as input_var(unsupervised) or based-on output_shape
        (supervised)
    strict_batch : bool
        whether it is necessary to enforce similar batch size for training for
        this function
    name : a string, None or list of string
        An optional identifiers to attach to this layer.

    '''

    def __init__(self, incoming, unsupervised,
                 strict_batch=False, name=None, **kwargs):
        super(OdinFunction, self).__init__()
        self._unsupervised = unsupervised
        self._strict_batch = strict_batch

        self.set_incoming(incoming)
        # ====== other properties ====== #
        if name is None:
            name = []
        elif not isinstance(name, (tuple, list)):
            name = [name]
        self._name = name

        self.params = OrderedDict()
        self.params_tags = OrderedDict()

        # this is dirty hack that allow other functions to modify the inputs
        # of this function right before getting the outputs
        self._intermediate_inputs = None

    # ==================== Layer utilities ==================== #
    def set_incoming(self, incoming):
        # ====== parse incoming layers ====== #
        if not isinstance(incoming, (tuple, list)) or \
           isinstance(incoming[-1], (int, long, float)):
           incoming = [incoming]
        input_function = []
        input_shape = []
        for i in incoming:
            if isinstance(i, (tuple, list)): # shape_tuple
                input_function.append(None)
                input_shape.append(
                    tuple([j if j is None else int(j) for j in i]))
            elif hasattr(i, 'output_shape'):
                input_function.append(i)
                outshape = i.output_shape
                # OdinFunction always return list
                if isinstance(outshape[0], (tuple, list)):
                    input_shape += outshape
                else: # other framework only return 1 output shape
                    input_shape.append(outshape)
            else:
                self.raise_arguments(
                    'Unsupport incomming type %s' % i.__class__)
        self._incoming = input_function
        self._input_shape = input_shape

        # store ALL placeholder necessary for inputs of this Function
        self._input_var = None
        #{index : placeholder}, store placeholder created by this Function
        self._local_input_var = {}
        self._output_var = None
        return self

    def set_intermediate_inputs(self, inputs):
        self._intermediate_inputs = inputs
        return self

    def get_roots(self):
        ''' Performing Depth First Search to return the OdinFunction
        which is the root (all functions that contains placeholder input
        variables) of this Funciton

        Return
        ------
        OdinFunction : list of OdinFunction which contains placeholder for
            this Function

        Note
        ----
        This function preseves order of inputs for multiple inputs function

        '''
        roots = []
        for i in self._incoming:
            if i is None:
                roots.append(self)
            elif isinstance(i, OdinFunction):
                roots += i.get_roots()
        if len(roots) == 0:
            return [self]
        return T.np_ordered_set(roots).tolist()

    def get_children(self):
        ''' Performing Depth First Search to return all the OdinFunction
        which act as inputs to this function

        Return
        ------
        list : list of OdinFunction which contains placeholder for
            this Function

        Note
        ----
        This function preseves order of inputs for multiple inputs function

        '''
        children = []
        for i in self._incoming:
            if isinstance(i, OdinFunction):
                children.append(i)
                children += i.get_children()
        return T.np_ordered_set(children).tolist()

    # ==================== Helper private functions ==================== #
    def _log_footprint(self, training, inputs, outputs, **kwargs):
        self.log(_FOOT_PRINT %
            (self.name, training, self.input_shape,
            inputs, self.output_shape, outputs), 20)

    def _validation_optimization_params(self, objective, optimizer):
        if objective is None or not hasattr(objective, '__call__'):
            raise ValueError('objectives must be a function!')
        if optimizer is not None and not hasattr(optimizer, '__call__'):
            raise ValueError('optimizer must be a function!')

    def _deterministic_optimization_procedure(self, objective, optimizer,
                                              globals, training):
        self._validation_optimization_params(objective, optimizer)
        y_pred = self(training=training)
        y_true = self.output_var
        # ====== caluclate objectives for each in-out pair ====== #
        obj = T.castX(0.)
        # in case of multiple output, we take the mean of loss for each output
        for yp, yt in zip(y_pred, y_true):
            o = objective(yp, yt)
            # if multiple-dimension cannot calculate gradients
            # hence, we take mean of the objective
            if T.ndim(o) > 0:
                o = T.mean(o)
            obj = obj + o
        obj = obj / len(y_pred)
        # ====== get optimizer ====== #
        if optimizer is None:
            opt = None
        else:
            params = self.get_params(globals=globals, trainable=True)
            if globals: # optimize all params
                grad = T.gradients(obj, params)
            else: # optimize only the params of this funtions
                grad = T.gradients(obj, params,
                    consider_constant=self._last_inputs)
            opt = optimizer(grad, params)
        return obj, opt

    # ==================== Abstract methods ==================== #
    @abstractproperty
    def output_shape(self):
        raise NotImplementedError

    @abstractmethod
    def __call__(self, training=False):
        raise NotImplementedError

    @abstractmethod
    def get_optimization(self, objective=None, optimizer=None,
                         globals=True, training=True):
        '''
        Parameters
        ----------
        objective : function
            often a function(y_pred, y_true) for supervised function, however,
            can have different form for unsupervised task
        optimizer : function (optional)
            function(loss_or_grads, params)
        globals : bool
            training on globals' parameters, or just optimize locals' parameters
        training : bool
            use output for training or output for prediction (in production)

        Return
        ------
        cost, updates : computational expression, OrderDict
            cost for monitoring the training process, and the update for the
            optimization
        '''
        raise NotImplementedError

    # ==================== Built-in ==================== #
    @property
    def name(self):
        if len(self._name) == 0:
            return self.__class__.__name__
        return ','.join(self._name)

    @property
    def strict_batch(self):
        return self._strict_batch

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def incoming(self):
        ''' list of placeholder for input of this function '''
        return self._incoming

    @property
    def unsupervised(self):
        return self._unsupervised

    @property
    def input_var(self):
        ''' list of placeholder for input of this function
        Note
        ----
        This property doesn't return appropriate inputs for this funcitons,
        just the placeholder to create T.function
        '''

        if self._input_var is None:
            self._input_var = []
            for idx, (i, j) in enumerate(zip(self._incoming, self._input_shape)):
                if i is None:
                    x = T.placeholder(shape=j,
                        name='in:%s:' % (str(j)) + self.name)
                    self._input_var.append(x)
                    self._local_input_var[idx] = x
                elif T.is_placeholder(i):
                    self._input_var.append(i)
                    self._local_input_var[idx] = x
                else: # input from API layers
                    api = API.get_object_api(i)
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
            self._input_var = T.np_ordered_set(self._input_var).tolist()
        return self._input_var

    @property
    def output_var(self):
        if self._output_var is None:
            self._output_var = []
            if self.unsupervised:
                pass
            else:
                for outshape in self.output_shape:
                    self._output_var.append(T.placeholder(shape=outshape,
                            name='out:%s:' % (str(outshape)) + self.name))
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
        training : bool
            whether in training mode or not

        Note
        ----
        DO NOT call this methods multiple times, it will create duplicated
        unnecessary computation node on graph, self._last_inputs if you the
        inputs already created

        '''
        # ====== Dirty hack, modify intermediate inputs of function ====== #
        if self._intermediate_inputs is not None:
            inputs = self._intermediate_inputs
            self._intermediate_inputs = None
            self._last_inputs = inputs
            return inputs
        # ====== getting the inputs from nested functions ====== #
        inputs = []
        self.input_var # make sure initialized all placeholder
        for idx, i in enumerate(self._incoming):
            # this is InputLayer
            if i is None or T.is_placeholder(i):
                inputs.append(self._local_input_var[idx])
            # this is expression
            else:
                api = API.get_object_api(i)
                if api == 'lasagne':
                    import lasagne
                    inputs.append(
                        lasagne.layers.get_output(i, deterministic=not training))
                elif api == 'keras':
                    inputs.append(i.get_output(train=training))
                elif api == 'odin':
                    o = i(training=training)
                    if isinstance(o, (tuple, list)):
                        inputs += o
                    else:
                        inputs.append(o)
                elif T.is_variable(self._incoming):
                    inputs.append(i)
        # cache the last calculated inputs (if you want to disconnect
        # gradient from this input downward, don't re-build the input
        # graph)
        self._last_inputs = inputs
        return inputs

    def get_params(self, globals, trainable=None, regularizable=None):
        params = []
        # ====== Get all params from nested functions if globals mode on ====== #
        if globals:
            for i in self._incoming:
                if i is not None:
                    params += i.get_params(globals, trainable, regularizable)

        # ====== Params from this layers ====== #
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
        if T.is_variable(spec) or T.is_placeholder(spec):
            # We cannot check the shape here, Theano expressions (even shared
            # variables) do not have a fixed compile-time shape. We can check the
            # dimensionality though.
            # Note that we cannot assign a name here. We could assign to the
            # `name` attribute of the variable, but the user may have already
            # named the variable and we don't want to override this.
            if shape is not None and T.ndim(spec) != len(shape):
                raise RuntimeError("parameter variable has %d dimensions, "
                                   "should be %d" % (spec.ndim, len(shape)))
        elif isinstance(spec, np.ndarray):
            if shape is not None and spec.shape != shape:
                raise RuntimeError("parameter array has shape %s, should be "
                                   "%s" % (spec.shape, shape))
            spec = T.variable(spec, name=name)
        elif hasattr(spec, '__call__'):
            shape = tuple(shape)  # convert to tuple if needed
            if any(d <= 0 for d in shape):
                raise ValueError((
                    "Cannot create param with a non-positive shape dimension. "
                    "Tried to create param with shape=%r, name=%r") %
                    (shape, name))

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

class OdinUnsupervisedFunction(OdinFunction):

    def __init__(self, incoming, strict_batch=False, name=None, **kwargs):
        super(OdinUnsupervisedFunction, self).__init__(
            incoming, unsupervised=True, strict_batch=strict_batch,
            name=name, **kwargs)
        self._reconstruction_mode = False

    def set_reconstruction_mode(self, reconstruct):
        self._reconstruction_mode = reconstruct
        return self
