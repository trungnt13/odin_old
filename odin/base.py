# ===========================================================================
# Some functions in this module adpats the idea from: Lasagne library
# Original idea Copyright (c) 2014-2015 Lasagne contributors
# Modified work Copyright 2016-2017 TrungNT
# ===========================================================================

from __future__ import print_function, division, absolute_import

from collections import OrderedDict, defaultdict
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

    def raise_runtime(self, msg):
        raise RuntimeError('[%s] ' % self.__class__.__name__ + msg)


class OdinParams(OdinObject):

    """ Wrapper for storing Parameters of both training and predicting process
    params can be OdinFunction, variables, expression, placeholder
    """

    def __init__(self, name, params, shape, trainable, regularizable):
        super(OdinParams, self).__init__()
        if name is None or shape is None or params is None:
            self.raise_arguments('OdinParams cannot have None name or shape.')
        if not T.is_variable(params) and not T.is_expression(params):
            self.raise_arguments('Only variable and expression are accepted '
                                 'to be parameters.')
        self._name = name
        self._shape = shape
        self._params = params
        self._trainable = trainable
        self._regularizable = regularizable

    @property
    def shape(self):
        return self._shape

    @property
    def trainable(self):
        return self._trainable

    @property
    def regularizable(self):
        return self._regularizable

    @property
    def name(self):
        return self._name

    def as_variables(self, globals, trainable, regularizable):
        ''' Return list of variables that create this parameters '''
        if trainable is not None and self._trainable != trainable:
            return []
        if regularizable is not None and self._regularizable != regularizable:
            return []
        if T.is_expression(self._params):
            return []
        if T.is_variable(self._params):
            return [self._params]


class OdinFunction(OdinObject):
    __metaclass__ = ABCMeta
    _ID = 0

    """
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
               keras :class:`Models` instance, variable, placeholder or
               shape tuple
        The layer feeding into this layer, or the expected input shape.
    unsupervised : bool
        whether or not this is unsupervised model, this affect the output_var
        will be the same as input_var(unsupervised) or based-on output_shape
        (supervised)
    name : a string, None or list of string
        An optional identifiers to attach to this layer.

    """

    def __init__(self, incoming, unsupervised, name=None, **kwargs):
        super(OdinFunction, self).__init__()
        self._unsupervised = unsupervised

        self.set_incoming(incoming)
        # ====== other properties ====== #
        if name is None:
            name = []
        elif not isinstance(name, (tuple, list)):
            name = [name]
        self._name = name

        # unique identity number of a function during execution
        self._function_id = OdinFunction._ID
        OdinFunction._ID += 1

        # you can have 2 different version of parameters, for training and for
        # making prediction
        self.params = OrderedDict()

        # this is dirty hack that allow other functions to modify the inputs
        # of this function right before getting the outputs
        self._intermediate_inputs = None

        # must store cache outputs, if you build a graph, duplicated output
        # significantly reduce the performance
        self._cache_inputs_train = None
        self._cache_inputs_pred = None

    # ==================== Layer utilities ==================== #
    def set_incoming(self, incoming):
        # ====== Accept None incoming ====== #
        input_function = []
        input_shape = []
        # flags for variables as incoming that can be learnt
        self._learnable_incoming = defaultdict(lambda: False)
        # ====== check if incoming contain list of acceptable info ====== #
        if incoming is not None:
            if not isinstance(incoming, (tuple, list)) or \
               isinstance(incoming[-1], (int, long, float)):
                incoming = [incoming]
            # ====== Parse incoming ====== #
            for i in incoming:
                if T.is_ndarray(i): # if ndarray, create wrapper variable
                    i = T.variable(i)

                # shape_tuple
                if isinstance(i, (tuple, list)):
                    input_function.append(None)
                    input_shape.append(
                        tuple([j if j is None else int(j) for j in i]))
                # output_shape(keras, odin, lasagne)
                elif hasattr(i, 'output_shape'):
                    input_function.append(i)
                    outshape = i.output_shape
                    # OdinFunction always return list
                    if isinstance(outshape[0], (tuple, list)):
                        input_shape += outshape
                    else: # other framework only return 1 output shape
                        input_shape.append(outshape)
                # variable or placeholder
                elif T.is_variable(i) or T.is_expression(i):
                    shape = T.eval(T.shape(i))
                    if any(j is None for j in shape[1:]):
                        self.raise_arguments('Only first dimension is allowed to'
                                             ' be None, shape:%s does not satisfy'
                                             ' the condition.' % str(shape))
                    input_shape.append(shape)
                    input_function.append(i)
                    # this variable cannot interact anything
                    if T.is_variable(i):
                        self.set_learnable_incoming(i, False, False)
                else:
                    self.raise_arguments(
                        'Unsupport incomming type: %s' % i.__class__)
        self._incoming = input_function
        self._input_shape = input_shape

        # store ALL placeholder necessary for inputs of this Function
        self._input_var = None
        #{index : placeholder}, store placeholder created by this Function
        self._local_input_var = {}
        self._output_var = None

        return self

    def set_learnable_incoming(self, variable, trainable, regularizable):
        '''If a variable is specified at the incoming, you can set it learnable
        by using this function.

        Note
        ----
        No cleaner way to do this
        '''
        if T.is_variable(variable):
            self._learnable_incoming[variable] = [trainable, regularizable]
        return self

    def set_intermediate_inputs(self, inputs, root=False):
        '''
        Parameters
        ----------
        inputs : list(tensor)
            inputs must be a list of anything you want this layer to process
        root : bool
            if True, go to roots of this Funciton and set their input to given
            inputs
        '''
        if not isinstance(inputs, (tuple, list)):
            inputs = [inputs]

        if not root:
            n_inputs = self.n_inputs
            if len(inputs) != n_inputs and len(inputs) != 1:
                self.raise_arguments("Cannot fit the given inputs to the "
                                     "inputs of this funciton, n_given_inputs={} "
                                     "!= n_function_inputs={}. Note: if you "
                                     "don't know the number of inputs, just give "
                                     "1 input and it will be duplicated to "
                                     "all other inputs of function.".format(
                                         len(inputs), n_inputs))
            elif len(inputs) == 1:
                inputs = inputs * n_inputs
            self._intermediate_inputs = inputs
        else:
            self.log("You changed the root inputs, we will reset the whole "
                     "function's tree for it takes effect. Bear in mind this "
                     "situation when you used set_intermediate_inputs.", 20)
            self.reset_cache(True)
            for i in self.get_roots():
                i.set_intermediate_inputs(inputs)
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

    def get_cache(self, training):
        '''Return last inputs returned by this funcitons'''
        if training:
            return self._cache_inputs_train
        return self._cache_inputs_pred

    def reset_cache(self, globals):
        ''' Each time you call this function, its inputs are cached for reused
        Reset cache will force the function to call the inputs again

        Parameters
        ----------
        globals : bool
            if globals=True, all children of this function also reset their
            cached inputs
        '''
        self._cache_inputs_pred = None
        self._cache_inputs_train = None
        if globals:
            for i in self.get_children():
                # a cycle graph may create infinite recursive, but who care
                # we doesn't support cycle graph anyway
                i.reset_cache(globals=globals)
            # also the parameters of this function can contain other functions
            for _, i in self.params.iteritems():
                if isinstance(i._params, OdinFunction):
                    i._params.reset_cache(globals=globals)

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

    def _validate_nD_input(self, n):
        '''All inputs which > n-dimension will be flatten and must have the
        same shape
        Returns
        -------
        full input shape after flattened
        '''
        shape = self.input_shape[0]
        i = tuple(shape[:(n - 1)]) + (np.prod(shape[(n - 1):]),)
        for j in self.input_shape:
            if i != tuple(j[:(n - 1)]) + (np.prod(j[(n - 1):]),):
                self.raise_arguments('All incoming inputs must be flatten to %d'
                                     ' dimension and have the same shape, but'
                                     ' %s != %s.' %
                                     (n, str(i), str(j)))
        # critical, keep the shape in int32
        return (i[0],) + tuple(int(i) for i in i[1:])

    # ==================== Abstract methods ==================== #
    @abstractproperty
    def output_shape(self):
        raise NotImplementedError

    @abstractmethod
    def __call__(self, training=False, **kwargs):
        raise NotImplementedError

    def get_optimization(self, objective, optimizer=None,
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
            if T.ndim(o) > 0 and optimizer is not None:
                self.log('The return objective has > 1 dimension which '
                         'cannot be used to calculate the gradients '
                         'for optimization, hence, we take the mean of '
                         'their values.', 30)
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
                    consider_constant=self.get_cache(training=training))
            opt = optimizer(grad, params)
        return obj, opt

    # ==================== Built-in ==================== #
    @property
    def name(self):
        function_id = '%03d_' % self._function_id
        if len(self._name) == 0:
            return function_id + self.__class__.__name__
        return function_id + ','.join(self._name)

    @property
    def n_inputs(self):
        '''Return expected number of inputs will be returned by
        get_input function
        '''
        # only OdinFunction can return multiple outputs
        return sum(len(i.output_shape)
                   if isinstance(i, OdinFunction) else 1
                   for i in self.incoming)

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
                    shape_str = '_'.join([str(k) for k in j])
                    x = T.placeholder(shape=j,
                        name='in.%s.%s' % (shape_str, self.name))
                    self._input_var.append(x)
                    self._local_input_var[idx] = x
                elif T.is_expression(i): # placeholder
                    self._input_var.append(i)
                    self._local_input_var[idx] = i
                elif T.is_variable(i): # don't need to do anything with variable
                    pass
                else: # input from API layers
                    api = API.get_object_api(i)
                    self._input_var += API.get_input_variables(i, api)
            # not duplicate
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
                    shape_str = '_'.join([str(k) for k in outshape])
                    self._output_var.append(T.placeholder(shape=outshape,
                            name='out.%s.%s' % (shape_str, self.name)))
        return self._output_var

    def get_input(self, training=True, **kwargs):
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
        This method always cache its inputs for training and predicting, if
        you want it calculate new inputs use ``reset_cache``
        '''
        # ====== Dirty hack, modify intermediate inputs of function ====== #
        if self._intermediate_inputs is not None:
            inputs = self._intermediate_inputs
            self._intermediate_inputs = None
            return inputs
        # ====== Get cached input ====== #
        if training and self._cache_inputs_train is not None:
            return self._cache_inputs_train
        elif not training and self._cache_inputs_pred is not None:
            return self._cache_inputs_pred
        # ====== getting the inputs from nested functions ====== #
        inputs = []
        self.input_var # make sure initialized all placeholder
        for idx, i in enumerate(self._incoming):
            # this is expression or InputLayer
            if i is None or T.is_expression(i):
                inputs.append(self._local_input_var[idx])
            # this is variable
            elif T.is_variable(i):
                inputs.append(i)
            # this is from API
            else:
                api = API.get_object_api(i)
                inputs += API.get_outputs(i, api, training)
        # cache the last calculated inputs (if you want to disconnect
        # gradient from this input downward, don't re-build the input
        # graph)
        if training:
            self._cache_inputs_train = inputs
        else:
            self._cache_inputs_pred = inputs
        return inputs

    def get_params(self, globals, trainable=None, regularizable=None):
        params = []
        # ====== Get all params from nested functions if globals mode on ====== #
        if globals:
            for i in self._incoming:
                if i is not None:
                    # variables that learnable
                    if T.is_variable(i):
                        learnable = self._learnable_incoming[i]
                        if learnable and \
                           (learnable[0] == trainable or trainable is None) and \
                           (learnable[1] == regularizable or regularizable is None):
                            params.append(i)
                    # other api
                    else:
                        api = API.get_object_api(i)
                        if api is not None:
                            params += API.get_params(
                                i, globals, trainable, regularizable)

        # ====== Params from this layers ====== #
        local_params = []
        for i, j in self.params.iteritems():
            local_params += j.as_variables(globals, trainable, regularizable)
        return params + local_params

    def get_params_value(self, globals, trainable=None, regularizable=None):
        return [T.get_value(x) for x in
        self.get_params(globals, trainable, regularizable)]

    def create_params(self, spec, shape, name, regularizable, trainable):
        if T.is_variable(spec):
            spec_shape = T.eval(T.shape(spec))
            if shape is None:
                shape = spec_shape
            elif tuple(shape) != tuple(spec_shape):
                self.raise_arguments('Given variable has different shape '
                                     'from requirement, %s != %s' %
                                     (str(spec_shape), str(shape)))

        elif T.is_expression(spec):
            # We cannot check the shape here, Theano expressions (even shared
            # variables) do not have a fixed compile-time shape. We can check the
            # dimensionality though.
            # Note that we cannot assign a name here. We could assign to the
            # `name` attribute of the variable, but the user may have already
            # named the variable and we don't want to override this.
            if shape is not None and T.ndim(spec) != len(shape):
                self.raise_arguments("parameter variable has %d dimensions, "
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
                if T.is_expression(arr):
                    # we do not support expression as params
                    arr = T.eval(arr)
                if arr.shape != shape:
                    raise RuntimeError("cannot initialize parameters: the "
                                       "provided callable did not return a value "
                                       "with the correct shape")
                spec = T.variable(arr, name=name)
        elif isinstance(spec, OdinParams):
            if spec.shape != shape:
                self.raise_arguments('Given OdinParams has different shape '
                                     'from required shape, %s != %s' %
                                    (str(spec.shape), shape))
        else:
            raise RuntimeError("cannot initialize parameters: 'spec' is not "
                               "a numpy array, a Theano expression, or a "
                               "callable")
        # ====== create and return params ====== #
        params = OdinParams(name, spec, shape, trainable, regularizable)
        if params.name in self.params:
            self.raise_arguments("Parameters' name already exist, choose other "
                                 "name for your parameters.")
        self.params[params.name] = params
        # return actual variable or expression
        return spec


class OdinUnsupervisedFunction(OdinFunction):

    def __init__(self, incoming, name=None, **kwargs):
        super(OdinUnsupervisedFunction, self).__init__(
            incoming, unsupervised=True, name=name, **kwargs)
        self._reconstruction_mode = False

    def set_reconstruction_mode(self, reconstruct):
        self._reconstruction_mode = reconstruct
        return self
