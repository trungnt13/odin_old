# ===========================================================================
# This module is adpated from: https://github.com/fchollet/keras
# Revision: @80927fa
# Original work Copyright (c) 2014-2015 keras contributors
# Some idea are also borrowed from Lasagne library
# Original work Copyright (c) 2014-2015 Lasagne contributors
# Modified work Copyright 2016-2017 TrungNT
# ===========================================================================
from __future__ import division

import theano
from theano import tensor as T
import numpy as np
from collections import OrderedDict

from .. import config

_FLOATX = config.floatX()
_EPSILON = config.epsilon()
theano.config.floatX = _FLOATX


# ===========================================================================
# INTERNAL UTILS
# ===========================================================================
def on_gpu():
    '''Return whether the session is set to
    run on GPU or not (i.e. on CPU).
    '''
    import theano.sandbox.cuda

    return 'gpu' in theano.config.device or \
    'cuda' in theano.config.device or \
    'gpu' in theano.config.contexts or \
    'cuda' in theano.config.contexts or \
    theano.sandbox.cuda.cuda_enabled

if on_gpu():
    '''Import cuDNN only if running on GPU:
    not having Cuda installed should not
    prevent from running the present code.
    '''
    # dummy initialization to remove the overhead of running libgpuarray backend
    T.zeros(0, dtype='int').eval()
    _ = theano.shared(value=np.asarray(1., dtype='float32'),
                     name='temporary_var')
    T.grad(2 * _, _).eval()
    _.set_value(None)
    del _


def get_session():
    return _on_gpu()


# ===========================================================================
# VARIABLE MANIPULATION
# ===========================================================================
def variable(value, dtype=_FLOATX, name=None, broadcastable=None, target='dev0'):
    '''Instantiate a tensor variable.
    '''
    value = np.asarray(value, dtype=dtype)
    if not on_gpu():
        target = None

    kwargs = {}
    if broadcastable is not None:
        kwargs['broadcastable'] = broadcastable
    if target is not None:
        kwargs['target'] = target

    return theano.shared(value=value, name=name, strict=False, **kwargs)


def zeros_var(shape, dtype=_FLOATX, name=None):
    '''Instantiate an all-zeros variable.
    '''
    return variable(np.zeros(shape), dtype, name)


def ones_var(shape, dtype=_FLOATX, name=None):
    '''Instantiate an all-ones variable.
    '''
    return variable(np.ones(shape), dtype, name)


def is_variable(v):
    return isinstance(v, theano.compile.SharedVariable)

_PLACEHOLDER_ID = 0
_PLACEHOLDER_SHAPE = {}


def placeholder(shape=None, ndim=None, dtype=_FLOATX, name=None):
    '''Instantiate an input data placeholder variable.
    '''
    if shape is None and ndim is None:
        raise Exception('Specify either a shape or ndim value.')
    if shape is not None:
        ndim = len(shape)
    broadcast = (False,) * ndim

    # ====== Modify add name prefix ====== #
    global _PLACEHOLDER_ID
    name_prefix = 'ID.%02d.' % _PLACEHOLDER_ID
    _PLACEHOLDER_ID += 1
    if name is None:
        name = ''
    name = name_prefix + name
    placeholder = T.TensorType(dtype, broadcast)(name)
    # store the predefined shape of placeholder
    _PLACEHOLDER_SHAPE[name] = \
        [None for _ in range(ndim)] if shape is None else shape
    return placeholder


def is_expression(v):
    '''placeholder also is an expression'''
    return isinstance(v, theano.tensor.TensorVariable)


def is_placeholder(v):
    if is_expression(v) and v.name in _PLACEHOLDER_SHAPE:
        return True
    return False


def eval(x):
    '''Run a graph.
    '''
    # just a hack to return placeholder shape when eval
    if x in _PLACEHOLDER_SHAPE:
        return _PLACEHOLDER_SHAPE[x]
    return x.eval()

# ===========================================================================
# Shape operator
# ===========================================================================


def shape(x):
    '''Return the shape of a tensor.

    Warning: type returned will be different for
    Theano backend (Theano tensor type) and TF backend (TF TensorShape).
    '''
    shape = x.shape
    # little to eval the shape of placeholder
    if hasattr(x, 'name'):
        if x.name in _PLACEHOLDER_SHAPE:
            _PLACEHOLDER_SHAPE[shape] = _PLACEHOLDER_SHAPE[x.name]
    return shape


def int_shape(x):
    return x.shape.eval()


def ndim(x):
    return x.ndim


def broadcastable(x):
    return x.broadcastable


def addbroadcast(x, *axes):
    return T.addbroadcast(x, *axes)

# ===========================================================================
# Predefined data
# ===========================================================================


def zeros(shape, dtype=_FLOATX, name=None):
    '''Instantiate an all-zeros variable.
    '''
    return T.zeros(shape=shape, dtype=dtype)


def ones(shape, dtype=_FLOATX, name=None):
    '''Instantiate an all-ones variable.
    '''
    return T.ones(shape=shape, dtype=dtype)


def ones_like(x):
    return T.ones_like(x)


def zeros_like(x):
    return T.zeros_like(x)


def count_params(x):
    '''Return number of scalars in a tensor.

    Return: numpy integer.
    '''
    return np.prod(x.shape.eval())


def cast(x, dtype):
    if 'theano.' in str(x.__class__):
        return T.cast(x, dtype)
    return np.cast[dtype](x)


def castX(x):
    return cast(x, _FLOATX)


# ===========================================================================
# LINEAR ALGEBRA
# Assumed overridden:
# +, -, /, *, +=, -=, *=, /=
# ===========================================================================
def dot(x, y):
    # TODO: float16 overflow
    if config.floatX() == 'float16':
        return T.dot(x.astype('float32'), y.astype('float32')).astype('float16')
    return T.dot(x, y)


def transpose(x):
    return T.transpose(x)


def gather(reference, indices):
    '''reference: a tensor.
    indices: an int tensor of indices.

    Return: a tensor of same type as reference.
    '''
    return reference[indices]


def diag(x):
    return T.diag(x)


def eye(n, dtype=_FLOATX):
    return T.eye(n, dtype=dtype)


# ===========================================================================
# ELEMENT-WISE OPERATIONS
# ===========================================================================
def var(x, axis=None, keepdims=False):
    return T.var(x, axis=axis, keepdims=keepdims)


def max(x, axis=None, keepdims=False):
    return T.max(x, axis=axis, keepdims=keepdims)


def min(x, axis=None, keepdims=False):
    return T.min(x, axis=axis, keepdims=keepdims)


def sum(x, axis=None, keepdims=False):
    '''Sum of the values in a tensor, alongside the specified axis.
    '''
    return T.sum(x, axis=axis, keepdims=keepdims)


def mul(x, y):
    return T.mul(x, y)


def prod(x, axis=None, keepdims=False):
    '''Multiply the values in a tensor, alongside the specified axis.
    '''
    return T.prod(x, axis=axis, keepdims=keepdims)


def mean(x, axis=None, keepdims=False):
    dtype = x.dtype
    if 'int' in dtype:
        dtype = _FLOATX
    return T.mean(x, axis=axis, keepdims=keepdims, dtype=dtype)


def std(x, axis=None, keepdims=False):
    return T.std(x, axis=axis, keepdims=keepdims)


def any(x, axis=None, keepdims=False):
    '''Bitwise reduction (logical OR).
    '''
    return T.any(x, axis=axis, keepdims=keepdims)


def argmax(x, axis=-1):
    return T.argmax(x, axis=axis, keepdims=False)


def argsort(x, axis=-1):
    return T.argsort(x, axis)


def argtop_k(x, k=1):
    # top-k accuracy
    top = T.argsort(x, axis=-1)
    # (Theano cannot index with [..., -top_k:], we need to simulate that)
    top = top[[slice(None) for _ in range(top.ndim - 1)] +
              [slice(-k, None)]]
    top = top[(slice(None),) * (top.ndim - 1) + (slice(None, None, -1),)]
    return top


def argmin(x, axis=-1):
    return T.argmin(x, axis=axis, keepdims=False)


def square(x):
    return T.sqr(x)


def abs(x):
    return T.abs_(x)


def sqrt(x):
    x = T.clip(x, 0., np.inf)
    return T.sqrt(x)


def exp(x):
    return T.exp(x)


def log(x):
    return T.log(x)


def round(x):
    return T.round(x)


def pow(x, a):
    return T.pow(x, a)


def clip(x, min_value, max_value):
    if max_value < min_value:
        max_value = min_value
    return T.clip(x, min_value, max_value)


def maximum(x, y):
    return T.maximum(x, y)


def minimum(x, y):
    return T.minimum(x, y)


# ===========================================================================
# SHAPE OPERATIONS
# ===========================================================================
def reverse(x, axis=-1):
    '''Apply [::-1] to appropriate axis'''
    if axis < 0:
        axis += x.ndim
    return x[(slice(None),) * axis + (slice(None, None, -1),)]


def concatenate(tensors, axis=-1):
    return T.concatenate(tensors, axis=axis)


def reshape(x, shape):
    shape = tuple([-1 if i is None else i for i in shape])
    return T.reshape(x, shape)


def dimshuffle(x, pattern):
    '''Transpose dimensions.

    pattern should be a tuple or list of
    dimension indices, e.g. [0, 2, 1].
    '''
    pattern = tuple(pattern)
    return x.dimshuffle(pattern)


def repeat_elements(x, rep, axis):
    '''Repeat the elements of a tensor along an axis, like np.repeat.

    If x has shape (s1, s2, s3) and axis=1, the output
    will have shape (s1, s2 * rep, s3).
    '''
    return T.repeat(x, rep, axis=axis)


def resize_images(X, height_factor, width_factor, dim_ordering):
    '''Resize the images contained in a 4D tensor of shape
    - [batch, channels, height, width] (for 'th' dim_ordering)
    - [batch, height, width, channels] (for 'tf' dim_ordering)
    by a factor of (height_factor, width_factor). Both factors should be
    positive integers.
    '''
    if dim_ordering == 'th':
        output = repeat_elements(X, height_factor, axis=2)
        output = repeat_elements(output, width_factor, axis=3)
        return output
    elif dim_ordering == 'tf':
        output = repeat_elements(X, height_factor, axis=1)
        output = repeat_elements(output, width_factor, axis=2)
        return output
    else:
        raise Exception('Invalid dim_ordering: ' + dim_ordering)


def repeat(x, n):
    '''Repeat a 2D tensor.

    If x has shape (samples, dim) and n=2,
    the output will have shape (samples, 2, dim).
    '''
    assert x.ndim == 2
    x = x.dimshuffle((0, 'x', 1))
    return T.extra_ops.repeat(x, n, axis=1)


def tile(x, n):
    return T.tile(x, n)


def flatten(x, outdim=2):
    return T.flatten(x, outdim)


def expand_dims(x, dim=-1):
    '''Add a 1-sized dimension at index "dim".
    '''
    pattern = [i for i in range(x.type.ndim)]
    if dim < 0:
        if x.type.ndim == 0:
            dim = 0
        else:
            dim = dim % x.type.ndim + 1
    pattern.insert(dim, 'x')
    return x.dimshuffle(pattern)


def squeeze(x, axis):
    '''Remove a 1-dimension from the tensor at index "axis".
    '''
    x = T.addbroadcast(x, axis)
    return T.squeeze(x)


def temporal_padding(x, padding=1):
    '''Pad the middle dimension of a 3D tensor
    with "padding" zeros left and right.

    Appologies for the inane API, but Theano makes this
    really hard.
    '''
    input_shape = x.shape
    output_shape = (input_shape[0],
                    input_shape[1] + 2 * padding,
                    input_shape[2])
    output = T.zeros(output_shape)
    return T.set_subtensor(output[:, padding:x.shape[1] + padding, :], x)


def spatial_2d_padding(x, padding=(1, 1), dim_ordering='th'):
    '''Pad the 2nd and 3rd dimensions of a 4D tensor
    with "padding[0]" and "padding[1]" (resp.) zeros left and right.
    '''
    input_shape = x.shape
    if dim_ordering == 'th':
        output_shape = (input_shape[0],
                        input_shape[1],
                        input_shape[2] + 2 * padding[0],
                        input_shape[3] + 2 * padding[1])
        output = T.zeros(output_shape)
        indices = (slice(None),
                   slice(None),
                   slice(padding[0], input_shape[2] + padding[0]),
                   slice(padding[1], input_shape[3] + padding[1]))

    elif dim_ordering == 'tf':
        output_shape = (input_shape[0],
                        input_shape[1] + 2 * padding[0],
                        input_shape[2] + 2 * padding[1],
                        input_shape[3])
        output = T.zeros(output_shape)
        indices = (slice(None),
                   slice(padding[0], input_shape[1] + padding[0]),
                   slice(padding[1], input_shape[2] + padding[1]),
                   slice(None))
    else:
        raise Exception('Invalid dim_ordering: ' + dim_ordering)
    return T.set_subtensor(output[indices], x)


def stack(*x):
    return T.stack(*x)


# ===========================================================================
# VALUE MANIPULATION
# ===========================================================================
def get_value(x, borrow=False):
    if not hasattr(x, 'get_value'):
        raise Exception("'get_value() can only be called on a variable. " +
                        "If you have an expression instead, use eval().")
    return x.get_value(borrow=borrow)


def set_value(x, value):
    x.set_value(np.asarray(value, dtype=x.dtype))


def set_subtensor(x, y):
    return T.set_subtensor(x, y)


# ===========================================================================
# GRAPH MANIPULATION
# ===========================================================================
_GLOBALS_UPDATES = OrderedDict()


def add_global_updates(variable, value):
    '''trick to update tensorflow variables anywhere
    This dictionary will be reseted after each time you create a function
    '''
    _GLOBALS_UPDATES[variable] = value


def reset_global_updates():
    global _GLOBALS_UPDATES
    _GLOBALS_UPDATES = OrderedDict()


class Function(object):

    def __init__(self, inputs, outputs, updates=[], **kwargs):
        if isinstance(updates, OrderedDict):
            updates = updates.items()
        # ====== add and reset global update ====== #
        updates += _GLOBALS_UPDATES.items()
        reset_global_updates()
        self.function = theano.function(
            inputs, outputs,
            updates=updates,
            # on_unused_input='ignore', # TODO: remove this when stop testing
            allow_input_downcast=True, **kwargs)

    def __call__(self, *inputs):
        return self.function(*inputs)


def function(inputs, outputs, updates=[]):
    return Function(inputs, outputs, updates=updates)


def grad_clip(x, clip):
    '''
    This clip the gradient of expression, used on forward pass but clip the
    gradient on backward pass

    This is an elemwise operation.

    Parameters
    ----------
    x: expression
        the variable we want its gradient inputs clipped
    lower_bound: float
        The lower bound of the gradient value
    upper_bound: float
        The upper bound of the gradient value.

    Example
    -------
    >>> x = theano.tensor.scalar()
    >>>
    >>> z = theano.tensor.grad(grad_clip(x, -1, 1)**2, x)
    >>> z2 = theano.tensor.grad(x**2, x)
    >>>
    >>> f = theano.function([x], outputs = [z, z2])
    >>>
    >>> print(f(2.0))  # output (1.0, 4.0)

    Note
    ----
    We register an opt in tensor/opt.py that remove the GradClip.
    So it have 0 cost in the forward and only do work in the grad.

    '''
    return theano.gradient.grad_clip(x, -clip, clip)


def gradients(loss, variables, consider_constant=None, known_grads=None):
    """
    Return symbolic gradients for one or more variables with respect to some
    cost.

    For more information about how automatic differentiation works in Theano,
    see :mod:`gradient`. For information on how to implement the gradient of
    a certain Op, see :func:`grad`.

    Parameters
    ----------
    cost : scalar (0-dimensional) tensor variable or None
        Value with respect to which we are differentiating.  May be
        `None` if known_grads is provided.
    wrt : variable or list of variables
        term[s] for which we want gradients
    consider_constant : list of expressions(variables)
        expressions not to backpropagate through
    known_grads : dict, optional
        A dictionary mapping variables to their gradients. This is
        useful in the case where you know the gradient on some
        variables but do not know the original cost.
    Returns
    -------
    variable or list/tuple of variables (matches `wrt`)
        symbolic expression of gradient of `cost` with respect to each
        of the `wrt` terms.  If an element of `wrt` is not
        differentiable with respect to the output, then a zero
        variable is returned.

    Example
    -------
    >>> # For consider_constant:
    >>> a = T.variable(1.2)
    >>> b = T.variable(1.3)
    >>> x = a * b
    >>>
    >>> y = T.variable(2.)
    >>> z = T.variable(1.)
    >>>
    >>> z_pred = x * y
    >>> loss = T.pow((z - z_pred), 2)
    >>>
    >>> G = T.gradients(loss, [a, b, y], consider_constant=[x])
    >>>
    >>> for g in G:
    >>>     print(g.eval())
    >>> # a_grad=0. b_grad=0. y_grad=6.614
    """
    # TODO: float16 overflow, unsupport DeepCopyOps
    return T.grad(loss, wrt=variables,
        consider_constant=consider_constant, known_grads=known_grads,
        disconnected_inputs='raise')


def jacobian(loss, variables):
    return theano.gradient.jacobian(loss, variables, disconnected_inputs='warn')


def hessian(loss, variables):
    return theano.gradient.hessian(loss, variables, disconnected_inputs='warn')

# ===========================================================================
# CONTROL FLOW
# ===========================================================================


def scan(step_fn, sequences=None, outputs_info=None, non_sequences=None,
    n_steps=None, truncate_gradient=-1, go_backwards=False):
    return theano.scan(step_fn,
        sequences=sequences,
        outputs_info=outputs_info,
        non_sequences=non_sequences,
        n_steps=n_steps, truncate_gradient=truncate_gradient,
        go_backwards=go_backwards,
        strict=False)


def loop(step_fn, n_steps,
    sequences=None, outputs_info=None, non_sequences=None,
    go_backwards=False):
    """
    Helper function to unroll for loops. Can be used to unroll theano.scan.
    The parameter names are identical to theano.scan, please refer to here
    for more information.

    Note that this function does not support the truncate_gradient
    setting from theano.scan.

    Parameters
    ----------
    step_fn : function
        Function that defines calculations at each step.

    sequences : TensorVariable or list of TensorVariables
        List of TensorVariable with sequence data. The function iterates
        over the first dimension of each TensorVariable.

    outputs_info : list of TensorVariables
        List of tensors specifying the initial values for each recurrent
        value. Specify output_info to None for non-arguments to
        the step_function

    non_sequences: list of TensorVariables
        List of theano.shared variables that are used in the step function.

    n_steps: int
        Number of steps to unroll.

    go_backwards: bool
        If true the recursion starts at sequences[-1] and iterates
        backwards.

    Returns
    -------
    List of TensorVariables. Each element in the list gives the recurrent
    values at each time step.

    """
    if not isinstance(sequences, (list, tuple)):
        sequences = [] if sequences is None else [sequences]

    # When backwards reverse the recursion direction
    counter = range(n_steps)
    if go_backwards:
        counter = counter[::-1]

    output = []
    # ====== check if outputs_info is None ====== #
    if outputs_info is not None:
        prev_vals = outputs_info
    else:
        prev_vals = []
    output_idx = [i for i in range(len(prev_vals)) if prev_vals[i] is not None]
    # ====== check if non_sequences is None ====== #
    if non_sequences is None:
        non_sequences = []
    # ====== Main loop ====== #
    for i in counter:
        step_input = [s[i] for s in sequences] + \
                     [prev_vals[idx] for idx in output_idx] + \
            non_sequences
        out_ = step_fn(*step_input)
        # The returned values from step can be either a TensorVariable,
        # a list, or a tuple.  Below, we force it to always be a list.
        if isinstance(out_, T.TensorVariable):
            out_ = [out_]
        if isinstance(out_, tuple):
            out_ = list(out_)
        output.append(out_)
        prev_vals = output[-1]

    # iterate over each scan output and convert it to same format as scan:
    # [[output11, output12,...output1n],
    # [output21, output22,...output2n],...]
    output_scan = []
    for i in range(len(output[0])):
        l = map(lambda x: x[i], output)
        output_scan.append(T.stack(*l))

    return output_scan


def rnn(step_function, inputs, initial_states,
        go_backwards=False, mask=None, constants=None):
    '''Iterates over the time dimension of a tensor.
    Parameters
    ----------
    inputs: tensor of temporal data of shape (samples, time, ...)
        (at least 3D).
    step_function:
        Parameters:
            input: tensor with shape (samples, ...) (no time dimension),
                representing input for the batch of samples at a certain
                time step.
            states: list of tensors.
        Returns:
            output: tensor with shape (samples, ...) (no time dimension),
            new_states: list of tensors, same length and shapes
                as 'states'.
    initial_states: tensor with shape (samples, ...) (no time dimension),
        containing the initial values for the states used in
        the step function.
    go_backwards: boolean. If True, do the iteration over
        the time dimension in reverse order.
    mask: binary tensor with shape (samples, time),
        with a zero for every element that is masked.
    constants: a list of constant values passed at each step.
    Returns
    -------
    A tuple (last_output, outputs, new_states).
        last_output: the latest output of the rnn, of shape (samples, ...)
        outputs: tensor with shape (samples, time, ...) where each
            entry outputs[s, t] is the output of the step function
            at time t for sample s.
        new_states: list of tensors, latest states returned by
            the step function, of shape (samples, ...).
    '''
    ndim = inputs.ndim
    assert ndim >= 3, 'Input should be at least 3D.'

    axes = [1, 0] + list(range(2, ndim))
    inputs = inputs.dimshuffle(axes)

    if mask is not None:
        if mask.ndim == ndim - 1:
            mask = expand_dims(mask)
        assert mask.ndim == ndim
        mask = mask.dimshuffle(axes)

        if constants is None:
            constants = []
        # build an all-zero tensor of shape (samples, output_dim)
        initial_output = step_function(inputs[0], initial_states + constants)[0] * 0
        # Theano gets confused by broadcasting patterns in the scan op
        initial_output = T.unbroadcast(initial_output, 0, 1)

        def _step(input, mask, output_tm1, *states):
            output, new_states = step_function(input, states)
            # output previous output if masked.
            output = T.switch(mask, output, output_tm1)
            return_states = []
            for state, new_state in zip(states, new_states):
                return_states.append(T.switch(mask, new_state, state))
            return [output] + return_states

        results, _ = theano.scan(
            _step,
            sequences=[inputs, mask],
            outputs_info=[initial_output] + initial_states,
            non_sequences=constants,
            go_backwards=go_backwards)
    else:
        def _step(input, *states):
            output, new_states = step_function(input, states)
            return [output] + new_states

        results, _ = theano.scan(
            _step,
            sequences=inputs,
            outputs_info=[None] + initial_states,
            non_sequences=constants,
            go_backwards=go_backwards)

    # deal with Theano API inconsistency
    if type(results) is list:
        outputs = results[0]
        states = results[1:]
    else:
        outputs = results
        states = []

    outputs = T.squeeze(outputs)
    last_output = outputs[-1]

    axes = [1, 0] + list(range(2, outputs.ndim))
    outputs = outputs.dimshuffle(axes)
    states = [T.squeeze(state[-1]) for state in states]
    return last_output, outputs, states


def switch(condition, then_expression, else_expression):
    '''condition: scalar tensor.
    '''
    return T.switch(condition, then_expression, else_expression)


# ===========================================================================
# Comparator
# ===========================================================================
def neq(a, b):
    """a != b"""
    return T.neq(a, b)


def eq(a, b):
    """a == b"""
    return T.eq(a, b)


def gt(a, b):
    """a > b"""
    return T.gt(a, b)


def ge(a, b):
    """a >= b"""
    return T.ge(a, b)


def lt(a, b):
    """a < b"""
    return T.lt(a, b)


def le(a, b):
    """a <= b"""
    return T.le(a, b)


def one_hot(x, nb_class):
    ''' x: 1D-integer vector '''
    ret = T.zeros((x.shape[0], nb_class), dtype=_FLOATX)
    ret = T.set_subtensor(ret[T.arange(x.shape[0]), x], 1)
    return ret


def confusion_matrix(y_pred, y_true, labels=None):
    """
    Computes the confusion matrix of given vectors containing
    actual observations and predicted observations.
    Parameters
    ----------
    pred : 1-d or 2-d tensor variable
    actual : 1-d or 2-d tensor variable
    labels : array, shape = [n_classes], optional
        List of labels to index the matrix. This may be used to reorder
        or select a subset of labels.
        If none is given, those that appear at least once
        in ``y_true`` or ``y_pred`` are used in sorted order.

    Returns
    -------
    conf_mat : Confusion matrix of actual and predictions observations as shown below.
               | Predicted
    ___________|___________
       Actual  |
               |
    Examples
    --------
    >>> import theano
    >>> from theano.tensor.nnet import confusion_matrix
    >>> x = theano.tensor.vector()
    >>> y = theano.tensor.vector()
    >>> f = theano.function([x, y], confusion_matrix(x, y))
    >>> a = [0, 1, 2, 1, 0]
    >>> b = [0, 0, 2, 1, 2]
    >>> print(f(a, b))
    [array([[0, 0, 1],
            [2, 1, 0],
            [0, 0, 1]]), array([ 0.,  1.,  2.])]
    """
    if y_true.ndim == 2:
        y_true = T.argmax(y_true, axis=-1)
    elif y_true.ndim != 1:
        raise ValueError('actual must be 1-d or 2-d tensor variable')
    if y_pred.ndim == 2:
        y_pred = T.argmax(y_pred, axis=-1)
    elif y_pred.ndim != 1:
        raise ValueError('pred must be 1-d or 2-d tensor variable')

    if labels is None:
        labels = T.extra_ops.Unique(False, False, False)(T.concatenate([y_true, y_pred]))

    colA = y_true.dimshuffle(0, 'x')
    colP = y_pred.dimshuffle(0, 'x')

    oneHotA = T.eq(colA, labels).astype('int64')
    oneHotP = T.eq(colP, labels).astype('int64')

    conf_mat = T.dot(oneHotA.T, oneHotP)
    return conf_mat


def one_hot_max(x, axis=-1):
    '''
    Example
    -------
    >>> Input: [[0.0, 0.0, 0.5],
    >>>         [0.0, 0.3, 0.1],
    >>>         [0.6, 0.0, 0.2]]
    >>> Output: [[0.0, 0.0, 1.0],
    >>>         [0.0, 1.0, 0.0],
    >>>         [1.0, 0.0, 0.0]]
    '''
    return T.cast(
        T.eq(T.arange(x.shape[axis])[None, :],
             T.argmax(x, axis=axis, keepdims=True)),
        _FLOATX
    )


def apply_mask(x, mask):
    '''
    x : 3D tensor
    mask : 2D tensor

    Example
    -------
    >>> Input: [128, 500, 120]
    >>> Mask:  [1, 1, 0]
    >>> Output: [128, 500, 0]
    '''
    return T.mul(x, expand_dims(mask, -1))
