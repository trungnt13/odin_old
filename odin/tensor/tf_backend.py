# ===========================================================================
# This module is adpated from: https://github.com/fchollet/keras
# Revision: @bec2701
# Original work Copyright (c) 2014-2015 keras contributors
# Some idea are also borrowed from Lasagne library
# Original work Copyright (c) 2014-2015 Lasagne contributors
# Some work are adapted from tensorfuse library
# Original work Copyright (c) [dementrock](https://github.com/dementrock)
# Modified work Copyright 2016-2017 TrungNT
# ===========================================================================

import tensorflow as tf
import numpy as np
import os
from collections import OrderedDict

from .. import config
from .numpy_backend import get_random_magic_seed

from six.moves import range, zip

_FLOATX = config.floatX()
_EPSILON = config.epsilon()

# ===========================================================================
# INTERNAL UTILS
# ===========================================================================
_SESSION = None


def get_session():
    global _SESSION
    if _SESSION is None:
        if not os.environ.get('OMP_NUM_THREADS'):
            _SESSION = tf.Session('')
        else:
            nb_thread = int(os.environ.get('OMP_NUM_THREADS'))
            _SESSION = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=nb_thread))
    return _SESSION


def _set_session(session):
    global _SESSION
    _SESSION = session

# From Theano


def _format_as(use_list, use_tuple, outputs):
    """
    Formats the outputs according to the flags `use_list` and `use_tuple`.
    If `use_list` is True, `outputs` is returned as a list (if `outputs`
    is not a list or a tuple then it is converted in a one element list).
    If `use_tuple` is True, `outputs` is returned as a tuple (if `outputs`
    is not a list or a tuple then it is converted into a one element tuple).
    Otherwise (if both flags are false), `outputs` is returned.
    """
    assert not (use_list and use_tuple), \
        "Both flags cannot be simultaneously True"
    if (use_list or use_tuple) and not isinstance(outputs, (list, tuple)):
        if use_list:
            return [outputs]
        else:
            return (outputs,)
    elif not (use_list or use_tuple) and isinstance(outputs, (list, tuple)):
        assert len(outputs) == 1, \
            "Wrong arguments. Expected a one element list"
        return outputs[0]
    elif use_list or use_tuple:
        if use_list:
            return list(outputs)
        else:
            return tuple(outputs)
    else:
        return outputs


def _wrap_into_list(x):
    """
    Wrap the input into a list if it is not already a list.
    """
    if x is None:
        return []
    elif not isinstance(x, (list, tuple)):
        return [x]
    else:
        return list(x)
# ===========================================================================
# VARIABLE MANIPULATION
# ===========================================================================


def variable(value, dtype=_FLOATX, name=None, broadcastable=None):
    v = tf.Variable(np.asarray(value, dtype=dtype), name=name)
    get_session().run(v.initializer)
    return v


def zeros_var(shape, dtype=_FLOATX, name=None):
    return variable(np.zeros(shape), dtype, name)


def ones_var(shape, dtype=_FLOATX, name=None):
    return variable(np.ones(shape), dtype, name)


def is_variable(v):
    return isinstance(v, tf.python.Variable)

_PLACEHOLDER_ID = 0
_PLACEHOLDER_SHAPE = {}


def placeholder(shape=None, ndim=None, dtype=_FLOATX, name=None):
    # name must match: [A-Za-z0-9.][A-Za-z0-9_.\-/]*
    if not shape:
        if ndim:
            shape = [None for _ in range(ndim)]

    # ====== Modify add name prefix ====== #
    global _PLACEHOLDER_ID
    name_prefix = 'ID.%02d.' % _PLACEHOLDER_ID
    _PLACEHOLDER_ID += 1
    if name is None:
        name = ''
    name = name_prefix + name
    placeholder = tf.placeholder(dtype, shape=shape, name=name)
    _PLACEHOLDER_SHAPE[placeholder.name] = shape
    return placeholder


def is_expression(v):
    return isinstance(v, tf.python.Tensor)


def is_placeholder(v):
    if is_expression(v) and v.name in _PLACEHOLDER_SHAPE:
        return True
    return False


def eval(x):
    '''Run a graph.
    '''
    if isinstance(x, tf.TensorShape):
        return x.as_list()
    return x.eval(session=get_session())

# ===========================================================================
# Shape operators
# ===========================================================================


def shape(x):
    return x.get_shape()


def int_shape(x):
    shape = x.get_shape()
    return tuple([i.__int__() for i in shape])


def ndim(x):
    return len(x.get_shape())


def broadcastable(x):
    return None


def addbroadcast(x, *axes):
    return x

# ===========================================================================
# Predefined data
# ===========================================================================


def zeros(shape, dtype=_FLOATX, name=None):
    return tf.zeros(shape, dtype=dtype, name=name)


def ones(shape, dtype=_FLOATX, name=None):
    return tf.ones(shape, dtype=dtype, name=name)


def ones_like(x, name=None):
    return tf.ones_like(x, name=name)


def zeros_like(x, name=None):
    return tf.zeros_like(x, name=name)


def count_params(x):
    '''Return number of scalars in a tensor.
    '''
    shape = x.get_shape()
    return np.prod([shape[i]._value for i in range(len(shape))])


def cast(x, dtype):
    if 'tensorflow.' in str(x.__class__):
        return tf.cast(x, dtype)
    return np.cast[dtype](x)


def castX(x):
    return cast(x, _FLOATX)

# ===========================================================================
# LINEAR ALGEBRA
# ===========================================================================


def dot(x, y):
    return tf.matmul(x, y)


def transpose(x):
    return tf.transpose(x)


def gather(reference, indices):
    '''
    # Arguments
        reference: a tensor.
        indices: an int tensor of indices.

    # Returns
        a tensor of same type as `reference`.
    '''
    return tf.gather(reference, indices)


# ===========================================================================
# ELEMENT-WISE OPERATIONS
# ===========================================================================
def normalize_axis(axis, ndim):
    if type(axis) is tuple:
        axis = list(axis)
    if type(axis) is list:
        for i, a in enumerate(axis):
            if a is not None and a < 0:
                axis[i] = a % ndim
    else:
        if axis is not None and axis < 0:
            axis = axis % ndim
    return axis


def max(x, axis=None, keepdims=False):
    axis = normalize_axis(axis, ndim(x))
    return tf.reduce_max(x, reduction_indices=axis, keep_dims=keepdims)


def min(x, axis=None, keepdims=False):
    axis = normalize_axis(axis, ndim(x))
    return tf.reduce_min(x, reduction_indices=axis, keep_dims=keepdims)


def sum(x, axis=None, keepdims=False):
    '''Sum of the values in a tensor, alongside the specified axis.
    '''
    axis = normalize_axis(axis, ndim(x))
    return tf.reduce_sum(x, reduction_indices=axis, keep_dims=keepdims)


def mul(x, y):
    return tf.mul(x, y)


def prod(x, axis=None, keepdims=False):
    '''Multiply the values in a tensor, alongside the specified axis.
    '''
    axis = normalize_axis(axis, ndim(x))
    return tf.reduce_prod(x, reduction_indices=axis, keep_dims=keepdims)


def var(x, axis=None, keepdims=False):
    axis = normalize_axis(axis, ndim(x))
    if x.dtype.base_dtype == tf.bool:
        x = tf.cast(x, _FLOATX)
    m = tf.reduce_mean(x, reduction_indices=axis, keep_dims=True)
    devs_squared = tf.square(x - m)
    return tf.reduce_mean(devs_squared,
                          reduction_indices=axis,
                          keep_dims=keepdims)


def std(x, axis=None, keepdims=False):
    axis = normalize_axis(axis, ndim(x))
    if x.dtype.base_dtype == tf.bool:
        x = tf.cast(x, _FLOATX)
    m = tf.reduce_mean(x, reduction_indices=axis, keep_dims=True)
    devs_squared = tf.square(x - m)
    return tf.sqrt(tf.reduce_mean(devs_squared,
                                  reduction_indices=axis,
                                  keep_dims=keepdims))


def mean(x, axis=None, keepdims=False):
    axis = normalize_axis(axis, ndim(x))
    if x.dtype.base_dtype == tf.bool:
        x = tf.cast(x, _FLOATX)
    return tf.reduce_mean(x, reduction_indices=axis, keep_dims=keepdims)


def any(x, axis=None, keepdims=False):
    '''Bitwise reduction (logical OR).

    Return array of uint8 (0s and 1s).
    '''
    axis = normalize_axis(axis, ndim(x))
    x = tf.cast(x, tf.bool)
    x = tf.reduce_any(x, reduction_indices=axis, keep_dims=keepdims)
    return tf.cast(x, tf.uint8)


def argmax(x, axis=-1):
    if axis < 0:
        axis = axis % len(x.get_shape())
    return tf.argmax(x, axis)


def argsort(x, axis=-1):
    raise NotImplementedError


def argtop_k(x, k=1):
    ''' See also: tf.nn.in_top_k '''
    return tf.nn.top_k(x, k)[1]


def argmin(x, axis=-1):
    if axis < 0:
        axis = axis % len(x.get_shape())
    return tf.argmin(x, axis)


def square(x):
    return tf.square(x)


def abs(x):
    return tf.abs(x)


def sqrt(x):
    x = tf.clip_by_value(x, tf.cast(0., dtype=_FLOATX),
                         tf.cast(np.inf, dtype=_FLOATX))
    return tf.sqrt(x)


def exp(x):
    return tf.exp(x)


def log(x):
    return tf.log(x)


def round(x):
    return tf.round(x)


def pow(x, a):
    return tf.pow(x, a)


def clip(x, min_value, max_value):
    if max_value < min_value:
        max_value = min_value
    return tf.clip_by_value(x, tf.cast(min_value, dtype=_FLOATX),
                            tf.cast(max_value, dtype=_FLOATX))


def maximum(x, y):
    return tf.maximum(x, y)


def minimum(x, y):
    return tf.minimum(x, y)


# ===========================================================================
# SHAPE OPERATIONS
# ===========================================================================
def reverse(x, axis=-1):
    '''Apply [::-1] to appropriate axis'''
    ndim = len(x.get_shape())
    dims = [False] * ndim
    if axis < 0:
        axis = axis % ndim
    dims[axis] = True
    return tf.reverse(x, dims)


def concatenate(tensors, axis=-1):
    if axis < 0:
        axis = axis % len(tensors[0].get_shape())
    return tf.concat(axis, tensors)


def reshape(x, shape):
    shape = [i.value
             if isinstance(i, tf.python.framework.tensor_shape.Dimension) else i
             for i in shape]
    shape = tuple([-1 if i is None else i for i in shape])
    return tf.reshape(x, shape)


def dimshuffle(x, pattern):
    '''
    # Arguments
        pattern: should be a tuple or list of
            dimension indices, e.g. [0, 2, 1].
    '''
    if 'x' in pattern:
        x = tf.transpose(x, perm=[i for i in pattern if i != 'x'])
    for i, p in enumerate(pattern):
        if p == 'x':
            x = tf.expand_dims(x, i)
    return x


def resize_images(X, height_factor, width_factor, dim_ordering):
    '''Resize the images contained in a 4D tensor of shape
    - [batch, channels, height, width] (for 'th' dim_ordering)
    - [batch, height, width, channels] (for 'tf' dim_ordering)
    by a factor of (height_factor, width_factor). Both factors should be
    positive integers.
    '''
    if dim_ordering == 'th':
        new_height = shape(X)[2].value * height_factor
        new_width = shape(X)[3].value * width_factor
        X = dimshuffle(X, [0, 2, 3, 1])
        X = tf.image.resize_nearest_neighbor(X, (new_height, new_width))
        return dimshuffle(X, [0, 3, 1, 2])
    elif dim_ordering == 'tf':
        new_height = shape(X)[1].value * height_factor
        new_width = shape(X)[2].value * width_factor
        return tf.image.resize_nearest_neighbor(X, (new_height, new_width))
    else:
        raise Exception('Invalid dim_ordering: ' + dim_ordering)


def repeat_elements(x, rep, axis):
    '''Repeats the elements of a tensor along an axis, like np.repeat

    If x has shape (s1, s2, s3) and axis=1, the output
    will have shape (s1, s2 * rep, s3)
    '''
    if axis < 0:
        axis = axis % len(x.get_shape())

    x_shape = x.get_shape().as_list()
    # slices along the repeat axis
    splits = tf.split(axis, x_shape[axis], x)
    # repeat each slice the given number of reps
    x_rep = [s for s in splits for i in range(rep)]
    return tf.concat(axis, x_rep)


def repeat(x, n):
    '''Repeat a 2D tensor:

    if x has shape (samples, dim) and n=2,
    the output will have shape (samples, 2, dim)
    '''
    assert ndim(x) == 2
    tensors = [x] * n
    stacked = tf.pack(tensors)
    return tf.transpose(stacked, (1, 0, 2))


def tile(x, n):
    return tf.tile(x, n)


def flatten(x, outdim=2):
    '''Turn a n-D tensor into a m-D tensor (m < n) where
    the first dimension is conserved.
    '''
    if outdim == 1:
        pattern = [-1]
    else:
        pattern = [-1, np.prod(x.get_shape()[(outdim - 1):].as_list())]
    return tf.reshape(x, pattern)


def expand_dims(x, dim=-1):
    '''Add a 1-sized dimension at index "dim".
    '''
    return tf.expand_dims(x, dim)


def squeeze(x, axis):
    '''Remove a 1-dimension from the tensor at index "axis".
    '''
    return tf.squeeze(x, [axis])


def temporal_padding(x, padding=1):
    '''Pad the middle dimension of a 3D tensor
    with "padding" zeros left and right.
    '''
    pattern = [[0, 0], [padding, padding], [0, 0]]
    return tf.pad(x, pattern)


def spatial_2d_padding(x, padding=(1, 1), dim_ordering='th'):
    '''Pad the 2nd and 3rd dimensions of a 4D tensor
    with "padding[0]" and "padding[1]" (resp.) zeros left and right.
    '''
    if dim_ordering == 'th':
        pattern = [[0, 0], [0, 0],
                   [padding[0], padding[0]], [padding[1], padding[1]]]
    else:
        pattern = [[0, 0],
                   [padding[0], padding[0]], [padding[1], padding[1]],
                   [0, 0]]
    return tf.pad(x, pattern)


def stack(*x):
    return tf.pack(x)

# ===========================================================================
# VALUE MANIPULATION
# ===========================================================================


def get_value(x, borrow=False):
    '''Technically the same as eval() for TF.
    '''
    return x.eval(session=get_session())


def set_value(x, value):
    tf.assign(x, np.asarray(value)).op.run(session=get_session())


def set_subtensor(x, y):
    raise NotImplementedError


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

    def __init__(self, inputs, outputs, updates=[]):
        assert type(inputs) in {list, tuple}
        if type(outputs) not in {list, tuple}:
            outputs = [outputs]
            self._return_list = False
        else:
            self._return_list = True
        if isinstance(updates, OrderedDict):
            updates = updates.items()
        assert type(updates) in {list, tuple}
        self.inputs = list(inputs)
        self.outputs = list(outputs)
        with tf.control_dependencies(self.outputs):
            self.updates = [tf.assign(p, new_p) for (p, new_p) in updates]
        # ====== add global_update ====== #
        self.global_update = [tf.assign(p, new_p) for (p, new_p) in _GLOBALS_UPDATES.items()]
        reset_global_updates()

    def __call__(self, *inputs):
        assert type(inputs) in {list, tuple}
        names = [v.name for v in self.inputs]
        feed_dict = dict(zip(names, inputs))
        session = get_session()
        # ====== add global updates ====== #
        updated = session.run(self.outputs + self.updates + self.global_update,
            feed_dict=feed_dict)
        if self._return_list:
            return updated[:len(self.outputs)]
        return updated[0]


def function(inputs, outputs, updates=[]):
    return Function(inputs, outputs, updates=updates)


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

    """
    if consider_constant is not None:
        for i in consider_constant:
            tf.stop_gradient(i)
        raise NotImplementedError
    grad = tf.gradients(loss, variables)
    if known_grads is not None:
        grad = [known_grads[i] if i in known_grads else j
                for i, j in zip(variables, grad)]
    return grad


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
    # TODO: no implementation for grad_clipping on tensorflow on forward pass
    return x


def jacobian(expression, wrt):
    # copying theano's implementation, which is based on scan
    #from theano.tensor import arange
    # Check inputs have the right format
    assert is_variable(expression), \
        "tensor.jacobian expects a Variable as `expression`"
    assert expression.ndim < 2, \
        ("tensor.jacobian expects a 1 dimensional variable as "
         "`expression`. If not use flatten to make it a vector")
    assert not is_variable(expression.shape[0]), \
        "shape of the expression must be known"

    using_list = isinstance(wrt, list)
    using_tuple = isinstance(wrt, tuple)

    if isinstance(wrt, (list, tuple)):
        wrt = list(wrt)
    else:
        wrt = [wrt]

    if expression.ndim == 0:
        # expression is just a scalar, use grad
        return _format_as(using_list, using_tuple, gradients(expression, wrt))

    def inner_function(*args):
        idx = args[0]
        expr = args[1]
        rvals = []
        for inp in args[2:]:
            try:
                rval = gradients(expr[idx], inp)
            except Exception as e:
                import ipdb; ipdb.set_trace()
            if rval is None:
                import ipdb; ipdb.set_trace()
            rvals.append(rval)
        return rvals
    # Computing the gradients does not affect the random seeds on any random
    # generator used n expression (because during computing gradients we are
    # just backtracking over old values. (rp Jan 2012 - if anyone has a
    # counter example please show me)
    jacobs, updates = scan(inner_function,
                           sequences=[range(expression.shape[0])],
                           non_sequences=[expression] + wrt,
                           n_steps=expression.shape[0])
    assert not updates
    return _format_as(using_list, using_tuple, jacobs)


def hessian(expression, wrt):
    raise NotImplementedError

# ===========================================================================
# CONTROL FLOW
# ===========================================================================


def scan(step_fn, sequences=None, outputs_info=None, non_sequences=None,
    n_steps=None, truncate_gradient=-1, go_backwards=False):
    from operator import itemgetter
    # n_steps must be provided under cgt or tensorflow
    if n_steps is None:
        raise ValueError(
            'n_steps must be provided for scan to work under TensorFlow')
    sequences = _wrap_into_list(sequences)
    non_sequences = _wrap_into_list(non_sequences)
    if outputs_info is not None:
        outputs_info = _wrap_into_list(outputs_info)
    if go_backwards and n_steps < 0:
        go_backwards = False
        n_steps = -n_steps
    if go_backwards or n_steps < 0:
        go_backwards = True
        n_steps = abs(n_steps)
    step_outputs = []
    cur_output = outputs_info
    loop_range = range(n_steps - 1, -1, -1) if go_backwards else range(n_steps)
    for i in loop_range:
        # Only pass output if needed
        if outputs_info is not None:
            cur_output = step_fn(*(map(itemgetter(i), sequences) + cur_output + non_sequences))
        else:
            cur_output = step_fn(*(map(itemgetter(i), sequences) + non_sequences))
        step_outputs.append(cur_output)
    outputs = []
    try:
        if len(step_outputs) > 0:
            if outputs_info is None:
                for i in range(len(step_outputs[0])):
                    outputs.append(tf.pack(map(itemgetter(i), step_outputs)))
                #outputs = step_outputs
            else:
                for i in range(len(outputs_info)):
                    outputs.append(tf.pack(map(itemgetter(i), step_outputs)))
        else:
            import ipdb; ipdb.set_trace()
    except Exception as e:
        raise e
    # This is quite ugly, but unfortunately it's what theano does
    if len(outputs) > 1:
        # update is not supported yet
        return outputs, None
    elif len(outputs) == 1:
        return outputs[0], None
    else:
        return None, None


def loop(step_fn, n_steps, sequences=None, outputs_info=None, non_sequences=None,
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

    # ====== check if outputs_info is None ====== #
    output = []
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
        if isinstance(out_, tf.python.Tensor):
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
        output_scan.append(tf.pack(l))
    return output_scan


def rnn(step_function, inputs, initial_states,
        go_backwards=False, mask=None, constants=None):
    '''Iterates over the time dimension of a tensor.
    # Arguments
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
        mask: binary tensor with shape (samples, time, 1),
            with a zero for every element that is masked.
        constants: a list of constant values passed at each step.
    # Returns
        A tuple (last_output, outputs, new_states).
            last_output: the latest output of the rnn, of shape (samples, ...)
            outputs: tensor with shape (samples, time, ...) where each
                entry outputs[s, t] is the output of the step function
                at time t for sample s.
            new_states: list of tensors, latest states returned by
                the step function, of shape (samples, ...).
    '''
    ndim = len(inputs.get_shape())
    assert ndim >= 3, "Input should be at least 3D."
    axes = [1, 0] + list(range(2, ndim))
    inputs = tf.transpose(inputs, (axes))
    input_list = tf.unpack(inputs)
    if constants is None:
        constants = []

    states = initial_states
    successive_states = []
    successive_outputs = []
    if go_backwards:
        input_list.reverse()

    if mask is not None:
        # Transpose not supported by bool tensor types, hence round-trip to uint8.
        mask = tf.cast(mask, tf.uint8)
        if len(mask.get_shape()) == ndim - 1:
            mask = expand_dims(mask)
        mask = tf.cast(tf.transpose(mask, axes), tf.bool)
        mask_list = tf.unpack(mask)

        for input, mask_t in zip(input_list, mask_list):
            output, new_states = step_function(input, states + constants)

            # tf.select needs its condition tensor to be the same shape as its two
            # result tensors, but in our case the condition (mask) tensor is
            # (nsamples, 1), and A and B are (nsamples, ndimensions). So we need to
            # broadcast the mask to match the shape of A and B. That's what the
            # tile call does, is just repeat the mask along its second dimension
            # ndimensions times.
            tiled_mask_t = tf.tile(mask_t, tf.pack([1, tf.shape(output)[1]]))

            if len(successive_outputs) == 0:
                prev_output = zeros_like(output)
            else:
                prev_output = successive_outputs[-1]

            output = tf.select(tiled_mask_t, output, prev_output)

            return_states = []
            for state, new_state in zip(states, new_states):
                # (see earlier comment for tile explanation)
                tiled_mask_t = tf.tile(mask_t, tf.pack([1, tf.shape(new_state)[1]]))
                return_states.append(tf.select(tiled_mask_t, new_state, state))

            states = return_states
            successive_outputs.append(output)
            successive_states.append(states)
    else:
        for input in input_list:
            output, states = step_function(input, states + constants)
            successive_outputs.append(output)
            successive_states.append(states)

    last_output = successive_outputs[-1]
    outputs = tf.pack(successive_outputs)
    new_states = successive_states[-1]

    axes = [1, 0] + list(range(2, len(outputs.get_shape())))
    outputs = tf.transpose(outputs, axes)
    return last_output, outputs, new_states


def switch(condition, then_expression, else_expression):
    '''Switch between two operations depending on a scalar value.

    # Arguments
        condition: scalar tensor.
        then_expression: TensorFlow operation.
        else_expression: TensorFlow operation.
    '''
    return tf.python.control_flow_ops.cond(condition,
                                           lambda: then_expression,
                                           lambda: else_expression)


# ===========================================================================
# NN OPERATIONS
# ===========================================================================
def relu(x, alpha=0., max_value=None):
    '''ReLU.
    # Arguments
        alpha: slope of negative section.
        max_value: saturation threshold.
    '''
    negative_part = tf.nn.relu(-x)
    x = tf.nn.relu(x)
    if max_value is not None:
        x = tf.clip_by_value(x, tf.cast(0., dtype=_FLOATX),
                             tf.cast(max_value, dtype=_FLOATX))
    if isinstance(alpha, (tuple, list, np.ndarray)) or np.isscalar(alpha):
        alpha = tf.constant(alpha, dtype=_FLOATX)
    x -= alpha * negative_part
    return x


def linear(x):
    return x


def softmax(x):
    return tf.nn.softmax(x)


def softplus(x):
    return tf.nn.softplus(x)


def categorical_crossentropy(output, target, from_logits=False):
    '''Note: tf.nn.softmax_cross_entropy_with_logits
    expects logits, Keras expects probabilities.
    '''
    if not from_logits:
        # scale preds so that the class probas of each sample sum to 1
        output /= tf.reduce_sum(output,
                                reduction_indices=len(output.get_shape()) - 1,
                                keep_dims=True)
        # manual computation of crossentropy
        output = tf.clip_by_value(output, tf.cast(_EPSILON, dtype=_FLOATX),
                                  tf.cast(1. - _EPSILON, dtype=_FLOATX))
        return - tf.reduce_sum(target * tf.log(output),
                               reduction_indices=len(output.get_shape()) - 1)
    else:
        return tf.nn.softmax_cross_entropy_with_logits(output, target)


def binary_crossentropy(output, target, from_logits=False):
    '''Note: tf.nn.sigmoid_cross_entropy_with_logits
    expects logits, Keras expects probabilities.
    '''
    if not from_logits:
        # transform back to logits
        output = tf.clip_by_value(output, tf.cast(_EPSILON, dtype=_FLOATX),
                                  tf.cast(1. - _EPSILON, dtype=_FLOATX))
        output = tf.log(output / (1 - output))
    return tf.nn.sigmoid_cross_entropy_with_logits(output, target)


def sigmoid(x):
    return tf.nn.sigmoid(x)


def hard_sigmoid(x):
    x = (0.2 * x) + 0.5
    x = tf.clip_by_value(x, tf.cast(0., dtype=_FLOATX),
                         tf.cast(1., dtype=_FLOATX))
    return x


def tanh(x):
    return tf.nn.tanh(x)


def dropout(x, level, rescale=True, noise_shape=None,
    seed=None, rng=None):
    """Computes dropout.

    With probability `keep_prob`, outputs the input element scaled up by
    `1 / keep_prob`, otherwise outputs `0`.  The scaling is so that the expected
    sum is unchanged.

    By default, each element is kept or dropped independently.  If `noise_shape`
    is specified, it must be
    [broadcastable](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
    to the shape of `x`, and only dimensions with `noise_shape[i] == shape(x)[i]`
    will make independent decisions.  For example, if `shape(x) = [k, l, m, n]`
    and `noise_shape = [k, 1, 1, n]`, each batch and channel component will be
    kept independently and each row and column will be kept or not kept together.

    Parameters
    ----------
    x: A tensor.
    level: float(0.-1.)
        probability dropout values in given tensor
    rescale: bool
        whether rescale the outputs by dividing the retain probablity
    noise_shape: A 1-D `Tensor` of type `int32`, representing the
      shape for randomly generated keep/drop flags.
    seed: int
        A Python integer. Used to create random seeds. See
    rng: `tensor.rng`
        random generator from tensor class
    """
    retain_prob = 1. - level
    if isinstance(rng, _RandomWrapper):
        seed = rng._rng.randint(10e6)
    elif seed is None:
        seed = get_random_magic_seed()

    if noise_shape is not None:
        # from tensorflow.python.ops import array_ops
        # shape_x = array_ops.shape(x)
        noise_shape = tuple([shape(x)[i].value if j is None or j < 0 else j
                            for i, j in enumerate(noise_shape)])
    # the dummy 1. works around a TF bug
    # (float32_ref vs. float32 incomptability)
    x = tf.nn.dropout(x * 1., retain_prob, noise_shape=noise_shape, seed=seed)
    if not rescale:
        x = x * retain_prob
    return x

# ==================== Regularizations ==================== #


def l2_normalize(x, axis):
    if axis < 0:
        axis = axis % len(x.get_shape())
    return tf.nn.l2_normalize(x, dim=axis)


def l2_regularize(x):
    return sum(tf.square(x))


def l1_regularize(x):
    return sum(tf.abs(x))


def jacobian_regularize(hidden, params):
    ''' Computes the jacobian of the hidden layer with respect to
    the input, reshapes are necessary for broadcasting the
    element-wise product on the right axis
    '''
    hidden = hidden * (1 - hidden)
    L = expand_dims(hidden, 1) * expand_dims(params, 0)
    # Compute the jacobian and average over the number of samples/minibatch
    L = sum(mean(tf.pow(L, 2), axis=0)) # avr over all samples in batch
    return mean(L)


def kl_gaussian(mean_, logsigma,
                prior_mean=0., prior_logsigma=0.,
                regularizer_scale=1.):
    ''' KL-divergence between two gaussians.
    Useful for Variational AutoEncoders. Use this as an activation regularizer
    Parameters:
    -----------
    mean, logsigma: parameters of the input distributions
    prior_mean, prior_logsigma: paramaters of the desired distribution (note the
        log on logsigma)
    regularizer_scale: Rescales the regularization cost. Keep this 1 for most cases.

    Note
    ----
    origin implementation from seya:
    https://github.com/Philip-Bachman/ICML-2015/blob/master/LogPDFs.py
    Copyright (c) Philip Bachman
    '''
    gauss_klds = 0.5 * (prior_logsigma - logsigma +
            ((tf.exp(logsigma) + pow((mean_ - prior_mean), 2.0)) / tf.exp(prior_logsigma)) - 1.0)
    return mean(gauss_klds)


def correntropy_regularize(x, sigma=1.):
    '''
    Note
    ----
    origin implementation from seya:
    https://github.com/EderSantana/seya/blob/master/seya/regularizers.py
    Copyright (c) EderSantana
    '''
    return -sum(mean(tf.exp(tf.pow(x, 2) / sigma), axis=0)) / tf.sqrt(2 * np.pi * sigma)

# ===========================================================================
# CONVOLUTIONS
# ===========================================================================


def conv2d(x, kernel, strides=(1, 1), border_mode='valid', dim_ordering='th',
           image_shape=None, filter_shape=None):
    '''Runs on cuDNN if available.

    # Arguments
        border_mode: string, "same" or "valid".
        dim_ordering: whether to use Theano or TensorFlow dimension ordering
        in inputs/kernels/ouputs.
    '''
    if border_mode == 'same':
        padding = 'SAME'
    elif border_mode == 'valid':
        padding = 'VALID'
    else:
        raise Exception('Invalid border mode: ' + str(border_mode))

    strides = (1,) + strides + (1,)

    if _FLOATX == 'float64':
        # tf conv2d only supports float32
        x = tf.cast(x, 'float32')
        kernel = tf.cast(kernel, 'float32')

    if dim_ordering == 'th':
        # TF uses the last dimension as channel dimension,
        # instead of the 2nd one.
        # TH input shape: (samples, input_depth, rows, cols)
        # TF input shape: (samples, rows, cols, input_depth)
        # TH kernel shape: (depth, input_depth, rows, cols)
        # TF kernel shape: (rows, cols, input_depth, depth)
        x = tf.transpose(x, (0, 2, 3, 1))
        kernel = tf.transpose(kernel, (2, 3, 1, 0))
        x = tf.nn.conv2d(x, kernel, strides, padding=padding)
        x = tf.transpose(x, (0, 3, 1, 2))
    elif dim_ordering == 'tf':
        x = tf.nn.conv2d(x, kernel, strides, padding=padding)
    else:
        raise Exception('Unknown dim_ordering: ' + str(dim_ordering))

    if _FLOATX == 'float64':
        x = tf.cast(x, 'float64')
    return x


def conv3d(x, kernel, strides=(1, 1, 1), border_mode='valid', dim_ordering='th',
           image_shape=None, filter_shape=None):
    raise NotImplementedError


def pool2d(x, pool_size, strides=(1, 1),
           border_mode='valid', dim_ordering='th', pool_mode='max'):
    '''
    # Arguments
        pool_size: tuple of 2 integers.
        strides: tuple of 2 integers.
        border_mode: one of "valid", "same".
        dim_ordering: one of "th", "tf".
    '''
    if border_mode == 'same':
        padding = 'SAME'
    elif border_mode == 'valid':
        padding = 'VALID'
    else:
        raise Exception('Invalid border mode: ' + str(border_mode))

    strides = (1,) + strides + (1,)
    pool_size = (1,) + pool_size + (1,)

    if _FLOATX == 'float64':
        # tf max_pool only supports float32
        x = tf.cast(x, 'float32')

    if dim_ordering in {'tf', 'th'}:
        if dim_ordering == 'th':
            # TF uses the last dimension as channel dimension,
            # instead of the 2nd one.
            # TH input shape: (samples, input_depth, rows, cols)
            # TF input shape: (samples, rows, cols, input_depth)
            # TH kernel shape: (depth, input_depth, rows, cols)
            # TF kernel shape: (rows, cols, input_depth, depth)
            x = tf.transpose(x, (0, 2, 3, 1))
        if pool_mode == 'max':
            x = tf.nn.max_pool(x, pool_size, strides, padding=padding)
        elif pool_mode == 'avg':
            x = tf.nn.avg_pool(x, pool_size, strides, padding=padding)
        else:
            raise Exception('Invalid pooling mode: ' + str(pool_mode))
        if dim_ordering == 'th':
            x = tf.transpose(x, (0, 3, 1, 2))
    else:
        raise Exception('Unknown dim_ordering: ' + str(dim_ordering))

    if _FLOATX == 'float64':
        x = tf.cast(x, 'float64')
    return x


def pool3d(x, pool_size, strides=(1, 1, 1),
           border_mode='valid', dim_ordering='th', pool_mode='max'):
    raise NotImplementedError

# ===========================================================================
# RANDOMNESS
# ===========================================================================


class _RandomWrapper(object):

    def __init__(self, rng):
        super(_RandomWrapper, self).__init__()
        self._rng = np.random.RandomState(rng)
        self._state = np.random.RandomState(rng)

    def randint(self):
        return self._state.randint(10e6)

    def normal(self, shape, mean, std, dtype=_FLOATX):
        return tf.random_normal(shape=shape, mean=mean, stddev=std,
            dtype=dtype, seed=self._rng.randint(10e6))

    def uniform(self, shape, low, high, dtype=_FLOATX):
        return tf.random_uniform(shape=shape, minval=low, maxval=high,
                             dtype=dtype, seed=self._rng.randint(10e6))

    def binomial(self, shape, p, dtype=_FLOATX):
        return tf.cast(
            tf.less(
                tf.random_uniform(shape=shape, minval=0., maxval=1.,
                             dtype=_FLOATX, seed=self._rng.randint(10e6)),
                p),
            dtype)


def rng(seed=None):
    if seed is None:
        seed = get_random_magic_seed()
    return _RandomWrapper(seed)


def random_normal(shape, mean=0.0, std=1.0, dtype=_FLOATX, seed=None):
    if seed is None:
        seed = get_random_magic_seed()
    return tf.random_normal(shape, mean=mean, stddev=std,
                            dtype=dtype, seed=seed)


def random_uniform(shape, low=0.0, high=1.0, dtype=_FLOATX, seed=None):
    if seed is None:
        seed = get_random_magic_seed()
    return tf.random_uniform(shape, minval=low, maxval=high,
                             dtype=dtype, seed=seed)


def random_binomial(shape, p, dtype=_FLOATX, seed=None):
    if seed is None:
        seed = get_random_magic_seed()
    return tf.cast(
        tf.less(
            tf.random_uniform(shape=shape, minval=0., maxval=1.,
                         dtype=dtype, seed=seed),
            p),
        dtype)

# ===========================================================================
# Comparator
# ===========================================================================


def eq(x, y):
    """a == b"""
    return tf.equal(x, y)


def neq(x, y):
    """a != b"""
    return tf.not_equal(x, y)


def gt(a, b):
    """a > b"""
    return tf.greater(a, b)


def ge(a, b):
    """a >= b"""
    return tf.greater_equal(a, b)


def lt(a, b):
    """a < b"""
    return tf.less(a, b)


def le(a, b):
    """a <= b"""
    return tf.less_equal(a, b)


def one_hot(x, nb_class):
    ''' x: 1D-integer vector '''
    shape = x.get_shape()
    ret = tf.zeros((shape[0].value, nb_class), dtype=_FLOATX)
    ret[np.arange(shape[0].value), x] = 1
    return ret


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
    if axis < 0:
        axis = axis % len(x.get_shape())
    shape = x.get_shape()[axis].value
    return tf.cast(
        tf.equal(tf.cast(tf.range(shape), 'int64'),
                expand_dims(tf.argmax(x, axis))),
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
    return tf.mul(x, tf.expand_dims(mask, -1))
