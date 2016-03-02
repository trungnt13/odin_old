# ===========================================================================
# This module is adpated from: https://github.com/fchollet/keras
# Revision: @80927fa
# Original work Copyright (c) 2014-2015 keras contributors
# Some idea are also borrowed from Lasagne library
# Original work Copyright (c) 2014-2015 Lasagne contributors
# Modified work Copyright 2016-2017 TrungNT
# ===========================================================================

import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.signal import pool
from theano.tensor.nnet import conv3d2d
import numpy as np

from .. import config
from .numpy_backend import get_random_magic_seed, get_random_magic_seed

_FLOATX = config.floatX()
_EPSILON = config.epsilon()

# ===========================================================================
# INTERNAL UTILS
# ===========================================================================
theano.config.floatX = _FLOATX

def _on_gpu():
    '''Return whether the session is set to
    run on GPU or not (i.e. on CPU).
    '''
    return theano.config.device[:3] == 'gpu' or theano.sandbox.cuda.cuda_enabled

if _on_gpu():
    '''Import cuDNN only if running on GPU:
    not having Cuda installed should not
    prevent from running the present code.
    '''
    from theano.sandbox.cuda import dnn

def get_session():
    return _on_gpu()

# ===========================================================================
# VARIABLE MANIPULATION
# ===========================================================================
def variable(value, dtype=_FLOATX, name=None, broadcastable=None):
    '''Instantiate a tensor variable.
    '''
    value = np.asarray(value, dtype=dtype)
    if broadcastable:
        return theano.shared(value=value, name=name, strict=False,
                             broadcastable=broadcastable)
    return theano.shared(value=value, name=name, strict=False)

def is_variable(v):
    return isinstance(v, theano.compile.SharedVariable)

_PLACEHOLDER_ID = 0
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
    name_prefix = 'ID.%d.' % _PLACEHOLDER_ID
    _PLACEHOLDER_ID += 1
    if name is None:
        name = ''
    name = name_prefix + name
    return T.TensorType(dtype, broadcast)(name)

def is_expression(v):
    return isinstance(v, theano.tensor.TensorVariable)

def eval(x):
    '''Run a graph.
    '''
    return x.eval()

# ===========================================================================
# Shape operator
# ===========================================================================
def shape(x):
    '''Return the shape of a tensor.

    Warning: type returned will be different for
    Theano backend (Theano tensor type) and TF backend (TF TensorShape).
    '''
    return x.shape

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
    return variable(np.zeros(shape), dtype, name)


def ones(shape, dtype=_FLOATX, name=None):
    '''Instantiate an all-ones variable.
    '''
    return variable(np.ones(shape), dtype, name)


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

# LINEAR ALGEBRA

'''
Assumed overridden:
+, -, /, *, +=, -=, *=, /=
'''


def dot(x, y):
    return T.dot(x, y)


def transpose(x):
    return T.transpose(x)


def gather(reference, indices):
    '''reference: a tensor.
    indices: an int tensor of indices.

    Return: a tensor of same type as reference.
    '''
    return reference[indices]


# ELEMENT-WISE OPERATIONS


def max(x, axis=None, keepdims=False):
    return T.max(x, axis=axis, keepdims=keepdims)


def min(x, axis=None, keepdims=False):
    return T.min(x, axis=axis, keepdims=keepdims)


def sum(x, axis=None, keepdims=False):
    '''Sum of the values in a tensor, alongside the specified axis.
    '''
    return T.sum(x, axis=axis, keepdims=keepdims)


def prod(x, axis=None, keepdims=False):
    '''Multiply the values in a tensor, alongside the specified axis.
    '''
    return T.prod(x, axis=axis, keepdims=keepdims)

def mean(x, axis=None, keepdims=False):
    dtype = None
    if 'int' in x.dtype:
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
class Function(object):

    def __init__(self, inputs, outputs, updates=[], **kwargs):
        self.function = theano.function(
            inputs, outputs,
            updates=updates,
            allow_input_downcast=True, **kwargs)

    def __call__(self, *inputs):
        return self.function(*inputs)


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
    return T.grad(loss, variables,
        consider_constant=consider_constant, known_grads=known_grads,
        disconnected_inputs='warn')

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
        sequences, outputs_info, non_sequences,
        n_steps=n_steps, truncate_gradient=truncate_gradient,
        go_backwards=go_backwards)

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
# NN OPERATIONS
# ===========================================================================
def relu(x, alpha=0., max_value=None):
    assert hasattr(T.nnet, 'relu'), ('It looks like like your version of '
                                     'Theano is out of date. '
                                     'Install the latest version with:\n'
                                     'pip install git+git://github.com/Theano/Theano.git --upgrade --no-deps')
    x = T.nnet.relu(x, alpha)
    if max_value is not None:
        x = T.minimum(x, max_value)
    return x

def softmax(x):
    return T.nnet.softmax(x)

def softplus(x):
    return T.nnet.softplus(x)

def linear(x):
    return x

def categorical_crossentropy(output, target, from_logits=False):
    if from_logits:
        output = T.nnet.softmax(output)
    else:
        # scale preds so that the class probas of each sample sum to 1
        output /= output.sum(axis=-1, keepdims=True)
    # avoid numerical instability with _EPSILON clipping
    output = T.clip(output, _EPSILON, 1.0 - _EPSILON)
    return T.nnet.categorical_crossentropy(output, target)


def binary_crossentropy(output, target, from_logits=False):
    if from_logits:
        output = T.nnet.sigmoid(output)
    # avoid numerical instability with _EPSILON clipping
    output = T.clip(output, _EPSILON, 1.0 - _EPSILON)
    return T.nnet.binary_crossentropy(output, target)


def sigmoid(x):
    return T.nnet.sigmoid(x)


def hard_sigmoid(x):
    return T.nnet.hard_sigmoid(x)


def tanh(x):
    return T.tanh(x)


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
    # ====== Validate arguments ====== #
    if seed is None:
        seed = get_random_magic_seed()
    if rng is None:
        rng = _RandomWrapper(RandomStreams(seed=seed),
                             np.random.RandomState(seed=seed))
    elif isinstance(rng, RandomStreams):
        rng = _RandomWrapper(rng, np.random.RandomState(seed=seed))
    # ====== Dropout ====== #
    retain_prob = 1. - level
    if noise_shape is None:
        x = x * rng.binomial(shape=x.shape, p=retain_prob, dtype=x.dtype)
    else:
        # validate remove all None or -1 dimension
        noise_shape = tuple([x.shape[i] if j is None or j < 0 else j
                       for i, j in enumerate(noise_shape)])
        # auto select broadcast shape
        broadcast = [i for i, j in enumerate(noise_shape) if j == 1]
        if len(broadcast) > 0:
            x = x * T.addbroadcast(
                rng.binomial(shape=noise_shape, p=retain_prob, dtype=x.dtype),
                *broadcast)
        else:
            x = x * rng.binomial(shape=noise_shape, p=retain_prob, dtype=x.dtype)
    if rescale:
        x /= retain_prob
    return x

# ==================== Regularizations ==================== #
def l2_normalize(x, axis):
    norm = T.sqrt(T.sum(T.square(x), axis=axis, keepdims=True))
    return x / norm

def l2_regularize(x):
    return T.sum(T.square(x))

def l1_regularize(x):
    return T.sum(T.abs_(x))

def jacobian_regularize(hidden, params):
    ''' Computes the jacobian of the hidden layer with respect to
    the input, reshapes are necessary for broadcasting the
    element-wise product on the right axis
    '''
    hidden = hidden * (1 - hidden)
    L = expand_dims(hidden, 1) * expand_dims(params, 0)
    # Compute the jacobian and average over the number of samples/minibatch
    L = T.sum(T.pow(L, 2)) / hidden.shape[0]
    return T.mean(L)

def kl_gaussian(mean, logsigma,
                prior_mean=0., prior_logsigma=0.):
    ''' KL-divergence between two gaussians.
    Useful for Variational AutoEncoders. Use this as an activation regularizer
    Parameters:
    -----------
    mean, logsigma: parameters of the input distributions
    prior_mean, prior_logsigma: paramaters of the desired distribution (note the
        log on logsigma)

    Note
    ----
    origin implementation from seya:
    https://github.com/EderSantana/seya/blob/master/seya/regularizers.py
    Copyright (c) EderSantana
    '''
    kl = (prior_logsigma - logsigma +
          0.5 * (-1 + T.exp(2 * logsigma) + (mean - prior_mean) ** 2) /
          T.exp(2 * prior_logsigma))
    return T.mean(kl)

def correntropy_regularize(x, sigma=1.):
    '''
    Note
    ----
    origin implementation from seya:
    https://github.com/EderSantana/seya/blob/master/seya/regularizers.py
    Copyright (c) EderSantana
    '''
    return -T.sum(T.mean(T.exp(x**2 / sigma), axis=0)) / T.sqrt(2 * np.pi * sigma)

# ===========================================================================
# CONVOLUTIONS
# ===========================================================================
def conv2d(x, kernel, strides=(1, 1),
           border_mode='valid', dim_ordering='th',
           image_shape=None, filter_shape=None):
    '''
    Run on cuDNN if available.
    border_mode: string, "same" or "valid".
    '''
    if dim_ordering not in {'th', 'tf'}:
        raise Exception('Unknown dim_ordering ' + str(dim_ordering))

    if dim_ordering == 'tf':
        # TF uses the last dimension as channel dimension,
        # instead of the 2nd one.
        # TH input shape: (samples, input_depth, rows, cols)
        # TH kernel shape: (depth, input_depth, rows, cols)
        # TF input shape: (samples, rows, cols, input_depth)
        # TF kernel shape: (rows, cols, input_depth, depth)
        x = x.dimshuffle((0, 3, 1, 2))
        kernel = kernel.dimshuffle((3, 2, 0, 1))
        if image_shape:
            image_shape = (image_shape[0], image_shape[3],
                           image_shape[1], image_shape[2])
        if filter_shape:
            filter_shape = (filter_shape[3], filter_shape[2],
                            filter_shape[0], filter_shape[1])

    if _on_gpu() and dnn.dnn_available():
        if border_mode == 'same':
            np_kernel = kernel.eval()
            # mode same and even filter
            if len([s for s in np_kernel.shape[2:] if s % 2 == 0]) > 0.:
                assert strides[0] <= np_kernel.shape[2], \
                    'strides should be smaller than the convolution window.'
                assert strides[1] <= np_kernel.shape[3], \
                    'strides should be smaller than the convolution window.'
                conv_out = dnn.dnn_conv(img=x,
                                        kerns=kernel,
                                        border_mode='full')
                shift_x = (np_kernel.shape[2] - strides[0]) // 2
                shift_y = (np_kernel.shape[3] - strides[1]) // 2
                expected_width = (x.shape[2] + strides[0] - 1) // strides[0]
                expected_height = (x.shape[3] + strides[1] - 1) // strides[1]
                conv_out = conv_out[:, :,
                                    shift_x: shift_x + expected_width,
                                    shift_y: shift_y + expected_height]
            else: # same mode and odd filter
                border_mode = tuple(s // 2 for s in np_kernel.shape[2:])
                conv_out = dnn.dnn_conv(img=x,
                                        kerns=kernel,
                                        border_mode=border_mode,
                                        subsample=strides)
        else:
            conv_out = dnn.dnn_conv(img=x,
                                    kerns=kernel,
                                    border_mode=border_mode,
                                    subsample=strides)
    else:
        if border_mode == 'same' or border_mode == 'full':
            th_border_mode = 'full'
            np_kernel = kernel.eval()
            assert strides[0] <= np_kernel.shape[2], 'strides should be smaller than the convolution window.'
            assert strides[1] <= np_kernel.shape[3], 'strides should be smaller than the convolution window.'
        elif border_mode == 'valid':
            th_border_mode = 'valid'
        elif isinstance(border_mode, (tuple, list)):
            th_border_mode = border_mode
        else:
            raise Exception('Border mode not supported: ' + str(border_mode))

        conv_out = T.nnet.conv2d(x, kernel,
                                 border_mode=th_border_mode,
                                 subsample=strides,
                                 input_shape=image_shape,
                                 filter_shape=filter_shape)
        if border_mode == 'same':
            shift_x = (np_kernel.shape[2] - strides[0]) // 2
            shift_y = (np_kernel.shape[3] - strides[1]) // 2
            expected_width = (x.shape[2] + strides[0] - 1) // strides[0]
            expected_height = (x.shape[3] + strides[1] - 1) // strides[1]

            conv_out = conv_out[:, :,
                                shift_x: shift_x + expected_width,
                                shift_y: shift_y + expected_height]
    if dim_ordering == 'tf':
        conv_out = conv_out.dimshuffle((0, 2, 3, 1))
    return conv_out

def conv3d(x, kernel, strides=(1, 1, 1),
           border_mode='valid', dim_ordering='th',
           image_shape=None, filter_shape=None):
    '''
    Run on cuDNN if available.
    border_mode: string, "same" or "valid".
    conv_mode: string, "conv" or "cross".
    '''
    if dim_ordering not in {'th', 'tf'}:
        raise Exception('Unknown dim_ordering ' + str(dim_ordering))

    if dim_ordering == 'tf':
        # TF uses the last dimension as channel dimension,
        # instead of the 2nd one.
        # TH input shape: (samples, input_depth, rows, cols, time)
        # TH kernel shape: (depth, input_depth, rows, cols, time)
        # TF input shape: (samples, rows, cols, time, input_depth)
        # TF kernel shape: (rows, cols, time, input_depth, depth)
        x = x.dimshuffle((0, 4, 1, 2, 3))
        kernel = kernel.dimshuffle((4, 3, 0, 1, 2))
        if image_shape:
            image_shape = (image_shape[0], image_shape[4],
                           image_shape[1], image_shape[2],
                           image_shape[3])
        if filter_shape:
            filter_shape = (filter_shape[4], filter_shape[3],
                            filter_shape[0], filter_shape[1],
                            filter_shape[2])

    if _on_gpu() and dnn.dnn_available():
        if border_mode == 'same':
            np_kernel = kernel.eval()
            border_mode = tuple(s // 2 for s in np_kernel.shape[2:])
        conv_out = dnn.dnn_conv3d(img=x,
                                kerns=kernel,
                                border_mode=border_mode,
                                subsample=strides)
    else:
        if border_mode == 'same':
            assert(strides == (1, 1, 1))
            pad_dim1 = (kernel.shape[2] - 1)
            pad_dim2 = (kernel.shape[3] - 1)
            pad_dim3 = (kernel.shape[4] - 1)
            output_shape = (x.shape[0], x.shape[1],
                            x.shape[2] + pad_dim1,
                            x.shape[3] + pad_dim2,
                            x.shape[4] + pad_dim3)
            output = T.zeros(output_shape)
            indices = (slice(None), slice(None),
                       slice(pad_dim1 // 2, x.shape[2] + pad_dim1 // 2),
                       slice(pad_dim2 // 2, x.shape[3] + pad_dim2 // 2),
                       slice(pad_dim3 // 2, x.shape[4] + pad_dim3 // 2))
            x = T.set_subtensor(output[indices], x)
            border_mode = 'valid'

        border_mode_3d = (border_mode, border_mode, border_mode)
        conv_out = conv3d2d.conv3d(signals=x.dimshuffle(0, 2, 1, 3, 4),
                                   filters=kernel.dimshuffle(0, 2, 1, 3, 4),
                                   border_mode=border_mode_3d)
        conv_out = conv_out.dimshuffle(0, 2, 1, 3, 4)

        # support strides by manually slicing the output
        if strides != (1, 1, 1):
            conv_out = conv_out[:, :, ::strides[0], ::strides[1], ::strides[2]]
    if dim_ordering == 'tf':
        conv_out = conv_out.dimshuffle((0, 2, 3, 1))
    return conv_out

def pool2d(x, pool_size, strides=(1, 1), border_mode='valid',
           dim_ordering='th', pool_mode='max'):
    # ====== dim ordering ====== #
    if dim_ordering not in {'th', 'tf'}:
        raise Exception('Unknown dim_ordering ' + str(dim_ordering))
    if dim_ordering == 'tf':
        x = x.dimshuffle((0, 3, 1, 2))
    # ====== border mode ====== #
    if border_mode == 'same':
        w_pad = pool_size[0] - 2 if pool_size[0] % 2 == 1 else pool_size[0] - 1
        h_pad = pool_size[1] - 2 if pool_size[1] % 2 == 1 else pool_size[1] - 1
        padding = (w_pad, h_pad)
    elif border_mode == 'valid':
        padding = (0, 0)
    elif isinstance(border_mode, (tuple, list)):
        padding = tuple(border_mode)
    else:
        raise Exception('Invalid border mode: ' + str(border_mode))

    # ====== pooling ====== #
    if _on_gpu() and dnn.dnn_available():
        pool_out = dnn.dnn_pool(x, pool_size,
                                stride=strides,
                                mode=pool_mode,
                                pad=padding)
    else: # CPU veresion support by theano
        pool_out = pool.pool_2d(x, ds=pool_size, st=strides,
                                ignore_border=True,
                                padding=padding,
                                mode=pool_mode)

    if dim_ordering == 'tf':
        pool_out = pool_out.dimshuffle((0, 2, 3, 1))
    return pool_out

def pool3d(x, pool_size, strides=(1, 1, 1), border_mode='valid',
           dim_ordering='th', pool_mode='max'):
    # ====== dim ordering ====== #
    if dim_ordering not in {'th', 'tf'}:
        raise Exception('Unknown dim_ordering ' + str(dim_ordering))
    if dim_ordering == 'tf':
        x = x.dimshuffle((0, 4, 1, 2, 3))
    # ====== border mode ====== #
    if border_mode == 'same':
        w_pad = pool_size[0] - 2 if pool_size[0] % 2 == 1 else pool_size[0] - 1
        h_pad = pool_size[1] - 2 if pool_size[1] % 2 == 1 else pool_size[1] - 1
        d_pad = pool_size[2] - 2 if pool_size[2] % 2 == 1 else pool_size[2] - 1
        padding = (w_pad, h_pad, d_pad)
    elif border_mode == 'valid':
        padding = (0, 0, 0)
    elif isinstance(border_mode, (tuple, list)):
        padding = tuple(border_mode)
    else:
        raise Exception('Invalid border mode: ' + str(border_mode))
    # ====== pooling ====== #
    if _on_gpu() and dnn.dnn_available():
        pool_out = dnn.dnn_pool(x, pool_size,
                                stride=strides,
                                mode=pool_mode,
                                pad=padding)
    else:
        padding = padding[:2]
        # pooling over conv_dim2, conv_dim1 (last two channels)
        output = pool.pool_2d(input=x.dimshuffle(0, 1, 4, 3, 2),
                              ds=(pool_size[1], pool_size[0]),
                              st=(strides[1], strides[0]),
                              ignore_border=True,
                              padding=padding,
                              mode=pool_mode)
        # pooling over conv_dim3
        pool_out = pool.pool_2d(input=output.dimshuffle(0, 1, 4, 3, 2),
                                ds=(1, pool_size[2]),
                                st=(1, strides[2]),
                                ignore_border=True,
                                padding=padding,
                                mode=pool_mode)
    # ====== output ====== #
    if dim_ordering == 'tf':
        pool_out = pool_out.dimshuffle((0, 2, 3, 4, 1))
    return pool_out

# ===========================================================================
# RANDOMNESS
# ===========================================================================
class _RandomWrapper(object):

    def __init__(self, rng, state):
        super(_RandomWrapper, self).__init__()
        self._rng = rng
        self._state = state

    def randint(self):
        return self._state.randint(10e6)

    def normal(self, shape, mean, std, dtype=_FLOATX):
        return self._rng.normal(size=shape, avg=mean, std=std, dtype=dtype)

    def uniform(self, shape, low, high, dtype=_FLOATX):
        return self._rng.uniform(size=shape, low=low, high=high, dtype=dtype)

    def binomial(self, shape, p, dtype=_FLOATX):
        return self._rng.binomial(size=shape, n=1, p=p, dtype=dtype)

def rng(seed=None):
    if seed is None:
        seed = get_random_magic_seed()
    return _RandomWrapper(RandomStreams(seed=seed),
                          np.random.RandomState(seed=seed))

def random_normal(shape, mean=0.0, std=1.0, dtype=_FLOATX, seed=None):
    if seed is None:
        seed = get_random_magic_seed()
    rng = RandomStreams(seed=seed)
    return rng.normal(size=shape, avg=mean, std=std, dtype=dtype)

def random_uniform(shape, low=0.0, high=1.0, dtype=_FLOATX, seed=None):
    if seed is None:
        seed = get_random_magic_seed()
    rng = RandomStreams(seed=seed)
    return rng.uniform(shape, low=low, high=high, dtype=dtype)

def random_binomial(shape, p, dtype=_FLOATX, seed=None):
    if seed is None:
        seed = get_random_magic_seed()
    rng = RandomStreams(seed=seed)
    return rng.binomial(size=shape, n=1, p=p, dtype=dtype)

'''
more TODO:

tensordot -> soon to be introduced in TF
batched_tensordot -> reimplement
'''
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
