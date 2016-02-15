from __future__ import absolute_import, print_function, division
import numpy as np
from . import tensor
from .config import epsilon

# ===========================================================================
# Differentiable
# ===========================================================================
def squared_error(y_pred, y_true):
    return tensor.square(y_pred - y_true)

def absolute_error(y_pred, y_true):
    return tensor.abs(y_pred - y_true)

def absolute_percentage_error(y_pred, y_true):
    diff = tensor.abs((y_true - y_pred) / tensor.clip(tensor.abs(y_true), epsilon(), np.inf))
    return 100. * diff

def squared_logarithmic_error(y_pred, y_true):
    first_log = tensor.log(tensor.clip(y_pred, epsilon(), np.inf) + 1.)
    second_log = tensor.log(tensor.clip(y_true, epsilon(), np.inf) + 1.)
    return tensor.square(first_log - second_log)

def squared_hinge(y_pred, y_true):
    return tensor.square(tensor.maximum(1. - y_true * y_pred, 0.))

def hinge(y_pred, y_true, binary=False, delta=1.):
    """Computes the binary hinge loss between predictions and targets.

    .. math:: L_i = \\max(0, \\delta - t_i p_i)

    Parameters
    ----------
    y_pred : Theano tensor
        Predictions in (0, 1), such as sigmoidal output of a neural network.
    y_true : Theano tensor
        Targets in {0, 1} (or in {-1, 1} depending on `binary`), such as
        ground truth labels.
    binary : bool, default True
        ``True`` if targets are in {0, 1}, ``False`` if they are in {-1, 1}
    delta : scalar, default 1
        The hinge loss margin

    Returns
    -------
    Theano tensor
        An expression for the element-wise binary hinge loss

    Notes
    -----
    This is an alternative to the binary cross-entropy loss for binary
    classification problems
    """
    if binary:
        y_true = 2 * y_true - 1
    return tensor.maximum(delta - y_true * y_pred, 0.)

def categorical_crossentropy(y_pred, y_true):
    '''Expects a binary class matrix instead of a vector of scalar classes.
    '''
    return tensor.categorical_crossentropy(y_pred, y_true)

def binary_crossentropy(y_pred, y_true):
    return tensor.binary_crossentropy(y_pred, y_true)

def poisson(y_pred, y_true):
    return y_pred - y_true * tensor.log(y_pred + epsilon())

def cosine_proximity(y_pred, y_true):
    y_true = tensor.l2_normalize(y_true, axis=-1)
    y_pred = tensor.l2_normalize(y_pred, axis=-1)
    return -y_true * y_pred

# ===========================================================================
# Not differentiable
# ===========================================================================
def binary_accuracy(y_pred, y_true, threshold=0.5):
    """ Non-differentiable """
    y_pred = tensor.ge(y_pred, threshold)
    return tensor.eq(tensor.cast(y_pred, 'int32'),
                     tensor.cast(y_true, 'int32'))

def categorical_accuracy(y_pred, y_true, top_k=1):
    """ Non-differentiable """
    if tensor.ndim(y_true) == tensor.ndim(y_pred):
        y_true = tensor.argmax(y_true, axis=-1)
    elif tensor.ndim(y_true) != tensor.ndim(y_pred) - 1:
        raise TypeError('rank mismatch between y_true and y_pred')

    if top_k == 1:
        # standard categorical accuracy
        top = tensor.argmax(y_pred, axis=-1)
        return tensor.eq(top, y_true)
    else:
        # top-k accuracy
        top = tensor.argtop_k(y_pred, top_k)
        y_true = tensor.expand_dims(y_true, dim=-1)
        return tensor.any(tensor.eq(top, y_true), axis=-1)
