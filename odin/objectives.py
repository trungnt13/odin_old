# ===========================================================================
# This module is created based on the code from 2 libraries: Lasagne and keras
# Original work Copyright (c) 2014-2015 keras contributors
# Original work Copyright (c) 2014-2015 Lasagne contributors
# Modified work Copyright 2016-2017 TrungNT
# ===========================================================================

from __future__ import absolute_import, print_function, division
import numpy as np
from . import tensor as T
from .config import epsilon

__all__ = [
    "squared_loss",
    "mean_squared_loss",
    "absolute_loss",
    "mean_absolute_loss",
    "absolute_percentage_loss",
    "squared_logarithmic_loss",
    "squared_hinge",
    "hinge",
    "categorical_crossentropy",
    "mean_categorical_crossentropy",
    "binary_crossentropy",
    "mean_binary_crossentropy",
    "poisson",
    "cosine_proximity",
    "binary_accuracy",
    "categorical_accuracy"
]

# ===========================================================================
# Differentiable
# ===========================================================================
def squared_loss(y_pred, y_true):
    return T.square(y_pred - y_true)

def mean_squared_loss(y_pred, y_true):
    return T.mean(T.square(y_pred - y_true))

def absolute_loss(y_pred, y_true):
    return T.abs(y_pred - y_true)

def mean_absolute_loss(y_pred, y_true):
    return T.mean(T.abs(y_pred - y_true))

def absolute_percentage_loss(y_pred, y_true):
    diff = T.abs((y_true - y_pred) / T.clip(T.abs(y_true), epsilon(), np.inf))
    return 100. * diff

def squared_logarithmic_loss(y_pred, y_true):
    first_log = T.log(T.clip(y_pred, epsilon(), np.inf) + 1.)
    second_log = T.log(T.clip(y_true, epsilon(), np.inf) + 1.)
    return T.square(first_log - second_log)

def squared_hinge(y_pred, y_true):
    return T.square(T.maximum(1. - y_true * y_pred, 0.))

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
    return T.maximum(delta - y_true * y_pred, 0.)

def categorical_crossentropy(y_pred, y_true):
    '''Expects a binary class matrix instead of a vector of scalar classes.
    '''
    return T.categorical_crossentropy(y_pred, y_true)

def mean_categorical_crossentropy(y_pred, y_true):
    '''Expects a binary class matrix instead of a vector of scalar classes.
    '''
    return T.mean(T.categorical_crossentropy(y_pred, y_true))

def binary_crossentropy(y_pred, y_true):
    return T.binary_crossentropy(y_pred, y_true)

def mean_binary_crossentropy(y_pred, y_true):
    return T.mean(T.binary_crossentropy(y_pred, y_true))

def poisson(y_pred, y_true):
    return y_pred - y_true * T.log(y_pred + epsilon())

def cosine_proximity(y_pred, y_true):
    y_true = T.l2_normalize(y_true, axis=-1)
    y_pred = T.l2_normalize(y_pred, axis=-1)
    return -y_true * y_pred

# ===========================================================================
# Stochastic optimization
# ===========================================================================
def contrastive_divergence(input_sampled, bias_sampled,
                           input_original, bias_original):
    ''' Contrastive divergence cost function
    The input and bias should be calculated as follow:
        input = T.dot(X, self.W) + hbias
        bias = T.dot(X, vbias)
    Then the free energy is calculated as:
        .. math:: F(x) = - \sum{ \log{1 + \exp{input}} }
    '''
    hidden_sampled = T.sum(T.log(1 + T.exp(input_sampled)), axis=1)
    hidden_original = T.sum(T.log(1 + T.exp(input_original)), axis=1)
    return T.mean(-hidden_original - bias_original) - \
    T.mean(-hidden_sampled - bias_sampled)

# ===========================================================================
# Not differentiable
# ===========================================================================
def binary_accuracy(y_pred, y_true, threshold=0.5):
    """ Non-differentiable """
    y_pred = T.ge(y_pred, threshold)
    return T.eq(T.cast(y_pred, 'int32'),
                T.cast(y_true, 'int32'))

def categorical_accuracy(y_pred, y_true, top_k=1):
    """ Non-differentiable """
    if T.ndim(y_true) == T.ndim(y_pred):
        y_true = T.argmax(y_true, axis=-1)
    elif T.ndim(y_true) != T.ndim(y_pred) - 1:
        raise TypeError('rank mismatch between y_true and y_pred')

    if top_k == 1:
        # standard categorical accuracy
        top = T.argmax(y_pred, axis=-1)
        return T.eq(top, y_true)
    else:
        # top-k accuracy
        top = T.argtop_k(y_pred, top_k)
        y_true = T.expand_dims(y_true, dim=-1)
        return T.any(T.eq(top, y_true), axis=-1)
