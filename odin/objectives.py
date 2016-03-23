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
    "MfoM"
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


def bayes_crossentropy(y_pred, y_true, distribution=None):
    '''Expects a binary class matrix instead of a vector of scalar classes.
    '''
    return T.bayes_crossentropy(y_pred, y_true, distribution)


def mean_bayes_crossentropy(y_pred, y_true, distribution=None):
    '''Expects a binary class matrix instead of a vector of scalar classes.
    '''
    return T.mean(T.bayes_crossentropy(y_pred, y_true, distribution))


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
def free_energy(input_sampled, bias_sampled,
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
# Special objective for Language and accent recognition
# ===========================================================================
def MfoM(y_pred, y_true, distribution,
    alpha=1., beta=0., eta=1.,
    Cmiss=1., Cfa=1., Ptar=0.5,
    epsilon=10e-8):
    nb_classes = len(distribution)
    # clip probability value so it not 0. or 1.
    y_pred = T.clip(y_pred, epsilon, 1 - epsilon)
    # if 1 dimension, transform to one-hot encoding
    if T.ndim(y_true) == 1:
        y_true = T.one_hot(y_true, nb_classes)
    # ====== calculate strategy function ====== #
    # trick to clip Non-nan and non-inf values
    _ = T.clip((nb_classes - 1) * y_pred, epsilon, 1 - epsilon)
    d = 1. / eta * (T.log(1 - _) - T.log(_))
    l = 1. / (1 + T.exp(-alpha * d - beta))
    # anti-model one-hot encoding matrix
    y_false = T.switch(y_true, 0, 1)
    # ====== calculate statistics ====== #
    FN = l * y_true
    TN = l * y_false
    TP = (1 - l) * y_true
    FP = (1 - l) * y_false
    # ====== miss and false alarm ====== #
    # sum over all samples (axis=0), plus epsilon so no divide by zero issue
    Pmiss = T.sum(FN / (TP + FN + epsilon), axis=0)
    Pfa = T.sum(FP / (FP + TN + epsilon), axis=0)
    # ====== main cost ====== #
    model_cost = Cmiss * Ptar * Pmiss
    anti_model = 1 / (nb_classes - 1) * (Cfa * (1 - Ptar) * Pfa)
    # Cavg now is vector of [1 x nb_classes], Cpair score for each language
    # within cluster.
    Cavg = 1 / nb_classes * (model_cost + anti_model)
    # now take mean of all languages
    Cavg = T.mean(Cavg)
    return Cavg
