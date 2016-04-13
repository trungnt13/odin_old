# ===========================================================================
# Collections of popular online dataset
# ===========================================================================
from __future__ import print_function, division, absolute_import
import os
from .ie import get_file

from .dataset import dataset
from .. import logger

__all__ = [
    'load_mnist'
]


def load_mnist(path='https://s3.amazonaws.com/ai-datasets/mnist.h5'):
    '''
    path : str
        local path or url to hdf5 datafile
    '''
    datapath = get_file('mnist.h5', path)
    return datapath


def load_imdb(path):
    pass


def load_reuters(path):
    pass
