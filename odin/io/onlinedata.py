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


@staticmethod
def load_mnist(path='https://s3.amazonaws.com/ai-datasets/mnist.h5'):
    '''
    path : str
        local path or url to hdf5 datafile
    '''
    datapath = get_file('mnist.h5', path)
    logger.info('Loading data from: %s' % datapath)
    try:
        ds = dataset(datapath, mode='r')
    except:
        if os.path.exists(datapath):
            os.remove(datapath)
        datapath = get_file('mnist.h5', path)
        ds = dataset(datapath, mode='r')
    return ds


def load_imdb(path):
    pass


def load_reuters(path):
    pass
