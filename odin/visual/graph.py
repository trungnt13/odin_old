from __future__ import print_function, division, absolute_import

import os
from .. import config
from .. import utils
from .. import tensor


def draw_computational_graph(var_or_func, save_path=None):
    '''
    Use d3viz to provide interactive html graph for theano
    For tensorflow, use: tensorboard --logdir [save_path]
    '''
    from matplotlib import pyplot as plt
    import matplotlib.image as mpimg

    if save_path is None:
        save_path = os.path.join(utils.get_tmp_dir(), 'tmp.html')

    if config.backend() == 'theano':
        import theano.d3viz as d3v
        d3v.d3viz(var_or_func, save_path)
        # theano.printing.pydotprint(
        #     var_or_func,
        #     outfile = save_path)
        return save_path
    elif config.backend() == 'tensorflow':
        import tensorflow as tf
        sess = tensor.get_session()
        writer = tf.python.training.summary_io.SummaryWriter(
            save_path, sess.graph_def)
        tensor.eval(var_or_func)
        writer.close()
        return save_path
