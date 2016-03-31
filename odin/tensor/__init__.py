from __future__ import absolute_import, print_function
import os

from .. import config
from .. import logger
from .numpy_backend import *


# ===========================================================================
# Load the config
# ===========================================================================
def _load_theano_config():
    if config.device() == 'cpu':
        flags = "mode=FAST_RUN,device=%s,floatX=%s" % (config.device(), config.floatX())
    else:
        contexts = ';'.join(['dev%d->cuda%d' % (i, int(_.replace('cuda', '')))
                             for i, _ in enumerate(config.device())])
        flags = "contexts=" + contexts + ",mode=FAST_RUN,floatX=%s" % config.floatX()
    # ====== others ====== #
    if config.verbose():
        flags += ',exception_verbosity=high'
    # Speedup CuDNNv4
    if config.fastcnn() and config.device() == 'gpu':
        flags += ',dnn.conv.algo_fwd=time_once,dnn.conv.algo_bwd_filter=time_once,dnn.conv.algo_bwd_data=time_once'
        logger.warning('Using fast cnn algorithm, only compatible with CuDNN v4.')
    os.environ['THEANO_FLAGS'] = flags


# ===========================================================================
# Load backend
# ===========================================================================
if config.backend() == 'theano':
    _load_theano_config()
    logger.critical('Using Theano backend, flags:%s' % os.environ['THEANO_FLAGS'])
    from .theano_backend import *
elif config.backend() == 'tensorflow':
    logger.critical('Using TensorFlow backend.')
    from .tf_backend import *
# ===========================================================================
# Auto load backend
# ===========================================================================
else:
    is_load_backend = False
    try:
        _load_theano_config()
        import theano
        logger.critical('Auto load theano_backend backend, flags:%s' % os.environ['THEANO_FLAGS'])
        from .theano_backend import *
        is_load_backend = True
        config.set_backend('theano')
    except Exception, e:
        logger.critical('Failed to load theano, error:' + str(e))

    if not is_load_backend:
        try:
            import tensorflow
            from .tf_backend import *
            logger.critical('Auto load tensorflow_backend backend.')
            is_load_backend = True
            config.set_backend('tensorflow')
        except Exception, e:
            logger.critical('Failed to load tensorflow, error:' + str(e))
    if not is_load_backend:
        raise Exception('Unknown backend: ' + str(config.backend()))
