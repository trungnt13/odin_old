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
        contexts = ""
        device = "device=%s" % config.device()
    else:
        contexts = "contexts="
        contexts += ';'.join(['dev%d->cuda%d' % (i, int(_.replace('cuda', '')))
                             for i, _ in enumerate(config.device())])
        # TODO: bizarre degradation in performance if not specify device=gpu
        device = 'device=gpu'
    flags = contexts + "," + device + ",mode=FAST_RUN,floatX=%s" % config.floatX()
    # ====== others ====== #
    if config.verbose():
        flags += ',exception_verbosity=high'
    # Speedup CuDNNv4
    if config.fastcnn() and isinstance(config.device(), (list, tuple)):
        flags += ',dnn.conv.algo_fwd=time_once,dnn.conv.algo_bwd_filter=time_once,dnn.conv.algo_bwd_data=time_once'
        logger.warning('Using fast cnn algorithm, only compatible with CuDNN v4.')
    # CNMEM
    if config.cnmem() > 0. and config.cnmem() <= 1.:
        flags += ',lib.cnmem=%.2f,allow_gc=True' % config.cnmem()
    os.environ['THEANO_FLAGS'] = flags


# ===========================================================================
# Load backend
# ===========================================================================
if config.backend() == 'theano':
    _load_theano_config()
    logger.critical('Using Theano backend, flags:%s' % os.environ['THEANO_FLAGS'])
    from .theano_backend import *
    from .nnet_theano import *
    from .mgi_theano import *
    from .stochastic_theano import *
elif config.backend() == 'tensorflow':
    logger.critical('Using TensorFlow backend.')
    from .tf_backend import *
    from .nnet_tf import *
    from .mgi_tf import *
    from .stochastic_tf import *
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
        from .nnet_theano import *
        from .mgi_theano import *
        from .stochastic_theano import *

        is_load_backend = True
        config.set_backend('theano')
    except Exception, e:
        logger.critical('Failed to load theano, error:' + str(e))
        import traceback; traceback.print_exc()

    if not is_load_backend:
        try:
            import tensorflow
            logger.critical('Auto load tensorflow_backend backend.')

            from .tf_backend import *
            from .nnet_tf import *
            from .mgi_tf import *
            from .stochastic_tf import *

            is_load_backend = True
            config.set_backend('tensorflow')
        except Exception, e:
            logger.critical('Failed to load tensorflow, error:' + str(e))
            import traceback; traceback.print_exc()
    if not is_load_backend:
        raise Exception('Unknown backend: ' + str(config.backend()))
