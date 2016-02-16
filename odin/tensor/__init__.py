from __future__ import absolute_import, print_function
import os

from .. import config
from .. import logger

def _load_theano_config():
    flags = "mode=FAST_RUN,device=%s,floatX=%s" % (config.device(), config.floatX())
    if config._VERBOSE:
        flags += ',exception_verbosity=high'
    os.environ['THEANO_FLAGS'] = flags

if config.backend() == 'theano':
    logger.critical('Using Theano backend.')
    _load_theano_config()
    from .theano_backend import *
elif config.backend() == 'tensorflow':
    logger.critical('Using TensorFlow backend.')
    from .tf_backend import *
else:
    is_load_backend = False
    try:
        _load_theano_config()
        import theano
        logger.critical('Auto load theano_backend backend.')
        from .theano_backend import *
        is_load_backend = True
    except:
        pass
    if not is_load_backend:
        try:
            import tensorflow
            from .tf_backend import *
            logger.critical('Auto load tensorflow_backend backend.')
            is_load_backend = True
        except:
            pass
    if not is_load_backend:
        raise Exception('Unknown backend: ' + str(config.backend()))

