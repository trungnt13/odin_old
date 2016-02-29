from __future__ import print_function, division, absolute_import

import os
from .logger import set_enable, info, set_print_level, error

__all__ = [
    'floatX',
    'backend',
    'epsilon'
]

_ODIN_FLAGS = os.getenv("ODIN", "")
_FLOATX = 'float32'
_BACKEND = None
_EPSILON = 10e-8
_DEVICE = 'cpu'
_VERBOSE = False
_FAST_CNN = False
_GRAPHIC = False

def _parse_config():
    global _FLOATX
    global _BACKEND
    global _EPSILON
    global _DEVICE
    global _VERBOSE
    global _FAST_CNN
    global _GRAPHIC

    s = _ODIN_FLAGS.split(',')
    for i in s:
        i = i.lower()
        if 'float' in i or 'int' in i:
            _FLOATX = i
        elif 'theano' in i:
            _BACKEND = 'theano'
        elif 'tensorflow' in i or 'tf' in i:
            _BACKEND = 'tensorflow'
        elif 'gpu' == i or 'cpu' == i:
            _DEVICE = i
        elif 'verbose' in i:
            _VERBOSE = True
            set_enable(True)
            try:
                if len(i) > len('verbose'):
                    log_level = int(i[len('verbose'):])
                    set_print_level(log_level)
            except Exception, e:
                error('Fail to read verbose level, error:' + str(e))
        elif 'fastcnn' in i:
            _FAST_CNN = True
        elif 'graphic' in i:
            _GRAPHIC = True
        else:
            try:
                i = float(i)
                _EPSILON = i
            except:
                pass

    # set non-graphic backend for matplotlib
    if not _GRAPHIC:
        try:
            import matplotlib
            matplotlib.use('Agg')
        except:
            pass

def set_backend(backend):
    global _BACKEND
    if _BACKEND is not None:
        raise ValueError('Cannot set backend after program started!')
    if 'theano' in backend:
        _BACKEND = 'theano'
    elif 'tensorflow' in backend or 'tf' in backend:
        _BACKEND = 'tensorflow'
    else:
        raise ValueError('Unsupport backend: %d!' % backend)

# ===========================================================================
# Parse and get configuration
# ===========================================================================
def floatX():
    return _FLOATX

def backend():
    return _BACKEND

def epsilon():
    return _EPSILON

def device():
    return _DEVICE

def verbose():
    return _VERBOSE

def fastcnn():
    return _FAST_CNN

_parse_config()
if verbose():
    info('[Config] Device : %s' % _DEVICE)
    info('[Config] Backend: %s' % _BACKEND)
    info('[Config] FloatX : %s' % _FLOATX)
    info('[Config] Epsilon: %s' % _EPSILON)
    info('[Config] Fast-cnn: %s' % _FAST_CNN)
    info('[Config] Graphic: %s' % _GRAPHIC)
