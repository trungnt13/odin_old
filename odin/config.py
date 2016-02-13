from __future__ import print_function

import os
from .logger import set_enable, info

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

def _parse_config():
    global _FLOATX
    global _BACKEND
    global _EPSILON
    global _DEVICE
    global _VERBOSE

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
        else:
            try:
                i = float(i)
                _EPSILON = i
            except:
                pass

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

_parse_config()
if verbose():
    info('[Config] Device : %s' % _DEVICE)
    info('[Config] Backend: %s' % _BACKEND)
    info('[Config] FloatX : %s' % _FLOATX)
    info('[Config] Epsilon: %s' % _EPSILON)
