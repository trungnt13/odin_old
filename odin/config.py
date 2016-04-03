from __future__ import print_function, division, absolute_import

import os
import re
from .logger import set_enable, info, set_print_level, error

__all__ = [
    'floatX',
    'backend',
    'epsilon'
]

_valid_device_name = re.compile('(cuda|gpu)\d+')
_valid_cnmem_name = re.compile('(cnmem)[=]?[10]?\.\d*')

_ODIN_FLAGS = os.getenv("ODIN", "")
_FLOATX = 'float32'
_BACKEND = None
_EPSILON = 10e-8
_DEVICE = []
_VERBOSE = False
_FAST_CNN = False
_GRAPHIC = False
_CNMEM = 0.


def _parse_config():
    global _FLOATX
    global _BACKEND
    global _EPSILON
    global _DEVICE
    global _VERBOSE
    global _FAST_CNN
    global _GRAPHIC
    global _CNMEM

    s = _ODIN_FLAGS.split(',')
    for i in s:
        i = i.lower()
        # ====== Data type ====== #
        if 'float' in i or 'int' in i:
            _FLOATX = i
            if _FLOATX == 'float16':
                _EPSILON = 10e-5
            elif _FLOATX == 'float32':
                _EPSILON = 10e-8
            elif _FLOATX == 'float64':
                _EPSILON = 10e-12
        # ====== Backend ====== #
        elif 'theano' in i:
            _BACKEND = 'theano'
        elif 'tensorflow' in i or 'tf' in i:
            _BACKEND = 'tensorflow'
        # ====== Devices ====== #
        elif 'cpu' == i and len(_DEVICE) == 0:
            _DEVICE = 'cpu'
        elif 'gpu' in i or 'cuda' in i:
            if isinstance(_DEVICE, str):
                raise ValueError('Device already setted to cpu')
            if i == 'gpu': i = 'gpu0'
            elif i == 'cuda': i = 'cuda0'
            if _valid_device_name.match(i) is None:
                raise ValueError('Unsupport device name: %s '
                                 '(must be "cuda"|"gpu" and optional number)'
                                 ', e.g: cuda0, gpu0, cuda, gpu, ...' % i)
            _DEVICE.append(i.replace('gpu', 'cuda'))
        # ====== others ====== #
        elif 'verbose' in i:
            _VERBOSE = True
            set_enable(True)
            try:
                if len(i) > len('verbose'):
                    log_level = int(i[len('verbose'):])
                    set_print_level(log_level)
            except Exception, e:
                error('Fail to read verbose level, error:' + str(e))
        # ====== fastcnn ====== #
        elif 'fastcnn' in i:
            _FAST_CNN = True
        # ====== graphic ====== #
        elif 'graphic' in i:
            _GRAPHIC = True
        # ====== cnmem ====== #
        elif 'cnmem' in i:
            match = _valid_cnmem_name.match(i)
            if match is None:
                raise ValueError('Unsupport CNMEM format: %s. '
                                 'Valid format must be: cnmem=0.75 or cnmem=.75 '
                                 ' or cnmem.75' % str(i))

            i = i[match.start():match.end()].replace('cnmem', '').replace('=', '')
            _CNMEM = float(i)

    # set non-graphic backend for matplotlib
    if not _GRAPHIC:
        try:
            import matplotlib
            matplotlib.use('Agg')
        except:
            pass
    # if DEVICE still len = 0, use cpu
    if len(_DEVICE) == 0:
        _DEVICE = 'cpu'


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
def cnmem():
    return float(_CNMEM)


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
    info('[Config] CNMEM: %s' % _CNMEM)
    info('[Config] Graphic: %s' % _GRAPHIC)
