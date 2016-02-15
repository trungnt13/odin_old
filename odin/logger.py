# ===========================================================================
# Docs
# VERBOSITY level:
#  - CRITICAL: 50
#  - ERROR   : 40
#  - WARNING : 30
#  - INFO    : 20
#  - DEBUG   : 10
#  - UNSET   : 0
# ===========================================================================
from __future__ import print_function, division, absolute_import

import sys
import logging
import time

__all__ = [
    "create_logger",

    "set_print_level",
    "set_enable",
    "is_enable",
    "set_save_path",

    "critical",
    "error",
    "warning",
    "debug",
    "info",
    "log",

    "progress"
]

# ===========================================================================
# Reusable function
# ===========================================================================
def create_logger(name=None, logging_path=None, mode='w', multiprocess=False):
    ''' All logger are created at DEBUG level

    Parameters
    ----------

    Example
    -------
    >>> logger.debug('This is a debug message')
    >>> logger.info('This is an info message')
    >>> logger.warning('This is a warning message')
    >>> logger.error('This is an error message')
    >>> logger.critical('This is a critical error message')

    Note
    ----
    if name is None or default, the created logger will be used as default
    logger for odin
    '''
    if name is None: name = 'default'
    log = logging.getLogger('odin.%s' % name)
    # remove all old handler
    log.handlers = []

    # ====== Print handler ====== #
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(logging.Formatter(
        fmt = '%(asctime)s %(levelname)s:  %(message)s',
        datefmt = '%d/%m/%Y %I:%M:%S'))
    sh.setLevel(logging.DEBUG)
    log.setLevel(logging.DEBUG)
    log.addHandler(sh)

    # ====== Saved log handler ====== #
    if type(logging_path) not in (tuple, list):
        logging_path = [logging_path]

    for path in logging_path:
        if path is not None:
            # saving path
            fh = logging.FileHandler(path, mode=mode)
            fh.setFormatter(logging.Formatter(
                fmt = '%(asctime)s %(levelname)s %(message)s',
                datefmt = '%d/%m/%Y %I:%M:%S'))
            fh.setLevel(logging.DEBUG)
            if multiprocess:
                import multiprocessing_logging
                log.addHandler(
                    multiprocessing_logging.MultiProcessingHandler('mpi', fh))
            else:
                log.addHandler(fh)

    # enable or disable
    if name == 'default':
        global _default_logger
        _default_logger = log
    return log

# ===========================================================================
# Main Code
# ===========================================================================
_logging = False
_default_logger = create_logger()

def set_enable(enable):
    global _logging
    _logging = enable

def is_enable():
    return _logging

def set_print_level(level):
    ''' VERBOSITY level:
     - CRITICAL: 50
     - ERROR   : 40
     - WARNING : 30
     - INFO    : 20
     - DEBUG   : 10
     - UNSET   : 0
    '''
    for h in _default_logger.handlers:
        h.setLevel(level)

def set_save_path(logging_path, mode='w', multiprocess=False):
    '''All old path will be ignored'''
    log = _default_logger
    log.handlers = [log.handlers[0]]

    if type(logging_path) not in (tuple, list):
        logging_path = [logging_path]

    for path in logging_path:
        if path is not None:
            # saving path
            fh = logging.FileHandler(path, mode=mode)
            fh.setFormatter(logging.Formatter(
                fmt = '%(asctime)s %(levelname)s  %(message)s',
                datefmt = '%d/%m/%Y %I:%M:%S'))
            fh.setLevel(logging.DEBUG)
            if multiprocess:
                import multiprocessing_logging
                log.addHandler(
                    multiprocessing_logging.MultiProcessingHandler('mpi', fh))
            else:
                log.addHandler(fh)

def warning(*anything, **kwargs):
    if not _logging: return
    _default_logger.warning(*anything)

def error(*anything, **kwargs):
    if not _logging: return
    _default_logger.error(*anything)

def critical(*anything, **kwargs):
    if not _logging: return
    _default_logger.critical(*anything)

def debug(*anything, **kwargs):
    if not _logging: return
    _default_logger.debug(*anything)

def info(*anything, **kwargs):
    if not _logging: return
    if len(anything) == 0: _default_logger.info('')
    else: _default_logger.info(*anything)

def log(*anything, **kwargs):
    '''This log is at INFO level'''
    if not _logging: return

    # format with only messages
    for h in _default_logger.handlers:
        h.setFormatter(logging.Formatter(fmt = '%(message)s'))
    if len(anything) == 0:
        _default_logger.info('')
    else:
        _default_logger.info(*anything)
    # format with time and level
    for h in _default_logger.handlers:
        h.setFormatter(logging.Formatter(
            fmt = '%(asctime)s %(levelname)s  %(message)s',
            datefmt = '%d/%m/%Y %I:%M:%S'))

# ===========================================================================
# Progress bar
# ===========================================================================
_last_progress_idx = None
_progress_map = {}

def progress(p, max_val=1.0,
             title='Progress', bar='=',
             newline=False, idx='default',
             titlelen=13):
    '''
    Parameters
    ----------
    p : number
        current progress value
    max_val : number
        maximum value progress can reach (be equal)
    idx : anything
        identification of current progress, if 2 progress is diffrent, print
        newline to switch to print new progress

    Notes
    -----
    This methods is not thread safe
    ETA: estimated time arrival
    ETD: estimated time duration
    '''
    # ====== validate args ====== #
    if not _logging: return
    if p < 0:
        p = 0.0
    elif p > max_val:
        p = max_val

    global _last_progress_idx

    p = 100. / max_val * p
    max_val = 100
    # ====== check old process ====== #
    if idx not in _progress_map or _progress_map[idx][0] > p:
        _progress_map[idx] = [0, -1., 1.] # last_value, last_time, etd
    last_value = _progress_map[idx][0]
    last_time = _progress_map[idx][1]
    last_etd = _progress_map[idx][2]

    if _last_progress_idx != idx: # re-update last time
        last_time = -1.

    _last_progress_idx = idx
    # ====== Config ====== #
    fmt_str = "\r%-" + str(titlelen) + "s (%3d/%3d)[%s] - ETD:%.2fs ETA:%.2fs "
    if newline:
        fmt_str = fmt_str[1:] + '\n'
    # ====== ETA: estimated time of arrival ====== #
    if last_time < 0:
        eta = (max_val - p) * last_etd
    else:
        last_etd = time.time() - last_time
        eta = (max_val - p) / max(1e-13, abs(p - last_value)) * last_etd
    last_value = p
    last_time = time.time()

    # ====== print ====== #
    max_val_bar = 20
    n_bar = int(p / max_val * max_val_bar)
    bar = '=' * n_bar + '>' + ' ' * (max_val_bar - n_bar)
    sys.stdout.write(fmt_str % (title, round(p), max_val, bar, last_etd, eta))
    sys.stdout.flush()

    # ====== Save new value ====== #
    _progress_map[idx][0] = last_value
    _progress_map[idx][1] = last_time
    _progress_map[idx][2] = last_etd
