from __future__ import print_function, division

import sys
import logging
import time
# ===========================================================================
# Main Code
# ===========================================================================

class logger():
    _last_value = 0
    _last_time = -1
    _default_logger = None
    _is_enable = True
    """docstring for Logger"""

    @staticmethod
    def set_enable(is_enable):
        logger._is_enable = is_enable

    @staticmethod
    def _check_init_logger():
        if logger._default_logger is None:
            logger.create_logger(logging_path=None)

    @staticmethod
    def set_print_level(level):
        ''' VERBOSITY level:
         - CRITICAL: 50
         - ERROR   : 40
         - WARNING : 30
         - INFO    : 20
         - DEBUG   : 10
         - UNSET   : 0
        '''
        logger._check_init_logger()
        logger._default_logger.handlers[0].setLevel(level)

    @staticmethod
    def set_save_path(logging_path, mode='w', multiprocess=False):
        '''All old path will be ignored'''
        logger._check_init_logger()
        log = logger._default_logger
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

    @staticmethod
    def warning(*anything, **kwargs):
        if not logger._is_enable: return
        logger._check_init_logger()
        logger._default_logger.warning(*anything)

    @staticmethod
    def error(*anything, **kwargs):
        if not logger._is_enable: return
        logger._check_init_logger()
        logger._default_logger.error(*anything)

    @staticmethod
    def critical(*anything, **kwargs):
        if not logger._is_enable: return
        logger._check_init_logger()
        logger._default_logger.critical(*anything)

    @staticmethod
    def debug(*anything, **kwargs):
        if not logger._is_enable: return
        logger._check_init_logger()
        logger._default_logger.debug(*anything)

    @staticmethod
    def info(*anything, **kwargs):
        if not logger._is_enable: return
        logger._check_init_logger()
        if len(anything) == 0: logger._default_logger.info('')
        else: logger._default_logger.info(*anything)

    @staticmethod
    def log(*anything, **kwargs):
        '''This log is at INFO level'''
        if not logger._is_enable: return
        logger._check_init_logger()
        # format with only messages
        for h in logger._default_logger.handlers:
            h.setFormatter(logging.Formatter(fmt = '%(message)s'))

        if len(anything) == 0:
            logger._default_logger.info('')
        else:
            logger._default_logger.info(*anything)

        # format with time and level
        for h in logger._default_logger.handlers:
            h.setFormatter(logging.Formatter(
                fmt = '%(asctime)s %(levelname)s  %(message)s',
                datefmt = '%d/%m/%Y %I:%M:%S'))

    _last_progress_idx = None

    @staticmethod
    def progress(p, max_val=1.0, title='Progress', bar='=', newline=False, idx=None):
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
        '''
        if not logger._is_enable:
            return
        # ====== Check same progress or not ====== #
        if logger._last_progress_idx != idx:
            print()
        logger._last_progress_idx = idx

        # ====== Config ====== #
        if p < 0: p = 0.0
        if p > max_val: p = max_val
        fmt_str = "\r%s (%.2f/%.2f)[%s] - ETA:%.2fs ETD:%.2fs"
        if max_val > 100:
            p = int(p)
            max_val = int(max_val)
            fmt_str = "\r%s (%d/%d)[%s] - ETA:%.2fs ETD:%.2fs"

        if newline:
            fmt_str = fmt_str[1:]
            fmt_str += '\n'
        # ====== ETA: estimated time of arrival ====== #
        if logger._last_time < 0:
            logger._last_time = time.time()
        eta = (max_val - p) / max(1e-13, abs(p - logger._last_value)) * (time.time() - logger._last_time)
        etd = time.time() - logger._last_time
        logger._last_value = p
        logger._last_time = time.time()
        # ====== print ====== #
        max_val_bar = 20
        n_bar = int(p / max_val * max_val_bar)
        bar = '=' * n_bar + '>' + ' ' * (max_val_bar - n_bar)
        sys.stdout.write(fmt_str % (title, p, max_val, bar, eta, etd))
        sys.stdout.flush()
        # if p >= max_val:
        #     sys.stdout.write("\n")

    @staticmethod
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
        logger for dnntoolkit
        '''
        if name is None:
            name = 'default'
        log = logging.getLogger('dnntoolkit.%s' % name)
        # remove all old handler
        log.handlers = []
        # print
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(logging.Formatter(
            fmt = '%(asctime)s %(levelname)s:  %(message)s',
            datefmt = '%d/%m/%Y %I:%M:%S'))
        sh.setLevel(logging.DEBUG)

        # add the current logger
        log.setLevel(logging.DEBUG)
        log.addHandler(sh)

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

        # enable or disable
        if name == 'default':
            logger._default_logger = log
            logger.set_enable(logger._is_enable)
        return log
