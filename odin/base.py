from __future__ import print_function, division

import numpy as np
from . import logger
from collections import OrderedDict

# ===========================================================================
# Based class design
# ===========================================================================
from abc import ABCMeta, abstractmethod

class OdinObject(object):
    __metaclass__ = ABCMeta
    _logging = True

    def get_config(self):
        config = OrderedDict()
        config['class'] = self.__class__.__name__
        return config

    @staticmethod
    def parse_config(config):
        raise NotImplementedError()

    def set_logging(self, enable):
        self._logging = enable

    def log(self, msg, level=20):
        if not self._logging:
            return
        msg = '[%s]: %s' % (self.__class__.__name__, str(msg))
        if level == 10:
            logger.debug(msg)
        elif level == 20:
            logger.info(msg)
        elif level == 30:
            logger.warning(msg)
        elif level == 40:
            logger.error(msg)
        elif level == 50:
            logger.critical(msg)
        else:
            logger.log(msg)
