# -*- coding: utf-8 -*-
# ===========================================================================
# This module is created based on the code from Lasagne library
# Original work Copyright (c) 2014-2015 Lasagne contributors
# Modified work Copyright 2016-2017 TrungNT
# ===========================================================================
from __future__ import print_function, division

from six.moves import zip_longest, zip, range
import numpy as np
from collections import defaultdict

from .. import tensor as T
from ..base import OdinFunction
from ..utils import as_tuple, as_index_map
from .dense import Dense
from .ops import Ops

__all__ = [
    "Gate",
    "Cell",
    "GRUCell",
    "Recurrent",
    "lstm_algorithm",
    "gru_algorithm",
    "simple_algorithm",
    "GRU",
    "LSTM",
]


# ===========================================================================
# Helper
# ===========================================================================
