# ======================================================================
# Author: TrungNT
# ======================================================================
from __future__ import print_function, division

from ..utils import function, get_magic_seed, frame
import unittest
import cPickle

import math
import random

# ===========================================================================
# Main Test
# ===========================================================================
def test(a=1, b=2):
    a = math.sqrt(a)
    return 'Shit %s over %s here!' % (str(a), str(b))

class UtilsTest(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_function_save_load(self):
        f = function(test, 3, 4)
        s1 = f()
        config = cPickle.dumps(f.get_config())
        config = cPickle.loads(config)
        f = function.parse_config(config)
        s2 = f()

        self.assertEqual(s1, s2)

    def test_frame(self):
        labels = ['a', 'b', 'c', 'd']
        random.seed(get_magic_seed())
        f = [frame(), frame(), frame()]
        for i in f:
            for k in xrange(10):
                i.record(k, *[random.choice(labels) for j in xrange(3)])

        x1 = f[0].select(['a', 'b'])
        x2 = f[0].select(['a', 'b', 'c'])
        x3 = f[0].select('d')

        # ====== Test pickle frame ====== #
        f = cPickle.loads(cPickle.dumps(f))
        y1 = f[0].select(['a', 'b'])
        y2 = f[0].select(['a', 'b', 'c'])
        y3 = f[0].select('d')

        self.assertEqual(x1, y1)
        self.assertEqual(x2, y2)
        self.assertEqual(x3, y3)

        # ====== Test merge ====== #
        original_len = sum([len(i) for i in f])
        x = f[0].select(['a']) + f[1].select(['a']) + f[2].select(['a'])
        x1 = f[0].select(['a', 'c'], absolute=True) + \
            f[1].select(['a', 'c'], absolute=True) + \
            f[2].select(['a', 'c'], absolute=True)
        x2 = f[0].select(['a', 'd'], absolute=True, filter_value=lambda x: x != 0) + \
            f[1].select(['a', 'd'], absolute=True, filter_value=lambda x: x != 0) + \
            f[2].select(['a', 'd'], absolute=True, filter_value=lambda x: x != 0)

        f = f[0].merge(f[1:])
        new_len = len(f)
        self.assertEqual(original_len, new_len)

        y = f.select(['a'])
        y1 = f.select(['a', 'c'], absolute=True)
        y2 = f.select(['a', 'd'], absolute=True, filter_value=lambda x: x != 0)
        self.assertEqual(sorted(x), sorted(y))
        self.assertEqual(sorted(x1), sorted(y1))
        self.assertEqual(sorted(x2), sorted(y2))

# ===========================================================================
# Main
# ===========================================================================
if __name__ == '__main__':
    print(' Use nosetests to run these tests ')
