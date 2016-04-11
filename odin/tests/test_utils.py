# ======================================================================
# Author: TrungNT
# ======================================================================
from __future__ import print_function, division
import os
from ..utils import function, frame, get_from_path
from ..tensor import get_magic_seed
from ..decorators import cache, typecheck
import unittest
import cPickle

import math
import random

# ===========================================================================
# Main Test
# ===========================================================================
CODE = '''
import math

def test(a=1, b=2):
    a = math.sqrt(a)
    return 'Shit %s over %s here!' % (str(a), str(b))
'''
idx = 0


def test(a=1, b=2):
    a = math.sqrt(a)
    return 'Shit %s over %s here!' % (str(a), str(b))


class UtilsTest(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_function_save_load(self):
        try:
            import h5py
        except:
            print('\n This test require h5py library.')
            return
        # ====== write code to file ====== #
        f = open('/tmp/tmp_code.py', 'w')
        f.write(CODE)
        f.close()

        # ====== save ====== #
        code = get_from_path('test', prefix='tmp_code', path='/tmp')[0]
        f = function(code, 3, 4)
        s1 = f()
        config = cPickle.dumps(f)
        file = h5py.File('/tmp/tmp.h5', 'w')
        file['function'] = config
        file.close()

        # ====== load ====== #
        os.remove('/tmp/tmp_code.py')
        file = h5py.File('/tmp/tmp.h5', 'r')
        config = file['function'].value
        os.remove('/tmp/tmp.h5')

        f = cPickle.loads(config)
        print('\n', f)
        s2 = f()

        self.assertEqual(s1, s2)

        # ====== exec the source of function ====== #
        exec(f.source)
        a = test()
        self.assertEqual(a, 'Shit 1.0 over 2 here!')

    def test_function_compare(self):
        f1 = function(test, 3, 4)
        f2 = function(test, 3, 4)
        f3 = function(test, 1, 4)

        self.assertEqual(f1 == f2, True)
        self.assertEqual(f1 == f3, False)

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

    def test_cache_decorator(self):
        @cache
        def function(a, b, c=3, d=4):
            global idx
            idx += 1
            return idx

        self.assertEqual(function(1, 2), function(1, 2, 3, 4))
        self.assertEqual(function(1, 2), function(1, 2, d=4, c=3))
        self.assertEqual(function(1, 2), function(1, 2))
        self.assertEqual(function(1, 2), function(1, 2))

        self.assertNotEqual(function(1, 2), function(1, 2, c=4))

        class ClassName(object):
            idx = 10
            arg = 1

            @cache
            def method(self, a, b=2):
                ClassName.idx += 1
                return ClassName.idx

            @cache('arg')
            def methodarg(self, a, b):
                ClassName.idx += 1
                return ClassName.idx

            @classmethod
            @cache
            def clmethod(cl, a, b):
                ClassName.idx += 1
                return ClassName.idx

        c = ClassName()
        self.assertEqual(c.method(1, 2), c.method(1, 2))
        self.assertEqual(c.method(1, 2), c.method(1))
        self.assertNotEqual(c.method(1, 2), c.method(1, 3))

        self.assertEqual(c.methodarg(1, 2), c.methodarg(1, 2))
        x1 = c.methodarg(1, 2)
        c.arg = 2
        self.assertNotEqual(x1, c.methodarg(1, 2))

        self.assertEqual(c.clmethod(1, 2), c.clmethod(1, 2))
        self.assertNotEqual(c.clmethod(1, 2), c.clmethod(1, 3))

    def test_typecheck(self):
        @typecheck(inputs=(int, float, str, int), outputs=(int, str), debug=2)
        def function(a, b, c=8, d=12):
            return 8, '12'
        self.assertEqual(function(1, 2., '8'), (8, '12'))

# ===========================================================================
# Main
# ===========================================================================
if __name__ == '__main__':
    print(' Use nosetests to run these tests ')
