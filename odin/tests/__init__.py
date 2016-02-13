import unittest

def run():
    tests = ['test_dataset', 'test_utils']
    for t in tests:
        exec('from . import %s' % t)
        tests = unittest.TestLoader().loadTestsFromModule(globals()[t])
        unittest.TextTestRunner(verbosity=2).run(tests)
