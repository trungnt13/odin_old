import unittest

def run(name=None):
    if name is None:
        tests = ['test_backend', 'test_dataset', 'test_utils', 'test_models',
                 'test_training', 'test_obj_opt', 'test_nnet']
    else:
        tests = []
        if not isinstance(name, (list, tuple)):
            name = [name]

        for i in name:
            if 'backend' in i or 'tensor' in i:
                tests.append('test_backend')
            elif 'funcs' in i or 'nnet' in i:
                tests.append('test_nnet')
            elif 'data' in i:
                tests.append('test_dataset')
            elif 'model' in i:
                tests.append('test_models')
            elif 'train' in i:
                tests.append('test_training')
            elif 'utils' in i:
                tests.append('test_utils')
            elif 'obj' in i or 'opt' in i:
                tests.append('test_obj_opt')

    # tests = ['test_backend']
    for t in tests:
        exec('from . import %s' % t)
        tests = unittest.TestLoader().loadTestsFromModule(globals()[t])
        unittest.TextTestRunner(verbosity=2).run(tests)
