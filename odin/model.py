from __future__ import print_function, division
from six.moves import range, zip

from .base import OdinObject
from .utils import function, frame
from . import tensor

import os
import numpy as np
import h5py

__all__ = [
    'model'
]

# ===========================================================================
# Helper
# ===========================================================================
def _hdf5_save_overwrite(hdf5, key, value):
    if key in hdf5:
        del hdf5[key]
    hdf5[key] = value

def _load_weights(hdf5, api):
    weights = []
    if 'nb_weights' in hdf5:
        if api == 'lasagne':
            for i in range(hdf5['nb_weights'].value):
                weights.append(hdf5['weight_%d' % i].value)
        elif api == 'keras':
            for i in range(hdf5['nb_weights'].value):
                w = []
                for j in range(hdf5['nb_layers_%d' % i].value):
                    w.append(hdf5['weight_%d_%d' % (i, j)].value)
                weights.append(w)
        else:
            raise ValueError('Currently not support API: %s' % api)
    return weights

def _save_weights(hdf5, api, weights):
    if api == 'lasagne':
        _hdf5_save_overwrite(hdf5, 'nb_weights', len(weights))
        for i, w in enumerate(weights):
            _hdf5_save_overwrite(hdf5, 'weight_%d' % i, w)
    elif api == 'keras':
        _hdf5_save_overwrite(hdf5, 'nb_weights', len(weights))
        for i, W in enumerate(weights):
            _hdf5_save_overwrite(hdf5, 'nb_layers_%d' % i, len(W))
            for j, w in enumerate(W):
                _hdf5_save_overwrite(hdf5, 'weight_%d_%d' % (i, j), w)
    else:
        raise ValueError('Currently not support API: %s' % api)

def _set_weights(model, weights, api):
    if api == 'lasagne':
        import lasagne
        lasagne.layers.set_all_param_values(model, weights)
    elif api == 'keras':
        for l, w in zip(model.layers, weights):
            l.set_weights(w)
    else:
        raise ValueError('Currently not support API: %s' % api)

def _get_weights(model):
    if 'lasagne' in str(model.__class__):
        import lasagne
        return lasagne.layers.get_all_param_values(model)
    elif 'keras' in str(model.__class__):
        weights = []
        for l in model.layers:
            weights.append(l.get_weights())
        return weights
    else:
        raise ValueError('Currently not support API')

def _convert_weights(model, weights, original_api, target_api):
    W = []
    if original_api == 'keras' and target_api == 'lasagne':
        for w in weights:
            W += w
    elif original_api == 'lasagne' and target_api == 'keras':
        count = 0
        for l in model.layers:
            n = len(l.trainable_weights + l.non_trainable_weights)
            W.append([weights[i] for i in range(count, count + n)])
            count += n
    else:
        raise ValueError('Currently not support API')
    return W

# ===========================================================================
# Model
# ===========================================================================
class model(OdinObject):

    """Supported API:
    - Lasagne
    - keras
    """

    def __init__(self, savepath=None):
        super(model, self).__init__()
        self._history = []
        self._working_history = None
        self._history_updated = False

        self._weights = []
        self._save_path = savepath

        # contain real model object
        self._need_update_model = False
        self._model = None
        self._model_func = None
        self._model_var = {} # variable for input, output of model
        self._api = ''

        # ====== functions ====== #
        self._pred_func = None

        self._cost_func_old = None
        self._cost_func = None

        self._updates_func_old = None
        self._objective_func_old = None
        self._updates_func = None

    # ==================== Weights ==================== #
    def set_weights(self, weights, api=None):
        '''
        Parameters
        ----------
        weights : list(np.ndarray)
            list of all numpy array contain parameters
        api : str (lasagne, keras)
            the api of the weights format

        Notes
        -----
        - if set_model is called, this methods will set_weights for your AI also
        '''
        # ====== Make sure model created ====== #
        if self._model_func:
            self.get_model()
            # set model weights
            if self._model:
                # convert weights between api
                if api and api != self._api:
                    self._weights = _convert_weights(
                        self._model, weights, api, self._api)
                else:
                    self._weights = weights
                _set_weights(self._model, self._weights, self._api)
            elif api is not None:
                raise ValueError('Cannot convert weights when model haven\'t created!')
        # ====== fetch new weights directly ====== #
        else:
            self._weights = []
            for w in weights:
                self._weights.append(w)

    def get_weights(self):
        ''' if set_model is called, and your AI is created, always return the
        newest weights from AI
        '''
        # create model to have weights
        if self._model_func:
            self.get_model()
        # always update the newest weights of model
        self._weights = _get_weights(self._model)
        return self._weights

    def get_nweights(self):
        ''' Return number of weights '''
        n = 0
        weights = self.get_weights()
        for W in weights:
            if type(W) not in (tuple, list):
                W = [W]
            for w in W:
                n += np.prod(w.shape)
        return n

    def get_nlayers(self):
        ''' Return number of layers (only layers with trainable params) '''
        if self._model is not None:
            if self._api == 'lasagne':
                import lasagne
                return len([i for i in lasagne.layers.get_all_layers(self._model)
                            if len(i.get_params(trainable=True)) > 0])
            elif self._api == 'keras':
                count = 0
                for l in self._model.layers:
                    if len(l.trainable_weights) > 0:
                        count += 1
                return count
            else:
                raise ValueError('Currently not support API: %s' % self._api)
        return 0

    def get_vars(self):
        return self._model_var

    # ==================== Model ==================== #
    def set_model(self, model, api, *args, **kwargs):
        ''' Save a function that create your model.

        Parameters
        ----------
        model : __call__ object
            main function that create your model
        api : str
            support: lasagne | keras | blocks
        args : *
            any arguments for your model creatation function
        kwargs : **
            any arguments for your model creatation function
        '''
        if not hasattr(model, '__call__'):
            raise NotImplementedError('Model must be a function return computational graph')
        api = api.lower()
        func = function(model, *args, **kwargs)
        if self._model_func and self._model_func != func:
            self._need_update_model = True
        # ====== Lasagne ====== #
        if 'lasagne' in api:
            self._api = 'lasagne'
            self._model_func = func
        # ====== Keras ====== #
        elif 'keras' in api:
            self._api = 'keras'
            self._model_func = func
        elif 'blocks' in api:
            self._api = 'blocks'
            raise ValueError('Currently not support API: %s' % self._api)

    def get_model(self, checkpoint=True):
        '''
        Parameters
        ----------
        checkpoint: bool
            if True, not only create new model but also create a saved
            checkpoint if NO weights have setted

        Notes
        -----
        The logic of this method is:
         - if already set_weights, old weights will be loaded into new model
         - if NO setted weights and new model creates, fetch all models weights
         and save it to file to create checkpoint of save_path available
        '''
        if self._model_func is None:
            raise ValueError("You must set_model first")
        if self._need_update_model or self._model is None:
            self.log('*** Creating network ... ***', 20)
            self._model = self._model_func()
            self._need_update_model = False
            # reset all function
            self._pred_func = None
            self._cost_func = None
            self._updates_func = None
            # ====== load old weight ====== #
            if len(self._weights) > 0:
                try:
                    _set_weights(self._model, self._weights, self._api)
                    self.log('*** Successfully load old weights ***', 20)
                except Exception, e:
                    self.log('*** Cannot load old weights ***', 50)
                    self.log(str(e), 40)
                    import traceback; traceback.print_exc();
            # ====== fetch new weights into model, create checkpoints ====== #
            else:
                weights = self.get_weights()
                if self._save_path is not None and checkpoint:
                    f = h5py.File(self._save_path, mode='a')
                    try:
                        _save_weights(f, self._api, weights)
                        self.log('*** Created checkpoint ... ***', 20)
                    except Exception, e:
                        self.log('Error Creating checkpoint: %s' % str(e), 40)
                        raise e
                    f.close()
            # ====== store model's variables ====== #
            if self._api == 'lasagne':
                import lasagne
                X = [l.input_var for l in lasagne.layers.find_layers(
                    self._model, types=lasagne.layers.InputLayer)]
                y_pred = lasagne.layers.get_output(self._model, deterministic=True)
                y_train = lasagne.layers.get_output(self._model, deterministic=False)
            elif self._api == 'keras':
                X = self._model.get_input(train=False)
                if type(X) not in (list, tuple):
                    X = [X]
                y_pred = self._model.get_output(train=False)
                y_train = self._model.get_output(train=True)
            y_true = tensor.placeholder(ndim=tensor.ndim(y_pred))
            self._model_var['X'] = X
            self._model_var['y_true'] = y_true
            self._model_var['y_pred'] = y_pred
            self._model_var['y_train'] = y_train
        return self._model

    # ==================== Network function ==================== #
    def create_pred(self):
        ''' Create prediction funciton '''
        self.get_model()
        # ====== Create prediction function ====== #
        if not self._pred_func:
            var = self._model_var
            if self._api == 'lasagne':
                updates = []
            elif self._api == 'keras':
                updates = self._model.state_updates
            else:
                raise ValueError('Currently not support API: %s' % self._api)
            # create prediction function
            self._pred_func = tensor.function(
                inputs=var['X'],
                outputs=var['y_pred'],
                updates=updates)
            self.log(
                '*** Successfully create [%s] prediction function ***' % self._api, 20)
        return self._pred_func

    def create_cost(self, cost_func):
        ''' Create cost funciton
        Parameters
        ----------
        cost_func: callable object
            cost funciton is a function(y_pred, y_true,...)
        args, kwargs: arguments, keyword arguments
            for given cost function
        '''
        if not cost_func or not hasattr(cost_func, '__call__'):
            raise ValueError(
                'Cost funciton must be callable and has at least 2 arguments')
        self.get_model()
        # ====== Create prediction function ====== #
        if not self._cost_func or \
           self._cost_func_old != cost_func:
            self._cost_func_old = cost_func
            var = self._model_var
            if self._api == 'lasagne':
                updates = []
            elif self._api == 'keras':
                updates = self._model.state_updates
            else:
                raise ValueError('Currently not support API: %s' % self._api)
            # create cost function
            cost = cost_func(var['y_pred'], var['y_true'])
            self._cost_func = tensor.function(
                inputs=var['X'] + [var['y_true']],
                outputs=cost,
                updates=updates)
            self.log(
                '*** Successfully create [%s] cost function ***' % self._api, 20)
        return self._cost_func

    def create_updates(self, objective_func, updates_func):
        ''' Create updates funciton
        Parameters
        ----------
        objective_func: callable object
            objective funciton is a function(loss, params)
        updates_func: callable object
            update funciton is a function(loss_or_grads, params,...)
        args, kwargs: arguments, keyword arguments
            for given cost function
        '''
        if not updates_func or not hasattr(updates_func, '__call__'):
            raise ValueError(
                'Updates funciton must be callable and has at least 2 arguments')
        self.get_model()
        # ====== Create updates function ====== #
        if not self._updates_func or \
           self._updates_func_old != updates_func or \
           self._objective_func_old != objective_func:
            self._updates_func_old = updates_func
            self._objective_func_old = objective_func
            var = self._model_var
            obj = tensor.mean(objective_func(var['y_pred'], var['y_true']))
            if self._api == 'lasagne':
                import lasagne
                params = lasagne.layers.get_all_params(self._model, trainable=True)
            elif self._api == 'keras':
                params = self._model.trainable_weights
            else:
                raise ValueError('Currently not support API: %s' % self._api)
            # create updates function
            updates = updates_func(obj, params)
            self._updates_func = tensor.function(
                inputs=var['X'] + [var['y_true']],
                outputs=obj,
                updates=updates)
            self.log(
                '*** Successfully create [%s] updates function ***' % self._api, 20)
        return self._updates_func

    # ==================== network actions ==================== #
    def pred(self, *X):
        '''
        Order of input will be keep in the same order when you create network
        '''
        self.create_pred()
        # ====== make prediction ====== #
        prediction = self._pred_func(X)
        return prediction

    def cost(self, *X):
        if not self._cost_func:
            raise ValueError('Haven\'t created cost function')
        # ====== caluclate cost ====== #
        return self._cost_func(X)

    def updates(self, *X):
        if not self._updates_func:
            raise ValueError('Haven\'t created cost function')
        return self._updates_func(X)

    def rollback(self):
        ''' Roll-back weights and history of model from last checkpoints
        (last saved path).
        '''
        if self._save_path is not None and os.path.exists(self._save_path):
            import cPickle
            f = h5py.File(self._save_path, 'r')

            # rollback weights
            weights = _load_weights(f, self._api)
            if len(weights) > 0:
                self._weights = weights
                if self._model is not None:
                    _set_weights(self._model, self._weights, self._api)
                    self.log(' *** Weights rolled-back! ***', 20)

            # rollback history
            if 'history' in f:
                self._history = cPickle.loads(f['history'].value)
                self.log(' *** History rolled-back! ***', 20)
                self._history_updated = True
        else:
            self.log('No checkpoint found! Ignored rollback!', 30)
        return self

    # ==================== History manager ==================== #
    def _check_current_working_history(self):
        if len(self._history) == 0:
            self._history.append(frame())
        if self._history_updated or self._working_history is None:
            self._working_history = self[:]

    def __getitem__(self, key):
        if len(self._history) == 0:
            self._history.append(frame())

        if isinstance(key, slice) or isinstance(key, int):
            h = self._history[key]
            if hasattr(h, '__len__'):
                if len(h) > 1: return h[0].merge(*h[1:])
                else: return h[0]
            return h
        elif isinstance(key, str):
            for i in self._history:
                if key == i.name:
                    return i
        elif type(key) in (tuple, list):
            h = [i for k in key for i in self._history if i.name == k]
            if len(h) > 1: return h[0].merge(*h[1:])
            else: return h[0]
        raise ValueError('Model index must be [slice],\
            [int] or [str], or list of string')

    def get_working_history(self):
        self._check_current_working_history()
        return self._working_history

    def new_frame(self, name=None, description=None):
        self._history.append(frame(name, description))
        self._history_updated = True

    def drop_frame(self):
        if len(self._history) < 2:
            self._history = []
        else:
            self._history = self._history[:-1]
        self._working_history = None
        self._history_updated = True

    def record(self, values, *tags):
        ''' Always write to the newest frame '''
        if len(self._history) == 0:
            self._history.append(frame())
        self._history[-1].record(values, *tags)
        self._history_updated = True

    # def update(self, tags, new, after=None, before=None, n=None, absolute=False):
    #     if len(self._history) == 0:
    #         self._history.append(frame())
    #     self._history[-1].update(self, tags, new,
    #         after, before, n, absolute)
    #     self._history_updated = True

    def select(self, tags, default=None, after=None, before=None, n=None,
        filter_value=None, absolute=False, newest=False, return_time=False):
        ''' Query in history, default working history is the newest frame.

        Parameters
        ----------
        tags : list, str, filter function or any comparable object
            get all values contain given tags
        default : object
            default return value of found nothing
        after, before : time constraint (in millisecond)
            after < t < before
        n : int
            number of record return
        filter_value : function(value)
            function to filter each value found
        absolute : boolean
            whether required the same set of tags or just contain
        newest : boolean
            returning order (newest first, default is False)
        time : boolean
            whether return time tags

        Returns
        -------
        return : list
            always return list, in case of no value, return empty list
        '''
        self._check_current_working_history()
        return self._working_history.select(tags, default, after, before, n,
            filter_value, absolute, newest, return_time)

    def print_history(self):
        self._check_current_working_history()
        self._working_history.print_history()

    def print_frames(self):
        if len(self._history) == 0:
            self._history.append(frame())

        for i in self._history:
            self.log(str(i), 0)

    def print_code(self):
        if self._model_func is not None:
            self.log(str(self._model_func), 0)

    def __str__(self):
        s = ''
        s += 'Model: %s' % self._save_path + '\n'
        # weight
        s += '======== Weights ========\n'
        nb_params = 0
        # it is critical to get_weights here to fetch weight from created model
        for w in self.get_weights():
            s += ' - shape:%s' % str(w.shape) + '\n'
            nb_params += np.prod(w.shape)
        s += ' => Total: %d (parameters)' % nb_params + '\n'
        s += ' => Size: %.2f MB' % (nb_params * 4. / 1024. / 1024.) + '\n'
        # history
        self._check_current_working_history()
        s += str(self._working_history)

        # model function
        s += '======== Code ========\n'
        s += ' - api:%s' % self._api + '\n'
        s += ' - name:%s' % self._model_name + '\n'
        s += ' - args:%s' % str(self._model_args) + '\n'
        s += ' - sandbox:%s' % str(self._sandbox) + '\n'
        return s[:-1]

    # ==================== Load & Save ==================== #
    def save(self, path=None):
        if path is None and self._save_path is None:
            raise ValueError("Save path haven't specified!")
        path = path if path is not None else self._save_path
        self._save_path = path

        import cPickle

        f = h5py.File(path, 'w')
        f['history'] = cPickle.dumps(self._history)
        # ====== Save model function ====== #
        if self._model_func is not None:
            f['model_func'] = cPickle.dumps(self._model_func.get_config())
            f['api'] = self._api
        # ====== save weights ====== #
        # check weights, always fetch newest weights from model
        weights = self.get_weights()
        _save_weights(f, self._api, weights)

        f.close()

    @staticmethod
    def load(path):
        ''' Load won't create any modification to original AI file '''
        if not os.path.exists(path):
            m = model()
            m._save_path = path
            return m
        import cPickle

        m = model()
        m._save_path = path

        f = h5py.File(path, 'r')
        # ====== Load history ====== #
        if 'history' in f:
            m._history = cPickle.loads(f['history'].value)
        else:
            m._history = []
        # ====== Load model ====== #
        if 'api' in f:
            m._api = f['api'].value
        else: m._api = None
        # load model_func code
        if 'model_func' in f:
            m._model_func = function.parse_config(
                cPickle.loads(f['model_func'].value))
        else: m._model_func = None
        # ====== load weights ====== #
        m._weights = _load_weights(f, m._api)

        f.close()
        return m
