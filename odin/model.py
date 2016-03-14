from __future__ import print_function, division, absolute_import
from six.moves import range, zip

from .base import OdinObject
from .utils import function, frame
from .utils import api as API
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


# ===========================================================================
# Eearly stopping
# ===========================================================================
def _check_gs(validation):
    ''' Generalization sensitive:
    validation is list of cost values (assumpt: lower is better)
    '''
    if len(validation) == 0:
        return 0, 0
    shouldStop = 0
    shouldSave = 0

    if validation[-1] > min(validation):
        shouldStop = 1
        shouldSave = -1
    else:
        shouldStop = -1
        shouldSave = 1

    return shouldSave, shouldStop


def _check_gl(validation, threshold=5):
    ''' Generalization loss:
    validation is list of cost values (assumpt: lower is better)
    Note
    ----
    This strategy prefer to keep the model remain when the cost unchange
    '''
    gl_exit_threshold = threshold
    epsilon = 1e-5

    if len(validation) == 0:
        return 0, 0
    shouldStop = 0
    shouldSave = 0

    gl_t = 100 * (validation[-1] / (min(validation) + epsilon) - 1)
    if gl_t <= 0 and np.argmin(validation) == (len(validation) - 1):
        shouldSave = 1
        shouldStop = -1
    elif gl_t > gl_exit_threshold:
        shouldStop = 1
        shouldSave = -1

    # check stay the same performance for so long
    if len(validation) > threshold:
        remain_detected = 0
        j = validation[-int(threshold)]
        for i in validation[-int(threshold):]:
            if abs(i - j) < epsilon:
                remain_detected += 1
        if remain_detected >= threshold:
            shouldStop = 1
    return shouldSave, shouldStop


def _check_hope_and_hop(validation):
    ''' Hope and hop:
    validation is list of cost values (assumpt: lower is better)
    '''
    patience = 5
    patience_increase = 0.5
    improvement_threshold = 0.998

    if len(validation) == 0:
        return 0, 0
    shouldStop = 0
    shouldSave = 0

    # one more iteration
    i = len(validation)
    if len(validation) == 1: # cold start
        shouldSave = 1
        shouldStop = -1
    else: # warm up
        last_best_validation = min(validation[:-1])
        # significant improvement
        if min(validation) < last_best_validation * improvement_threshold:
            patience += i * patience_increase
            shouldSave = 1
            shouldStop = -1
        # punish
        else:
            # the more increase the faster we running out of patience
            rate = validation[-1] / last_best_validation
            patience -= i * patience_increase * rate
            # if still little bit better, just save it
            if min(validation) < last_best_validation:
                shouldSave = 1
            else:
                shouldSave = -1

    if patience <= 0:
        shouldStop = 1
        shouldSave = -1
    return shouldSave, shouldStop


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
        self._api = ''

        # ====== functions ====== #
        self._pred_func = None

        self._cost_func_old = None
        self._cost_func = None

        self._updates_func_old = None
        self._objective_func_old = None
        self._updates_func = None

    # ==================== DNN operators ==================== #
    def earlystop(self, tags, generalization_loss = False,
        generalization_sensitive=False, hope_hop=False,
        threshold=None):
        ''' Early stop.

        Parameters
        ----------
        generalization_loss : type
            note
        generalization_sensitive : type
            note
        hope_hop : type
            note

        Returns
        -------
        return : boolean, boolean
            shouldSave, shouldStop

        '''
        costs = self.select(tags,
            filter_value=lambda x: isinstance(x, float) or isinstance(x, int))

        if len(costs) == 0:
            shouldSave, shouldStop = False, False
        else:
            values = costs
            shouldSave = 0
            shouldStop = 0
            if generalization_loss:
                if threshold is not None:
                    save, stop = _check_gl(values, threshold)
                else:
                    save, stop = _check_gl(values)
                shouldSave += save
                shouldStop += stop
            if generalization_sensitive:
                save, stop = _check_gs(values)
                shouldSave += save
                shouldStop += stop
            if hope_hop:
                save, stop = _check_hope_and_hop(values)
                shouldSave += save
                shouldStop += stop
            shouldSave, shouldStop = shouldSave > 0, shouldStop > 0
        self.log(
            'Earlystop: shouldSave=%s, shouldStop=%s' % (shouldSave, shouldStop),
            50)
        return shouldSave, shouldStop

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
                    self._weights = API.convert_weights(
                        self._model, weights, api, self._api)
                else:
                    self._weights = weights
                API.set_weights(self._model, self._api, self._weights)
            elif api is not None:
                raise ValueError('Cannot convert weights when model haven\'t created!')
        # ====== fetch new weights directly ====== #
        else:
            self._weights = []
            for w in weights:
                self._weights.append(w)

    def get_params(self):
        ''' if set_model is called, and your AI is created, always return the
        newest weights from AI
        '''
        # create model to have weights
        self.get_model()
        # always update the newest weights of model
        self._weights = API.get_params_for_saving(self._model, self._api)
        return self._weights

    def get_nparams(self):
        ''' Return number of weights '''
        n = 0
        weights = self.get_params()
        for W in weights:
            if type(W) not in (tuple, list):
                W = [W]
            for w in W:
                n += np.prod(w.shape)
        return n

    def get_nlayers(self):
        ''' Return number of layers (only layers with trainable params) '''
        self.get_model()
        return API.get_nlayers(self._model, self._api)

    # ==================== Model ==================== #
    def set_model(self, model, *args, **kwargs):
        ''' Save a function that create your model.

        Parameters
        ----------
        model : __call__ object
            main function that create your model
        api : str
            support: lasagne | keras | odin
        args : *
            any arguments for your model creatation function
        kwargs : **
            any arguments for your model creatation function
        '''
        if not hasattr(model, '__call__'):
            raise NotImplementedError('Model must be a function return '
                                      'computational graph.')
        func = function(model, *args, **kwargs)
        if self._model_func and self._model_func != func:
            self._need_update_model = True
        self._model_func = func

    def get_api(self):
        self.get_model()
        return self._api

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
            self.log('*** Creating network ... ***', 10)
            self._model = self._model_func()
            if self._model is None:
                raise ValueError(
                    'Model\'s creator function doesn\'t return appropriate model')
            self._api = API.get_object_api(self._model)
            self._need_update_model = False
            # reset all function
            self._pred_func = None
            self._cost_func = None
            self._updates_func = None
            # ====== load old weight ====== #
            if len(self._weights) > 0:
                try:
                    API.set_weights(self._model, self._api, self._weights)
                    self.log('*** Successfully load old weights ***', 10)
                except Exception, e:
                    self.log('*** Cannot load old weights ***', 40)
                    self.log(str(e), 40)
                    import traceback; traceback.print_exc()
            # ====== fetch new weights into model, create checkpoints ====== #
            else:
                weights = self.get_params()
                if self._save_path is not None and checkpoint:
                    f = h5py.File(self._save_path, mode='a')
                    try:
                        API.save_weights(f, self._api, weights)
                        self.log('*** Created checkpoint ... ***', 10)
                    except Exception, e:
                        self.log('Error Creating checkpoint: %s' % str(e), 40)
                        raise e
                    f.close()
        return self._model

    # ==================== Network function ==================== #
    def create_pred(self):
        ''' Create prediction funciton '''
        self.get_model()
        # ====== Create prediction function ====== #
        if not self._pred_func:
            input_var = API.get_input_variables(self._model, self._api)
            output_var = API.get_outputs(self._model, self._api, False)
            # create prediction function
            self._pred_func = tensor.function(
                inputs=input_var,
                outputs=output_var)
            self.log(
                '*** Successfully create [%s] prediction function ***' % self._api, 10)
        return self._pred_func

    # ==================== network actions ==================== #
    def pred(self, *X):
        '''
        Order of input will be keep in the same order when you create network
        Returns
        -------
        list of results
        '''
        self.create_pred()
        # ====== make prediction ====== #
        prediction = self._pred_func(*X)
        return prediction

    def rollback(self):
        ''' Roll-back weights and history of model from last checkpoints
        (last saved path).
        '''
        if self._save_path is not None and os.path.exists(self._save_path):
            import cPickle
            f = h5py.File(self._save_path, 'r')

            # rollback weights
            weights = API.load_weights(f, self._api)
            if len(weights) > 0:
                self._weights = weights
                if self._model is not None:
                    API.set_weights(self._model, self._api, self._weights)
                    self.log(' *** Weights rolled-back! ***', 10)

            # rollback history
            if 'history' in f:
                self._history = cPickle.loads(f['history'].value)
                self.log(' *** History rolled-back! ***', 10)
                self._history_updated = True
        else:
            self.log('No checkpoint found! Ignored rollback!', 10)
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

    def select(self, tags, default=None, after=None, before=None,
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
        return self._working_history.select(tags, default, after, before,
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
        for w in self.get_params():
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
        weights = self.get_params()
        API.save_weights(f, self._api, weights)

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
        m._weights = API.load_weights(f, m._api)

        f.close()
        return m
