to_import = [
    'modelutils',
    'search.bayesian'
    ]

from os.path import dirname, abspath
project_name = dirname(abspath(__file__)).split('\\')[-1]

import sys
import types
import pickle
import importlib
from hyperopt import space_eval
for module in to_import:
    module_name = module.split('.')[-1]
    new_module = importlib.import_module(name = f'.{module}', package = project_name)
    sys.modules[__name__].__dict__.update({module_name: new_module})

################################################################################

import numpy as np
from sklearn.metrics import f1_score
import logging
logger = logging.getLogger('pipeline')

from sklearn import metrics
from sklearn.metrics import f1_score, fbeta_score, recall_score
import joblib
flatten = lambda t: [item for sublist in t for item in sublist]

def calc_score(y_true, y_pred):
    return f1_score(y_true, y_pred, labels= ['B', 'I'], average= 'weighted')

class SklearnModel1:
    @modelutils.catch('SKELARNMODEL_INITERROR')
    def __init__(self, **hparams):
        
        print(f'sklearn_model hparameters: {hparams}')
        self.model = hparams.pop('model_class', None)
        self.model_class = self.model
        self.minimize_metric = hparams.pop('minimize_metric', True)
        
        if self.model is not None:
            self.model = self.model(**hparams)

    @modelutils.catch('SKELARNMODEL_FITERROR')
    def fit(self, x):
        logger.info(f'Fitting sklearn model...')
        
        X_train, y_train = x['train_data']
        X_val, y_val = x['val_data']
        
        if isinstance(X_train, types.FunctionType):
            X_train = X_train()
        if isinstance(y_train, types.FunctionType):
            y_train = y_train()
        if isinstance(X_val, types.FunctionType):
            X_val = X_val()
        if isinstance(y_val, types.FunctionType):
            y_val = y_val()

        """try:
            y_train = np.array(y_train.todense()).ravel()
            y_val = np.array(y_val.todense()).ravel()
        except:
            y_train = np.array(y_train).ravel()
            y_val = np.array(y_val).ravel()"""
        
        try:
            self.model.fit(X_train, y_train)
            
        except ZeroDivisionError: # *** SVM BUG ***
            logger.info(f"SVM error, returning 0.")
            return {'metric': 0.0}
        
        y_pred = self.model.predict_proba(X_val)
        th = y_train.mean() * 0.75
        y_pred = y_pred[:, 1] > th
        print(f'sklearn model threshold: {th}')

        #metric = calc_score(flatten(y_val), flatten(y_pred))
        #metric = fbeta_score(y_val, y_pred, beta = 2.0, average='binary', pos_label = 1)
        metric = recall_score(y_val, y_pred, pos_label = 1)
        #metric = metrics.accuracy_score(y_val, y_pred)
        #self.model.score(X_val, y_val)
        
        self.th = th
        logger.info(f"Metric: {metric}")
        return {'metric': metric, 'thresh': th}
    
    @modelutils.catch('SKELARNMODEL_TESTERROR')
    def test(self, x):
        test_data = x['test_data']

        #test_preds = self.model.predict()
        return self.model.score(test_data[0], test_data[1])

    @modelutils.catch('SKELARNMODEL_PREDICTERROR')
    def predict(self, x, debug = False, **kwargs):
        if debug:
            logger.info(f'Model input: {x}')

        for x_example in x:
            x_example.update({'output': self.model.predict_proba(x_example['output'])})
        return x

    @modelutils.catch('SKELARNMODEL_LOADERROR')
    def load(self, model_path = 'pipeline_model.joblib'):
        # Load model
        try:
            self.model = joblib.load(model_path)
        except FileNotFoundError:
            logger.error(f'Model file {model_path} does not exist.')

    @modelutils.catch('SKELARNMODEL_SAVEERROR')
    def save(self, model_path = 'pipeline_model.joblib'):
        try:
            joblib.dump(self.model, model_path)
        except Exception as e:
            logger.error(e)

    @modelutils.catch('SKELARNMODEL_FITBESTERROR')
    def fit_best(self, x, search_space, trials_path = 'trials_sklearn'):

        # Load trials
        trials = pickle.load(open(trials_path, "rb"))

        # Get best search hparams
        search_params = space_eval(search_space, {k: v[0] for k,v in trials.best_trial['misc']['vals'].items()})
        logger.info(f'Best search hparameters: {search_params}')

        # Join fixed parameters and search parameters
        params = model_defaults.model_defaults[self.model_tag]['search_fixed']
        params.update(search_params)

        self.__init__(
            self.model_class, **params
            )
        return self.fit(x)

    @modelutils.catch('SKELARNMODEL_SEARCHERROR')
    def search(self, x, search_space, search_fixed, num_iter = 25, trials_path = 'trials_sklearn'):
        
        # Search
        res_search = bayesian.bayesian_search(
            self.__init__,
            self.fit,
            x,
            search_space,
            search_fixed,
            num_iter = num_iter,
            mode = 'bayesian',
            minimize = self.minimize_metric,
            trials_path = trials_path,
            ) # model_tag = self.model_tag
        
        return res_search