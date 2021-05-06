to_import = [
    'utils',
    'model_defaults',
    'base.pipeline.pipeline',
    'base.model.lgbm',
    'base.model.sklearn_model',
]

from os.path import dirname, abspath
package_path = dirname(dirname(dirname(abspath(__file__)))).split('\\')[-1]

import sys
import importlib
for module in to_import:
    module_name = module.split('.')[-1]
    new_module = importlib.import_module(name = f'.{module}', package = package_path)
    sys.modules[__name__].__dict__.update({module_name: new_module})

################################################################################

import os
import keras.backend as K
import pickle
import gc

from sklearn.model_selection import KFold
from hyperopt import space_eval

import logging
logger = logging.getLogger('pipeline')

# Experiment logging
try:
    from mlflow import log_metric, log_param, log_artifact
    mlflow_available = True
except:
    logger.warning('MLFLOW NOT AVAILABLE.')
    mlflow_available = False

sklearn_models = ['svm', 'lr', 'nb']
all_models = sklearn_models + ['mlp', 'lgbm']

class MultiModel:
    @utils.catch('MULTIMODEL_INITERROR')
    def __init__(self, **kwargs):
        def_args = dict(
            model_type = None,
            model_method = 'predict',
            mlflow_logging = mlflow_available,
            input_size = None,
            num_search_iterations = 25,
            data_dir = '',
            package_path = '',
            minimize_metric = False
        )
        # Extract related arguments
        for k, def_val in def_args.items():
            self.__dict__.update({k: kwargs.get(k, def_val)})

        # Set paths for model files
        self.model_paths = dict()
        if self.model_type in sklearn_models:
            self.model_paths['model_path'] = os.path.join(self.package_path, 'pipeline_model.joblib')
            self.trials_path = os.path.join(self.package_path, 'trials_sklearn')

        elif self.model_type == 'mlp':
            self.model_paths['hparams_path'] = os.path.join(self.package_path, 'mlp_hparams')
            self.model_paths['weights_path'] = os.path.join(self.package_path, 'mlp_weights.h5')
            self.trials_path = os.path.join(self.package_path, 'trials_mlp')
        else:
            raise NotImplementedError()
        
        self.model = model_defaults.model_defaults[self.model_type]['model'](
            model_tag = self.model_type,
            input_size = self.input_size,
            **model_defaults.model_defaults[self.model_type]['hparams']
            )

        # Initialize model
        if self.model_method in ['predict', 'test']:
            logger.info('Loading model...')
            self.model.load(**self.model_paths)
    
    @utils.catch('MULTIMODEL_CALLERROR')
    def __call__(self, x, debug = False):
        if debug:
            logger.info(f'Model input: {x}')
        return getattr(self, self.model_method)(x)
    
    @utils.catch('MULTIMODEL_FITERROR')
    def fit(self, x, **kwargs):
        assert(x is not None)

        # Train model
        res = self.model.fit(x, **kwargs)

        # Save model
        self.model.save(**self.model_paths)

        # Log result
        self.log_model(res)

        return res
    
    @utils.catch('MULTIMODEL_PREDICTERROR')
    def predict(self, x):
        for x_example in x:
            x_example.update({'output': self.model.predict(x_example['output'])})
        return x


    @utils.catch('MULTIMODEL_LOGMODELERROR')
    def log_model(self, res):
        if self.mlflow_logging:
            for k, v in self.model_paths.items():
                log_artifact(v)

            log_param("model_type", self.model_type)
            log_param("model_method", self.model_method)
            log_metric("metric", res['metric'])

    @utils.catch('MULTIMODEL_SEARCHERROR')
    def search(self, x, **kwargs):
        self.model.search(
            x,
            num_iter = self.num_search_iterations,
            trials_path = self.trials_path,
            **kwargs
            )
        
        res = self.fit_best(x)
        self.model.save(**self.model_paths)
        self.log_model(res)
        return res
    
    @utils.catch('MULTIMODEL_FITBESTERROR')
    def fit_best(self, x):
        return self.model.fit_best(
            x,
            trials_path = self.trials_path
            )