to_import = [
    'utils',
    'model_defaults',
    'base.model.search.bayesian',
    'base.model.kerasmodel']

from os.path import dirname, abspath
project_name = dirname(dirname(dirname(abspath(__file__)))).split('\\')[-1]

import sys
import importlib
from hyperopt import space_eval
for module in to_import:
    module_name = module.split('.')[-1]
    new_module = importlib.import_module(name = f'.{module}', package = project_name)
    sys.modules[__name__].__dict__.update({module_name: new_module})

################################################################################

import keras.backend as K
import keras.applications

from keras.utils import to_categorical
from keras.models import Sequential
from keras import layers
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import tensorflow as tf

import numpy as np
import pandas as pd
import math
from IPython.display import display

import pickle
import logging
logger = logging.getLogger('pipeline')

class MLP(kerasmodel.BaseKerasModel):
    @utils.catch('MLP_INITERROR')
    def __init__(self, **kwargs):
        """Create a Keras MLP model. No need for input shape.
        
        Parameters
        ----------
            node_counts - list of int
                Neuron counts in each layer, including the output.
                
            learning_rate - float
            metric - str
            loss - str
                
        """
        def_args = dict(
            n_layers = 1,
            size = 10,
            shrink_rate = 0.7,
            dropout = 0.01,
            learning_rate = 1e-3,
            output_size = 1,
            input_size = 100,
            beta_1 = 0.95,
            beta_2 = 0.999,
            epochs = 50,
            batch_size = 25,
            verbose = 0,
            metric = 'accuracy',
            loss = 'binary_crossentropy',
            ohe = True,
            activation = 'relu',
            minimize_metric = True,
            use_callback_checkpoint = True,
            last_activation = 'sigmoid',
            metric_func = None
        )
        self.def_args = def_args

        # Extract related arguments
        for k, def_val in def_args.items():
            self.__dict__.update({k: kwargs.get(k, def_val)})
            
        if self.input_size is None:
            raise AttributeError('input_size must be specified for mlp. It can be passed to multi_model.')
        
        hparam_dict = {k: self.__dict__.get(k) for k in def_args.keys()}
        print(f'MLP Hparameters: {hparam_dict}')

        super().__init__(**self.__dict__)
        self.create_mlp()

    @utils.catch('MLP_CREATEMLPERROR')
    def create_mlp(self):
        logger.info('Creating MLP...')
        logger.info(f'Input size: {self.input_size}')

        # Delete all previous models to free memory
        K.clear_session()
        tf.reset_default_graph()
        
        opt = Adam(lr=self.learning_rate, beta_1=self.beta_1, beta_2=self.beta_2,
                   epsilon=None, decay=0.0, amsgrad=False)
        
        node_counts = [int(math.ceil(self.size*(self.shrink_rate)**(i_layer+1)))\
                       for i_layer in range(int(self.n_layers))]
        
        # Create keras model
        first_layer = True
        model = Sequential()
        logger.info(f'Layers: {node_counts}')
        
        for node_count in node_counts:
            assert(node_count > 0)
            
            # Add input size if first layer
            if first_layer:
                model.add(layers.Dense(node_count,
                                       input_dim= self.input_size,
                                       activation= self.activation))
                
                model.add(layers.Dropout(rate = self.dropout))
                first_layer = False
            else:
                model.add(layers.Dense(node_count, activation=self.activation))
                model.add(layers.Dropout(rate = self.dropout))
                    
        # Last layer has softmax activation
        model.add(layers.Dense(self.output_size, activation=self.last_activation))
        
        logging.info(f'Metric: {self.metric}')
        logging.info(f'Loss: {self.loss}')
        logging.info(f'Learning rate: {self.learning_rate}')
        
        model.compile(loss=self.loss,
                      optimizer='adam',
                      metrics=[self.metric])
        
        self.model = model
        logger.info('Model initialized.')

    @utils.catch('MLP_FEATUREIMPORTANCESERROR')
    def feature_importances(
            self, feature_list = None, plot = True, sort = True, 
            limit = 50):
        
        weight_abs_sum = np.sum(np.abs(self.model.layers[0].get_weights()[0]), axis = 1)

        importance = pd.DataFrame({'importance': weight_abs_sum})

        if feature_list is not None:
            importance['feature'] = feature_list

        if sort:
            importance = importance.sort_values(by = 'importance', ascending = False)

        if plot:
            display(importance.iloc[:limit].style.background_gradient(cmap = 'coolwarm'))
        
        # Return only a number of features
        return importance.iloc[:limit]

    @utils.catch('MLP_FITERROR')
    def fit(self, x, **fit_params):
        X_train = x['train_data'][0]
        y_train = x['train_data'][1]
        X_val = x['val_data'][0]
        y_val = x['val_data'][1]
        
        logger.info('Converting labels to ohe...')
        if self.ohe:
            try:
                y_train = to_categorical(np.array(y_train.todense()))
                y_val = to_categorical(np.array(y_val.todense()))
            except:
                y_train = to_categorical(np.array(y_train))
                y_val = to_categorical(np.array(y_val))
                
            assert(y_train.shape[1] == y_val.shape[1])
            
        logger.info('Fitting MLP...')
        metrics = self.model.fit(
            X_train,
            y_train,
            batch_size = int(self.batch_size),
            validation_data = (X_val, y_val),
            epochs = int(self.epochs),
            initial_epoch = 0,
            callbacks = self.callbacks,
            verbose = self.verbose)
        
        keys_metric = {
            'accuracy': 'val_acc',
            'mean_squared_error': 'val_mean_squared_error'
        }
        
        # Use keras function for metric
        if self.metric_func is None:
            #print(metrics.history.keys())
            metric_hist = metrics.history[keys_metric[self.metric]]

            return {
                'metric': np.min(metric_hist) if self.minimize_metric else np.max(metric_hist)
            }
        
        # Use your own metric function
        else:
            return {
                'metric': self.metric_func(np.array(y_val).ravel(), self.model.predict(X_val.values).ravel())
            }
    
    @utils.catch('MLP_SAVEERROR')
    def save(self, hparams_path = 'mlp_hparams', weights_path = 'mlp_weights.h5'):
        # 1- Save hyperparameters to pickle
        arg_keys = self.def_args.keys()
        hparams = {k: self.__dict__[k] for k in arg_keys}
        
        try:
            # Save weights and hparameters
            pickle.dump(hparams, open(hparams_path, "wb"))
            self.model.save_weights(weights_path)
        except Exception as e:
            logger.error(e)
        else:
            logger.info('Successfully saved mlp.')
    
    @utils.catch('MLP_LOADERROR')
    def load(self, hparams_path = 'mlp_hparams', weights_path = 'mlp_weights.h5'):
        # 1- Read hparams from pickle
        try:
            with open(hparams_path, "rb") as fp:
                hparams = pickle.load(fp)
                self.__dict__.update(hparams)

            logger.info(f'Initialize MLP with hparameters: {hparams}')
        except FileNotFoundError:
            logger.error(f'Hyperparameter file {hparams_path} does not exist.')
            return
        try:
            # Init model
            self.create_mlp()

            # Load model
            self.model.load_weights(weights_path)
            
        except FileNotFoundError:
            logger.error(f'Weights file {weights_path} does not exist.')
        else:
            logger.info('Successfully loaded mlp.')

    @utils.catch('MLP_FITBESTERROR')
    def fit_best(
            self, x, ohe = True,
            trials_path = f'trials_mlp', **fit_params):

        # Get search space
        search_space = model_defaults.model_defaults['mlp']['search_space']

        # Load trials
        try:
            trials = pickle.load(open(trials_path, "rb"))
        except FileNotFoundError:
            logger.error(f'Trials file {trials_path} does not exist.')

        search_params = space_eval(search_space, {k: v[0] for k,v in trials.best_trial['misc']['vals'].items()})
        logger.info(f'Best search hparameters: {search_params}')
        
        # Join fixed parameters and search parameters
        params = model_defaults.model_defaults['mlp']['search_fixed']
        params.update(search_params)
        
        logger.info(f'Fitting mlp with params: \n{params}')
        self.__dict__.update(params)
        self.create_mlp()

        return self.fit(x, **fit_params)
    
    @utils.catch('MLP_SEARCHERROR')
    def search(self, x, num_iter = 25, trials_path = 'trials_mlp', fixed_params = None):        
        # Get default hparams
        search_space = model_defaults.model_defaults['mlp']['search_space']
        fixed_params = model_defaults.model_defaults['mlp']['search_fixed'] if fixed_params is None else fixed_params
        # Search
        res_search = bayesian.bayesian_search(
            self.__init__,
            self.fit,
            x,
            search_space,
            fixed_params,
            num_iter = num_iter,
            mode = 'bayesian',
            minimize = self.minimize_metric,
            trials_path = trials_path,
            input_size = self.input_size
            )
            
        return res_search