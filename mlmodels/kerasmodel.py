to_import = ['utils']

from os.path import dirname, abspath
project_name = dirname(dirname(dirname(abspath(__file__)))).split('\\')[-1]

import sys
import importlib
for module in to_import:
    module_name = module.split('.')[-1]
    new_module = importlib.import_module(name = f'.{module}', package = project_name)
    sys.modules[__name__].__dict__.update({module_name: new_module})

################################################################################

import matplotlib.pyplot as plt
import keras
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import logging
import os
import tensorflow as tf
logger = logging.getLogger('pipeline')

class BaseKerasModel():
    @utils.catch('BASEMODEL_INITERROR')
    def __init__(self, **kwargs):
        
        def_args = {
            'use_callback_checkpoint': True
        }
        
        self.callbacks = []
        self.model = None
        self.graph = tf.get_default_graph()
        
        # Extract related arguments
        for k, def_val in def_args.items():
            self.__dict__.update({k: kwargs.get(k, def_val)})
        
        # By default, add all callbacks
        self.add_callback_reducelr()
        self.add_callback_earlystopping()
        
        if self.use_callback_checkpoint:
            self.add_callback_checkpoint()
    
    @utils.catch('BASEMODEL_SETMODELERROR')
    def set_model(self, model):
        logger.info('Warning: Model already exists, replacing.')
        self.model = model
    
    @utils.catch('BASEMODEL_FITGENERROR')
    def fit_generator(
            self, x, steps_per_epoch = 1, epochs = 1,
            validation_steps = 1, **kwargs):
        """ Trains model, returns validation loss in {'val_loss': val_loss}
        """      
        history = self.model.fit_generator(
                    x['train_data'],
                    steps_per_epoch = steps_per_epoch,
                    epochs = epochs,
                    validation_data = x['val_data'],
                    validation_steps = validation_steps,
                    callbacks = self.callbacks,
                    **kwargs)
        
        # Create a dict for metric
        return {'val_loss': min(history.history['val_loss'])}
    
    @utils.catch('BASEMODEL_PREDICTERROR')
    def predict(self, x, **kwargs):
        return self.model.predict(x, **kwargs)
    
    @utils.catch('MLP_FITERROR')
    def fit(self, x, steps_per_epoch, validation_steps, 
            plot_history = False, **fit_params):
        
        # Train model
        metrics = self.model.fit(
            x['train_data'][0],
            x['train_data'][1],
            verbose = True,
            batch_size = self.batch_size,
            validation_data = x['val_data'],
            epochs = 1,
            initial_epoch = 0,
            callbacks = self.callbacks,
            **fit_params)
        
        # Plot metrics over epochs
        if plot_history:
            history = metrics
            acc = history.history['acc']
            val_acc = history.history['val_acc']
            loss = history.history['loss']
            val_loss = history.history['val_loss']
            x = range(1, len(acc) + 1)

            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(x, acc, 'b', label='Training acc')
            plt.plot(x, val_acc, 'r', label='Validation acc')
            plt.title('Training and validation accuracy')
            plt.legend()
            plt.subplot(1, 2, 2)
            plt.plot(x, loss, 'b', label='Training loss')
            plt.plot(x, val_loss, 'r', label='Validation loss')
            plt.title('Training and validation loss')
            plt.legend()
        
        return metrics
    
    @utils.catch('BASEMODEL_EVALUATEERROR')
    def test_generator(self, dataset):
        """Test for pipelines with a data generator.
        
        Parameters
        ----------
            dataset - generator
                Generator object from retriever or preprocessor
        Returns
        -------
            metrics - dict
                Evaluation results.
                
        """
        steps = dataset.samples // dataset.batch_size \
            + (dataset.samples % dataset.batch_size > 0)
        
        metrics = self.model.evaluate_generator(
                    dataset,
                    steps=steps,
                    max_queue_size=10,
                    workers=1,
                    use_multiprocessing=False,
                    verbose=0)
        return metrics
    
    @utils.catch('MLP_TESTERROR')
    def test(self, test_data):
        """Test for pipelines without a data generator.
        
        Parameters
        ----------
            test_data - tuple of X and y : (X_test, y_test)
                Test dataset, with X and y.
                
        Returns
        -------
            metrics - dict
                Evaluation results.
                
        """
        return self.model.evaluate(test_data[0], test_data[1])
                                
    @utils.catch('BASEMODEL_ADDCHECKPOINTERROR')
    def add_callback_checkpoint(self, path = None, monitor = 'val_loss', period = 1):
        if path is None:
            path = "logs//keras_loss-{val_loss:.4f}_ep-{epoch:03d}.h5"
        self.callbacks.append(ModelCheckpoint(
                                path,
                                monitor = monitor,
                                save_weights_only = True,
                                save_best_only = True,
                                period = period))
        
    @utils.catch('BASEMODEL_ADDREDUCELRERROR')
    def add_callback_reducelr(self, monitor = 'val_loss', factor = 0.2,
                              patience = 3, verbose = 1):
        
        self.callbacks.append(ReduceLROnPlateau(
                                monitor = monitor,
                                factor = factor,
                                patience = patience,
                                verbose = verbose))
        
    @utils.catch('BASEMODEL_ADDEARLYSTOPPINGERROR')
    def add_callback_earlystopping(self, monitor = 'val_loss', min_delta = 0,
                                   patience = 5, verbose = 1):
        
        self.callbacks.append(EarlyStopping(
                                monitor = monitor,
                                min_delta = min_delta,
                                patience = patience,
                                verbose = verbose,
                                restore_best_weights = True
                                ))