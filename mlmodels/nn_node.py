# This is not the latest version

to_import = [
    'utils',
    'model_defaults',
    'base.model.search.bayesian'
    'lib'
]

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

import os
import time
import lib
import torch, torch.nn as nn
import torch.nn.functional as F
from qhoptim.pyt import QHAdam

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from tqdm import tqdm
from IPython.display import clear_output
import numpy as np
import pandas as pd
import math
from IPython.display import display

import pickle
import logging
logger = logging.getLogger('pipeline')

class NODE():
    @utils.catch('NODE_INITERROR')
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
            experiment_name = str(uuid.uuid4()),
            beta1 = 0.95,
            beta2 = 0.998,
            nu1 = 0.7,
            nu2 = 1.0,
            hidden_size = 128,
            num_layers = 8,
            tree_dim = 3,
            depth = 6,
            batch_size = 200,
            metric_func = None,
            ohe = False,
            max_epochs = 50,
            minimize_metric = True
        )
        self.def_args = def_args

        # Extract related arguments
        for k, def_val in def_args.items():
            self.__dict__.update({k: kwargs.get(k, def_val)})
        
        hparam_dict = {k: self.__dict__.get(k) for k in def_args.keys()}
        print(f'NODE Hparameters: {hparam_dict}')
        
        self.create_model()

    @utils.catch('NODE_CREATEMODELERROR')
    def create_model(self, x):
        self.model = nn.Sequential(
            lib.DenseBlock(in_features, self.hidden_size, num_layers=self.num_layers, tree_dim=self.tree_dim, depth=self.depth, flatten_output=False,
                           choice_function=lib.entmax15, bin_function=lib.entmoid15),
            lib.Lambda(lambda x: x[..., 0].mean(dim=-1)),  # average first channels of every tree
        ).to(device)

        with torch.no_grad():
            res = self.model(torch.as_tensor(x['train_data'][0][:10000].values, device=device))
            # trigger data-aware init

        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
            
        self.optimizer_params = { 'nus':(self.nu1, self.nu2), 'betas':(self.beta1, self.beta2) }
        
        trainer = lib.Trainer(
            model=self.model, loss_function=F.mse_loss,
            experiment_name=self.modelexperiment_name,
            warm_start=False,
            Optimizer=QHAdam,
            optimizer_params=self.modeloptimizer_params,
            verbose=True,
            n_last_checkpoints=5
        )

    @utils.catch('MLP_FITERROR')
    def fit(self, x, **fit_params):
        X_train = x['train_data'][0]
        y_train = x['train_data'][1]
        X_val = x['val_data'][0]
        y_val = x['val_data'][1]
        
        loss_history, metric_history = [], []
        best_metric = float('inf')
        best_step_metric = 0
        early_stopping_rounds = 5000
        report_frequency = 100
        
        logger.info('Converting labels to ohe...')
        if self.ohe:
            try:
                y_train = to_categorical(np.array(y_train.todense()))
                y_val = to_categorical(np.array(y_val.todense()))
            except:
                y_train = to_categorical(np.array(y_train))
                y_val = to_categorical(np.array(y_val))
                
            assert(y_train.shape[1] == y_val.shape[1])
            
            
        # Training loop
        for batch in lib.iterate_minibatches(x['train_data'][0].values, x['train_data'][1].values, batch_size=batch_size, 
                                                shuffle=True, epochs=float('inf')):
            metrics = trainer.train_on_batch(*batch, device=device)

            loss_history.append(metrics['loss'])

            if trainer.step % report_frequency == 0:
                trainer.save_checkpoint()
                trainer.average_checkpoints(out_tag='avg')
                trainer.load_checkpoint(tag='avg')

                metric = - trainer.evaluate_numerai(
                    x['val_data'][0].values, x['val_data'][1].values, device=device, eras=fit_params['val_eras'], batch_size=1000) #val_eras

                if metric < best_metric:
                    best_metric = metric
                    best_step_metric = trainer.step
                    trainer.save_checkpoint(tag='best_metric')
                metric_history.append(metric)

                trainer.load_checkpoint()  # last
                trainer.remove_old_temp_checkpoints()

                clear_output(True)
                plt.figure(figsize=[18, 6])
                plt.subplot(1, 2, 1)
                plt.plot(loss_history)
                plt.title('Loss')
                plt.grid()
                plt.subplot(1, 2, 2)
                plt.plot(metric_history)
                plt.title('Negative correlation')
                plt.grid()
                plt.show()
                print("Loss %.5f" % (metrics['loss']))
                print("Val metric: %0.5f" % (metric))
            if (trainer.step > best_step_metric + early_stopping_rounds) or (trainer.step // report_frequency > max_epochs):
                print('BREAK. There is no improvment for {} steps'.format(early_stopping_rounds))
                print("Best step: ", best_step_metric)
                print("Best Val metric: %0.5f" % (best_metric))
                break
        
        val_metric = best_metric
        else:
            return {
                'metric': val_metric
            }
    
    @utils.catch('NODE_SEARCHERROR')
    def search(self, x, num_iter = 25, trials_path = 'trials_node', fixed_params = None):        
        # Get default hparams
        search_space = model_defaults.model_defaults['node']['search_space']
        fixed_params = model_defaults.model_defaults['node']['search_fixed'] if fixed_params is None else fixed_params
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
            input_size = x['train_data'][0].shape[1]
            )
            
        return res_search