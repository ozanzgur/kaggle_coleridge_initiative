import functools
import logging
import traceback
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json

logger = logging.getLogger('pipeline')

def set_logger(mode = 'error'):
    if logger.handlers:
        logger.handlers = []
    
    level = None
    if mode == 'error':
        level = logging.ERROR
    elif mode == 'info':
        level = logging.INFO
    else:
        raise NotImplementedError('mode must be one of ["error", "info"]')

    # set logger level
    logger.setLevel(level)

    logFormatter = logging.Formatter(fmt=' %(name)s :: %(levelname)-8s :: %(message)s')

    # create console handler
    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(level)
    consoleHandler.setFormatter(logFormatter)

    # Add console handler to logger
    logger.addHandler(consoleHandler)

def catch(msg):
    """Catch errors and add a message.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                
                logger.error(traceback.format_exc())
                raise Exception(msg + r'//' + str(e))
            else:
                return result 
        return wrapper
    return decorator

def debug(f = None, *, level ='debug'):
    
    if f is None:
        return functools.partial(debug, level=level)
    
    @functools.wraps(f)
    def log_f_as_called(*args, **kwargs):
        logging.info(f'{f} was called with arguments={args} and kwargs={kwargs}')
        value = f(*args, **kwargs)
        logging.info(f'{f} return value {value}')
        return value
    return log_f_as_called
    
def explain_confidences(test_res_dict):
    preds_df = pd.DataFrame({'conf': test_res_dict['confidences'],
                             'pred': test_res_dict['preds'],
                             'label': test_res_dict['labels']})

    plt.figure(figsize = (15,7))
    sns.distplot(preds_df[preds_df['pred'] == preds_df['label']]['conf'], kde = False, bins = 15)
    sns.distplot(preds_df[preds_df['pred'] != preds_df['label']]['conf'], kde = False, bins = 15)
    plt.show()
    return preds_df
    