from hyperopt import hp
from hyperopt.pyll.base import scope
   
fixed = {
    'bagging_fraction': 0.5827412807931113,
    'feature_fraction': 0.7542845804413821,
    'learning_rate': 0.019308353919896423,
    'max_depth': 14.0,
    'min_data_in_leaf': 557.0,
    'min_sum_hessian_in_leaf': 0.303032034177885,
    'num_leaves': 5,
    'reg_alpha': 0.07257855387297459,
    'reg_lambda': 0.3751782050564545,
    'unbalanced_sets ': 1,
    
    ### These are fixed ###
    'num_iterations' : 1000,
    'random_state' : 42,
    'bagging_freq' : 5,
    'bagging_seed' : 42,
    #early_stopping_round = 500,
    'objective' : 'regression',
    'metric' : 'rmse',
    #verbose = 0,
    'num_class' : 1,
    'minimize_metric' : True,
    'metric_func' : None,
    'folds' : KFold(n_splits=5, random_state=42, shuffle=True),
    'is_walk_forward' : False,
    'transform_walk_forward' : None,
    'target_name' : None,
    'transform_cols': None # Walk-forward generated features
}
search_space = dict(
    num_leaves = scope.int(hp.quniform('num_leaves', 2, 100, 1)),
    max_depth = scope.int(hp.quniform('max_depth', 2, 20, 1)),
    min_data_in_leaf = scope.int(hp.quniform('min_data_in_leaf', 1, 100, 1)),
    bagging_fraction = hp.uniform('bagging_fraction', 0.025, 1.0),
    learning_rate = hp.loguniform('learning_rate', -4, 1),
    reg_alpha = hp.loguniform('reg_alpha', -7, 3),
    reg_lambda = hp.loguniform('reg_lambda', -7, 3),
    min_sum_hessian_in_leaf = hp.loguniform('min_sum_hessian_in_leaf', -7, 3),
    feature_fraction = hp.uniform('feature_fraction', 0.001, 1.0),
    unbalanced_sets = hp.choice('unbalanced_sets ', [True, False]),
    num_iterations = scope.int(hp.quniform('num_iterations', 250, 3000, 1)),
)
search_fixed = dict(
    folds = KFold(n_splits=5, random_state=42, shuffle=True),
    
    random_state = 42,
    bagging_freq = 5,
    bagging_seed = 42,
    #early_stopping_round = 500,
    objective = 'regression',
    metric = 'rmse',
    #verbose = 0,
    num_class = 1,
    minimize_metric = True,
    metric_func = None,
    is_walk_forward = False,
    transform_walk_forward = None,
    target_name = None,
    transform_cols = None
)