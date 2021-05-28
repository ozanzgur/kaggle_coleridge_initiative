from hyperopt import hp
from hyperopt.pyll.base import scope
from sklearn.linear_model import LogisticRegression
   
fixed = dict(
    penalty = 'l2',
    tol = 0.0001,
    C = 0.12345,
    class_weight = None, # 'balanced'
    random_state = 42,
    solver = 'lbfgs',
    max_iter = 500,
    multi_class = 'auto',
    verbose = 0,
    n_jobs = -1
)
search_space = dict(
    C = hp.loguniform('C', -7, 3),
    #class_weight =  hp.choice('class_weight', ['balanced', None]),
    #solver =  hp.choice('solver', ['saga']), # 'newton-cg', 'lbfgs', 'sag'
    tol = hp.loguniform('tol', -7, 1)
)
search_fixed = dict(     
    max_iter = 500,
    verbose = 0,
    n_jobs = -1,
    class_weight = 'balanced',
    penalty = 'l2',
    solver = 'lbfgs',
    multi_class = 'auto',
    random_state=42,
    minimize_metric = False,
    model_class = LogisticRegression
)