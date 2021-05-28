from hyperopt import hp
from hyperopt.pyll.base import scope
from sklearn.naive_bayes import MultinomialNB
   
fixed = dict(
            alpha = 1.0
        )
search_space = dict(
            alpha = hp.loguniform('alpha', -7, 3)
        )
search_fixed = dict(
    model_class = MultinomialNB
)