from hyperopt import hp
from hyperopt.pyll.base import scope
from sklearn_crfsuite import CRF

search_space =  hp.choice('type',
                [{
                    'algorithm': 'lbfgs',
                    'delta': hp.loguniform('delta_1', -7, -1),
                    'epsilon':  hp.loguniform('epsilon_1', -7, -1),
                    'c2': hp.loguniform('c2_1', -7, 7),
                    'all_possible_transitions': hp.choice('all_possible_transitions_1', [True, False]),
                    'all_possible_states': hp.choice('all_possible_states_1', [True, False]),
                    'min_freq': scope.int(hp.quniform('min_freq_1', 0, 20, 1)),
                    'c1': hp.loguniform('c1', -7, 7),
                    'num_memories': scope.int(hp.quniform('num_memories', 3, 25, 1)),
                    'period': scope.int(hp.quniform('period', 5, 15, 1)),
                    'linesearch': hp.choice('linesearch', ['MoreThuente', 'Backtracking', 'StrongBacktracking']),
                    'max_linesearch': scope.int(hp.quniform('max_linesearch', 3, 50, 1)),
                 },
                {
                    'algorithm': 'l2sgd',
                    'c2': hp.loguniform('c2_2', -7, 5),
                    'delta': hp.loguniform('delta_2', -7, -3),
                    'all_possible_transitions': hp.choice('all_possible_transitions_2', [True, False]),
                    'all_possible_states': hp.choice('all_possible_states_2', [True, False]),
                    'min_freq': scope.int(hp.quniform('min_freq_2', 0, 4, 1)),
                    'calibration_eta': hp.loguniform('calibration_eta', -2, 0),
                    'calibration_rate': hp.loguniform('calibration_rate', -1, 1),
                    'calibration_samples': scope.int(hp.quniform('calibration_samples', 300, 2000, 1)),
                    'calibration_candidates': scope.int(hp.quniform('calibration_candidates', 5, 15, 1)),
                    'calibration_max_trials': scope.int(hp.quniform('calibration_max_trials', 10, 30, 1)),
                 },
                 {
                     'algorithm': 'pa',
                     'epsilon':  hp.loguniform('epsilon_2', -7, -3),
                     'all_possible_transitions': hp.choice('all_possible_transitions_3', [True, False]),
                     'all_possible_states': hp.choice('all_possible_states_3', [True, False]),
                     'min_freq': scope.int(hp.quniform('min_freq_3', 0, 4, 1)),
                     'pa_type': hp.choice('pa_type', [0, 1, 2]),
                     
                     'c': hp.loguniform('c', -1, 1),
                     'error_sensitive': hp.choice('error_sensitive', [True, False]),
                     'averaging': hp.choice('averaging', [True, False]),
                 },
                 {
                     'algorithm': 'arow',
                     'epsilon':  hp.loguniform('epsilon_3', -7, -3),
                     'all_possible_transitions': hp.choice('all_possible_transitions_4', [True, False]),
                     'all_possible_states': hp.choice('all_possible_states_4', [True, False]),
                     'min_freq': scope.int(hp.quniform('min_freq_4', 0, 4, 1)),
                     'variance': hp.loguniform('variance', -0.5, 0.5),
                     'gamma': hp.loguniform('gamma', -2, 2),
                 },
                 {
                     'algorithm': 'ap',
                     'epsilon':  hp.loguniform('epsilon_4', -7, -3),
                     'all_possible_transitions': hp.choice('all_possible_transitions_5', [True, False]),
                     'all_possible_states': hp.choice('all_possible_states_5', [True, False]),
                     'min_freq': scope.int(hp.quniform('min_freq_5', 0, 4, 1)),
                 }]
        )
search_fixed = dict(model_class = CRF)