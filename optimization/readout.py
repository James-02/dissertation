import sys
import os

# add relative path to allow importing of utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import optuna
import numpy as np

from optuna.visualization.matplotlib import plot_slice

from reservoirpy.nodes import ScikitLearnNode
from sklearn.linear_model import RidgeClassifier, LogisticRegression, Perceptron
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from utils.preprocessing import load_npz
from utils.classification import evaluate_performance
from utils.logger import Logger
from optimization.base import evaluate_study, research, plot_results

def objective(trial, **kwargs):
    trained_states, Y_train, tested_states, Y_test = load_data(kwargs['load_file'])

    classifier_name = trial.suggest_categorical('classifier', kwargs['classifiers'])

    if classifier_name == "Ridge":
        hypers = {
            "alpha": trial.suggest_float('ridge_alpha', 1e-5, 1e2, log=True),
            "fit_intercept": trial.suggest_categorical('ridge_fit_intercept', [True, False]),
            "copy_X": trial.suggest_categorical('ridge_copy_X', [True, False]),
            "tol": trial.suggest_float('ridge_tol', 1e-5, 1e-1, log=True),
            "solver": trial.suggest_categorical('ridge_solver', ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']),
        }
        clf = RidgeClassifier

    elif classifier_name == "Bayes":
        hypers = {
            "var_smoothing": trial.suggest_float('bayes_var_smoothing', 1e-12, 1e-4),
        }
        clf = GaussianNB

    elif classifier_name == "LR":
        hypers = {
            "tol": trial.suggest_float('lr_tol', 1e-5, 1e-1, log=True),
            "C": trial.suggest_float('lr_C', 1e-5, 1e5, log=True),
            "fit_intercept": trial.suggest_categorical('lr_fit_intercept', [True, False]),
            "intercept_scaling": trial.suggest_float('lr_intercept_scaling', 0.1, 10),
            "solver": trial.suggest_categorical('lr_solver', ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga']),
            "max_iter": 1000,
        }
        clf = LogisticRegression

    elif classifier_name == "Perceptron":
        hypers = {
            "penalty": trial.suggest_categorical('perceptron_penalty', [None, 'l2', 'l1', 'elasticnet']),
            "alpha": trial.suggest_float('perceptron_alpha', 1e-5, 1e-2, log=True),
            "l1_ratio": trial.suggest_float('perceptron_l1_ratio', 0, 1),
            "fit_intercept": trial.suggest_categorical('perceptron_fit_intercept', [True, False]),
            "tol": trial.suggest_float('perceptron_tol', 1e-5, 1e-1, log=True),
            "eta0": trial.suggest_float('perceptron_eta0', 1e-4, 1, log=True),
        }
        clf = Perceptron

    elif classifier_name == "SVM":
        kernel = trial.suggest_categorical('svc_kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
        hypers = {
            "C": trial.suggest_float('svc_C', 1e-5, 1e4, log=True),
            "kernel": kernel,
            "degree": trial.suggest_int('svc_degree', 1, 5),
            "gamma": trial.suggest_categorical('svc_gamma', ['scale', 'auto']) if kernel != 'linear' else 'scale',
            "tol": trial.suggest_float('svc_tol', 1e-5, 1e-2, log=True),
        }
        clf = SVC

    elif classifier_name == "MLP":
        hypers = {
            "hidden_layer_sizes": trial.suggest_int('mlp_hidden_layer_sizes', 100, 300),
            "activation": trial.suggest_categorical('mlp_activation', ['logistic', 'tanh', 'relu']),
            "solver": trial.suggest_categorical('mlp_solver', ['lbfgs', 'adam']),
            "alpha": trial.suggest_float('mlp_alpha', 1e-5, 1e-3, log=True),
            "learning_rate": trial.suggest_categorical('mlp_learning_type', ['constant', 'invscaling']),
            "learning_rate_init": trial.suggest_float('mlp_learning_rate_init', 1e-5, 1e-3, log=True),
            "power_t": trial.suggest_float('mlp_power_t', 0.1, 1.0),
            "tol": trial.suggest_float('mlp_tol', 1e-5, 1e-1, log=True),
            "momentum": trial.suggest_float('mlp_momentum', 0.1, 0.9),
            "epsilon": trial.suggest_float('mlp_epsilon', 1e-8, 1e-6, log=True),
            "max_iter": 1000,
        }
        clf = MLPClassifier

    elif classifier_name == "KNN":
        hypers = {
            "n_neighbors": trial.suggest_int('knn_n_neighbors', 1, 20),
            "weights": trial.suggest_categorical('knn_weights', ['uniform', 'distance']),
            "algorithm": trial.suggest_categorical('knn_algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute']),
            "leaf_size": trial.suggest_int('knn_leaf_size', 10, 50),
            "p": trial.suggest_float('knn_p', 1.0, 2.0)
        }
        clf = KNeighborsClassifier

    elif classifier_name == "DT":
        hypers = {
            "criterion": trial.suggest_categorical('dt_criterion', ["gini", "entropy", "log_loss"]),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        }
        clf = DecisionTreeClassifier

    elif classifier_name == "RF":
        hypers = {
                "n_estimators": trial.suggest_int('rf_n_estimators', 100, 500),
                "criterion": trial.suggest_categorical('rf_criterion', ["gini", "entropy", "log_loss"]),
                "oob_score": trial.suggest_categorical('rf_oob_score', [True, False]),
            }
        clf = RandomForestClassifier

    elif classifier_name == "GB":
        hypers = {
            "loss": trial.suggest_categorical('gb_loss', ['log_loss', 'exponential']),
            "learning_rate": trial.suggest_float('gb_learning_rate', 0.001, 1.0),
            "n_estimators": trial.suggest_int('gb_n_estimators', 100, 500),
            "subsample": trial.suggest_float('gb_subsample', 0.1, 1.0),
            "criterion": trial.suggest_categorical('gb_criterion', ['friedman_mse', 'squared_error']),
        }
        clf = GradientBoostingClassifier

    node = ScikitLearnNode(clf, model_hypers=hypers)

    logger.debug(f"Fitting {classifier_name}")
    node.fit(trained_states, Y_train)

    logger.debug(f"Running {classifier_name}")
    Y_pred = node.run(tested_states)
    return evaluate_performance(Y_pred, Y_test)['f1']

def load_data(filename):
    data = load_npz(filename)
    trained_states = data['train_states']
    tested_states = data['test_states']
    Y_train = data['Y_train']
    Y_test = data['Y_test']

    # Reshape arrays
    trained_states = trained_states.reshape(trained_states.shape[0], 1, -1)
    tested_states = tested_states.reshape(tested_states.shape[0], 1, -1)
    Y_train = Y_train.reshape(Y_train.shape[0], 1, -1)
    Y_test = Y_test.reshape(Y_test.shape[0], 1, -1)

    return trained_states, Y_train, tested_states, Y_test

if __name__ == "__main__":
    filename = "results/runs/OscillatorReservoir-250-3215-fold-0.npz"

    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=100)
    parser.add_argument('--study_name', type=str, default="readout")
    parser.add_argument('--job_id', type=int, default=0)
    parser.add_argument('--load_file', type=str, default=filename)
    parser.add_argument('--processes', type=int, default=1)
    args = parser.parse_args()

    logger = Logger(log_file=f"logs/readout-optimization-{args.job_id}.log")
    db_name = f"logs/optuna-{args.study_name}.db"

    storage = optuna.storages.RDBStorage(f'sqlite:///{db_name}')
    seed = np.random.randint(0, 10000)
    sampler = optuna.samplers.RandomSampler(seed=seed)

    study = optuna.create_study(
        study_name=args.study_name,
        direction='maximize',
        storage=storage,
        sampler=sampler,
        load_if_exists=True
    )

    params = {
        "classifiers": ["Ridge", "Bayes", "LR", "Perceptron", "SVM", "MLP", "KNN", "DT", "RF", "GB"],
        "study_name": args.study_name,
        "job_id": args.job_id,
        "load_file": args.load_file
    }

    study = research(study, args.trials, objective, processes=args.processes, **params)

    # Run optimization for each classifier
    for classifier_name in params['classifiers']:
        study_df = study.trials_dataframe()
        classifier_study = study_df[study_df['params_classifier'] == classifier_name]
        evaluate_study(classifier_study, "F1 Score")

    plot_results(study, plot_slice, params=["ridge_alpha", "ridge_fit_intercept", "ridge_copy_X", "ridge_solver", "ridge_tol"], filename="ridge")
    plot_results(study, plot_slice, params=["lr_tol", "lr_fit_intercept", "lr_intercept_scaling", "lr_C", "lr_solver"], filename="lr")
    plot_results(study, plot_slice, params=["perceptron_penalty", "perceptron_alpha", "perceptron_l1_ratio", "perceptron_tol", "perceptron_eta0", "perceptron_fit_intercept"], filename="perceptron")
    plot_results(study, plot_slice, params=["svc_kernel", "svc_C", "svc_gamma", "svc_tol", "svc_degree"], filename="svm")
    plot_results(study, plot_slice, params=["bayes_var_smoothing"], filename="bayes")
    plot_results(study, plot_slice, params=["mlp_hidden_layer_sizes", "mlp_solver", "mlp_alpha", "mlp_activation", "mlp_tol", "mlp_learning_rate_init", "mlp_power_t", "mlp_momentum", "mlp_epsilon"], filename="mlp")
    plot_results(study, plot_slice, params=["knn_n_neighbors", "knn_weights", "knn_leaf_size", "knn_p", "knn_algorithm"], filename="knn")
    plot_results(study, plot_slice, params=["dt_criterion", "min_samples_split", "min_samples_leaf"], filename="dt")
    plot_results(study, plot_slice, params=["rf_n_estimators", "rf_criterion", "rf_oob_score"], filename="rf")
    plot_results(study, plot_slice, params=["gb_loss", "gb_learning_rate", "gb_criterion", "gb_subsample", "gb_n_estimators"], filename="gb")
