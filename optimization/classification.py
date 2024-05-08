import sys
import os

# add relative path to allow importing of utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import optuna

from reservoirpy.nodes import Ridge
from optuna.visualization.matplotlib import plot_param_importances, plot_slice

from reservoir.reservoir import OscillatorReservoir
from utils.preprocessing import load_ecg_data
from utils.classification import classify
from utils.logger import Logger

from optimization.base import evaluate_study, research, plot_results


def objective(trial, **kwargs):
    X_train, Y_train, X_test, Y_test = load_ecg_data(
        rows=kwargs['instances'],
        test_ratio=kwargs['test_ratio'],
        binary=kwargs['binary']
    )

    noise_ratio = trial.suggest_float("noise_ratio", 0, 1)
    noise_rate = trial.suggest_float("noise_rate", 0, 1)

    reservoir = OscillatorReservoir(
        units=trial.suggest_float("nodes", 0, 300),
        timesteps=X_train[0].shape[0],
        sr=trial.suggest_float("sr", 0, 2),
        warmup=trial.suggest_float("warmup", 0, 100),
        coupling=trial.suggest_float("coupling", 0, 1),
        rc_scaling=trial.suggest_float("rc_scaling", 0, 1),
        input_connectivity=trial.suggest_float("input_connectivity", 0, 1),
        rc_connectivity=trial.suggest_float("rc_connectivity", 0, 1),
        input_scaling=trial.suggest_float("input_scaling", 0, 1),
        bias_scaling=trial.suggest_float("bias_scaling", 0, 1),
        node_kwargs={'delay': kwargs['delay']},
        seed=kwargs['seed'])

    readout = Ridge(ridge=kwargs['ridge'])

    filename = f"results/runs/{kwargs['study_name']}-{trial.number}-{kwargs['idx']}"

    metrics = classify(reservoir, readout, X_train, Y_train, X_test, Y_test, folds=kwargs['folds'],
                       save_file=filename, noise_rate=noise_rate, noise_ratio=noise_ratio)

    return metrics['f1']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=5)
    parser.add_argument('--study_name', type=str, default="final")
    parser.add_argument('--instances', type=int, default=4015)
    parser.add_argument('--binary', type=bool, default=False)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--processes', type=int, default=1)
    parser.add_argument('--folds', type=int, default=0)
    parser.add_argument('--job_id', type=int, default=0)
    args = parser.parse_args()

    logger = Logger(log_file=f"logs/classification-optimization-{args.job_id}.log")

    # Define the grid of parameter values as a dictionary
    param_values = {
        'sr': [0.9, 1.0, 1.1],
        'nodes': [200, 225, 250],
        'input_connectivity': [0.1, 0.3, 0.5],
        'rc_connectivity': [0.1, 0.3, 0.5],
        'coupling': [1e-3, 5e-2, 5e-3],
        'warmup': [40, 50, 60],
        'rc_scaling': [4e-6, 8e-6],
        'input_scaling': [0.9, 1.0, 1.1],
        'bias_scaling': [0.9, 1.0, 1.1],
        'noise_ratio': [0.1, 0.2, 0.3],
        'noise_rate': [0.1, 0.2, 0.3]
    }

    # Create a list of parameter sets as dictionaries
    params = {
        'delay': 10,
        'seed': 1337,
        'ridge': 1e-5,
        'instances': args.instances,
        'binary': args.binary,
        'test_ratio': args.test_ratio,
        'folds': args.folds,
        'idx': args.job_id
    }

    # Create the study with the GridSampler
    sampler = optuna.samplers.GridSampler(param_values)

    log_name = f"logs/optuna-{args.study_name}.db"
    storage = optuna.storages.RDBStorage(f'sqlite:///{log_name}')
    study = optuna.create_study(
        study_name=args.study_name,
        direction='maximize',
        storage=storage,
        sampler=sampler,
        load_if_exists=True
    )

    # perform objective research
    study = research(study, args.trials, objective, args.processes, **params)

    # evaluate study results
    evaluate_study(study.trials_dataframe(), objective_str="F1 Score")

    # plot results
    plot_results(study, plot_slice, filename="classification-slice.png")
    plot_results(study, plot_param_importances, filename="classification-importance.png")
