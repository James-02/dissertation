import os
import time
import argparse
import multiprocessing

import joblib
import optuna
import reservoirpy as rpy
import matplotlib.pyplot as plt

from reservoirpy.nodes import Ridge
from optuna.visualization.matplotlib import plot_contour, plot_parallel_coordinate, plot_param_importances, plot_rank, plot_slice

from reservoir.reservoir import OscillatorReservoir
from utils.preprocessing import load_ecg_data
from utils.classification import classify
from utils.logger import Logger

rpy.verbosity(0)

def objective(trial, dataset, params):
    X_train, Y_train, X_test, Y_test = dataset

    sr = trial.suggest_float("sr", 0, 2)
    input_connectivity = trial.suggest_float("input_connectivity", 0, 1)
    rc_connectivity = trial.suggest_float("rc_connectivity", 0, 1)
    coupling = trial.suggest_float("coupling", 1e-6, 1e-2)
    rc_scaling = trial.suggest_float("rc_scaling", 1e-7, 1e-2)

    reservoir = OscillatorReservoir(units=params['nodes'],
                                    timesteps=X_train[0].shape[0],
                                    delay=params['delay'],
                                    sr=sr,
                                    coupling=coupling,
                                    rc_scaling=rc_scaling,
                                    input_connectivity=input_connectivity,
                                    rc_connectivity=rc_connectivity,
                                    input_scaling=params['input_scaling'],
                                    seed=params['seed'])

    readout = Ridge(ridge=params['ridge'])

    metrics = classify(reservoir, readout, X_train, Y_train, X_test, Y_test)
    return metrics['f1']


def plot_results(study, save=True):
    if save:
        results_dir = "results/metrics"
        os.makedirs(results_dir, exist_ok=True)

    def plot_and_save(plot_func, filename):
        plot_func(study)
        if save:
            plt.savefig(os.path.join(results_dir, filename))
        plt.show()
        plt.close()

    plot_and_save(plot_slice, "slice.png")
    plot_and_save(plot_rank, "rank.png")
    plot_and_save(plot_parallel_coordinate, "parallel_coordinates.png")
    plot_and_save(plot_param_importances, "param_importances.png")
    plot_and_save(plot_contour, "contour.png")


def optimize_study(study, trials, dataset, params):
    study.optimize(lambda trial: objective(trial, dataset, params), n_trials=trials)


def research_multi(trials, processes):
    print(f"Optimization with n_process = {processes}")
    start_time = time.time()

    trials_per_process = trials // processes

    studies = joblib.Parallel(n_jobs=processes)(
        joblib.delayed(optimize_study)(trials_per_process) for _ in range(processes))

    end_time = time.time()
    print(f"Time Elapsed: {end_time - start_time}s")
    return studies

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=5)
    parser.add_argument('--study_name', type=str, default="optimization-study")
    parser.add_argument('--nodes', type=int, default=100)
    parser.add_argument('--instances', type=int, default=1000)
    parser.add_argument('--binary', type=bool, default=False)
    parser.add_argument('--use_oscillators', type=bool, default=True)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--processes', type=int, default=1)
    parser.add_argument('--job_id', type=int, default=0)
    args = parser.parse_args()

    logger = Logger(log_file=f"logs/optimization-{args.job_id}.log")
    dataset = load_ecg_data(
        rows=args.instances,
        test_ratio=args.test_ratio,
        normalize=True,
        encode_labels=True,
        shuffle=True,
        repeat_targets=False,
        binary=args.binary)

    # Define the grid of parameter values as a dictionary
    param_values = {
        'sr': [0.7, 0.9, 1.1],
        'input_connectivity': [0.3, 0.5, 1.0],
        'rc_connectivity': [0.3, 0.5, 1.0],
        'coupling': [1e-2, 5e-3, 1e-4],
        'rc_scaling': [1e-6, 8e-6, 1e-7]
    }

    # Create a list of parameter sets as dictionaries
    params = {
        'nodes': args.nodes,
        'input_scaling': 1.0,
        'delay': 10,
        'seed': 1337,
        'ridge': 1e-5
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

    print("Processes Available: ", multiprocessing.cpu_count())
    if args.processes > 1:
        research_multi(args.trials, args.processes)
    else:
        optimize_study(study, args.trials, dataset, params)

    trials_df = study.trials_dataframe()
    best_trial = trials_df.loc[trials_df['value'].idxmax()]

    print("Best Parameters:")
    for param_name in best_trial.index:
        if param_name.startswith('params_'):
            param_value = best_trial[param_name]
            param_name = param_name[len('params_'):]
            print(f" - {param_name}: {param_value}")
    print("F1 Score: ", best_trial["value"])

    plot_results(study)
