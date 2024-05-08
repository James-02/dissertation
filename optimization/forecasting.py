import sys
import os

# add relative path to allow importing of utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import optuna

import numpy as np
from reservoirpy.nodes import Ridge
from optimization.base import evaluate_study, plot_results, research
from optuna.visualization.matplotlib import plot_param_importances, plot_slice

from reservoir.reservoir import OscillatorReservoir
from utils.preprocessing import load_ecg_forecast
from utils.logger import Logger


def objective(trial, **kwargs):
    X_train, Y_train, X_test, Y_test = load_ecg_forecast(
        timesteps=kwargs["timesteps"],
        test_ratio=kwargs["test_ratio"],
        forecast=kwargs["forecast"])

    reservoir = OscillatorReservoir(
        units=trial.suggest_int("nodes", 25, 150),
        timesteps=params['timesteps'],
        sr=trial.suggest_float("sr", 0, 1),
        coupling=trial.suggest_float("coupling", 1e-6, 1e-2),
        rc_scaling=trial.suggest_float("rc_scaling", 1e-7, 1e-2),
        input_connectivity=trial.suggest_float("input_connectivity", 0, 1),
        rc_connectivity=trial.suggest_float("rc_connectivity", 0, 1),
        input_scaling=trial.suggest_float("input_scaling", 0, 1.0),
        node_kwargs={'delay': trial.suggest_int("delay", 1, 10)},
        seed=kwargs['seed'])

    readout = Ridge(ridge=params['ridge'])

    # construct echo state network model
    esn = reservoir >> readout

    # fit model to training data (no forecast)
    esn.fit(X_train, Y_train)

    # test model on data with forecast
    Y_pred = esn.run(X_test)

    # calulate mean squared error loss
    mse = np.mean((Y_test - Y_pred) ** 2)

    # use root mean squared error as objective
    return np.sqrt(mse)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=25)
    parser.add_argument('--study_name', type=str, default="forecasting")
    parser.add_argument('--timesteps', type=int, default=2000)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--forecast', type=int, default=10)
    parser.add_argument('--processes', type=int, default=1)
    parser.add_argument('--job_id', type=int, default=0)
    args = parser.parse_args()
    
    logger = Logger(log_file=f"logs/forecast-optimization-{args.job_id}.log")

    # Create a list of parameter sets as dictionaries
    params = {
        'seed': 1337,
        'ridge': 1e-5,
        'timesteps': args.timesteps,
        'forecast': args.forecast,
        'test_ratio': args.test_ratio
    }

    # Create the study with the GridSampler
    sampler = optuna.samplers.RandomSampler()

    log_name = f"logs/optuna-{args.study_name}.db"
    storage = optuna.storages.RDBStorage(f'sqlite:///{log_name}')
    study = optuna.create_study(
        study_name=args.study_name,
        direction='minimize',
        storage=storage,
        sampler=sampler,
        load_if_exists=True
    )

    # perform objective research
    study = research(study, args.trials, objective, args.processes, **params)

    # evaluate study results
    evaluate_study(study.trials_dataframe(), objective_str="RMSE")

    # plot results
    plot_results(study, plot_slice, filename="forecasting-slice.png")
    plot_results(study, plot_param_importances, filename="forecasting-importance.png")
