from typing import Callable, Dict, List, Optional
import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt

import reservoirpy as rpy
from optuna.study import Study

def _optimize_study(study: Study, trials: int, objective_func: Callable, **kwargs) -> Study:
    """Optimizes the study with a given objective function.

    Args:
        study (Study): The Optuna study object.
        trials (int): Number of trials to optimize.
        objective_func (Callable): The objective function to optimize.
        **kwargs: Additional keyword arguments for the objective function.

    Returns:
        Study: The optimized Optuna study object.
    """
    rpy.verbosity(kwargs['verbosity'] if kwargs['verbosity'] is not None else 0)
    study.optimize(objective_func, n_trials=trials, catch=(Exception,), **kwargs)
    return study

def research(study: Study, trials: int, objective_func: Callable, processes: Optional[int] = None, **kwargs) -> List[Study]:
    """Conduct research by optimizing the study through hyperparameter sampling.

    Args:
        study (Study): The Optuna study object.
        trials (int): Number of trials to optimize.
        objective_func (Callable): The objective function to optimize.
        processes (int, optional): Number of processes for parallel optimization. Defaults to None.
        **kwargs: Additional keyword arguments for the optimization function.

    Returns:
        List[Study]: List of optimized Optuna study objects.
    """
    if not processes:
        return [_optimize_study(study, trials, objective_func, **kwargs)]

    print(f"Optimizing with {processes} processes")
    return joblib.Parallel(n_jobs=processes)(
        joblib.delayed(_optimize_study)(study, trials // processes, objective_func, **kwargs) for _ in range(processes))

def evaluate_study(study: Study, objective_str: str = "Score") -> None:
    """Prints the best parameters, their distributions, best objective value, and top 5 trials.

    Args:
        study (Study): The Optuna study object.
        objective_str (str, optional): The name of the objective being optimized. Defaults to "Score".
    """
    trials_df = study.trials_dataframe()
    best_trial = trials_df.loc[trials_df['value'].idxmax()]

    print("Best Parameters:")
    for param_name in best_trial.index:
        if param_name.startswith('params_'):
            param_value = best_trial[param_name]
            param_name = param_name[len('params_'):]
            print(f" - {param_name}: {param_value}")

    print("\nHyperparameter Distributions:")
    for param_name in study.best_params.keys():
        param_values = study.trials_dataframe()['params_' + param_name]
        param_distribution = param_values.value_counts(normalize=True)
        print(f"\n{param_name} Distribution:")
        print(param_distribution)

    print(f"\nBest {objective_str}: ", best_trial["value"])
    print("Trial: ", best_trial["number"])

    print("\nTop 5 Trials:")
    top_trials = trials_df.nlargest(5, 'value')
    for _, row in top_trials.iterrows():
        print(f"\nTrial {row['number']}:")
        for param_name in row.index:
            if param_name.startswith('params_') and not pd.isna(param_value):
                param_value = row[param_name]
                param_name = param_name[len('params_'):]
                print(f" - {param_name}: {param_value}")
        print("Score: ", row['value'])

def plot_results(study: Study, plot_func: Callable, params: Optional[Dict] = None, filename: Optional[str] = None) -> None:
    """Plots the results of the Optuna study.

    Args:
        study (Study): The Optuna study object.
        plot_func (Callable): The function to generate the plot.
        params (Dict, optional): Additional parameters for the plot function. Defaults to None.
        filename (str, optional): The filename to save the plot. Defaults to None.
    """
    plot_func(study, params if params else {})

    if filename:
        results_dir = "results/metrics/"
        os.makedirs(results_dir, exist_ok=True)
        plt.savefig(os.path.join(results_dir, f"{filename}.png"), bbox_inches='tight', dpi=800)

    plt.show()
    plt.close()