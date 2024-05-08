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
    rpy.verbosity(kwargs.get('verbosity', 0))

    # Optimize your study using the wrapped objective function
    study.optimize(lambda trial: objective_func(trial, **kwargs), n_trials=trials, catch=(Exception,))
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
    if processes <= 1:
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
    best_trial = study.loc[study['value'].idxmax()]

    print("\nBest Parameters:")
    for param_name in best_trial.index:
        if param_name.startswith('params_') and not pd.isna(best_trial[param_name]):
            param_value = best_trial[param_name]
            param_name = param_name[len('params_'):]
            print(f" - {param_name}: {param_value}")

    print(f"\nBest {objective_str}: ", best_trial["value"])
    print("Trial: ", best_trial["number"])

def plot_results(study: Study, plot_func: Callable, params: Optional[Dict] = None, filename: Optional[str] = None) -> None:
    """Plots the results of the Optuna study.

    Args:
        study (Study): The Optuna study object.
        plot_func (Callable): The function to generate the plot, must take "study" and "params" as arguments.
        params (Dict, optional): Additional parameters for the plot function. Defaults to None.
        filename (str, optional): The filename to save the plot. Defaults to None.
    """
    plot_func(study=study, params=params)

    if filename:
        results_dir = "results/optimization/"
        os.makedirs(results_dir, exist_ok=True)
        plt.savefig(os.path.join(results_dir, f"{filename}"), bbox_inches='tight', dpi=800)

    plt.show()
    plt.close()
