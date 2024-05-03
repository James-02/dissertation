from functools import partial
import time
import multiprocessing
from typing import Tuple, List, Dict, Optional

import numpy as np
from reservoirpy import Node
from sklearn.model_selection import KFold

from utils.logger import Logger
from utils.preprocessing import save_npz, augment_data, standardize_data
from utils.analysis import log_params, log_metrics, compute_mean_metrics, evaluate_performance

logger = Logger()

def train(reservoir: Node, X_train: np.ndarray) -> List[np.ndarray]:
    """
    Train the reservoir with the provided input data.

    Args:
        reservoir (Node): Reservoir node.
        X_train (np.ndarray): Training input data.

    Returns:
        List[np.ndarray]: List of trained reservoir states.
    """
    logger.info(f"Training Reservoir of {reservoir.units} nodes with {len(X_train)} instances")
    return [reservoir.run(x, reset=True)[-1, np.newaxis] for x in X_train]

def fit_readout(readout: Node, trained_states: List[np.ndarray], Y_train: np.ndarray) -> None:
    """
    Fit the readout layer using the trained reservoir states and target labels.

    Args:
        readout (Node): Readout node.
        trained_states (List[np.ndarray]): List of trained reservoir states.
        Y_train (np.ndarray): Target labels.
    """
    logger.info(f"Fitting readout layer with {len(trained_states)} states")
    readout.fit(trained_states, Y_train)

def predict(readout: Node, states: np.ndarray) -> List[np.ndarray]:
    """
    Predict target labels using the trained readout and input states.

    Args:
        readout (Node): Readout node.
        states (np.ndarray): Trained states.

    Returns:
        List[np.ndarray]: Predicted target labels.
    """
    logger.info(f"Predicting with {len(states)} instances.")
    return [readout.run(state) for state in states]

def __run_fold(reservoir: Node, readout: Node, X_train_fold: np.ndarray, Y_train_fold: np.ndarray, 
             X_val: np.ndarray, Y_val: np.ndarray, fold_index: int, save_file: Optional[str] = None) -> Dict[str, any]:
    """
    Private method allocated to each process to run a single fold of cross-validation with a copy of the reservoir and readout nodes.

    Args:
        reservoir (Node): Reservoir node.
        readout (Node): Readout node.
        X_train_fold (np.ndarray): Training input data for the fold.
        Y_train_fold (np.ndarray): Training target labels for the fold.
        X_val (np.ndarray): Validation input data.
        Y_val (np.ndarray): Validation target labels.
        fold_index (int): Index of the fold.
        save_file (Optional[str]): File to save the results (default is None).

    Returns:
        Dict[str, any]: Performance metrics for the fold.
    """
    logger.debug(f"Running cross validation fold: {str(fold_index)}")

    reservoir = reservoir.copy()
    readout = readout.copy()

    start_time = time.time()

    train_states = train(reservoir, X_train_fold)
    fit_readout(readout, train_states, Y_train_fold)

    test_states = train(reservoir, X_val)
    Y_pred = predict(readout, test_states)

    end_time = time.time()

    runtime = round(end_time - start_time, 4)
    fold_metrics = evaluate_performance(Y_val, Y_pred, runtime)

    if save_file:
        data = {"model-hypers": {**reservoir.hypers, **readout.hypers}, "train_states": train_states, "test_states": test_states, 
                "Y_train": Y_train_fold, "Y_test": Y_val, "Y_pred": Y_pred, "metrics": fold_metrics}
        save_npz(filename=f"{save_file}-fold-{fold_index}.npz", **data)

    return fold_metrics

def cross_validate(reservoir: Node, readout: Node, X: np.ndarray, Y: np.ndarray, folds: int = 5, 
                   save_file: Optional[str] = None, noise_rate: float = 0, noise_ratio: float = 0) -> Dict[str, any]:
    """
    Perform cross-validation.

    Args:
        reservoir (Node): Reservoir node.
        readout (Node): Readout node.
        X (np.ndarray): Input data.
        Y (np.ndarray): Target labels.
        folds (int): Number of folds (default is 5).
        save_file (Optional[str]): File to save the results (default is None).
        noise_rate (float): Rate of noise augmentation (default is 0).
        noise_ratio (float): Ratio of noise augmentation (default is 0).

    Returns:
        Dict[str, any]: Mean performance metrics across all folds.
    """
    kf = KFold(n_splits=folds, shuffle=True)
    fold_results = []
    fold_metrics = []

    run_fold_partial = partial(__run_fold, reservoir, readout)

    with multiprocessing.Pool(processes=folds) as pool:
        for i, (train_index, val_index) in enumerate(kf.split(X)):
            X_train_fold, X_val = X[train_index], X[val_index]
            Y_train_fold, Y_val = Y[train_index], Y[val_index]

            X_train_fold, Y_train_fold = augment_data(X_train_fold, Y_train_fold, noise_rate, noise_ratio)
            X_train_fold, X_val = standardize_data(X_train_fold, X_val)

            fold_result = pool.apply_async(run_fold_partial, args=(X_train_fold, Y_train_fold, X_val, Y_val, i, save_file))
            fold_results.append(fold_result)

        fold_metrics = [result.get() for result in fold_results]

    return compute_mean_metrics(fold_metrics)

def classify(reservoir: Node, readout: Node, X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray, 
             Y_test: np.ndarray, folds: Optional[int] = None, save_file: Optional[str] = None, 
             noise_rate: float = 0, noise_ratio: float = 0) -> Dict[str, any]:
    """
    Perform classification using reservoir computing, optionally specifying the folds for k-fold cross-validation.

    If `folds` is specified, k-fold cross-validation is performed. In this case, the dataset must be unaugmented,
    and if noise is required, it must be specified here. This is because we concatenate the training and test set here,
    and then apply augmentation and split into training and validation.

    The metrics returned will be averaged over all folds if given.
    If a `save_file` is given, the model hyperparameters, training and testing states, true and predicted labels, and metrics
    are saved to the file.

    Args:
        reservoir (Node): Reservoir node.
        readout (Node): Readout node.
        X_train (np.ndarray): Training input data.
        Y_train (np.ndarray): Training target labels.
        X_test (np.ndarray): Testing input data.
        Y_test (np.ndarray): Testing target labels.
        folds (Optional[int]): Number of folds for cross-validation (default is None).
        save_file (Optional[str]): File to save the results (default is None).
        noise_rate (float): Rate of noise augmentation (default is 0).
        noise_ratio (float): Ratio of noise augmentation (default is 0).

    Returns:
        Dict[str, Any]: Averaged performance metrics.
    """
    # Log model hyper-parameters
    log_params({**reservoir.hypers, **readout.hypers}, title="Reservoir Hyperparameters")

    if folds:
        X = np.concatenate((X_train, X_test), axis=0)
        Y = np.concatenate((Y_train, Y_test), axis=0)
        metrics = cross_validate(reservoir, readout, X, Y, folds=folds, save_file=save_file, noise_rate=noise_rate, noise_ratio=noise_ratio)
    else:
        start_time = time.time()

        train_states = train(reservoir, X_train)
        fit_readout(readout, train_states, Y_train)

        test_states = train(reservoir, X_test)
        Y_pred = predict(readout, test_states)

        end_time = time.time()

        runtime = round(end_time - start_time, 4)
        metrics = evaluate_performance(Y_test, Y_pred, runtime)

        # Save performance metrics to file
        if save_file:
            data = {"model-hypers": {**reservoir.hypers, **readout.hypers}, "train_states": train_states, "Y_train": Y_train, 
                    "test_states": test_states, "Y_test": Y_test, "Y_pred": Y_pred, "metrics": metrics}
            save_npz(filename=save_file + ".npz", **data)

    log_metrics(metrics)
    return metrics
