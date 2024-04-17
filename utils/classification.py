from functools import partial
import time
import multiprocessing

import numpy as np
from reservoirpy import Node
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, r2_score, classification_report

from utils.logger import Logger
from utils.preprocessing import save_npz
from utils.analysis import log_params, log_metrics, compute_mean_metrics

logger = Logger()

def train(reservoir: Node, X_train: np.ndarray):
    logger.info(f"Training Reservoir of {reservoir.units} nodes with {len(X_train)} instances")
    return [reservoir.run(x, reset=True)[-1, np.newaxis] for x in X_train]

def fit_readout(readout: Node, trained_states, Y_train):
    logger.info(f"Fitting readout layer with {len(trained_states)} states")
    readout.fit(trained_states, Y_train)

def predict(reservoir: Node, readout: Node, X_test: np.ndarray):
    logger.info(f"Predicting with {len(X_test)} instances.")
    return [readout.run(reservoir.run(x, reset=True)[-1, np.newaxis]) for x in X_test]

def run_fold(reservoir, readout, X_train_fold, Y_train_fold, X_val, Y_val, fold_index, save_file):
    logger.debug(f"Running cross validation fold: {str(fold_index)}")

    # Copy reservoir and readout for each fold
    reservoir = reservoir.copy()
    readout = readout.copy()

    start_time = time.time()
    trained_states = train(reservoir, X_train_fold)
    fit_readout(readout, trained_states, Y_train_fold)
    Y_pred = predict(reservoir, readout, X_val)
    end_time = time.time()

    runtime = round(end_time - start_time, 4)
    fold_metrics = evaluate_performance(Y_val, Y_pred, runtime)

    if save_file:
        data = {"X_train": X_train_fold, "X_test": X_val, "Y_train": Y_train_fold, "Y_test": Y_val,
                "reservoir-hypers": reservoir.hypers, "readout-hypers": readout.hypers,
                "X_trained": trained_states, "Y_predicted": Y_pred, "metrics": fold_metrics}
        save_npz(filename=f"{save_file}-fold-{fold_index}.npz", **data)

    return fold_metrics

def cross_validate(reservoir: Node, readout: Node, X: np.ndarray, Y: np.ndarray,
                   folds: int = 5, shuffle: bool = True, save_file: str = None):
    kf = KFold(n_splits=folds, shuffle=shuffle)
    fold_results = []
    fold_metrics = []

    # Define a partial function with fixed reservoir and readout arguments
    run_fold_partial = partial(run_fold, reservoir, readout)

    with multiprocessing.Pool(processes=folds) as pool:
        # Iterate over the folds and run each fold in parallel
        for i, (train_index, val_index) in enumerate(kf.split(X)):
            X_train_fold, X_val = X[train_index], X[val_index]
            Y_train_fold, Y_val = Y[train_index], Y[val_index]
            fold_result = pool.apply_async(run_fold_partial, args=(X_train_fold, Y_train_fold, X_val, Y_val, i, save_file))
            fold_results.append(fold_result)

        # Get the results from all the folds
        fold_metrics = [result.get() for result in fold_results]

    return compute_mean_metrics(fold_metrics)

def classify(reservoir: Node, readout: Node, X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray, 
             Y_test: np.ndarray, folds: int = None, save_file: str = None):
    
    # log model hyper-parameters
    log_params({**reservoir.hypers, **readout.hypers}, title="Reservoir Hyper-parameters")

    # perform cross validation classification
    if folds:
        X = np.concatenate((X_train, X_test), axis=0)
        Y = np.concatenate((Y_train, Y_test), axis=0)
        metrics = cross_validate(reservoir, readout, X, Y, folds=folds, save_file=save_file)

    # perform train / test classification
    else:
        start_time = time.time()
        trained_states = train(reservoir, X_train)
        fit_readout(readout, trained_states, Y_train)
        Y_pred = predict(reservoir, readout, X_test)
        end_time = time.time()

        runtime = round(end_time - start_time, 4)
        metrics = evaluate_performance(Y_test, Y_pred, runtime)

        # save performance metrics to file
        if save_file:
            data = {"X_train": X_train, "X_test": X_test, "Y_train": Y_train, "Y_test": Y_test,
                    "reservoir-hypers": reservoir.hypers, "readout-hypers": readout.hypers,
                    "X_trained": trained_states, "Y_predicted": Y_pred, "metrics": metrics}
            save_npz(filename=save_file, **data)
    
    log_metrics(metrics)
    return metrics

def evaluate_performance(Y_true: np.ndarray, Y_pred: np.ndarray, time: float):
    logger.info("Calculating model performance metrics")
    Y_true = np.array([np.argmax(y_t) for y_t in Y_true])
    Y_pred = np.array([np.argmax(y_p) for y_p in Y_pred])

    return {
        "runtime": time,
        "accuracy": accuracy_score(Y_true, Y_pred),
        "f1": f1_score(Y_true, Y_pred, average='weighted'),
        "recall": recall_score(Y_true, Y_pred, average='weighted'),
        "precision": precision_score(Y_true, Y_pred, average='weighted'),
        "mse": np.mean((Y_true - Y_pred) ** 2),
        "rmse": np.sqrt(np.mean((Y_true - Y_pred) ** 2)),
        "r2_score": r2_score(Y_true, Y_pred),
        "confusion_matrix": confusion_matrix(Y_true, Y_pred),
        "class_metrics": classification_report(Y_true, Y_pred, output_dict=True)
    }
