from typing import List

import numpy as np

from utils.logger import Logger

logger = Logger()

def log_metrics(metrics: dict):
    logger.info("----- Classification Report -----")
    for key, value in metrics.items():
        if isinstance(value, (int, float, str)):
            logger.info(f"{key}: {value:.3f}")
    logger.info("---------------------------------")

def log_params(params, title="Hyper-parameters"):
    logger.debug(f"----- {title} -----")
    for key, value in params.items():
        logger.debug(f"{key}: {value}")
    logger.debug("--------------------------------")

def compute_mean_metrics(metrics_list):
    sum_metrics = {}
    count = len(metrics_list)
    for metric in metrics_list:
        for key, value in metric.items():
            if isinstance(value, (int, float)):
                if key not in sum_metrics:
                    sum_metrics[key] = 0
                sum_metrics[key] += value

    return {key: sum_value / count for key, sum_value in sum_metrics.items()}

def count_labels(Y: List) -> dict:
    # Always use the first row each target, in case the targets are repeated
    Y = np.array([targets[0] for targets in Y])

    # If Y is encoded, revert the encoding
    if len(Y.shape) != 1:
        Y = np.argmax(Y, axis=1)

    return {label: count for label, count in zip(*np.unique(Y, return_counts=True))}


def measure_dataset_deviation(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return mean, std

def measure_class_deviation(X, Y):
    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)
    class_means = {}
    class_stds = {}
    
    # Calculate mean and std for each class
    for label in np.unique(Y):
        indices = np.where(Y == label)[0]
        class_means[label] = round(np.mean(X[indices], axis=0).tolist()[0], 4)
        class_stds[label] = round(np.std(X[indices], axis=0).tolist()[0], 4)

    return class_means, class_stds
